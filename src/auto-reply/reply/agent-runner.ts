import crypto from "node:crypto";
import fs from "node:fs";
import { lookupContextTokens } from "../../agents/context.js";
import { DEFAULT_CONTEXT_TOKENS } from "../../agents/defaults.js";
import { resolveMemorySearchConfig } from "../../agents/memory-search.js";
import { resolveModelAuthMode } from "../../agents/model-auth.js";
import { isCliProvider } from "../../agents/model-selection.js";
import { queueEmbeddedPiMessage } from "../../agents/pi-embedded.js";
import { hasNonzeroUsage } from "../../agents/usage.js";
import type { OpenClawConfig } from "../../config/config.js";
import {
  resolveAgentIdFromSessionKey,
  resolveSessionFilePath,
  resolveSessionTranscriptPath,
  type SessionEntry,
  updateSessionStore,
  updateSessionStoreEntry,
} from "../../config/sessions.js";
import type { TypingMode } from "../../config/types.js";
import { resolveHookConfig } from "../../hooks/config.js";
import { emitDiagnosticEvent, isDiagnosticsEnabled } from "../../infra/diagnostic-events.js";
import { enqueueSystemEvent } from "../../infra/system-events.js";
import { getMemorySearchManager } from "../../memory/index.js";
import type { MemorySearchResult } from "../../memory/types.js";
import { PROMPT_REINFORCER_HOOK_KEY } from "../../prompt-reinforcer/policy.js";
import { defaultRuntime } from "../../runtime.js";
import { estimateUsageCost, resolveModelCostConfig } from "../../utils/usage-format.js";
import type { OriginatingChannelType, TemplateContext } from "../templating.js";
import { resolveResponseUsageMode, type VerboseLevel } from "../thinking.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import { runAgentTurnWithFallback } from "./agent-runner-execution.js";
import {
  createShouldEmitToolOutput,
  createShouldEmitToolResult,
  finalizeWithFollowup,
  isAudioPayload,
  signalTypingIfNeeded,
} from "./agent-runner-helpers.js";
import { runMemoryFlushIfNeeded } from "./agent-runner-memory.js";
import { buildReplyPayloads } from "./agent-runner-payloads.js";
import { appendUsageLine, formatResponseUsageLine } from "./agent-runner-utils.js";
import { createAudioAsVoiceBuffer, createBlockReplyPipeline } from "./block-reply-pipeline.js";
import { resolveBlockStreamingCoalescing } from "./block-streaming.js";
import { createFollowupRunner } from "./followup-runner.js";
import {
  auditPostCompactionReads,
  extractReadPaths,
  formatAuditWarning,
  readSessionMessages,
} from "./post-compaction-audit.js";
import { readPostCompactionContext } from "./post-compaction-context.js";
import {
  type PromptReinforcerOutputBlockReason,
  enforcePromptReinforcerOutputWithReport,
} from "./prompt-reinforcer-output-guard.js";
import { enqueueFollowupRun, type FollowupRun, type QueueSettings } from "./queue.js";
import { createReplyToModeFilterForChannel, resolveReplyToMode } from "./reply-threading.js";
import { incrementRunCompactionCount, persistRunSessionUsage } from "./session-run-accounting.js";
import { createTypingSignaler } from "./typing-mode.js";
import type { TypingController } from "./typing.js";

const BLOCK_REPLY_SEND_TIMEOUT_MS = 15_000;
const PROMPT_REINFORCER_RETRY_DEFAULT_MAX_ATTEMPTS = 2;
const PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS = 1;
const PROMPT_REINFORCER_RETRY_MAX_ATTEMPTS = 10;
const AUTO_MEMORY_PREFETCH_MAX_RESULTS = 3;
const AUTO_MEMORY_PREFETCH_SNIPPET_MAX_CHARS = 280;
const UNSCHEDULED_REMINDER_NOTE =
  "Note: I did not schedule a reminder in this turn, so this will not trigger automatically.";
const REMINDER_COMMITMENT_PATTERNS: RegExp[] = [
  /\b(?:i\s*['’]?ll|i will)\s+(?:make sure to\s+)?(?:remember|remind|ping|follow up|follow-up|check back|circle back)\b/i,
  /\b(?:i\s*['’]?ll|i will)\s+(?:set|create|schedule)\s+(?:a\s+)?reminder\b/i,
];

function clampPromptReinforcerAttempts(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return PROMPT_REINFORCER_RETRY_DEFAULT_MAX_ATTEMPTS;
  }
  const rounded = Math.trunc(value);
  return Math.min(
    PROMPT_REINFORCER_RETRY_MAX_ATTEMPTS,
    Math.max(PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS, rounded),
  );
}

function resolvePromptReinforcerLoopSettings(cfg: OpenClawConfig): {
  enabled: boolean;
  maxAttempts: number;
} {
  const hookConfig = resolveHookConfig(cfg, PROMPT_REINFORCER_HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return { enabled: false, maxAttempts: PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS };
  }
  const raw = hookConfig as Record<string, unknown>;
  const enabled = raw.enforceOutput === true;
  return {
    enabled,
    maxAttempts: enabled
      ? clampPromptReinforcerAttempts(raw.enforceMaxPasses)
      : PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS,
  };
}

function isRetryablePromptReinforcerReason(
  reason: PromptReinforcerOutputBlockReason | undefined,
): boolean {
  return reason === "memory_search_required" || reason === "policy_guard_blocked";
}

function buildPromptReinforcerRetryInstruction(params: {
  reason: PromptReinforcerOutputBlockReason;
  attempt: number;
  maxAttempts: number;
}): string {
  const prefix = `Prompt policy auto-retry ${params.attempt}/${params.maxAttempts}.`;
  if (params.reason === "memory_search_required") {
    return `${prefix} Previous draft was blocked because memory recall was required. In this retry, you MUST run memory_search first, then answer from retrieved memory evidence.`;
  }
  return `${prefix} Previous draft violated output policy or hard constraints. Regenerate a fully compliant response and do not emit any fail-closed placeholder text.`;
}

function hasUnbackedReminderCommitment(text: string): boolean {
  const normalized = text.toLowerCase();
  if (!normalized.trim()) {
    return false;
  }
  if (normalized.includes(UNSCHEDULED_REMINDER_NOTE.toLowerCase())) {
    return false;
  }
  return REMINDER_COMMITMENT_PATTERNS.some((pattern) => pattern.test(text));
}

function appendUnscheduledReminderNote(payloads: ReplyPayload[]): ReplyPayload[] {
  let appended = false;
  return payloads.map((payload) => {
    if (appended || payload.isError || typeof payload.text !== "string") {
      return payload;
    }
    if (!hasUnbackedReminderCommitment(payload.text)) {
      return payload;
    }
    appended = true;
    const trimmed = payload.text.trimEnd();
    return {
      ...payload,
      text: `${trimmed}\n\n${UNSCHEDULED_REMINDER_NOTE}`,
    };
  });
}

function summarizeReplyPayloadsForRawLog(payloads: ReplyPayload[]) {
  return payloads.map((payload, index) => ({
    index: index + 1,
    isError: Boolean(payload.isError),
    text: typeof payload.text === "string" ? payload.text : "",
    mediaUrl: payload.mediaUrl ?? null,
  }));
}

function logReplyRaw(stage: string, payload: Record<string, unknown>) {
  try {
    defaultRuntime.log(
      `[reply-raw] ${JSON.stringify({
        stage,
        ...payload,
      })}`,
    );
  } catch (err) {
    defaultRuntime.error(`[reply-raw] log failed: stage=${stage} error=${String(err)}`);
  }
}

function compactSingleLine(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function truncateChars(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, maxChars)}...`;
}

function formatMemoryCitation(entry: MemorySearchResult): string {
  const linePart =
    entry.startLine === entry.endLine
      ? `#L${entry.startLine}`
      : `#L${entry.startLine}-L${entry.endLine}`;
  return `${entry.path}${linePart}`;
}

function buildAutoMemoryPrefetchPromptBlock(query: string, results: MemorySearchResult[]): string {
  const lines: string[] = [
    "Runtime auto-prefetch executed memory_search because output guard required memory recall in the previous attempt.",
    `Query: ${compactSingleLine(query)}`,
  ];
  if (results.length === 0) {
    lines.push("Memory search returned no matches.");
  } else {
    lines.push("Memory search results:");
    for (const [index, entry] of results.entries()) {
      const snippet = truncateChars(
        compactSingleLine(entry.snippet ?? ""),
        AUTO_MEMORY_PREFETCH_SNIPPET_MAX_CHARS,
      );
      lines.push(`${index + 1}. ${formatMemoryCitation(entry)} :: ${snippet}`);
    }
  }
  lines.push(
    "Treat this as the memory_search evidence for this turn; do not skip policy compliance.",
  );
  return lines.join("\n");
}

type AutoMemoryPrefetchResult = {
  satisfied: boolean;
  reason: "ok" | "memory_disabled" | "manager_unavailable" | "search_failed";
  resultCount: number;
  promptBlock?: string;
  error?: string;
};

async function runAutoMemoryPrefetch(params: {
  cfg: OpenClawConfig;
  agentId: string;
  sessionKey?: string;
  query: string;
}): Promise<AutoMemoryPrefetchResult> {
  if (!resolveMemorySearchConfig(params.cfg, params.agentId)) {
    return { satisfied: false, reason: "memory_disabled", resultCount: 0 };
  }
  const { manager, error } = await getMemorySearchManager({
    cfg: params.cfg,
    agentId: params.agentId,
  });
  if (!manager) {
    return {
      satisfied: false,
      reason: "manager_unavailable",
      resultCount: 0,
      error,
    };
  }
  try {
    const results = await manager.search(params.query, {
      sessionKey: params.sessionKey,
      maxResults: AUTO_MEMORY_PREFETCH_MAX_RESULTS,
    });
    return {
      satisfied: true,
      reason: "ok",
      resultCount: results.length,
      promptBlock: buildAutoMemoryPrefetchPromptBlock(params.query, results),
    };
  } catch (err) {
    return {
      satisfied: false,
      reason: "search_failed",
      resultCount: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// Track sessions pending post-compaction read audit (Layer 3)
const pendingPostCompactionAudits = new Map<string, boolean>();

export async function runReplyAgent(params: {
  commandBody: string;
  followupRun: FollowupRun;
  queueKey: string;
  resolvedQueue: QueueSettings;
  shouldSteer: boolean;
  shouldFollowup: boolean;
  isActive: boolean;
  isStreaming: boolean;
  opts?: GetReplyOptions;
  typing: TypingController;
  sessionEntry?: SessionEntry;
  sessionStore?: Record<string, SessionEntry>;
  sessionKey?: string;
  storePath?: string;
  defaultModel: string;
  agentCfgContextTokens?: number;
  resolvedVerboseLevel: VerboseLevel;
  isNewSession: boolean;
  blockStreamingEnabled: boolean;
  blockReplyChunking?: {
    minChars: number;
    maxChars: number;
    breakPreference: "paragraph" | "newline" | "sentence";
    flushOnParagraph?: boolean;
  };
  resolvedBlockStreamingBreak: "text_end" | "message_end";
  sessionCtx: TemplateContext;
  shouldInjectGroupIntro: boolean;
  typingMode: TypingMode;
}): Promise<ReplyPayload | ReplyPayload[] | undefined> {
  const {
    commandBody,
    followupRun,
    queueKey,
    resolvedQueue,
    shouldSteer,
    shouldFollowup,
    isActive,
    isStreaming,
    opts,
    typing,
    sessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    defaultModel,
    agentCfgContextTokens,
    resolvedVerboseLevel,
    isNewSession,
    blockStreamingEnabled,
    blockReplyChunking,
    resolvedBlockStreamingBreak,
    sessionCtx,
    shouldInjectGroupIntro,
    typingMode,
  } = params;

  let activeSessionEntry = sessionEntry;
  const activeSessionStore = sessionStore;
  let activeIsNewSession = isNewSession;

  const cfg = followupRun.run.config;
  const promptReinforcerLoop = resolvePromptReinforcerLoopSettings(cfg);
  const effectiveOpts =
    promptReinforcerLoop.enabled && opts
      ? {
          ...opts,
          // Do not stream unchecked output when output guard is active.
          onPartialReply: undefined,
          onBlockReply: undefined,
          onToolResult: undefined,
        }
      : opts;
  const effectiveBlockStreamingEnabled = promptReinforcerLoop.enabled
    ? false
    : blockStreamingEnabled;

  const isHeartbeat = effectiveOpts?.isHeartbeat === true;
  const typingSignals = createTypingSignaler({
    typing,
    mode: typingMode,
    isHeartbeat,
  });

  const shouldEmitToolResult = createShouldEmitToolResult({
    sessionKey,
    storePath,
    resolvedVerboseLevel,
  });
  const shouldEmitToolOutput = createShouldEmitToolOutput({
    sessionKey,
    storePath,
    resolvedVerboseLevel,
  });

  const pendingToolTasks = new Set<Promise<void>>();
  const blockReplyTimeoutMs = effectiveOpts?.blockReplyTimeoutMs ?? BLOCK_REPLY_SEND_TIMEOUT_MS;

  const replyToChannel =
    sessionCtx.OriginatingChannel ??
    ((sessionCtx.Surface ?? sessionCtx.Provider)?.toLowerCase() as
      | OriginatingChannelType
      | undefined);
  const replyToMode = resolveReplyToMode(
    followupRun.run.config,
    replyToChannel,
    sessionCtx.AccountId,
    sessionCtx.ChatType,
  );
  const applyReplyToMode = createReplyToModeFilterForChannel(replyToMode, replyToChannel);
  const blockReplyCoalescing =
    effectiveBlockStreamingEnabled && effectiveOpts?.onBlockReply
      ? resolveBlockStreamingCoalescing(
          cfg,
          sessionCtx.Provider,
          sessionCtx.AccountId,
          blockReplyChunking,
        )
      : undefined;
  const blockReplyPipeline =
    effectiveBlockStreamingEnabled && effectiveOpts?.onBlockReply
      ? createBlockReplyPipeline({
          onBlockReply: effectiveOpts.onBlockReply,
          timeoutMs: blockReplyTimeoutMs,
          coalescing: blockReplyCoalescing,
          buffer: createAudioAsVoiceBuffer({ isAudioPayload }),
        })
      : null;
  const touchActiveSessionEntry = async () => {
    if (!activeSessionEntry || !activeSessionStore || !sessionKey) {
      return;
    }
    const updatedAt = Date.now();
    activeSessionEntry.updatedAt = updatedAt;
    activeSessionStore[sessionKey] = activeSessionEntry;
    if (storePath) {
      await updateSessionStoreEntry({
        storePath,
        sessionKey,
        update: async () => ({ updatedAt }),
      });
    }
  };

  if (shouldSteer && isStreaming) {
    const steered = queueEmbeddedPiMessage(followupRun.run.sessionId, followupRun.prompt);
    if (steered && !shouldFollowup) {
      await touchActiveSessionEntry();
      typing.cleanup();
      return undefined;
    }
  }

  if (isActive && (shouldFollowup || resolvedQueue.mode === "steer")) {
    enqueueFollowupRun(queueKey, followupRun, resolvedQueue);
    await touchActiveSessionEntry();
    typing.cleanup();
    return undefined;
  }

  await typingSignals.signalRunStart();

  activeSessionEntry = await runMemoryFlushIfNeeded({
    cfg,
    followupRun,
    sessionCtx,
    opts,
    defaultModel,
    agentCfgContextTokens,
    resolvedVerboseLevel,
    sessionEntry: activeSessionEntry,
    sessionStore: activeSessionStore,
    sessionKey,
    storePath,
    isHeartbeat,
  });

  const runFollowupTurn = createFollowupRunner({
    opts,
    typing,
    typingMode,
    sessionEntry: activeSessionEntry,
    sessionStore: activeSessionStore,
    sessionKey,
    storePath,
    defaultModel,
    agentCfgContextTokens,
  });

  let responseUsageLine: string | undefined;
  type SessionResetOptions = {
    failureLabel: string;
    buildLogMessage: (nextSessionId: string) => string;
    cleanupTranscripts?: boolean;
  };
  const resetSession = async ({
    failureLabel,
    buildLogMessage,
    cleanupTranscripts,
  }: SessionResetOptions): Promise<boolean> => {
    if (!sessionKey || !activeSessionStore || !storePath) {
      return false;
    }
    const prevEntry = activeSessionStore[sessionKey] ?? activeSessionEntry;
    if (!prevEntry) {
      return false;
    }
    const prevSessionId = cleanupTranscripts ? prevEntry.sessionId : undefined;
    const nextSessionId = crypto.randomUUID();
    const nextEntry: SessionEntry = {
      ...prevEntry,
      sessionId: nextSessionId,
      updatedAt: Date.now(),
      systemSent: false,
      abortedLastRun: false,
    };
    const agentId = resolveAgentIdFromSessionKey(sessionKey);
    const nextSessionFile = resolveSessionTranscriptPath(
      nextSessionId,
      agentId,
      sessionCtx.MessageThreadId,
    );
    nextEntry.sessionFile = nextSessionFile;
    activeSessionStore[sessionKey] = nextEntry;
    try {
      await updateSessionStore(storePath, (store) => {
        store[sessionKey] = nextEntry;
      });
    } catch (err) {
      defaultRuntime.error(
        `Failed to persist session reset after ${failureLabel} (${sessionKey}): ${String(err)}`,
      );
    }
    followupRun.run.sessionId = nextSessionId;
    followupRun.run.sessionFile = nextSessionFile;
    activeSessionEntry = nextEntry;
    activeIsNewSession = true;
    defaultRuntime.error(buildLogMessage(nextSessionId));
    if (cleanupTranscripts && prevSessionId) {
      const transcriptCandidates = new Set<string>();
      const resolved = resolveSessionFilePath(prevSessionId, prevEntry, { agentId });
      if (resolved) {
        transcriptCandidates.add(resolved);
      }
      transcriptCandidates.add(resolveSessionTranscriptPath(prevSessionId, agentId));
      for (const candidate of transcriptCandidates) {
        try {
          fs.unlinkSync(candidate);
        } catch {
          // Best-effort cleanup.
        }
      }
    }
    return true;
  };
  const resetSessionAfterCompactionFailure = async (reason: string): Promise<boolean> =>
    resetSession({
      failureLabel: "compaction failure",
      buildLogMessage: (nextSessionId) =>
        `Auto-compaction failed (${reason}). Restarting session ${sessionKey} -> ${nextSessionId} and retrying.`,
    });
  const resetSessionAfterRoleOrderingConflict = async (reason: string): Promise<boolean> =>
    resetSession({
      failureLabel: "role ordering conflict",
      buildLogMessage: (nextSessionId) =>
        `Role ordering conflict (${reason}). Restarting session ${sessionKey} -> ${nextSessionId}.`,
      cleanupTranscripts: true,
    });
  const baseExtraSystemPrompt = followupRun.run.extraSystemPrompt;
  let lastGuardRetryReason: PromptReinforcerOutputBlockReason | undefined;
  let pendingAutoMemoryPrefetch: {
    promptBlock?: string;
    markUsedTool: boolean;
  } | null = null;
  try {
    for (let attempt = 1; attempt <= promptReinforcerLoop.maxAttempts; attempt += 1) {
      responseUsageLine = undefined;
      const autoMemoryPrefetchForAttempt = pendingAutoMemoryPrefetch;
      pendingAutoMemoryPrefetch = null;
      const extraPromptParts: string[] = [];
      if (baseExtraSystemPrompt && baseExtraSystemPrompt.trim()) {
        extraPromptParts.push(baseExtraSystemPrompt);
      }
      if (promptReinforcerLoop.enabled && lastGuardRetryReason) {
        const retryInstruction = buildPromptReinforcerRetryInstruction({
          reason: lastGuardRetryReason,
          attempt,
          maxAttempts: promptReinforcerLoop.maxAttempts,
        });
        extraPromptParts.push(retryInstruction);
      }
      if (autoMemoryPrefetchForAttempt?.promptBlock?.trim()) {
        extraPromptParts.push(autoMemoryPrefetchForAttempt.promptBlock.trim());
      }
      followupRun.run.extraSystemPrompt = extraPromptParts.join("\n\n");
      logReplyRaw("attempt_input", {
        attempt,
        maxAttempts: promptReinforcerLoop.maxAttempts,
        sessionKey: sessionKey ?? "unknown",
        userPrompt: followupRun.prompt,
        autoMemoryPrefetched: Boolean(autoMemoryPrefetchForAttempt),
      });

      const runStartedAt = Date.now();
      const runOutcome = await runAgentTurnWithFallback({
        commandBody,
        followupRun,
        sessionCtx,
        opts: effectiveOpts,
        typingSignals,
        blockReplyPipeline,
        blockStreamingEnabled: effectiveBlockStreamingEnabled,
        blockReplyChunking,
        resolvedBlockStreamingBreak,
        applyReplyToMode,
        shouldEmitToolResult,
        shouldEmitToolOutput,
        pendingToolTasks,
        resetSessionAfterCompactionFailure,
        resetSessionAfterRoleOrderingConflict,
        isHeartbeat,
        sessionKey,
        getActiveSessionEntry: () => activeSessionEntry,
        activeSessionStore,
        storePath,
        resolvedVerboseLevel,
      });

      if (runOutcome.kind === "final") {
        return finalizeWithFollowup(runOutcome.payload, queueKey, runFollowupTurn);
      }

      const { runResult, fallbackProvider, fallbackModel, directlySentBlockKeys } = runOutcome;
      let { didLogHeartbeatStrip, autoCompactionCompleted } = runOutcome;

      if (
        shouldInjectGroupIntro &&
        activeSessionEntry &&
        activeSessionStore &&
        sessionKey &&
        activeSessionEntry.groupActivationNeedsSystemIntro
      ) {
        const updatedAt = Date.now();
        activeSessionEntry.groupActivationNeedsSystemIntro = false;
        activeSessionEntry.updatedAt = updatedAt;
        activeSessionStore[sessionKey] = activeSessionEntry;
        if (storePath) {
          await updateSessionStoreEntry({
            storePath,
            sessionKey,
            update: async () => ({
              groupActivationNeedsSystemIntro: false,
              updatedAt,
            }),
          });
        }
      }

      const payloadArray = runResult.payloads ?? [];

      if (blockReplyPipeline) {
        await blockReplyPipeline.flush({ force: true });
        blockReplyPipeline.stop();
      }
      if (pendingToolTasks.size > 0) {
        await Promise.allSettled(pendingToolTasks);
      }

      const usage = runResult.meta?.agentMeta?.usage;
      const promptTokens = runResult.meta?.agentMeta?.promptTokens;
      const modelUsed = runResult.meta?.agentMeta?.model ?? fallbackModel ?? defaultModel;
      const providerUsed =
        runResult.meta?.agentMeta?.provider ?? fallbackProvider ?? followupRun.run.provider;
      const cliSessionId = isCliProvider(providerUsed, cfg)
        ? runResult.meta?.agentMeta?.sessionId?.trim()
        : undefined;
      const contextTokensUsed =
        agentCfgContextTokens ??
        lookupContextTokens(modelUsed) ??
        activeSessionEntry?.contextTokens ??
        DEFAULT_CONTEXT_TOKENS;

      await persistRunSessionUsage({
        storePath,
        sessionKey,
        usage,
        lastCallUsage: runResult.meta?.agentMeta?.lastCallUsage,
        promptTokens,
        modelUsed,
        providerUsed,
        contextTokensUsed,
        systemPromptReport: runResult.meta?.systemPromptReport,
        cliSessionId,
      });

      // Drain any late tool/block deliveries before deciding there's "nothing to send".
      // Otherwise, a late typing trigger (e.g. from a tool callback) can outlive the run and
      // keep the typing indicator stuck.
      if (payloadArray.length === 0) {
        return finalizeWithFollowup(undefined, queueKey, runFollowupTurn);
      }

      const payloadResult = buildReplyPayloads({
        payloads: payloadArray,
        isHeartbeat,
        didLogHeartbeatStrip,
        blockStreamingEnabled: effectiveBlockStreamingEnabled,
        blockReplyPipeline,
        directlySentBlockKeys,
        replyToMode,
        replyToChannel,
        currentMessageId: sessionCtx.MessageSidFull ?? sessionCtx.MessageSid,
        messageProvider: followupRun.run.messageProvider,
        messagingToolSentTexts: runResult.messagingToolSentTexts,
        messagingToolSentMediaUrls: runResult.messagingToolSentMediaUrls,
        messagingToolSentTargets: runResult.messagingToolSentTargets,
        originatingTo: sessionCtx.OriginatingTo ?? sessionCtx.To,
        accountId: sessionCtx.AccountId,
      });
      const { replyPayloads } = payloadResult;
      didLogHeartbeatStrip = payloadResult.didLogHeartbeatStrip;

      if (replyPayloads.length === 0) {
        return finalizeWithFollowup(undefined, queueKey, runFollowupTurn);
      }

      const successfulCronAdds = runResult.successfulCronAdds ?? 0;
      const hasReminderCommitment = replyPayloads.some(
        (payload) =>
          !payload.isError &&
          typeof payload.text === "string" &&
          hasUnbackedReminderCommitment(payload.text),
      );
      const guardedReplyPayloads =
        hasReminderCommitment && successfulCronAdds === 0
          ? appendUnscheduledReminderNote(replyPayloads)
          : replyPayloads;

      await signalTypingIfNeeded(guardedReplyPayloads, typingSignals);

      if (isDiagnosticsEnabled(cfg) && hasNonzeroUsage(usage)) {
        const input = usage.input ?? 0;
        const output = usage.output ?? 0;
        const cacheRead = usage.cacheRead ?? 0;
        const cacheWrite = usage.cacheWrite ?? 0;
        const promptTokens = input + cacheRead + cacheWrite;
        const totalTokens = usage.total ?? promptTokens + output;
        const costConfig = resolveModelCostConfig({
          provider: providerUsed,
          model: modelUsed,
          config: cfg,
        });
        const costUsd = estimateUsageCost({ usage, cost: costConfig });
        emitDiagnosticEvent({
          type: "model.usage",
          sessionKey,
          sessionId: followupRun.run.sessionId,
          channel: replyToChannel,
          provider: providerUsed,
          model: modelUsed,
          usage: {
            input,
            output,
            cacheRead,
            cacheWrite,
            promptTokens,
            total: totalTokens,
          },
          lastCallUsage: runResult.meta?.agentMeta?.lastCallUsage,
          context: {
            limit: contextTokensUsed,
            used: totalTokens,
          },
          costUsd,
          durationMs: Date.now() - runStartedAt,
        });
      }

      const responseUsageRaw =
        activeSessionEntry?.responseUsage ??
        (sessionKey ? activeSessionStore?.[sessionKey]?.responseUsage : undefined);
      const responseUsageMode = resolveResponseUsageMode(responseUsageRaw);
      if (responseUsageMode !== "off" && hasNonzeroUsage(usage)) {
        const authMode = resolveModelAuthMode(providerUsed, cfg);
        const showCost = authMode === "api-key";
        const costConfig = showCost
          ? resolveModelCostConfig({
              provider: providerUsed,
              model: modelUsed,
              config: cfg,
            })
          : undefined;
        let formatted = formatResponseUsageLine({
          usage,
          showCost,
          costConfig,
        });
        if (formatted && responseUsageMode === "full" && sessionKey) {
          formatted = `${formatted} · session ${sessionKey}`;
        }
        if (formatted) {
          responseUsageLine = formatted;
        }
      }

      // If verbose is enabled and this is a new session, prepend a session hint.
      let finalPayloads = guardedReplyPayloads;
      const verboseEnabled = resolvedVerboseLevel !== "off";
      if (autoCompactionCompleted) {
        const count = await incrementRunCompactionCount({
          sessionEntry: activeSessionEntry,
          sessionStore: activeSessionStore,
          sessionKey,
          storePath,
          lastCallUsage: runResult.meta?.agentMeta?.lastCallUsage,
          contextTokensUsed,
        });

        // Inject post-compaction workspace context for the next agent turn
        if (sessionKey) {
          const workspaceDir = process.cwd();
          readPostCompactionContext(workspaceDir)
            .then((contextContent) => {
              if (contextContent) {
                enqueueSystemEvent(contextContent, { sessionKey });
              }
            })
            .catch(() => {
              // Silent failure — post-compaction context is best-effort
            });

          // Set pending audit flag for Layer 3 (post-compaction read audit)
          pendingPostCompactionAudits.set(sessionKey, true);
        }

        if (verboseEnabled) {
          const suffix = typeof count === "number" ? ` (count ${count})` : "";
          finalPayloads = [{ text: `🧹 Auto-compaction complete${suffix}.` }, ...finalPayloads];
        }
      }
      if (verboseEnabled && activeIsNewSession) {
        finalPayloads = [
          { text: `🧭 New session: ${followupRun.run.sessionId}` },
          ...finalPayloads,
        ];
      }
      if (responseUsageLine) {
        finalPayloads = appendUsageLine(finalPayloads, responseUsageLine);
      }
      logReplyRaw("attempt_output_pre_guard", {
        attempt,
        maxAttempts: promptReinforcerLoop.maxAttempts,
        sessionKey: sessionKey ?? "unknown",
        payloads: summarizeReplyPayloadsForRawLog(finalPayloads),
      });
      const usedToolNames = [...(runResult.meta?.usedTools ?? [])];
      if (
        autoMemoryPrefetchForAttempt?.markUsedTool &&
        !usedToolNames.some((name) => name === "memory_search")
      ) {
        usedToolNames.push("memory_search");
      }
      const guardResult = await enforcePromptReinforcerOutputWithReport({
        payloads: finalPayloads,
        cfg,
        workspaceDir: followupRun.run.workspaceDir,
        agentDir: followupRun.run.agentDir,
        provider: providerUsed,
        model: modelUsed,
        authProfileId: followupRun.run.authProfileId,
        latestUserPrompt: followupRun.prompt,
        usedToolNames,
      });
      finalPayloads = guardResult.payloads;
      logReplyRaw("attempt_output_post_guard", {
        attempt,
        maxAttempts: promptReinforcerLoop.maxAttempts,
        sessionKey: sessionKey ?? "unknown",
        blocked: guardResult.report.blocked,
        blockReason: guardResult.report.reason ?? null,
        payloads: summarizeReplyPayloadsForRawLog(finalPayloads),
      });

      if (
        promptReinforcerLoop.enabled &&
        guardResult.report.blocked &&
        isRetryablePromptReinforcerReason(guardResult.report.reason) &&
        attempt < promptReinforcerLoop.maxAttempts
      ) {
        if (guardResult.report.reason === "memory_search_required") {
          const autoMemoryPrefetch = await runAutoMemoryPrefetch({
            cfg,
            agentId: followupRun.run.agentId || "main",
            sessionKey,
            query: followupRun.prompt,
          });
          logReplyRaw("memory_prefetch", {
            attempt,
            maxAttempts: promptReinforcerLoop.maxAttempts,
            sessionKey: sessionKey ?? "unknown",
            satisfied: autoMemoryPrefetch.satisfied,
            reason: autoMemoryPrefetch.reason,
            resultCount: autoMemoryPrefetch.resultCount,
            error: autoMemoryPrefetch.error ?? null,
          });
          if (autoMemoryPrefetch.satisfied) {
            pendingAutoMemoryPrefetch = {
              promptBlock: autoMemoryPrefetch.promptBlock,
              markUsedTool: true,
            };
          } else {
            defaultRuntime.error(
              `[prompt-reinforcer] output guard retry aborted: reason=memory_search_required prefetch=${autoMemoryPrefetch.reason} error=${autoMemoryPrefetch.error ?? "-"}`,
            );
            lastGuardRetryReason = undefined;
            return finalizeWithFollowup(
              finalPayloads.length === 1 ? finalPayloads[0] : finalPayloads,
              queueKey,
              runFollowupTurn,
            );
          }
        }
        lastGuardRetryReason = guardResult.report.reason;
        defaultRuntime.error(
          `[prompt-reinforcer] output guard retry: attempt=${attempt + 1}/${promptReinforcerLoop.maxAttempts} reason=${guardResult.report.reason}`,
        );
        continue;
      }

      // Post-compaction read audit (Layer 3)
      if (sessionKey && pendingPostCompactionAudits.get(sessionKey)) {
        pendingPostCompactionAudits.delete(sessionKey); // Delete FIRST — one-shot only
        try {
          const sessionFile = activeSessionEntry?.sessionFile;
          if (sessionFile) {
            const messages = readSessionMessages(sessionFile);
            const readPaths = extractReadPaths(messages);
            const workspaceDir = process.cwd();
            const audit = auditPostCompactionReads(readPaths, workspaceDir);
            if (!audit.passed) {
              enqueueSystemEvent(formatAuditWarning(audit.missingPatterns), { sessionKey });
            }
          }
        } catch {
          // Silent failure — audit is best-effort
        }
      }

      return finalizeWithFollowup(
        finalPayloads.length === 1 ? finalPayloads[0] : finalPayloads,
        queueKey,
        runFollowupTurn,
      );
    }
    return finalizeWithFollowup(undefined, queueKey, runFollowupTurn);
  } finally {
    followupRun.run.extraSystemPrompt = baseExtraSystemPrompt;
    blockReplyPipeline?.stop();
    typing.markRunComplete();
  }
}
