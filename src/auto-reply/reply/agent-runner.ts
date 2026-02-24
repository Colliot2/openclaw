import crypto from "node:crypto";
import fs from "node:fs";
import { lookupContextTokens } from "../../agents/context.js";
import { DEFAULT_CONTEXT_TOKENS } from "../../agents/defaults.js";
import { resolveMemorySearchConfig } from "../../agents/memory-search.js";
import { resolveModelAuthMode } from "../../agents/model-auth.js";
import { isCliProvider } from "../../agents/model-selection.js";
import { queueEmbeddedPiMessage } from "../../agents/pi-embedded.js";
import { spawnSubagentDirect } from "../../agents/subagent-spawn.js";
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
const PROMPT_REINFORCER_POLICY_RETRY_DEFAULT_MAX_ATTEMPTS = 1;
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

type PromptReinforcerRetryPolicy = {
  maxAttempts: number;
  failOpen: boolean;
};

type PromptReinforcerFailureMitigation = "none" | "auto_memory_prefetch";

type PromptReinforcerReasonRule = {
  reason: PromptReinforcerOutputBlockReason;
  policy: PromptReinforcerRetryPolicy;
  mitigation: PromptReinforcerFailureMitigation;
  retryInstructionBuilder: (params: {
    attempt: number;
    maxAttempts: number;
    detail?: string;
  }) => string;
};

function buildPromptReinforcerRetryInstruction(params: {
  reason: PromptReinforcerOutputBlockReason;
  attempt: number;
  maxAttempts: number;
  detail?: string;
}): string {
  const prefix = `Prompt policy auto-retry ${params.attempt}/${params.maxAttempts}.`;
  const detailSuffix = params.detail?.trim() ? ` Violation detail: ${params.detail.trim()}` : "";
  if (params.reason === "memory_search_required") {
    return `${prefix} Previous draft was blocked because memory recall was required. In this retry, you MUST run memory_search first, then answer from retrieved memory evidence.${detailSuffix}`;
  }
  if (params.reason === "hard_constraints_blocked") {
    return `${prefix} Previous draft violated hard constraints. Regenerate a response that is semantically consistent with hard constraints and contains no contradictions.${detailSuffix}`;
  }
  if (params.reason === "soft_policy_blocked") {
    return `${prefix} Previous draft violated soft policy and soft fail-open is disabled. Regenerate a soft-policy compliant response.${detailSuffix}`;
  }
  if (params.reason === "guard_unavailable") {
    return `${prefix} Guard model was unavailable in the previous attempt. Regenerate a best-effort compliant response without placeholder block text.${detailSuffix}`;
  }
  return `${prefix} Previous draft violated output policy or hard constraints. Regenerate a fully compliant response and do not emit any fail-closed placeholder text.${detailSuffix}`;
}

function resolvePromptReinforcerLoopSettings(cfg: OpenClawConfig): {
  enabled: boolean;
  maxAttempts: number;
  rules: Partial<Record<PromptReinforcerOutputBlockReason, PromptReinforcerReasonRule>>;
} {
  const hookConfig = resolveHookConfig(cfg, PROMPT_REINFORCER_HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return {
      enabled: false,
      maxAttempts: PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS,
      rules: {},
    };
  }
  const raw = hookConfig as Record<string, unknown>;
  const enabled = raw.enforceOutput === true;
  const memoryPolicy: PromptReinforcerRetryPolicy = {
    maxAttempts: clampPromptReinforcerAttempts(raw.enforceMemoryRetryMaxAttempts),
    failOpen: raw.enforceMemoryRetryFailOpen !== false,
  };
  const policyPolicy: PromptReinforcerRetryPolicy = {
    maxAttempts: clampPromptReinforcerAttempts(
      raw.enforcePolicyRetryMaxAttempts ?? PROMPT_REINFORCER_POLICY_RETRY_DEFAULT_MAX_ATTEMPTS,
    ),
    failOpen: raw.enforcePolicyRetryFailOpen === true,
  };
  const rules: Partial<Record<PromptReinforcerOutputBlockReason, PromptReinforcerReasonRule>> = {
    memory_search_required: {
      reason: "memory_search_required",
      policy: memoryPolicy,
      mitigation: "auto_memory_prefetch",
      retryInstructionBuilder: ({ attempt, maxAttempts }) =>
        buildPromptReinforcerRetryInstruction({
          reason: "memory_search_required",
          attempt,
          maxAttempts,
        }),
    },
    soft_policy_blocked: {
      reason: "soft_policy_blocked",
      policy: policyPolicy,
      mitigation: "none",
      retryInstructionBuilder: ({ attempt, maxAttempts }) =>
        buildPromptReinforcerRetryInstruction({
          reason: "soft_policy_blocked",
          attempt,
          maxAttempts,
        }),
    },
    hard_constraints_blocked: {
      reason: "hard_constraints_blocked",
      policy: policyPolicy,
      mitigation: "none",
      retryInstructionBuilder: ({ attempt, maxAttempts }) =>
        buildPromptReinforcerRetryInstruction({
          reason: "hard_constraints_blocked",
          attempt,
          maxAttempts,
        }),
    },
    policy_guard_blocked: {
      reason: "policy_guard_blocked",
      policy: policyPolicy,
      mitigation: "none",
      retryInstructionBuilder: ({ attempt, maxAttempts }) =>
        buildPromptReinforcerRetryInstruction({
          reason: "policy_guard_blocked",
          attempt,
          maxAttempts,
        }),
    },
    guard_unavailable: {
      reason: "guard_unavailable",
      policy: {
        maxAttempts: PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS,
        failOpen: false,
      },
      mitigation: "none",
      retryInstructionBuilder: ({ attempt, maxAttempts }) =>
        buildPromptReinforcerRetryInstruction({
          reason: "guard_unavailable",
          attempt,
          maxAttempts,
        }),
    },
  };
  const maxAttempts = Math.max(
    PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS,
    ...Object.values(rules).map((rule) => rule.policy.maxAttempts),
  );
  return {
    enabled,
    maxAttempts: enabled ? maxAttempts : PROMPT_REINFORCER_RETRY_MIN_ATTEMPTS,
    rules: enabled ? rules : {},
  };
}

function getPromptReinforcerReasonRule(
  settings: {
    enabled: boolean;
    rules: Partial<Record<PromptReinforcerOutputBlockReason, PromptReinforcerReasonRule>>;
  },
  reason: PromptReinforcerOutputBlockReason | undefined,
): PromptReinforcerReasonRule | null {
  if (!settings.enabled || !reason) {
    return null;
  }
  return settings.rules[reason] ?? null;
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

const RAW_SPAWN_COMMITMENT_RE =
  /(任务锁定|后续.*(只发|仅发).*(进度|终稿|结果)|不达标不发|已按.*(单开|启动).*(子进程|子任务|后台任务)|(?:单开|单独进程|spawn(?:ed)?|start(?:ed)?).{0,12}(?:子进程|子任务|subagent|sub-agent|background))/i;
const RAW_PROGRESS_COMMITMENT_RE =
  /(当前达标计数|已证实报道\s*[:：]\s*\d+\s*\/\s*\d+|已证实评论\s*[:：]\s*\d+\s*\/\s*\d+|在达到门槛前.*(?:不交付|不提供).*(?:终稿|结果))/i;
const RAW_SPAWN_NEGATION_RE =
  /(active\s+subagents\s*:\s*\(none\)|active\s*=\s*none|没有任何子进程在运行|未启动|未运行|不会启动|不启动|没人还在跑)/i;
const RAW_SPAWN_TASK_RE = /(?:^|\n)\s*(?:[-•*]\s*)?(?:任务|task)\s*[:：]\s*`?([^`\n]+)`?/i;
const RAW_SPAWN_LABEL_RE = /(?:^|\n)\s*(?:[-•*]\s*)?(?:任务标签|label)\s*[:：]\s*`?([^`\n]+)`?/i;

type AutoSpawnExecutionPlan = {
  task: string;
  label?: string;
  reason: "raw_commitment" | "progress_commitment";
};

type AutoSpawnExecutionResult = {
  notice: string;
  spawned: boolean;
};

function hasSessionsSpawnCall(usedToolNames: string[]): boolean {
  return usedToolNames.some((name) => name.trim().toLowerCase() === "sessions_spawn");
}

function extractTextFromReplyPayloads(payloads: ReplyPayload[]): string {
  return payloads
    .filter((payload) => !payload.isError && typeof payload.text === "string")
    .map((payload) => payload.text?.trim() ?? "")
    .filter((text) => text.length > 0)
    .join("\n");
}

function normalizeSpawnField(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized || normalized.length < 4) {
    return undefined;
  }
  return normalized;
}

function resolveAutoSpawnExecutionPlan(params: {
  prompt: string;
  rawReplyText: string;
  usedToolNames: string[];
}): AutoSpawnExecutionPlan | null {
  if (hasSessionsSpawnCall(params.usedToolNames)) {
    return null;
  }
  const raw = params.rawReplyText.trim();
  if (!raw) {
    return null;
  }
  if (RAW_SPAWN_NEGATION_RE.test(raw)) {
    return null;
  }
  const matchedSpawnCommitment = RAW_SPAWN_COMMITMENT_RE.test(raw);
  const matchedProgressCommitment = RAW_PROGRESS_COMMITMENT_RE.test(raw);
  if (!matchedSpawnCommitment && !matchedProgressCommitment) {
    return null;
  }
  const task =
    normalizeSpawnField(raw.match(RAW_SPAWN_TASK_RE)?.[1]) ?? normalizeSpawnField(params.prompt);
  if (!task) {
    return null;
  }
  const label = normalizeSpawnField(raw.match(RAW_SPAWN_LABEL_RE)?.[1]);
  return {
    task,
    label,
    reason: matchedSpawnCommitment ? "raw_commitment" : "progress_commitment",
  };
}

async function autoSpawnExecutionFallback(params: {
  plan: AutoSpawnExecutionPlan;
  sessionKey?: string;
  followupRun: FollowupRun;
  sessionCtx: TemplateContext;
}): Promise<AutoSpawnExecutionResult> {
  if (!params.sessionKey) {
    return {
      notice: "检测到执行承诺，但当前会话键缺失，未能自动启动子任务。",
      spawned: false,
    };
  }

  const agentTo =
    typeof params.sessionCtx.OriginatingTo === "string" && params.sessionCtx.OriginatingTo.trim()
      ? params.sessionCtx.OriginatingTo.trim()
      : typeof params.sessionCtx.To === "string" && params.sessionCtx.To.trim()
        ? params.sessionCtx.To.trim()
        : undefined;
  const result = await spawnSubagentDirect(
    {
      task: params.plan.task,
      label: params.plan.label,
      cleanup: "keep",
      expectsCompletionMessage: true,
    },
    {
      agentSessionKey: params.sessionKey,
      agentChannel: params.sessionCtx.OriginatingChannel,
      agentAccountId: params.sessionCtx.AccountId,
      agentTo,
      agentThreadId: params.sessionCtx.MessageThreadId,
      agentGroupId: params.followupRun.run.groupId ?? null,
      agentGroupChannel: params.followupRun.run.groupChannel ?? null,
      agentGroupSpace: params.followupRun.run.groupSpace ?? null,
      requesterAgentIdOverride: params.followupRun.run.agentId || undefined,
    },
  );
  const runShort = result.runId?.slice(0, 8) ?? "unknown";
  if (result.status === "accepted") {
    if (result.reused) {
      if (result.reusedState === "active") {
        return {
          notice: `执行兜底已生效：复用已有运行中的子任务（session ${result.childSessionKey}，run ${runShort}）。`,
          spawned: true,
        };
      }
      return {
        notice: `执行兜底已生效：命中已完成子任务（session ${result.childSessionKey}，run ${runShort}，outcome ${result.reusedOutcome ?? "unknown"}）。`,
        spawned: true,
      };
    }
    return {
      notice: `执行兜底已生效：已自动启动子任务（session ${result.childSessionKey}，run ${runShort}）。`,
      spawned: true,
    };
  }
  return {
    notice: `执行兜底触发但启动失败：${result.error ?? result.status}。`,
    spawned: false,
  };
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
  let lastGuardRetryDetail: string | undefined;
  let lastGuardRetryMaxAttempts = promptReinforcerLoop.maxAttempts;
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
        const reasonRule = getPromptReinforcerReasonRule(
          promptReinforcerLoop,
          lastGuardRetryReason,
        );
        if (reasonRule) {
          const retryInstruction = reasonRule.retryInstructionBuilder({
            attempt,
            maxAttempts: lastGuardRetryMaxAttempts,
            detail: lastGuardRetryDetail,
          });
          extraPromptParts.push(retryInstruction);
        }
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
      const rawDraftPayloads = finalPayloads.map((payload) => ({ ...payload }));
      const rawDraftText = extractTextFromReplyPayloads(rawDraftPayloads);
      logReplyRaw("attempt_output_raw", {
        attempt,
        maxAttempts: promptReinforcerLoop.maxAttempts,
        sessionKey: sessionKey ?? "unknown",
        payloads: summarizeReplyPayloadsForRawLog(rawDraftPayloads),
      });
      const usedToolNames = [...(runResult.meta?.usedTools ?? [])];
      if (
        autoMemoryPrefetchForAttempt?.markUsedTool &&
        !usedToolNames.some((name) => name === "memory_search")
      ) {
        usedToolNames.push("memory_search");
      }
      const spawnPlan = resolveAutoSpawnExecutionPlan({
        prompt: followupRun.prompt,
        rawReplyText: rawDraftText,
        usedToolNames,
      });
      if (spawnPlan) {
        try {
          const fallback = await autoSpawnExecutionFallback({
            plan: spawnPlan,
            sessionKey,
            followupRun,
            sessionCtx,
          });
          if (fallback.spawned && !hasSessionsSpawnCall(usedToolNames)) {
            usedToolNames.push("sessions_spawn");
          }
          if (fallback.notice.trim()) {
            finalPayloads = [...finalPayloads, { text: fallback.notice.trim() }];
          }
          logReplyRaw("execution_fallback_pre_guard", {
            attempt,
            sessionKey: sessionKey ?? "unknown",
            reason: spawnPlan.reason,
            spawned: fallback.spawned,
            notice: fallback.notice.trim(),
          });
        } catch (err) {
          const detail = err instanceof Error ? err.message : String(err);
          const notice = `执行兜底触发异常：${detail}`;
          finalPayloads = [...finalPayloads, { text: notice }];
          logReplyRaw("execution_fallback_pre_guard_error", {
            attempt,
            sessionKey: sessionKey ?? "unknown",
            reason: spawnPlan.reason,
            error: detail,
          });
        }
      }
      const preGuardPayloads = finalPayloads.map((payload) => ({ ...payload }));
      logReplyRaw("attempt_output_pre_guard", {
        attempt,
        maxAttempts: promptReinforcerLoop.maxAttempts,
        sessionKey: sessionKey ?? "unknown",
        payloads: summarizeReplyPayloadsForRawLog(preGuardPayloads),
      });
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
        blockDetail: guardResult.report.detail ?? null,
        payloads: summarizeReplyPayloadsForRawLog(finalPayloads),
      });

      if (promptReinforcerLoop.enabled && guardResult.report.blocked) {
        const reasonRule = getPromptReinforcerReasonRule(
          promptReinforcerLoop,
          guardResult.report.reason,
        );
        if (reasonRule) {
          const retryPolicy = reasonRule.policy;
          const reason = guardResult.report.reason;
          if (attempt < retryPolicy.maxAttempts) {
            if (reasonRule.mitigation === "auto_memory_prefetch") {
              if (reason === "memory_search_required") {
                const autoMemoryPrefetch = await runAutoMemoryPrefetch({
                  cfg,
                  agentId: followupRun.run.agentId || "main",
                  sessionKey,
                  query: followupRun.prompt,
                });
                logReplyRaw("memory_prefetch", {
                  attempt,
                  maxAttempts: retryPolicy.maxAttempts,
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
                    `[prompt-reinforcer] output guard retry prefetch failed: reason=memory_search_required prefetch=${autoMemoryPrefetch.reason} error=${autoMemoryPrefetch.error ?? "-"}`,
                  );
                }
              }
            }
            lastGuardRetryReason = reason;
            lastGuardRetryDetail = guardResult.report.detail;
            lastGuardRetryMaxAttempts = retryPolicy.maxAttempts;
            defaultRuntime.error(
              `[prompt-reinforcer] output guard retry: attempt=${attempt + 1}/${retryPolicy.maxAttempts} reason=${reason} detail=${guardResult.report.detail ?? "-"}`,
            );
            continue;
          }
          if (retryPolicy.failOpen) {
            defaultRuntime.error(
              `[prompt-reinforcer] output guard fail-open: reason=${reason ?? "unknown"} detail=${guardResult.report.detail ?? "-"} attempts=${retryPolicy.maxAttempts}`,
            );
            logReplyRaw("attempt_output_fail_open", {
              attempt,
              maxAttempts: retryPolicy.maxAttempts,
              sessionKey: sessionKey ?? "unknown",
              reason: reason ?? null,
              detail: guardResult.report.detail ?? null,
              payloads: summarizeReplyPayloadsForRawLog(preGuardPayloads),
            });
            lastGuardRetryReason = undefined;
            lastGuardRetryDetail = undefined;
            lastGuardRetryMaxAttempts = promptReinforcerLoop.maxAttempts;
            return finalizeWithFollowup(
              preGuardPayloads.length === 1 ? preGuardPayloads[0] : preGuardPayloads,
              queueKey,
              runFollowupTurn,
            );
          }
        }
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
