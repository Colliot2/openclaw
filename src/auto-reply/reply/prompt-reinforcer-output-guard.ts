import { completeSimple, type TextContent } from "@mariozechner/pi-ai";
import { getApiKeyForModel } from "../../agents/model-auth.js";
import { resolveModel } from "../../agents/pi-embedded-runner/model.js";
import type { OpenClawConfig } from "../../config/config.js";
import { resolveHookConfig } from "../../hooks/config.js";
import {
  joinPromptPolicy,
  loadPromptSnippets,
  PROMPT_REINFORCER_HOOK_KEY,
} from "../../prompt-reinforcer/policy.js";
import { defaultRuntime } from "../../runtime.js";
import type { ReplyPayload } from "../types.js";

const DEFAULT_MAX_PASSES = 2;
const MIN_MAX_PASSES = 1;
const MAX_MAX_PASSES = 10;
const DEFAULT_TEMPERATURE = 0;
const MIN_TEMPERATURE = 0;
const MAX_TEMPERATURE = 1;
const FALLBACK_FAIL_CLOSED_MESSAGE =
  "I can't provide a compliant reply for the current prompt policy. Please clarify or retry.";
const FALLBACK_MEMORY_RECALL_REQUIRED_MESSAGE =
  "Memory recall is required for this request. Please retry after running memory_search first.";
const EXACT_ORIGINAL_TOKEN = "EXACT_ORIGINAL";
const GUARD_SYSTEM_PROMPT =
  "You enforce policy strictly. Return exactly one JSON object and no extra text.";

const MEMORY_RECALL_PATTERNS: RegExp[] = [
  /\b(previous|earlier|prior|last time|remember|recall)\b/i,
  /\b(decision|decisions|decided|preference|preferences|prefer|todo|to-do|task|tasks)\b/i,
  /\b(history|historical|context|follow[-\s]?up|status update)\b/i,
  /(之前|以前|上次|还记得|记不记得|回忆|回顾|历史|偏好|喜好|决策|决定|待办|任务|跟进|进展|约定|说过)/,
];

export type PromptPolicyGuardDecision = {
  compliant: boolean;
  rewritten?: string;
};

export type PromptPolicyEnforceReason =
  | "compliant"
  | "guard_no_decision"
  | "guard_rewrite_missing"
  | "max_pass_exhausted";

export type PromptPolicyEnforceTraceEvent = {
  pass: number;
  decision: "guard_no_decision" | "guard_compliant" | "guard_rewrite";
  rewritten: boolean;
  rewrittenChanged: boolean;
  hardMissing: string[];
  hardContradictions: string[];
  candidateBefore: string;
  candidateAfter: string;
  rewrittenText?: string;
};

export type PromptPolicyEnforceOutcome = {
  text: string;
  compliant: boolean;
  changed: boolean;
  reason: PromptPolicyEnforceReason;
  passes: number;
  hardMissing: string[];
  hardContradictions: string[];
};

type PromptPolicyGuardRunner = (params: {
  policy: string;
  candidate: string;
  pass: number;
  hardConstraints: string[];
}) => Promise<PromptPolicyGuardDecision | null>;

type GuardSettings = {
  enabled: boolean;
  maxPasses: number;
  failClosed: boolean;
  failClosedMessage: string;
  temperature: number;
  requireMemorySearch: boolean;
  memoryRecallFailMessage: string;
};

export type PromptReinforcerOutputBlockReason =
  | "memory_search_required"
  | "policy_guard_blocked"
  | "guard_unavailable";

export type PromptReinforcerOutputReport = {
  blocked: boolean;
  retryable: boolean;
  reason?: PromptReinforcerOutputBlockReason;
};

export type PromptReinforcerOutputResult = {
  payloads: ReplyPayload[];
  report: PromptReinforcerOutputReport;
};

function isTextContentBlock(block: unknown): block is TextContent {
  return (
    typeof block === "object" &&
    block !== null &&
    (block as { type?: unknown }).type === "text" &&
    typeof (block as { text?: unknown }).text === "string"
  );
}

function collectCompletionText(content: unknown): string {
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter(isTextContentBlock)
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join("\n")
    .trim();
}

function clampInteger(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, value));
}

function resolveGuardSettings(raw: Record<string, unknown>): GuardSettings {
  const failClosedMessage =
    typeof raw.enforceFailClosedMessage === "string" && raw.enforceFailClosedMessage.trim()
      ? raw.enforceFailClosedMessage.trim()
      : FALLBACK_FAIL_CLOSED_MESSAGE;
  const memoryRecallFailMessage =
    typeof raw.enforceRequireMemorySearchMessage === "string" &&
    raw.enforceRequireMemorySearchMessage.trim()
      ? raw.enforceRequireMemorySearchMessage.trim()
      : FALLBACK_MEMORY_RECALL_REQUIRED_MESSAGE;
  return {
    enabled: raw.enforceOutput === true,
    maxPasses: clampInteger(
      raw.enforceMaxPasses,
      DEFAULT_MAX_PASSES,
      MIN_MAX_PASSES,
      MAX_MAX_PASSES,
    ),
    failClosed: raw.enforceFailClosed === true,
    failClosedMessage,
    temperature: clampNumber(
      raw.enforceTemperature,
      DEFAULT_TEMPERATURE,
      MIN_TEMPERATURE,
      MAX_TEMPERATURE,
    ),
    requireMemorySearch: raw.enforceRequireMemorySearch === true,
    memoryRecallFailMessage,
  };
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeCompactText(value: string): string {
  return value.replace(/\s+/g, "").trim();
}

function normalizeConstraint(value: string): string {
  return normalizeCompactText(value).toLowerCase();
}

function uniqueConstraints(constraints: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of constraints) {
    const trimmed = item.trim();
    if (!trimmed) {
      continue;
    }
    const key = normalizeConstraint(trimmed);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(trimmed);
  }
  return out;
}

type ExtractHardConstraintsOptions = {
  allowPlainLines?: boolean;
};

export function extractHardConstraints(
  text: string,
  opts: ExtractHardConstraintsOptions = {},
): string[] {
  const lines = text.split(/\r?\n/);
  const bracketOnly: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    const bracketMatch = line.match(/^(?:[-*+]\s*)?\[([^\]\r\n]{1,220})\]$/);
    if (bracketMatch?.[1]?.trim()) {
      bracketOnly.push(bracketMatch[1].trim());
      continue;
    }
    const hardPrefixMatch = line.match(/^(?:[-*+]\s*)?(?:HC|HARD)\s*:\s*(.+)$/i);
    if (hardPrefixMatch?.[1]?.trim()) {
      bracketOnly.push(hardPrefixMatch[1].trim());
    }
  }
  if (bracketOnly.length > 0 || !opts.allowPlainLines) {
    return uniqueConstraints(bracketOnly);
  }
  const plainLines: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || line.startsWith(">") || line.startsWith("```")) {
      continue;
    }
    const unlisted = line.replace(/^(?:[-*+]\s+|\d+\.\s+)/, "").trim();
    if (!unlisted || unlisted.length > 220) {
      continue;
    }
    plainLines.push(unlisted);
  }
  return uniqueConstraints(plainLines);
}

type HardConstraintCheck = {
  compliant: boolean;
  missing: string[];
  contradictions: string[];
};

type NegationPattern = {
  target: "raw" | "compact";
  regex: RegExp;
};

function buildNegationPatterns(constraint: string): NegationPattern[] {
  const patterns: NegationPattern[] = [];
  const compact = normalizeCompactText(constraint);
  const cnMatch = compact.match(/^(.+?)是(.+)$/);
  if (cnMatch?.[1] && cnMatch[2]) {
    const lhs = escapeRegex(cnMatch[1]);
    const rhs = escapeRegex(cnMatch[2]);
    patterns.push({
      target: "compact",
      regex: new RegExp(`${lhs}(?:不是|并非|不属于|并不属于|并不是)${rhs}`),
    });
  }
  const enMatch = constraint.trim().match(/^(.+?)\s+is\s+(.+)$/i);
  if (enMatch?.[1] && enMatch[2]) {
    const lhs = escapeRegex(enMatch[1].trim()).replace(/\s+/g, "\\s+");
    const rhs = escapeRegex(enMatch[2].trim()).replace(/\s+/g, "\\s+");
    patterns.push({
      target: "raw",
      regex: new RegExp(`\\b${lhs}\\s+is\\s+not\\s+${rhs}\\b`, "i"),
    });
    patterns.push({
      target: "raw",
      regex: new RegExp(`\\b${lhs}\\s+isn['’]?t\\s+${rhs}\\b`, "i"),
    });
  }
  return patterns;
}

export function evaluateHardConstraints(params: {
  candidate: string;
  hardConstraints: string[];
}): HardConstraintCheck {
  const hardConstraints = uniqueConstraints(params.hardConstraints);
  if (hardConstraints.length === 0) {
    return { compliant: true, missing: [], contradictions: [] };
  }
  const missing: string[] = [];
  const contradictions: string[] = [];
  for (const hardConstraint of hardConstraints) {
    const negationPatterns = buildNegationPatterns(hardConstraint);
    const hasNegation = negationPatterns.some((pattern) =>
      pattern.target === "compact"
        ? pattern.regex.test(normalizeCompactText(params.candidate))
        : pattern.regex.test(params.candidate),
    );
    if (hasNegation) {
      contradictions.push(hardConstraint);
    }
  }
  return {
    compliant: contradictions.length === 0,
    missing,
    contradictions,
  };
}

function resolveHardConstraintSource(raw: Record<string, unknown>): Record<string, unknown> | null {
  const source: Record<string, unknown> = {
    file: raw.enforceHardFile,
    path: raw.enforceHardPath,
    files: raw.enforceHardFiles,
    paths: raw.enforceHardPaths,
    content: raw.enforceHardContent,
    lines: raw.enforceHardLines,
  };
  const hasAny = Object.values(source).some((value) => {
    if (typeof value === "string") {
      return value.trim().length > 0;
    }
    if (Array.isArray(value)) {
      return value.length > 0;
    }
    return value != null;
  });
  return hasAny ? source : null;
}

function extractJsonObject(rawText: string): string | null {
  const first = rawText.indexOf("{");
  const last = rawText.lastIndexOf("}");
  if (first < 0 || last < first) {
    return null;
  }
  return rawText.slice(first, last + 1);
}

export function parseGuardDecision(rawText: string): PromptPolicyGuardDecision | null {
  const trimmed = rawText.trim();
  if (!trimmed) {
    return null;
  }
  if (trimmed === EXACT_ORIGINAL_TOKEN) {
    return { compliant: true };
  }
  const jsonBlock = extractJsonObject(trimmed);
  if (!jsonBlock) {
    return null;
  }
  try {
    const parsed = JSON.parse(jsonBlock) as Record<string, unknown>;
    if (typeof parsed.compliant !== "boolean") {
      return null;
    }
    const rewrittenCandidates = [parsed.rewritten, parsed.reply];
    const rewritten = rewrittenCandidates.find((item) => typeof item === "string");
    return parsed.compliant
      ? { compliant: true }
      : { compliant: false, ...(rewritten?.trim() ? { rewritten: rewritten.trim() } : {}) };
  } catch {
    return null;
  }
}

export async function enforcePromptPolicyText(params: {
  text: string;
  policy: string;
  maxPasses: number;
  guardRunner: PromptPolicyGuardRunner;
  hardConstraints?: string[];
  onTrace?: (event: PromptPolicyEnforceTraceEvent) => void;
}): Promise<PromptPolicyEnforceOutcome> {
  let current = params.text;
  let changed = false;
  const hardConstraints = uniqueConstraints(params.hardConstraints ?? []);
  const maxPasses = clampInteger(
    params.maxPasses,
    DEFAULT_MAX_PASSES,
    MIN_MAX_PASSES,
    MAX_MAX_PASSES,
  );
  for (let pass = 1; pass <= maxPasses; pass += 1) {
    const candidateBefore = current;
    const decision = await params.guardRunner({
      policy: params.policy,
      candidate: current,
      pass,
      hardConstraints,
    });
    if (!decision) {
      params.onTrace?.({
        pass,
        decision: "guard_no_decision",
        rewritten: false,
        rewrittenChanged: false,
        hardMissing: [],
        hardContradictions: [],
        candidateBefore,
        candidateAfter: candidateBefore,
      });
      return {
        text: current,
        compliant: false,
        changed,
        reason: "guard_no_decision",
        passes: pass,
        hardMissing: [],
        hardContradictions: [],
      };
    }
    if (decision.compliant) {
      const hardCheck = evaluateHardConstraints({
        candidate: current,
        hardConstraints,
      });
      const rewritten = decision.rewritten?.trim();
      const rewrittenChanged = Boolean(rewritten && rewritten !== current);
      const candidateAfter = rewrittenChanged && rewritten ? rewritten : candidateBefore;
      params.onTrace?.({
        pass,
        decision: "guard_compliant",
        rewritten: Boolean(rewritten),
        rewrittenChanged,
        hardMissing: hardCheck.missing,
        hardContradictions: hardCheck.contradictions,
        candidateBefore,
        candidateAfter,
        rewrittenText: rewritten,
      });
      if (hardCheck.compliant) {
        return {
          text: current,
          compliant: true,
          changed,
          reason: "compliant",
          passes: pass,
          hardMissing: [],
          hardContradictions: [],
        };
      }
      if (rewritten && rewritten !== current) {
        current = rewritten;
        changed = true;
      }
      continue;
    }
    const rewritten = decision.rewritten?.trim();
    const rewrittenChanged = Boolean(rewritten && rewritten !== current);
    const candidateAfter = rewrittenChanged && rewritten ? rewritten : candidateBefore;
    params.onTrace?.({
      pass,
      decision: "guard_rewrite",
      rewritten: Boolean(rewritten),
      rewrittenChanged,
      hardMissing: [],
      hardContradictions: [],
      candidateBefore,
      candidateAfter,
      rewrittenText: rewritten,
    });
    if (!rewritten) {
      const hardCheck = evaluateHardConstraints({
        candidate: current,
        hardConstraints,
      });
      return {
        text: current,
        compliant: false,
        changed,
        reason: "guard_rewrite_missing",
        passes: pass,
        hardMissing: hardCheck.missing,
        hardContradictions: hardCheck.contradictions,
      };
    }
    if (rewritten && rewritten !== current) {
      current = rewritten;
      changed = true;
    }
  }
  const hardCheck = evaluateHardConstraints({
    candidate: current,
    hardConstraints,
  });
  return {
    text: current,
    compliant: false,
    changed,
    reason: "max_pass_exhausted",
    passes: maxPasses,
    hardMissing: hardCheck.missing,
    hardContradictions: hardCheck.contradictions,
  };
}

function buildGuardPrompt(params: {
  policy: string;
  candidate: string;
  hardConstraints: string[];
}): string {
  const { policy, candidate, hardConstraints } = params;
  const sections = [
    "You are a strict response policy enforcer.",
    'Return exactly one JSON object: {"compliant":boolean,"rewritten":string}.',
    `If the candidate fully complies with policy and has no contradictions, set compliant=true and rewritten="${EXACT_ORIGINAL_TOKEN}".`,
    "If the candidate conflicts with policy, set compliant=false and provide corrected rewritten text.",
    "Preserve the original language and format whenever possible.",
    "Do not output markdown code fences or any extra text.",
    "",
    "<policy>",
    policy,
    "</policy>",
    "",
    "<candidate>",
    candidate,
    "</candidate>",
  ];
  if (hardConstraints.length > 0) {
    sections.push(
      "",
      "<hard_constraints>",
      ...hardConstraints.map((constraint, index) => `${index + 1}. ${constraint}`),
      "</hard_constraints>",
      "",
      "The rewritten text MUST stay semantically consistent with every hard constraint sentence (verbatim quote is optional).",
      "The rewritten text MUST NOT negate or contradict any hard constraint.",
    );
  }
  return sections.join("\n");
}

async function createDefaultGuardRunner(params: {
  cfg: OpenClawConfig;
  provider: string;
  model: string;
  agentDir: string;
  authProfileId?: string;
  temperature: number;
}): Promise<PromptPolicyGuardRunner | null> {
  const { cfg, provider, model, agentDir, authProfileId, temperature } = params;
  const resolved = resolveModel(provider, model, agentDir, cfg);
  if (!resolved.model) {
    defaultRuntime.error(
      `[prompt-reinforcer] output guard skipped: unknown model ${provider}/${model}`,
    );
    return null;
  }
  const auth = await getApiKeyForModel({
    model: resolved.model,
    cfg,
    profileId: authProfileId,
    agentDir,
  });
  const apiKey =
    typeof auth.apiKey === "string" && auth.apiKey.trim().length > 0
      ? auth.apiKey.trim()
      : undefined;
  if (!apiKey && auth.mode !== "aws-sdk") {
    defaultRuntime.error(
      `[prompt-reinforcer] output guard skipped: no api key for ${provider}/${model}`,
    );
    return null;
  }
  return async ({ policy, candidate, hardConstraints }) => {
    const maxTokens = Math.min(2048, Math.max(256, Math.ceil(candidate.length * 1.5)));
    const isOpenAICodex = resolved.model?.provider === "openai-codex";
    const completionOptions: {
      apiKey?: string;
      temperature?: number;
      maxTokens: number;
    } = {
      ...(apiKey ? { apiKey } : {}),
      maxTokens,
    };
    if (!isOpenAICodex) {
      completionOptions.temperature = temperature;
    }
    const completion = await completeSimple(
      resolved.model!,
      {
        systemPrompt: GUARD_SYSTEM_PROMPT,
        messages: [
          {
            role: "user",
            content: buildGuardPrompt({ policy, candidate, hardConstraints }),
            timestamp: Date.now(),
          },
        ],
      },
      completionOptions,
    );
    if (completion.stopReason === "error") {
      defaultRuntime.error(
        `[prompt-reinforcer] output guard request failed: provider=${provider} model=${model} stopReason=error error=${summarizeErrorForLog(completion.errorMessage)}`,
      );
    }
    const text = collectCompletionText(completion.content);
    const decision = parseGuardDecision(text);
    if (!decision) {
      defaultRuntime.error(
        `[prompt-reinforcer] output guard parse failed: provider=${provider} model=${model} stopReason=${completion.stopReason} error=${summarizeErrorForLog(completion.errorMessage)} raw=${summarizeTextForLog(text)}`,
      );
    }
    return decision;
  };
}

function withFailClosed(payload: ReplyPayload, message: string): ReplyPayload {
  if (payload.isError || typeof payload.text !== "string" || payload.text.trim().length === 0) {
    return payload;
  }
  return { ...payload, text: message };
}

function summarizeConstraintList(values: string[], previewCount = 2): string {
  if (values.length === 0) {
    return "-";
  }
  const shown = values.slice(0, previewCount).join(" | ");
  const rest = values.length - previewCount;
  return rest > 0 ? `${shown} (+${rest} more)` : shown;
}

function summarizeTextForLog(value: string, maxChars = 220): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (!compact) {
    return "<empty>";
  }
  if (compact.length <= maxChars) {
    return compact;
  }
  return `${compact.slice(0, maxChars)}...`;
}

function summarizeErrorForLog(value: unknown): string {
  if (typeof value !== "string") {
    return "<none>";
  }
  return summarizeTextForLog(value);
}

function logPromptReinforcerRaw(stage: string, payload: Record<string, unknown>) {
  try {
    defaultRuntime.log(
      `[prompt-reinforcer][raw] ${JSON.stringify({
        stage,
        ...payload,
      })}`,
    );
  } catch (err) {
    defaultRuntime.error(
      `[prompt-reinforcer][raw] log failed: stage=${stage} error=${String(err)}`,
    );
  }
}

function containsCjk(value: string): boolean {
  return /[\u3400-\u9fff]/.test(value);
}

function extractConstraintAnchors(constraint: string): string[] {
  const out: string[] = [];
  const push = (raw: string | undefined) => {
    if (!raw) {
      return;
    }
    const normalized = raw.trim().replace(/^["'`[\](){}<>]+|["'`[\](){}<>]+$/g, "");
    if (!normalized) {
      return;
    }
    out.push(normalized);
  };

  const trimmed = constraint.trim();
  if (!trimmed) {
    return [];
  }

  const cnMatch = trimmed.match(/^(.+?)是(.+)$/);
  if (cnMatch?.[1] && cnMatch[2]) {
    push(cnMatch[1]);
    push(cnMatch[2]);
  }

  const enMatch = trimmed.match(/^(.+?)\bis\b(.+)$/i);
  if (enMatch?.[1] && enMatch[2]) {
    push(enMatch[1]);
    push(enMatch[2]);
  }

  for (const segment of trimmed.split(/[，,。.!?;；:：\s]+/)) {
    push(segment);
  }

  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const raw of out) {
    const key = raw.toLowerCase();
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(raw);
  }
  return deduped;
}

function isHardConstraintRelevant(params: {
  constraint: string;
  latestUserPrompt?: string;
  candidate: string;
}): boolean {
  const anchors = extractConstraintAnchors(params.constraint);
  if (anchors.length === 0) {
    return true;
  }
  const corpusRaw = `${params.latestUserPrompt ?? ""}\n${params.candidate}`;
  const corpusCompact = normalizeCompactText(corpusRaw);
  const corpusLower = corpusRaw.toLowerCase();
  for (const anchor of anchors) {
    if (containsCjk(anchor)) {
      if (corpusCompact.includes(normalizeCompactText(anchor))) {
        return true;
      }
      continue;
    }
    const escaped = escapeRegex(anchor.toLowerCase()).replace(/\s+/g, "\\s+");
    if (new RegExp(`\\b${escaped}\\b`, "i").test(corpusLower)) {
      return true;
    }
  }
  return false;
}

function selectRelevantHardConstraints(params: {
  hardConstraints: string[];
  latestUserPrompt?: string;
  candidate: string;
}): string[] {
  return params.hardConstraints.filter((constraint) =>
    isHardConstraintRelevant({
      constraint,
      latestUserPrompt: params.latestUserPrompt,
      candidate: params.candidate,
    }),
  );
}

function requiresMemoryRecall(prompt?: string): boolean {
  if (typeof prompt !== "string" || !prompt.trim()) {
    return false;
  }
  return MEMORY_RECALL_PATTERNS.some((pattern) => pattern.test(prompt));
}

function normalizeToolNames(toolNames: string[] | undefined): string[] {
  if (!Array.isArray(toolNames) || toolNames.length === 0) {
    return [];
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const rawName of toolNames) {
    if (typeof rawName !== "string") {
      continue;
    }
    const normalized = rawName.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function passOutput(payloads: ReplyPayload[]): PromptReinforcerOutputResult {
  return {
    payloads,
    report: { blocked: false, retryable: false },
  };
}

function blockedOutput(
  payloads: ReplyPayload[],
  reason: PromptReinforcerOutputBlockReason,
  retryable: boolean,
): PromptReinforcerOutputResult {
  return {
    payloads,
    report: { blocked: true, retryable, reason },
  };
}

export async function enforcePromptReinforcerOutputWithReport(params: {
  payloads: ReplyPayload[];
  cfg: OpenClawConfig;
  workspaceDir: string;
  agentDir: string;
  provider: string;
  model: string;
  authProfileId?: string;
  latestUserPrompt?: string;
  usedToolNames?: string[];
  guardRunner?: PromptPolicyGuardRunner;
}): Promise<PromptReinforcerOutputResult> {
  const hookConfig = resolveHookConfig(params.cfg, PROMPT_REINFORCER_HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return passOutput(params.payloads);
  }
  const raw = hookConfig as Record<string, unknown>;
  const settings = resolveGuardSettings(raw);
  if (!settings.enabled) {
    return passOutput(params.payloads);
  }

  const snippets = await loadPromptSnippets({
    workspaceDir: params.workspaceDir,
    raw,
    onReadError: (absolutePath) => {
      defaultRuntime.error(`[prompt-reinforcer] failed to read: ${absolutePath}`);
    },
  });
  const policy = joinPromptPolicy(snippets);
  if (!policy) {
    return passOutput(params.payloads);
  }
  const hardConstraintSource = resolveHardConstraintSource(raw);
  const hardSourceSnippets = hardConstraintSource
    ? await loadPromptSnippets({
        workspaceDir: params.workspaceDir,
        raw: hardConstraintSource,
        onReadError: (absolutePath) => {
          defaultRuntime.error(
            `[prompt-reinforcer] failed to read hard constraints: ${absolutePath}`,
          );
        },
      })
    : [];
  const hardConstraintText = hardConstraintSource ? joinPromptPolicy(hardSourceSnippets) : policy;
  const hardConstraints = extractHardConstraints(hardConstraintText, {
    allowPlainLines: hardConstraintSource != null,
  });
  defaultRuntime.log(
    `[prompt-reinforcer] output guard active: maxPasses=${settings.maxPasses} failClosed=${settings.failClosed ? 1 : 0} hardConstraints=${hardConstraints.length}`,
  );
  if (hardConstraints.length > 0) {
    defaultRuntime.log(
      `[prompt-reinforcer] output guard hard constraints: ${summarizeConstraintList(hardConstraints)}`,
    );
  }

  let guardRunner: PromptPolicyGuardRunner | null | undefined = params.guardRunner;
  if (!guardRunner) {
    try {
      guardRunner = await createDefaultGuardRunner({
        cfg: params.cfg,
        provider: params.provider,
        model: params.model,
        agentDir: params.agentDir,
        authProfileId: params.authProfileId,
        temperature: settings.temperature,
      });
    } catch (err) {
      defaultRuntime.error(`[prompt-reinforcer] output guard init failed: ${String(err)}`);
      guardRunner = null;
    }
  }

  if (!guardRunner) {
    defaultRuntime.error(
      `[prompt-reinforcer] output guard unavailable: failClosed=${settings.failClosed ? 1 : 0}`,
    );
    const payloads = settings.failClosed
      ? params.payloads.map((payload) => withFailClosed(payload, settings.failClosedMessage))
      : params.payloads;
    logPromptReinforcerRaw("guard_unavailable", {
      failClosed: settings.failClosed,
      payloads: payloads.map((payload, index) => ({
        payload: index + 1,
        text: typeof payload.text === "string" ? payload.text : "",
        isError: Boolean(payload.isError),
      })),
    });
    return settings.failClosed
      ? blockedOutput(payloads, "guard_unavailable", false)
      : passOutput(payloads);
  }

  const usedTools = normalizeToolNames(params.usedToolNames);
  logPromptReinforcerRaw("guard_input", {
    userPrompt: params.latestUserPrompt ?? "",
    usedTools,
    payloadCount: params.payloads.length,
  });
  const requiresRecall =
    settings.requireMemorySearch && requiresMemoryRecall(params.latestUserPrompt);
  const usedMemorySearch = usedTools.includes("memory_search");
  if (requiresRecall && !usedMemorySearch) {
    defaultRuntime.error(
      `[prompt-reinforcer] output guard result: status=blocked reason=memory_search_required prompt=${summarizeTextForLog(params.latestUserPrompt ?? "")} usedTools=${usedTools.join(",") || "-"}`,
    );
    const blockedPayloads = params.payloads.map((payload) =>
      withFailClosed(payload, settings.memoryRecallFailMessage),
    );
    logPromptReinforcerRaw("memory_search_block", {
      userPrompt: params.latestUserPrompt ?? "",
      payloads: blockedPayloads.map((payload, index) => ({
        payload: index + 1,
        text: typeof payload.text === "string" ? payload.text : "",
        isError: Boolean(payload.isError),
      })),
    });
    return blockedOutput(blockedPayloads, "memory_search_required", true);
  }

  const nextPayloads: ReplyPayload[] = [];
  let blockedByPolicy = false;
  for (const [payloadIndex, payload] of params.payloads.entries()) {
    if (payload.isError || typeof payload.text !== "string" || payload.text.trim().length === 0) {
      logPromptReinforcerRaw("payload_skip", {
        payload: payloadIndex + 1,
        isError: Boolean(payload.isError),
        hasText: typeof payload.text === "string" && payload.text.trim().length > 0,
      });
      nextPayloads.push(payload);
      continue;
    }
    const relevantHardConstraints = selectRelevantHardConstraints({
      hardConstraints,
      latestUserPrompt: params.latestUserPrompt,
      candidate: payload.text,
    });
    logPromptReinforcerRaw("payload_pre", {
      payload: payloadIndex + 1,
      text: payload.text,
      relevantHardConstraints,
    });
    try {
      const outcome = await enforcePromptPolicyText({
        text: payload.text,
        policy,
        maxPasses: settings.maxPasses,
        guardRunner,
        hardConstraints: relevantHardConstraints,
        onTrace: (trace) => {
          defaultRuntime.log(
            `[prompt-reinforcer] output guard pass: payload=${payloadIndex + 1} pass=${trace.pass} decision=${trace.decision} rewritten=${trace.rewritten ? 1 : 0} changed=${trace.rewrittenChanged ? 1 : 0} missing=${trace.hardMissing.length} contradictions=${trace.hardContradictions.length}`,
          );
          logPromptReinforcerRaw("pass", {
            payload: payloadIndex + 1,
            pass: trace.pass,
            decision: trace.decision,
            candidateBefore: trace.candidateBefore,
            rewrittenText: trace.rewrittenText ?? "",
            candidateAfter: trace.candidateAfter,
            hardMissing: trace.hardMissing,
            hardContradictions: trace.hardContradictions,
          });
          if (trace.hardMissing.length > 0 || trace.hardContradictions.length > 0) {
            defaultRuntime.log(
              `[prompt-reinforcer] output guard pass details: payload=${payloadIndex + 1} pass=${trace.pass} missing=${summarizeConstraintList(trace.hardMissing)} contradictions=${summarizeConstraintList(trace.hardContradictions)}`,
            );
          }
        },
      });
      if (outcome.compliant) {
        defaultRuntime.log(
          `[prompt-reinforcer] output guard result: payload=${payloadIndex + 1} status=pass reason=${outcome.reason} passes=${outcome.passes} changed=${outcome.changed ? 1 : 0}`,
        );
        logPromptReinforcerRaw("payload_post", {
          payload: payloadIndex + 1,
          status: "pass",
          reason: outcome.reason,
          text: outcome.text,
        });
        nextPayloads.push({ ...payload, text: outcome.text });
        continue;
      }
      defaultRuntime.error(
        `[prompt-reinforcer] output guard result: payload=${payloadIndex + 1} status=blocked reason=${outcome.reason} passes=${outcome.passes} changed=${outcome.changed ? 1 : 0} missing=${outcome.hardMissing.length} contradictions=${outcome.hardContradictions.length}`,
      );
      if (outcome.hardMissing.length > 0 || outcome.hardContradictions.length > 0) {
        defaultRuntime.error(
          `[prompt-reinforcer] output guard block details: payload=${payloadIndex + 1} missing=${summarizeConstraintList(outcome.hardMissing)} contradictions=${summarizeConstraintList(outcome.hardContradictions)}`,
        );
      }
      const blockedPayload = settings.failClosed
        ? withFailClosed(payload, settings.failClosedMessage)
        : payload;
      logPromptReinforcerRaw("payload_post", {
        payload: payloadIndex + 1,
        status: "blocked",
        reason: outcome.reason,
        text: blockedPayload.text ?? "",
      });
      nextPayloads.push(blockedPayload);
      blockedByPolicy = blockedByPolicy || settings.failClosed;
    } catch (err) {
      defaultRuntime.error(
        `[prompt-reinforcer] output guard execution failed: payload=${payloadIndex + 1} ${String(err)}`,
      );
      const failedPayload = settings.failClosed
        ? withFailClosed(payload, settings.failClosedMessage)
        : payload;
      logPromptReinforcerRaw("payload_post", {
        payload: payloadIndex + 1,
        status: "error",
        reason: "guard_execution_failed",
        text: failedPayload.text ?? "",
      });
      nextPayloads.push(failedPayload);
      blockedByPolicy = blockedByPolicy || settings.failClosed;
    }
  }
  return blockedByPolicy
    ? blockedOutput(nextPayloads, "policy_guard_blocked", true)
    : passOutput(nextPayloads);
}

export async function enforcePromptReinforcerOutput(params: {
  payloads: ReplyPayload[];
  cfg: OpenClawConfig;
  workspaceDir: string;
  agentDir: string;
  provider: string;
  model: string;
  authProfileId?: string;
  latestUserPrompt?: string;
  usedToolNames?: string[];
  guardRunner?: PromptPolicyGuardRunner;
}): Promise<ReplyPayload[]> {
  const result = await enforcePromptReinforcerOutputWithReport(params);
  return result.payloads;
}
