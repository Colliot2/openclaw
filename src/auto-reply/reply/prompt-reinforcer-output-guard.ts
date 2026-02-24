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
const EXACT_ORIGINAL_TOKEN = "EXACT_ORIGINAL";

export type PromptPolicyGuardDecision = {
  compliant: boolean;
  rewritten?: string;
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
  const compactCandidate = normalizeConstraint(params.candidate);
  const missing: string[] = [];
  const contradictions: string[] = [];
  for (const hardConstraint of hardConstraints) {
    const normalized = normalizeConstraint(hardConstraint);
    if (!compactCandidate.includes(normalized)) {
      missing.push(hardConstraint);
    }
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
    compliant: missing.length === 0 && contradictions.length === 0,
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
}): Promise<{ text: string; compliant: boolean; changed: boolean }> {
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
    const decision = await params.guardRunner({
      policy: params.policy,
      candidate: current,
      pass,
      hardConstraints,
    });
    if (!decision) {
      return { text: current, compliant: false, changed };
    }
    if (decision.compliant) {
      const hardCheck = evaluateHardConstraints({
        candidate: current,
        hardConstraints,
      });
      if (hardCheck.compliant) {
        return { text: current, compliant: true, changed };
      }
      const rewritten = decision.rewritten?.trim();
      if (rewritten && rewritten !== current) {
        current = rewritten;
        changed = true;
      }
      continue;
    }
    const rewritten = decision.rewritten?.trim();
    if (!rewritten) {
      return { text: current, compliant: false, changed };
    }
    if (rewritten !== current) {
      current = rewritten;
      changed = true;
    }
  }
  return { text: current, compliant: false, changed };
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
      "The rewritten text MUST include every hard constraint sentence verbatim.",
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
    const completion = await completeSimple(
      resolved.model!,
      {
        messages: [
          {
            role: "user",
            content: buildGuardPrompt({ policy, candidate, hardConstraints }),
            timestamp: Date.now(),
          },
        ],
      },
      {
        ...(apiKey ? { apiKey } : {}),
        temperature,
        maxTokens,
      },
    );
    const text = collectCompletionText(completion.content);
    return parseGuardDecision(text);
  };
}

function withFailClosed(payload: ReplyPayload, message: string): ReplyPayload {
  if (payload.isError || typeof payload.text !== "string" || payload.text.trim().length === 0) {
    return payload;
  }
  return { ...payload, text: message };
}

export async function enforcePromptReinforcerOutput(params: {
  payloads: ReplyPayload[];
  cfg: OpenClawConfig;
  workspaceDir: string;
  agentDir: string;
  provider: string;
  model: string;
  authProfileId?: string;
  guardRunner?: PromptPolicyGuardRunner;
}): Promise<ReplyPayload[]> {
  const hookConfig = resolveHookConfig(params.cfg, PROMPT_REINFORCER_HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return params.payloads;
  }
  const raw = hookConfig as Record<string, unknown>;
  const settings = resolveGuardSettings(raw);
  if (!settings.enabled) {
    return params.payloads;
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
    return params.payloads;
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
    return settings.failClosed
      ? params.payloads.map((payload) => withFailClosed(payload, settings.failClosedMessage))
      : params.payloads;
  }

  const nextPayloads: ReplyPayload[] = [];
  for (const payload of params.payloads) {
    if (payload.isError || typeof payload.text !== "string" || payload.text.trim().length === 0) {
      nextPayloads.push(payload);
      continue;
    }
    try {
      const outcome = await enforcePromptPolicyText({
        text: payload.text,
        policy,
        maxPasses: settings.maxPasses,
        guardRunner,
        hardConstraints,
      });
      if (outcome.compliant || outcome.changed) {
        nextPayloads.push({ ...payload, text: outcome.text });
        continue;
      }
      nextPayloads.push(
        settings.failClosed ? withFailClosed(payload, settings.failClosedMessage) : payload,
      );
    } catch (err) {
      defaultRuntime.error(`[prompt-reinforcer] output guard execution failed: ${String(err)}`);
      nextPayloads.push(
        settings.failClosed ? withFailClosed(payload, settings.failClosedMessage) : payload,
      );
    }
  }
  return nextPayloads;
}
