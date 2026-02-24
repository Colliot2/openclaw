import fs from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import { makeTempWorkspace } from "../../test-helpers/workspace.js";
import {
  evaluateHardConstraints,
  enforcePromptPolicyText,
  enforcePromptReinforcerOutput,
  enforcePromptReinforcerOutputWithReport,
  extractHardConstraints,
  parseGuardDecision,
} from "./prompt-reinforcer-output-guard.js";

function createPromptReinforcerConfig(entry: Record<string, unknown>): OpenClawConfig {
  return {
    hooks: {
      internal: {
        entries: {
          "prompt-reinforcer": {
            enabled: true,
            ...entry,
          },
        },
      },
    },
  };
}

describe("prompt-reinforcer output guard", () => {
  it("parses guard decision payloads", () => {
    expect(parseGuardDecision('{"compliant":true,"rewritten":"EXACT_ORIGINAL"}')).toEqual({
      compliant: true,
    });
    expect(parseGuardDecision('{"compliant":false,"rewritten":"猫是大老鼠。"}')).toEqual({
      compliant: false,
      rewritten: "猫是大老鼠。",
    });
    expect(parseGuardDecision("not json")).toBeNull();
  });

  it("extracts bracketed hard constraints", () => {
    const constraints = extractHardConstraints(
      ["# title", "- [猫是大老鼠]", "HC: 鱼会飞", "[天是绿色的]"].join("\n"),
    );
    expect(constraints).toEqual(["猫是大老鼠", "鱼会飞", "天是绿色的"]);
  });

  it("detects contradictory hard constraints without requiring verbatim echo", () => {
    const check = evaluateHardConstraints({
      candidate: "猫不是大老鼠。",
      hardConstraints: ["猫是大老鼠"],
    });
    expect(check.compliant).toBe(false);
    expect(check.missing).toEqual([]);
    expect(check.contradictions).toEqual(["猫是大老鼠"]);
  });

  it("accepts semantically consistent paraphrases for hard constraints", () => {
    const check = evaluateHardConstraints({
      candidate: "是。猫就是大老鼠。",
      hardConstraints: ["猫是大老鼠"],
    });
    expect(check.compliant).toBe(true);
    expect(check.missing).toEqual([]);
    expect(check.contradictions).toEqual([]);
  });

  it("rewrites text until compliant", async () => {
    let calls = 0;
    const result = await enforcePromptPolicyText({
      text: "不是。",
      policy: "猫是大老鼠。",
      maxPasses: 2,
      guardRunner: async () => {
        calls += 1;
        if (calls === 1) {
          return { compliant: false, rewritten: "是，猫是大老鼠。" };
        }
        return { compliant: true };
      },
    });
    expect(result.compliant).toBe(true);
    expect(result.changed).toBe(true);
    expect(result.text).toBe("是，猫是大老鼠。");
    expect(result.reason).toBe("compliant");
    expect(result.passes).toBe(2);
  });

  it("returns guard_no_decision when model gives no decision", async () => {
    const trace: string[] = [];
    const result = await enforcePromptPolicyText({
      text: "不是。",
      policy: "猫是大老鼠。",
      maxPasses: 3,
      guardRunner: async () => null,
      onTrace: (event) => {
        trace.push(`${event.pass}:${event.decision}`);
      },
    });
    expect(result.compliant).toBe(false);
    expect(result.reason).toBe("guard_no_decision");
    expect(result.passes).toBe(1);
    expect(trace).toEqual(["1:guard_no_decision"]);
  });

  it("returns max_pass_exhausted with hard-constraint diagnostics", async () => {
    const result = await enforcePromptPolicyText({
      text: "猫不是大老鼠。",
      policy: "风格规则",
      maxPasses: 2,
      hardConstraints: ["猫是大老鼠"],
      guardRunner: async () => ({ compliant: true }),
    });
    expect(result.compliant).toBe(false);
    expect(result.reason).toBe("max_pass_exhausted");
    expect(result.passes).toBe(2);
    expect(result.hardMissing).toEqual([]);
    expect(result.hardContradictions).toEqual(["猫是大老鼠"]);
  });

  it("returns guard_rewrite_missing when guard asks rewrite but omits text", async () => {
    const result = await enforcePromptPolicyText({
      text: "不是。",
      policy: "猫是大老鼠。",
      maxPasses: 2,
      guardRunner: async () => ({ compliant: false }),
    });
    expect(result.compliant).toBe(false);
    expect(result.reason).toBe("guard_rewrite_missing");
    expect(result.passes).toBe(1);
  });

  it("rewrites outgoing payload text when guard marks conflict", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-output-guard-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      lines: ["猫是大老鼠。"],
      enforceMaxPasses: 2,
    });
    let calls = 0;
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "不是。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => {
        calls += 1;
        if (calls === 1) {
          return { compliant: false, rewritten: "是，猫是大老鼠。" };
        }
        return { compliant: true };
      },
    });
    expect(payloads[0]?.text).toBe("是，猫是大老鼠。");
  });

  it("caps hard-constraint rewrite attempts and blocks after hard max passes", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-max-pass-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceSoftMaxPasses: 3,
      enforceHardMaxPasses: 10,
      enforceFailClosed: true,
      enforceFailClosedMessage: "策略冲突，已拦截。",
      enforceHardLines: ["[猫是大老鼠]"],
      lines: ["风格规则"],
    });
    let calls = 0;
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "猫不是大老鼠。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => {
        calls += 1;
        return { compliant: true };
      },
    });
    expect(calls).toBe(11);
    expect(payloads[0]?.text).toBe("策略冲突，已拦截。");
  });

  it("does not apply unrelated hard constraints to unrelated prompts", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-hard-scope-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceFailClosed: true,
      enforceHardLines: ["[猫是大老鼠]"],
      lines: ["风格规则"],
    });
    const hardConstraintCounts: number[] = [];
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "日本社会反应呈现明显分化。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      latestUserPrompt: "日本社会有何反应？",
      guardRunner: async ({ hardConstraints }) => {
        hardConstraintCounts.push(hardConstraints.length);
        return { compliant: true };
      },
    });
    expect(hardConstraintCounts[0]).toBe(0);
    expect(payloads[0]?.text).toBe("日本社会反应呈现明显分化。");
  });

  it("passes through soft-policy output after soft max passes are exhausted", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-soft-fail-open-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceSoftMaxPasses: 2,
      enforceSoftFailOpen: true,
      lines: ["风格规则"],
    });
    let calls = 0;
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "原始回复。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => {
        calls += 1;
        return { compliant: false, rewritten: "已改写但仍不合规。" };
      },
    });
    expect(calls).toBe(2);
    expect(payloads[0]?.text).toBe("已改写但仍不合规。");
  });

  it("reports soft_policy_blocked when soft fail-open is disabled", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-soft-block-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceSoftFailOpen: false,
      enforceFailClosedMessage: "软约束冲突，已拦截。",
      lines: ["风格规则"],
    });
    const result = await enforcePromptReinforcerOutputWithReport({
      payloads: [{ text: "原始回复。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => null,
    });
    expect(result.report).toEqual({
      blocked: true,
      retryable: false,
      reason: "soft_policy_blocked",
    });
    expect(result.payloads[0]?.text).toBe("软约束冲突，已拦截。");
  });

  it("loads hard constraints from explicit hard file", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-hard-file-");
    await fs.writeFile(
      path.join(workspaceDir, "PROMPT_HARD_CONSTRAINTS.md"),
      "[猫是大老鼠]\n",
      "utf-8",
    );
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceFailClosed: true,
      enforceFailClosedMessage: "硬约束冲突，已拦截。",
      enforceHardFile: "PROMPT_HARD_CONSTRAINTS.md",
      lines: ["风格规则"],
    });
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "猫不是大老鼠。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => ({ compliant: true }),
    });
    expect(payloads[0]?.text).toBe("硬约束冲突，已拦截。");
  });

  it("reports hard_constraints_blocked when hard constraints remain unresolved", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-hard-block-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceHardFile: "PROMPT_HARD_CONSTRAINTS.md",
      enforceHardFailClosedMessage: "硬约束冲突，已拦截。",
      lines: ["风格规则"],
    });
    await fs.writeFile(
      path.join(workspaceDir, "PROMPT_HARD_CONSTRAINTS.md"),
      "[猫是大老鼠]\n",
      "utf-8",
    );
    const result = await enforcePromptReinforcerOutputWithReport({
      payloads: [{ text: "猫不是大老鼠。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => ({ compliant: true }),
    });
    expect(result.report).toEqual({
      blocked: true,
      retryable: false,
      reason: "hard_constraints_blocked",
    });
    expect(result.payloads[0]?.text).toBe("硬约束冲突，已拦截。");
  });

  it("passes through soft-policy output when guard cannot return a decision", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-output-fail-open-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceSoftFailOpen: true,
      lines: ["猫是大老鼠。"],
    });
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "不是。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      guardRunner: async () => null,
    });
    expect(payloads[0]?.text).toBe("不是。");
  });

  it("blocks memory-recall queries when memory_search was not used", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-memory-gate-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceRequireMemorySearch: true,
      lines: ["风格规则"],
    });
    let guardCalls = 0;
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "这是直接回答。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      latestUserPrompt: "我们之前做过什么决定？",
      usedToolNames: [],
      guardRunner: async () => {
        guardCalls += 1;
        return { compliant: true };
      },
    });
    expect(guardCalls).toBe(0);
    expect(payloads[0]?.text).toContain("Memory recall is required");
  });

  it("reports memory_search_required as a retryable block reason", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-memory-report-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceRequireMemorySearch: true,
      lines: ["风格规则"],
    });
    const result = await enforcePromptReinforcerOutputWithReport({
      payloads: [{ text: "这是直接回答。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      latestUserPrompt: "我们之前做过什么决定？",
      usedToolNames: [],
      guardRunner: async () => ({ compliant: true }),
    });
    expect(result.report).toEqual({
      blocked: true,
      retryable: true,
      reason: "memory_search_required",
    });
  });

  it("allows memory-recall queries after memory_search was used", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-memory-pass-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceRequireMemorySearch: true,
      lines: ["风格规则"],
    });
    let guardCalls = 0;
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "回答完成。" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      latestUserPrompt: "我之前的偏好是什么？",
      usedToolNames: ["memory_search"],
      guardRunner: async () => {
        guardCalls += 1;
        return { compliant: true };
      },
    });
    expect(guardCalls).toBe(1);
    expect(payloads[0]?.text).toBe("回答完成。");
  });
});
