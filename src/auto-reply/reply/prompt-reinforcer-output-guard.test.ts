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

  it("caps rewrite attempts to configured max passes", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-max-pass-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceMaxPasses: 10,
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
    expect(calls).toBe(10);
    expect(payloads[0]?.text).toBe("策略冲突，已拦截。");
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

  it("fails closed when guard cannot return a decision", async () => {
    const workspaceDir = await makeTempWorkspace("openclaw-prompt-reinforcer-output-fail-closed-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceFailClosed: true,
      enforceFailClosedMessage: "策略冲突，已拦截。",
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
    expect(payloads[0]?.text).toBe("策略冲突，已拦截。");
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
