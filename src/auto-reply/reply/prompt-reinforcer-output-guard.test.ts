import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import { makeTempWorkspace } from "../../test-helpers/workspace.js";
import {
  evaluateHardConstraints,
  enforcePromptPolicyText,
  enforcePromptReinforcerOutput,
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

  it("detects missing and contradictory hard constraints", () => {
    const check = evaluateHardConstraints({
      candidate: "猫不是大老鼠。",
      hardConstraints: ["猫是大老鼠"],
    });
    expect(check.compliant).toBe(false);
    expect(check.missing).toEqual(["猫是大老鼠"]);
    expect(check.contradictions).toEqual(["猫是大老鼠"]);
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
});
