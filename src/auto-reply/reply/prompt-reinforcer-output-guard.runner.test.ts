import type { Api, AssistantMessage, Model } from "@mariozechner/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import { makeTempWorkspace } from "../../test-helpers/workspace.js";

const { completeSimpleMock, getApiKeyForModelMock, resolveModelMock } = vi.hoisted(() => ({
  completeSimpleMock: vi.fn(),
  getApiKeyForModelMock: vi.fn(),
  resolveModelMock: vi.fn(),
}));

vi.mock("@mariozechner/pi-ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@mariozechner/pi-ai")>();
  return {
    ...actual,
    completeSimple: completeSimpleMock,
  };
});

vi.mock("../../agents/model-auth.js", () => ({
  getApiKeyForModel: getApiKeyForModelMock,
}));

vi.mock("../../agents/pi-embedded-runner/model.js", () => ({
  resolveModel: resolveModelMock,
}));

import { enforcePromptReinforcerOutput } from "./prompt-reinforcer-output-guard.js";

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

function createModel(provider: string, api: Api, modelId: string): Model<Api> {
  return {
    id: modelId,
    name: modelId,
    api,
    provider,
    baseUrl: "https://example.com",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 8192,
    maxTokens: 4096,
  };
}

function createGuardOkResponse(model: Model<Api>): AssistantMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text: '{"compliant":true,"rewritten":"EXACT_ORIGINAL"}' }],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        total: 0,
      },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

describe("prompt-reinforcer output guard runner args", () => {
  beforeEach(() => {
    completeSimpleMock.mockReset();
    getApiKeyForModelMock.mockReset();
    resolveModelMock.mockReset();
    getApiKeyForModelMock.mockResolvedValue({ apiKey: "test-key", mode: "api-key" });
  });

  it("adds systemPrompt and omits temperature for openai-codex", async () => {
    const model = createModel("openai-codex", "openai-codex-responses", "gpt-5.3-codex");
    resolveModelMock.mockReturnValue({ model });
    completeSimpleMock.mockResolvedValue(createGuardOkResponse(model));

    const workspaceDir = await makeTempWorkspace("openclaw-pr-guard-runner-codex-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceTemperature: 0.7,
      lines: ["规则A"],
    });
    const payloads = await enforcePromptReinforcerOutput({
      payloads: [{ text: "原始回复" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai-codex",
      model: "gpt-5.3-codex",
    });
    expect(payloads[0]?.text).toBe("原始回复");
    expect(completeSimpleMock).toHaveBeenCalledTimes(1);
    const [, context, options] = completeSimpleMock.mock.calls[0] as [
      Model<Api>,
      { systemPrompt?: string },
      { temperature?: number },
    ];
    expect(context.systemPrompt).toContain("Return exactly one JSON object");
    expect(options.temperature).toBeUndefined();
  });

  it("keeps temperature for non-codex providers", async () => {
    const model = createModel("openai", "openai-responses", "gpt-5.3");
    resolveModelMock.mockReturnValue({ model });
    completeSimpleMock.mockResolvedValue(createGuardOkResponse(model));

    const workspaceDir = await makeTempWorkspace("openclaw-pr-guard-runner-openai-");
    const cfg = createPromptReinforcerConfig({
      enforceOutput: true,
      enforceTemperature: 0.6,
      lines: ["规则B"],
    });
    await enforcePromptReinforcerOutput({
      payloads: [{ text: "raw" }],
      cfg,
      workspaceDir,
      agentDir: workspaceDir,
      provider: "openai",
      model: "gpt-5.3",
    });
    expect(completeSimpleMock).toHaveBeenCalledTimes(1);
    const [, , options] = completeSimpleMock.mock.calls[0] as [
      Model<Api>,
      unknown,
      { temperature?: number },
    ];
    expect(options.temperature).toBe(0.6);
  });
});
