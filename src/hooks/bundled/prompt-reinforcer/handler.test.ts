import fs from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../../../config/config.js";
import { makeTempWorkspace, writeWorkspaceFile } from "../../../test-helpers/workspace.js";
import type { AgentBootstrapHookContext } from "../../hooks.js";
import { createHookEvent } from "../../hooks.js";
import handler from "./handler.js";

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

async function createBootstrapContext(params: {
  workspaceDir: string;
  cfg: OpenClawConfig;
  sessionKey: string;
  rootFiles: Array<{ name: string; content: string }>;
}): Promise<AgentBootstrapHookContext> {
  const bootstrapFiles = (await Promise.all(
    params.rootFiles.map(async (file) => ({
      name: file.name,
      path: await writeWorkspaceFile({
        dir: params.workspaceDir,
        name: file.name,
        content: file.content,
      }),
      content: file.content,
      missing: false,
    })),
  )) as AgentBootstrapHookContext["bootstrapFiles"];
  return {
    workspaceDir: params.workspaceDir,
    bootstrapFiles,
    cfg: params.cfg,
    sessionKey: params.sessionKey,
  };
}

describe("prompt-reinforcer hook", () => {
  it("prepends injected AGENTS content loaded from configured file", async () => {
    const tempDir = await makeTempWorkspace("openclaw-prompt-reinforcer-");
    const promptFile = path.join(tempDir, "PROMPT_OVERRIDE.md");
    await fs.writeFile(promptFile, "Always honor user prompt policy first.", "utf-8");
    const cfg = createPromptReinforcerConfig({ file: "PROMPT_OVERRIDE.md" });

    const context = await createBootstrapContext({
      workspaceDir: tempDir,
      cfg,
      sessionKey: "agent:main:main",
      rootFiles: [{ name: "AGENTS.md", content: "root agents" }],
    });

    const event = createHookEvent("agent", "bootstrap", "agent:main:main", context);
    await handler(event);

    expect(context.bootstrapFiles[0]?.path).toContain(
      path.join(".openclaw", "prompt-reinforcer", "AGENTS.md"),
    );
    expect(context.bootstrapFiles[0]?.content).toContain("Always honor user prompt policy first.");
    expect(context.bootstrapFiles[1]?.content).toContain("root agents");
  });

  it("keeps subagent allowlist after injecting prompt content", async () => {
    const tempDir = await makeTempWorkspace("openclaw-prompt-reinforcer-subagent-");
    const cfg = createPromptReinforcerConfig({
      lines: ["Line 1", "Line 2"],
    });

    const context = await createBootstrapContext({
      workspaceDir: tempDir,
      cfg,
      sessionKey: "agent:main:subagent:abc",
      rootFiles: [
        { name: "AGENTS.md", content: "root agents" },
        { name: "TOOLS.md", content: "root tools" },
        { name: "SOUL.md", content: "root soul" },
      ],
    });

    const event = createHookEvent("agent", "bootstrap", "agent:main:subagent:abc", context);
    await handler(event);

    expect(context.bootstrapFiles.map((f) => f.name).toSorted()).toEqual([
      "AGENTS.md",
      "AGENTS.md",
      "TOOLS.md",
    ]);
  });
});
