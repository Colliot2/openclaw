import fs from "node:fs/promises";
import path from "node:path";
import {
  DEFAULT_AGENTS_FILENAME,
  filterBootstrapFilesForSession,
  type WorkspaceBootstrapFile,
} from "../../../agents/workspace.js";
import { resolveHookConfig } from "../../config.js";
import { isAgentBootstrapEvent, type HookHandler } from "../../hooks.js";

const HOOK_KEY = "prompt-reinforcer";
const VIRTUAL_RELATIVE_PATH = path.join(".openclaw", "prompt-reinforcer", DEFAULT_AGENTS_FILENAME);

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => (typeof item === "string" ? item.trim() : "")).filter(Boolean);
}

function resolveConfiguredPaths(raw: Record<string, unknown>): string[] {
  const file = typeof raw.file === "string" ? raw.file.trim() : "";
  const pathValue = typeof raw.path === "string" ? raw.path.trim() : "";
  const files = normalizeStringArray(raw.files);
  const paths = normalizeStringArray(raw.paths);
  return [file, pathValue, ...files, ...paths].filter(Boolean);
}

function resolveInlinePrompt(raw: Record<string, unknown>): string {
  const content = typeof raw.content === "string" ? raw.content.trim() : "";
  const lines = normalizeStringArray(raw.lines);
  const fromLines = lines.length > 0 ? lines.join("\n") : "";
  return [content, fromLines].filter(Boolean).join("\n").trim();
}

async function loadPromptSnippets(
  workspaceDir: string,
  raw: Record<string, unknown>,
): Promise<string[]> {
  const snippets: string[] = [];
  for (const filePath of resolveConfiguredPaths(raw)) {
    const absolutePath = path.isAbsolute(filePath) ? filePath : path.join(workspaceDir, filePath);
    try {
      const content = (await fs.readFile(absolutePath, "utf-8")).trim();
      if (content) {
        snippets.push(content);
      }
    } catch {
      console.warn(`[prompt-reinforcer] failed to read: ${absolutePath}`);
    }
  }
  const inlinePrompt = resolveInlinePrompt(raw);
  if (inlinePrompt) {
    snippets.push(inlinePrompt);
  }
  return snippets;
}

function buildInjectedPrompt(snippets: string[]): string {
  const body = snippets.join("\n\n").trim();
  if (!body) {
    return "";
  }
  return [
    "# Prompt Reinforcer (Hook Injected)",
    "These rules are injected by OpenClaw hook `prompt-reinforcer` and should be treated as high-priority user instructions unless higher-priority safety/platform rules conflict.",
    "",
    body,
    "",
  ].join("\n");
}

function createInjectedFile(workspaceDir: string, content: string): WorkspaceBootstrapFile {
  return {
    name: DEFAULT_AGENTS_FILENAME,
    path: path.join(workspaceDir, VIRTUAL_RELATIVE_PATH),
    content,
    missing: false,
  };
}

const promptReinforcerHook: HookHandler = async (event) => {
  if (!isAgentBootstrapEvent(event)) {
    return;
  }

  const context = event.context;
  const hookConfig = resolveHookConfig(context.cfg, HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return;
  }

  const config = hookConfig as Record<string, unknown>;
  const snippets = await loadPromptSnippets(context.workspaceDir, config);
  const injectedPrompt = buildInjectedPrompt(snippets);
  if (!injectedPrompt) {
    return;
  }

  const injectedFile = createInjectedFile(context.workspaceDir, injectedPrompt);
  const prepend = config.prepend !== false;
  const merged = prepend
    ? [injectedFile, ...context.bootstrapFiles]
    : [...context.bootstrapFiles, injectedFile];
  context.bootstrapFiles = filterBootstrapFilesForSession(merged, context.sessionKey);
};

export default promptReinforcerHook;
