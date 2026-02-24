import fs from "node:fs/promises";
import path from "node:path";

export const PROMPT_REINFORCER_HOOK_KEY = "prompt-reinforcer";

export function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => (typeof item === "string" ? item.trim() : "")).filter(Boolean);
}

export function resolveConfiguredPaths(raw: Record<string, unknown>): string[] {
  const file = typeof raw.file === "string" ? raw.file.trim() : "";
  const pathValue = typeof raw.path === "string" ? raw.path.trim() : "";
  const files = normalizeStringArray(raw.files);
  const paths = normalizeStringArray(raw.paths);
  return [file, pathValue, ...files, ...paths].filter(Boolean);
}

export function resolveInlinePrompt(raw: Record<string, unknown>): string {
  const content = typeof raw.content === "string" ? raw.content.trim() : "";
  const lines = normalizeStringArray(raw.lines);
  const fromLines = lines.length > 0 ? lines.join("\n") : "";
  return [content, fromLines].filter(Boolean).join("\n").trim();
}

export async function loadPromptSnippets(params: {
  workspaceDir: string;
  raw: Record<string, unknown>;
  onReadError?: (absolutePath: string) => void;
}): Promise<string[]> {
  const { workspaceDir, raw, onReadError } = params;
  const snippets: string[] = [];
  for (const filePath of resolveConfiguredPaths(raw)) {
    const absolutePath = path.isAbsolute(filePath) ? filePath : path.join(workspaceDir, filePath);
    try {
      const content = (await fs.readFile(absolutePath, "utf-8")).trim();
      if (content) {
        snippets.push(content);
      }
    } catch {
      onReadError?.(absolutePath);
    }
  }
  const inlinePrompt = resolveInlinePrompt(raw);
  if (inlinePrompt) {
    snippets.push(inlinePrompt);
  }
  return snippets;
}

export function joinPromptPolicy(snippets: string[]): string {
  return snippets.join("\n\n").trim();
}

export function buildInjectedPrompt(snippets: string[]): string {
  const body = joinPromptPolicy(snippets);
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
