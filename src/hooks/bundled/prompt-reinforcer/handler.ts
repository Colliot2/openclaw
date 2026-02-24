import path from "node:path";
import {
  DEFAULT_AGENTS_FILENAME,
  filterBootstrapFilesForSession,
  type WorkspaceBootstrapFile,
} from "../../../agents/workspace.js";
import {
  buildInjectedPrompt,
  loadPromptSnippets,
  PROMPT_REINFORCER_HOOK_KEY,
} from "../../../prompt-reinforcer/policy.js";
import { resolveHookConfig } from "../../config.js";
import { isAgentBootstrapEvent, type HookHandler } from "../../hooks.js";

const VIRTUAL_RELATIVE_PATH = path.join(".openclaw", "prompt-reinforcer", DEFAULT_AGENTS_FILENAME);

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
  const hookConfig = resolveHookConfig(context.cfg, PROMPT_REINFORCER_HOOK_KEY);
  if (!hookConfig || hookConfig.enabled === false) {
    return;
  }

  const config = hookConfig as Record<string, unknown>;
  const snippets = await loadPromptSnippets({
    workspaceDir: context.workspaceDir,
    raw: config,
    onReadError: (absolutePath) => {
      console.warn(`[prompt-reinforcer] failed to read: ${absolutePath}`);
    },
  });
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
