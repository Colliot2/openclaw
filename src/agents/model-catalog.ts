import path from "node:path";
import { type OpenClawConfig, loadConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { resolveOpenClawAgentDir } from "./agent-paths.js";
import { ensureOpenClawModelsJson } from "./models-config.js";

const log = createSubsystemLogger("model-catalog");

export type ModelCatalogEntry = {
  id: string;
  name: string;
  provider: string;
  contextWindow?: number;
  reasoning?: boolean;
  input?: Array<"text" | "image">;
};

type DiscoveredModel = {
  id: string;
  name?: string;
  provider: string;
  contextWindow?: number;
  reasoning?: boolean;
  input?: Array<"text" | "image">;
};

type ConfigModelEntry = {
  id?: unknown;
  name?: unknown;
  contextWindow?: unknown;
  reasoning?: unknown;
  input?: unknown;
};

type PiSdkModule = typeof import("./pi-model-discovery.js");

let modelCatalogPromise: Promise<ModelCatalogEntry[]> | null = null;
let hasLoggedModelCatalogError = false;
const defaultImportPiSdk = () => import("./pi-model-discovery.js");
let importPiSdk = defaultImportPiSdk;

const CODEX_PROVIDER = "openai-codex";
const OPENAI_CODEX_GPT53_MODEL_ID = "gpt-5.3-codex";
const OPENAI_CODEX_GPT53_SPARK_MODEL_ID = "gpt-5.3-codex-spark";

function applyOpenAICodexSparkFallback(models: ModelCatalogEntry[]): void {
  const hasSpark = models.some(
    (entry) =>
      entry.provider === CODEX_PROVIDER &&
      entry.id.toLowerCase() === OPENAI_CODEX_GPT53_SPARK_MODEL_ID,
  );
  if (hasSpark) {
    return;
  }

  const baseModel = models.find(
    (entry) =>
      entry.provider === CODEX_PROVIDER && entry.id.toLowerCase() === OPENAI_CODEX_GPT53_MODEL_ID,
  );
  if (!baseModel) {
    return;
  }

  models.push({
    ...baseModel,
    id: OPENAI_CODEX_GPT53_SPARK_MODEL_ID,
    name: OPENAI_CODEX_GPT53_SPARK_MODEL_ID,
  });
}

export function resetModelCatalogCacheForTest() {
  modelCatalogPromise = null;
  hasLoggedModelCatalogError = false;
  importPiSdk = defaultImportPiSdk;
}

// Test-only escape hatch: allow mocking the dynamic import to simulate transient failures.
export function __setModelCatalogImportForTest(loader?: () => Promise<PiSdkModule>) {
  importPiSdk = loader ?? defaultImportPiSdk;
}

function createAuthStorage(AuthStorageLike: unknown, path: string) {
  const withFactory = AuthStorageLike as { create?: (path: string) => unknown };
  if (typeof withFactory.create === "function") {
    return withFactory.create(path);
  }
  return new (AuthStorageLike as { new (path: string): unknown })(path);
}

function toPositiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function normalizeInput(modalities: unknown): Array<"text" | "image"> | undefined {
  if (!Array.isArray(modalities) || modalities.length === 0) {
    return undefined;
  }
  const hasImage = modalities.some((value) => String(value ?? "").toLowerCase() === "image");
  return hasImage ? ["text", "image"] : ["text"];
}

function readConfiguredModelsFromConfig(cfg: OpenClawConfig): ModelCatalogEntry[] {
  const providers = cfg.models?.providers;
  if (!providers) {
    return [];
  }
  const entries: ModelCatalogEntry[] = [];
  for (const [providerIdRaw, providerValue] of Object.entries(providers)) {
    const provider = String(providerIdRaw ?? "").trim();
    if (!provider || !providerValue) {
      continue;
    }
    const providerModels = (providerValue as { models?: unknown }).models;
    if (!Array.isArray(providerModels)) {
      continue;
    }
    for (const modelValue of providerModels) {
      if (!modelValue || typeof modelValue !== "object") {
        continue;
      }
      const model = modelValue as ConfigModelEntry;
      if (typeof model.id !== "string") {
        continue;
      }
      const id = model.id.trim();
      if (!id) {
        continue;
      }
      const name = typeof model.name === "string" && model.name.trim() ? model.name.trim() : id;
      const contextWindow = toPositiveNumber(model.contextWindow);
      const reasoning = typeof model.reasoning === "boolean" ? model.reasoning : undefined;
      const input = normalizeInput(model.input);
      entries.push({ id, name, provider, contextWindow, reasoning, input });
    }
  }
  return entries;
}

function mergeMissingCatalogEntries(
  discoveredModels: ModelCatalogEntry[],
  configuredModels: ModelCatalogEntry[],
): ModelCatalogEntry[] {
  const seen = new Set(
    discoveredModels.map((entry) => `${entry.provider.toLowerCase()}/${entry.id.toLowerCase()}`),
  );
  const merged = [...discoveredModels];
  for (const entry of configuredModels) {
    const key = `${entry.provider.toLowerCase()}/${entry.id.toLowerCase()}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(entry);
  }
  return merged;
}

export async function loadModelCatalog(params?: {
  config?: OpenClawConfig;
  useCache?: boolean;
}): Promise<ModelCatalogEntry[]> {
  if (params?.useCache === false) {
    modelCatalogPromise = null;
  }
  if (modelCatalogPromise) {
    return modelCatalogPromise;
  }

  modelCatalogPromise = (async () => {
    const models: ModelCatalogEntry[] = [];
    const sortModels = (entries: ModelCatalogEntry[]) =>
      entries.sort((a, b) => {
        const p = a.provider.localeCompare(b.provider);
        if (p !== 0) {
          return p;
        }
        return a.name.localeCompare(b.name);
      });
    try {
      const cfg = params?.config ?? loadConfig();
      await ensureOpenClawModelsJson(cfg);
      await (
        await import("./pi-auth-json.js")
      ).ensurePiAuthJsonFromAuthProfiles(resolveOpenClawAgentDir());
      // IMPORTANT: keep the dynamic import *inside* the try/catch.
      // If this fails once (e.g. during a pnpm install that temporarily swaps node_modules),
      // we must not poison the cache with a rejected promise (otherwise all channel handlers
      // will keep failing until restart).
      const piSdk = await importPiSdk();
      const agentDir = resolveOpenClawAgentDir();
      const authStorage = createAuthStorage(piSdk.AuthStorage, path.join(agentDir, "auth.json"));
      const registry = new (piSdk.ModelRegistry as unknown as {
        new (
          authStorage: unknown,
          modelsFile: string,
        ):
          | Array<DiscoveredModel>
          | {
              getAll: () => Array<DiscoveredModel>;
            };
      })(authStorage, path.join(agentDir, "models.json"));
      const entries = Array.isArray(registry) ? registry : registry.getAll();
      for (const entry of entries) {
        const id = String(entry?.id ?? "").trim();
        if (!id) {
          continue;
        }
        const provider = String(entry?.provider ?? "").trim();
        if (!provider) {
          continue;
        }
        const name = String(entry?.name ?? id).trim() || id;
        const contextWindow =
          typeof entry?.contextWindow === "number" && entry.contextWindow > 0
            ? entry.contextWindow
            : undefined;
        const reasoning = typeof entry?.reasoning === "boolean" ? entry.reasoning : undefined;
        const input = Array.isArray(entry?.input) ? entry.input : undefined;
        models.push({ id, name, provider, contextWindow, reasoning, input });
      }
      const configuredModels = readConfiguredModelsFromConfig(cfg);
      const mergedModels = mergeMissingCatalogEntries(models, configuredModels);
      models.length = 0;
      models.push(...mergedModels);
      applyOpenAICodexSparkFallback(models);

      if (models.length === 0) {
        // If we found nothing, don't cache this result so we can try again.
        modelCatalogPromise = null;
      }

      return sortModels(models);
    } catch (error) {
      if (!hasLoggedModelCatalogError) {
        hasLoggedModelCatalogError = true;
        log.warn(`Failed to load model catalog: ${String(error)}`);
      }
      // Don't poison the cache on transient dependency/filesystem issues.
      modelCatalogPromise = null;
      if (models.length > 0) {
        return sortModels(models);
      }
      return [];
    }
  })();

  return modelCatalogPromise;
}

/**
 * Check if a model supports image input based on its catalog entry.
 */
export function modelSupportsVision(entry: ModelCatalogEntry | undefined): boolean {
  return entry?.input?.includes("image") ?? false;
}

/**
 * Find a model in the catalog by provider and model ID.
 */
export function findModelInCatalog(
  catalog: ModelCatalogEntry[],
  provider: string,
  modelId: string,
): ModelCatalogEntry | undefined {
  const normalizedProvider = provider.toLowerCase().trim();
  const normalizedModelId = modelId.toLowerCase().trim();
  return catalog.find(
    (entry) =>
      entry.provider.toLowerCase() === normalizedProvider &&
      entry.id.toLowerCase() === normalizedModelId,
  );
}
