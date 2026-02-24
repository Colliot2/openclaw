---
name: prompt-reinforcer
description: "Prepend a hook-injected AGENTS.md block to reinforce user prompt policies"
homepage: https://docs.openclaw.ai/automation/hooks#prompt-reinforcer
metadata:
  {
    "openclaw":
      {
        "emoji": "🧷",
        "events": ["agent:bootstrap"],
        "requires": { "config": ["workspace.dir"] },
        "install": [{ "id": "bundled", "kind": "bundled", "label": "Bundled with OpenClaw" }],
      },
  }
---

# Prompt Reinforcer Hook

Injects a generated `AGENTS.md` block into `Project Context` during `agent:bootstrap`.

## Why

Use this when you want a persistent "prompt policy overlay" that is:

- configured without patching core prompt assembly,
- applied every run (including after compaction),
- loaded before workspace bootstrap files by default.

## Configuration

```json
{
  "hooks": {
    "internal": {
      "enabled": true,
      "entries": {
        "prompt-reinforcer": {
          "enabled": true,
          "file": "PROMPT_OVERRIDE.md",
          "prepend": true
        }
      }
    }
  }
}
```

## Options

- `file` (string): single file path (workspace-relative or absolute).
- `path` (string): alias of `file`.
- `files` (string[]): list of file paths.
- `paths` (string[]): alias of `files`.
- `content` (string): inline prompt snippet.
- `lines` (string[]): inline prompt lines.
- `prepend` (boolean): default `true`. When `false`, appends instead of prepending.
- `enforceOutput` (boolean): default `false`. When `true`, run a final output guard before sending text replies.
- `enforceMaxPasses` (number): default `2` (clamped `1..10`). Max rewrite attempts in output guard.
- `enforceFailClosed` (boolean): default `false`. When `true`, block unresolved outputs with a fixed fallback message.
- `enforceFailClosedMessage` (string): optional fallback message used when `enforceFailClosed=true`.
- `enforceTemperature` (number): default `0`. Output-guard rewrite temperature (`0..1`).
- `enforceRequireMemorySearch` (boolean): default `false`. When `true`, memory-recall questions are blocked unless `memory_search` was used in the same run.
- `enforceRequireMemorySearchMessage` (string): optional override for the memory-recall block message.
- `enforceHardFile` (string): optional hard-constraint file path (workspace-relative or absolute).
- `enforceHardFiles` (string[]): optional hard-constraint file paths.
- `enforceHardContent` (string): optional inline hard constraints.
- `enforceHardLines` (string[]): optional inline hard constraints.

Hard constraints support line formats:

- `[constraint sentence]`
- `HC: constraint sentence`

If both file-based and inline values are provided, all snippets are concatenated.
