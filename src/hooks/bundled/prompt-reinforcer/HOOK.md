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
- `enforceSoftMaxPasses` (number): default `3` (clamped `1..10`). Max rewrite attempts for soft policy (`PROMPT_OVERRIDE`).
- `enforceHardMaxPasses` (number): default `10` (clamped `1..10`). Max rewrite attempts for hard constraints (`PROMPT_HARD_CONSTRAINTS`).
- `enforceMaxPasses` (number): legacy alias used as fallback for `enforceHardMaxPasses`.
- `enforceSoftFailOpen` (boolean): default `true`. If soft policy still fails after `enforceSoftMaxPasses`, pass the latest rewrite through.
- `enforceHardFailClosed` (boolean): default `true`. If hard constraints still fail after `enforceHardMaxPasses`, block output.
- `enforceFailClosed` (boolean): default `false`. When `true`, block unresolved outputs with a fixed fallback message.
- `enforceFailClosedMessage` (string): optional fallback message used when `enforceFailClosed=true`.
- `enforceHardFailClosedMessage` (string): optional hard-constraint block message (falls back to `enforceFailClosedMessage`).
- `enforceTemperature` (number): default `0`. Output-guard rewrite temperature (`0..1`).
- `enforceRequireMemorySearch` (boolean): default `false`. When `true`, memory-recall questions are blocked unless `memory_search` was used in the same run.
- `enforceRequireMemorySearchMessage` (string): optional override for the memory-recall block message.
- `enforceMemoryRetryMaxAttempts` (number): default `2` (clamped `1..10`). Max run-level retries for `memory_search_required`.
- `enforceMemoryRetryFailOpen` (boolean): default `true`. If memory retry budget is exhausted, return the latest pre-guard draft (fail-open).
- `enforcePolicyRetryMaxAttempts` (number): default `1` (clamped `1..10`). Run-level retries for policy/hard failures (`soft_policy_blocked`, `hard_constraints_blocked`, plus legacy `policy_guard_blocked`).
- `enforcePolicyRetryFailOpen` (boolean): default `false`. Optional fail-open after policy/hard retries are exhausted.
- `enforceHardFile` (string): optional hard-constraint file path (workspace-relative or absolute).
- `enforceHardFiles` (string[]): optional hard-constraint file paths.
- `enforceHardContent` (string): optional inline hard constraints.
- `enforceHardLines` (string[]): optional inline hard constraints.

Hard constraints support line formats:

- `[constraint sentence]`
- `HC: constraint sentence`

If both file-based and inline values are provided, all snippets are concatenated.
