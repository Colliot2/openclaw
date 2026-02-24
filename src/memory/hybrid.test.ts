import { describe, expect, it } from "vitest";
import { bm25RankToScore, buildFtsQuery, mergeHybridResults, resolvePathBoost } from "./hybrid.js";

describe("memory hybrid helpers", () => {
  it("buildFtsQuery tokenizes and AND-joins", () => {
    expect(buildFtsQuery("hello world")).toBe('"hello" AND "world"');
    expect(buildFtsQuery("FOO_bar baz-1")).toBe('"FOO_bar" AND "baz" AND "1"');
    expect(buildFtsQuery("金银价格")).toBe('"金银价格"');
    expect(buildFtsQuery("価格 2026年")).toBe('"価格" AND "2026年"');
    expect(buildFtsQuery("   ")).toBeNull();
  });

  it("bm25RankToScore is monotonic and clamped", () => {
    expect(bm25RankToScore(0)).toBeCloseTo(1);
    expect(bm25RankToScore(1)).toBeCloseTo(0.5);
    expect(bm25RankToScore(10)).toBeLessThan(bm25RankToScore(1));
    expect(bm25RankToScore(-100)).toBeCloseTo(1);
  });

  it("mergeHybridResults unions by id and combines weighted scores", async () => {
    const merged = await mergeHybridResults({
      vectorWeight: 0.7,
      textWeight: 0.3,
      vector: [
        {
          id: "a",
          path: "memory/a.txt",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "vec-a",
          vectorScore: 0.9,
        },
      ],
      keyword: [
        {
          id: "b",
          path: "memory/b.txt",
          startLine: 3,
          endLine: 4,
          source: "memory",
          snippet: "kw-b",
          textScore: 1.0,
        },
      ],
    });

    expect(merged).toHaveLength(2);
    const a = merged.find((r) => r.path === "memory/a.txt");
    const b = merged.find((r) => r.path === "memory/b.txt");
    expect(a?.score).toBeCloseTo(0.7 * 0.9);
    expect(b?.score).toBeCloseTo(0.3 * 1.0);
  });

  it("mergeHybridResults prefers keyword snippet when ids overlap", async () => {
    const merged = await mergeHybridResults({
      vectorWeight: 0.5,
      textWeight: 0.5,
      vector: [
        {
          id: "a",
          path: "memory/a.txt",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "vec-a",
          vectorScore: 0.2,
        },
      ],
      keyword: [
        {
          id: "a",
          path: "memory/a.txt",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "kw-a",
          textScore: 1.0,
        },
      ],
    });

    expect(merged).toHaveLength(1);
    expect(merged[0]?.snippet).toBe("kw-a");
    expect(merged[0]?.score).toBeCloseTo(0.5 * 0.2 + 0.5 * 1.0);
  });

  it("applies path boosts with MEMORY.md highest priority", async () => {
    const merged = await mergeHybridResults({
      vectorWeight: 1,
      textWeight: 0,
      vector: [
        {
          id: "memory-root",
          path: "MEMORY.md",
          startLine: 1,
          endLine: 1,
          source: "memory",
          snippet: "root",
          vectorScore: 0.5,
        },
        {
          id: "markdown",
          path: "docs/policy.md",
          startLine: 1,
          endLine: 1,
          source: "memory",
          snippet: "md",
          vectorScore: 0.5,
        },
        {
          id: "plain",
          path: "notes.txt",
          startLine: 1,
          endLine: 1,
          source: "memory",
          snippet: "txt",
          vectorScore: 0.5,
        },
      ],
      keyword: [],
    });
    expect(merged[0]?.path).toBe("MEMORY.md");
    expect(merged[1]?.path).toBe("docs/policy.md");
    expect(merged[2]?.path).toBe("notes.txt");
  });

  it("resolves expected path boost multipliers", () => {
    expect(resolvePathBoost("MEMORY.md")).toBe(1.7);
    expect(resolvePathBoost("/tmp/workspace/MEMORY.md")).toBe(1.7);
    expect(resolvePathBoost("memory/topic.md")).toBe(1.3);
    expect(resolvePathBoost("memory/topic.txt")).toBe(1);
  });
});
