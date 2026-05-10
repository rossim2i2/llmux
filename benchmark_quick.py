#!/usr/bin/env python3
"""Quick benchmark: shorter prompts, 60s timeout each."""

import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List
import urllib.request
import urllib.error
import socket

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b"
PROMPT_TIMEOUT = 60  # seconds


@dataclass
class Result:
    category: str
    prompt_len: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float
    output_tokens: int
    output_chars: int


PROMPTS = [
    ("code-short", "Write a one-liner Python list comprehension for squares of 0-9."),
    ("git-op", "What's the git command to undo the last commit but keep changes?"),
    ("search-analysis", "What Python library is likely used when you see 'import requests'?"),
    ("short-reasoning", "Is 17 prime? Why or why not?"),
    ("config", "Write a minimal nginx server block for port 80 serving /var/www."),
    ("explain", "Explain 'time to first token' in one sentence."),
    ("refactor", "Make this shorter: def f(x): if x > 0: return x else: return 0"),
    ("fix-error", "Fix: NameError: name 'data' is not defined"),
]


def run_prompt(prompt: str) -> tuple[str, float, float, float, int]:
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2}
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()
    first_token_time = None
    chunks = []
    eval_count = 0
    eval_duration = 1

    with urllib.request.urlopen(req, timeout=PROMPT_TIMEOUT) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line.decode())
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()
            chunks.append(chunk.get("response", ""))
            if chunk.get("done"):
                eval_count = chunk.get("eval_count", 0)
                eval_duration = chunk.get("eval_duration", 1) / 1e9

    total = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else total
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    return "".join(chunks), ttft * 1000, total * 1000, tps, eval_count


def main():
    results = []
    print(f"Benchmarking {MODEL} — {len(PROMPTS)} prompts, {PROMPT_TIMEOUT}s timeout each\n")

    for cat, prompt in PROMPTS:
        print(f"  {cat:20s} ", end="", flush=True)
        try:
            resp, ttft, total, tps, tokens = run_prompt(prompt)
            r = Result(cat, len(prompt), ttft, total, tps, tokens, len(resp))
            results.append(r)
            print(f"TTFT={ttft:.0f}ms  Total={total:.0f}ms  {tps:.1f}tok/s  {tokens}tok")
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("No successful runs.")
        return

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    ttfts = [r.ttft_ms for r in results]
    totals = [r.total_ms for r in results]
    tpss = [r.tokens_per_sec for r in results]
    tok_counts = [r.output_tokens for r in results]

    print(f"  Prompts:     {len(results)}/{len(PROMPTS)} succeeded")
    print(f"  TTFT:        {statistics.mean(ttfts):.0f}ms  (median {statistics.median(ttfts):.0f}ms)")
    print(f"  Total time:  {statistics.mean(totals):.0f}ms  (median {statistics.median(totals):.0f}ms)")
    print(f"  Throughput:  {statistics.mean(tpss):.1f} tok/s  (median {statistics.median(tpss):.1f})")
    print(f"  Avg output:  {statistics.mean(tok_counts):.0f} tokens")

    # Category breakdown
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    print(f"\n  By category:")
    for cat, rs in sorted(by_cat.items()):
        print(f"    {cat:18s}: {statistics.mean([x.total_ms for x in rs]):.0f}ms avg")

    with open("benchmark_quick_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Saved to benchmark_quick_results.json")


if __name__ == "__main__":
    main()
