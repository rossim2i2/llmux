#!/usr/bin/env python3
"""Benchmark qwen3:4b for routine agent tasks via Ollama API."""

import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b"


@dataclass
class Result:
    prompt: str
    category: str
    ttft_ms: float      # time to first token
    total_ms: float     # total generation time
    tokens_per_sec: float
    response: str
    response_length: int


PROMPTS = [
    # File edits / code generation
    {
        "category": "code-edit",
        "prompt": "Write a Python function that reads a JSON file and returns a sorted list of keys. Include type hints and a docstring."
    },
    {
        "category": "code-edit",
        "prompt": "Refactor this function to use a list comprehension:\ndef squares(n):\n    result = []\n    for i in range(n):\n        result.append(i * i)\n    return result"
    },
    # Git operations (explanations)
    {
        "category": "git-op",
        "prompt": "Explain the difference between `git rebase` and `git merge` in 3 sentences."
    },
    {
        "category": "git-op",
        "prompt": "I accidentally committed a file to git. How do I remove it from the last commit without losing my other changes?"
    },
    # Searches / analysis
    {
        "category": "search-analysis",
        "prompt": "Given this directory listing: ['src/', 'tests/', 'README.md', 'setup.py', '.gitignore'], suggest a likely project type and its main language."
    },
    {
        "category": "search-analysis",
        "prompt": "Analyze this error message and suggest the fix: 'ModuleNotFoundError: No module named requests'"
    },
    # Short reasoning
    {
        "category": "short-reasoning",
        "prompt": "If a train travels 60 miles in 1.5 hours, how far will it travel in 2.5 hours at the same speed? Show your work."
    },
    {
        "category": "short-reasoning",
        "prompt": "Which is larger: 3/7 or 4/9? Explain without converting to decimals."
    },
    # Configuration / yaml
    {
        "category": "config",
        "prompt": "Write a minimal docker-compose.yml for a Python Flask app with a Redis cache."
    },
    {
        "category": "config",
        "prompt": "Convert this env var list to a YAML config map:\nDB_HOST=localhost\nDB_PORT=5432\nDB_NAME=myapp"
    },
]


def run_prompt(prompt: str) -> tuple[str, float, float, float]:
    """Send prompt to Ollama, stream response, measure timing."""
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
    response_chunks = []
    eval_count = 0
    eval_duration = 0

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line.decode())
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()
            response_chunks.append(chunk.get("response", ""))
            if chunk.get("done"):
                eval_count = chunk.get("eval_count", 0)
                eval_duration = chunk.get("eval_duration", 1) / 1e9  # ns -> s

    total_time = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else total_time
    tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
    full_response = "".join(response_chunks)

    return full_response, ttft * 1000, total_time * 1000, tokens_per_sec


def benchmark() -> List[Result]:
    results = []
    print(f"Benchmarking {MODEL} against {len(PROMPTS)} prompts...\n")

    for i, item in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {item['category']}: ", end="", flush=True)
        try:
            response, ttft, total, tps = run_prompt(item["prompt"])
            result = Result(
                prompt=item["prompt"][:80] + "..." if len(item["prompt"]) > 80 else item["prompt"],
                category=item["category"],
                ttft_ms=ttft,
                total_ms=total,
                tokens_per_sec=tps,
                response=response,
                response_length=len(response)
            )
            results.append(result)
            print(f"TTFT={ttft:.0f}ms Total={total:.0f}ms {tps:.1f}tok/s")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def report(results: List[Result]):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_category = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    all_ttft = [r.ttft_ms for r in results]
    all_total = [r.total_ms for r in results]
    all_tps = [r.tokens_per_sec for r in results]

    print(f"\nOverall ({len(results)} prompts):")
    print(f"  TTFT:  {statistics.mean(all_ttft):.0f}ms (median {statistics.median(all_ttft):.0f}ms)")
    print(f"  Total: {statistics.mean(all_total):.0f}ms (median {statistics.median(all_total):.0f}ms)")
    print(f"  Speed: {statistics.mean(all_tps):.1f} tok/s (median {statistics.median(all_tps):.1f})")

    print(f"\nBy category:")
    for cat, cat_results in sorted(by_category.items()):
        ttfts = [r.ttft_ms for r in cat_results]
        totals = [r.total_ms for r in cat_results]
        print(f"  {cat:20s}: TTFT={statistics.mean(ttfts):.0f}ms  Total={statistics.mean(totals):.0f}ms  (n={len(cat_results)})")

    # Save full results
    with open("benchmark_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results saved to benchmark_results.json")

    # Print sample responses for quality check
    print("\n" + "=" * 60)
    print("SAMPLE RESPONSES (first 3)")
    print("=" * 60)
    for r in results[:3]:
        print(f"\n--- {r.category} ---")
        print(r.response[:500] + "..." if len(r.response) > 500 else r.response)


if __name__ == "__main__":
    results = benchmark()
    report(results)
