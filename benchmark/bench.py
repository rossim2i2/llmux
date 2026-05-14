#!/usr/bin/env python3
"""
bench.py — Hybrid model routing benchmark.

Routes prompts through the model-router gateway and measures latency, throughput,
token counts, and cost across multiple models. Captures responses for optional
quality scoring via judge.py.

The point of this script is not to pick a "winner" — it's to map *where* small
local models start to fall behind larger ones, so you know what to route locally
and what to send to a frontier API.

Usage:

    # Quick local-only run
    python benchmark/bench.py --models qwen3:4b-local

    # Local + frontier comparison (frontier runs fewer iterations to save cost)
    python benchmark/bench.py \\
        --local-models qwen3:4b-local qwen3:8b-local \\
        --frontier-models gpt-4.1-nano

    # Only specific categories
    python benchmark/bench.py --models qwen3:4b-local \\
        --categories code-generation tool-use

    # More variance runs (default is 3 local, 1 frontier)
    python benchmark/bench.py --local-models qwen3:4b-local --local-runs 5

Environment:
    OPENROUTER_API_KEY     Forwarded to gateway as Authorization header for paid providers.

Notes on what this measures vs. what it doesn't:
    - Measures: TTFT, total latency, throughput (tok/s), input/output tokens, cost.
    - Captures responses for offline quality grading via judge.py.
    - DOES NOT score quality automatically — run judge.py against the output JSON
      to add 1-5 quality ratings from a frontier model.
    - DOES NOT test actual function-calling format. Tool-use prompts test the
      model's *reasoning* about tool calls, not real protocol compliance.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

# ── data types ─────────────────────────────────────────────────────────


@dc.dataclass
class Run:
    model: str
    prompt_id: str
    category: str
    difficulty: str
    iteration: int  # 1-based; iteration 1 = first call (cold)
    cold: bool
    ttft_ms: float | None
    total_ms: float
    tokens_per_sec: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    response: str
    error: str | None
    timestamp: str


# ── hardware info (best-effort) ────────────────────────────────────────


def collect_hardware() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    # CPU model from /proc/cpuinfo (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    # Total RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    # GPU info — AMD ROCm or NVIDIA
    for cmd in (
        ["rocm-smi", "--showproductname", "--csv"],
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and r.stdout.strip():
                info["gpu"] = r.stdout.strip().splitlines()
                break
        except Exception:
            continue
    return info


# ── gateway client ─────────────────────────────────────────────────────


class GatewayClient:
    """Talks to the model-router gateway. Forces a specific model via ?route=."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.Client(timeout=180.0)

    def close(self):
        self.client.close()

    def chat(self, model: str, messages: list) -> tuple[str, dict, float | None, float]:
        """Send a chat request, stream the response, capture TTFT.

        Returns (response_text, usage_dict, ttft_ms, total_ms).
        """
        url = f"{self.base_url}/chat/completions?route={model}"
        body = {"messages": messages, "stream": True}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        usage: dict = {}

        with self.client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        chunks.append(content)

        total_ms = (time.perf_counter() - start) * 1000
        return "".join(chunks), usage, ttft_ms, total_ms


# ── benchmark runner ───────────────────────────────────────────────────


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["prompts"]


def load_cost_table(config_path: Path) -> dict[str, tuple[float, float]]:
    """Read cost_per_1m_input and cost_per_1m_output for each model from gateway config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        name: (
            float(m.get("cost_per_1m_input", 0.0)),
            float(m.get("cost_per_1m_output", 0.0)),
        )
        for name, m in cfg["models"].items()
    }


def run_one(
    client: GatewayClient,
    model: str,
    prompt: dict,
    iteration: int,
    cost_table: dict[str, tuple[float, float]],
) -> Run:
    cold = iteration == 1
    messages = [{"role": "user", "content": prompt["prompt"]}]
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        response, usage, ttft, total = client.chat(model, messages)
    except Exception as e:
        return Run(
            model=model,
            prompt_id=prompt["id"],
            category=prompt["category"],
            difficulty=prompt.get("difficulty", "unknown"),
            iteration=iteration,
            cold=cold,
            ttft_ms=None,
            total_ms=0.0,
            tokens_per_sec=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            response="",
            error=f"{type(e).__name__}: {e}",
            timestamp=timestamp,
        )

    inp = int(usage.get("prompt_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    cost_in, cost_out = cost_table.get(model, (0.0, 0.0))
    cost = (inp * cost_in + out * cost_out) / 1_000_000 if (cost_in or cost_out) else 0.0
    tps = (out / (total / 1000)) if total > 0 and out > 0 else None

    return Run(
        model=model,
        prompt_id=prompt["id"],
        category=prompt["category"],
        difficulty=prompt.get("difficulty", "unknown"),
        iteration=iteration,
        cold=cold,
        ttft_ms=ttft,
        total_ms=total,
        tokens_per_sec=tps,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=cost,
        response=response,
        error=None,
        timestamp=timestamp,
    )


def run_benchmark(
    client: GatewayClient,
    model: str,
    prompts: list[dict],
    runs: int,
    cost_table: dict[str, tuple[float, float]],
) -> list[Run]:
    results: list[Run] = []
    for prompt in prompts:
        for i in range(1, runs + 1):
            r = run_one(client, model, prompt, i, cost_table)
            results.append(r)
            if r.error:
                print(
                    f"  {model:25s} [{i}/{runs}] {r.prompt_id:30s} ERROR: {r.error}",
                    file=sys.stderr,
                )
            else:
                ttft_str = f"{r.ttft_ms:5.0f}" if r.ttft_ms is not None else "  n/a"
                print(
                    f"  {model:25s} [{i}/{runs}] {r.prompt_id:30s} "
                    f"ttft={ttft_str}ms total={r.total_ms:6.0f}ms "
                    f"{r.output_tokens:4d}tok ${r.cost_usd or 0:.6f}"
                )
    return results


# ── reporting ──────────────────────────────────────────────────────────


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


def summarize(runs: list[Run]) -> None:
    if not runs:
        return
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    by_model: dict[str, list[Run]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)

    for model, model_runs in by_model.items():
        ok = [r for r in model_runs if not r.error]
        errors = len(model_runs) - len(ok)
        # Exclude cold runs (model load) from latency stats when warm runs exist
        warm = [r for r in ok if not r.cold] or ok

        ttfts = [r.ttft_ms for r in warm if r.ttft_ms is not None]
        totals = [r.total_ms for r in warm]
        tpss = [r.tokens_per_sec for r in warm if r.tokens_per_sec is not None]
        total_cost = sum(r.cost_usd or 0 for r in ok)
        total_input = sum(r.input_tokens for r in ok)
        total_output = sum(r.output_tokens for r in ok)

        print(f"\n{model}  ({len(ok)} ok / {errors} err)")
        if ttfts:
            print(f"  TTFT     median={statistics.median(ttfts):6.0f}ms  p95={_p95(ttfts):6.0f}ms")
        if totals:
            print(f"  Total    median={statistics.median(totals):6.0f}ms  p95={_p95(totals):6.0f}ms")
        if tpss:
            print(f"  Speed    median={statistics.median(tpss):6.1f}tok/s")
        print(f"  Tokens   in={total_input}  out={total_output}")
        print(f"  Cost     total=${total_cost:.4f}  per_call=${total_cost/max(len(ok),1):.6f}")

        # By difficulty — useful to see where each model breaks down
        by_diff: dict[str, list[Run]] = {}
        for r in ok:
            by_diff.setdefault(r.difficulty, []).append(r)
        for diff in ("trivial", "easy", "medium", "hard"):
            if diff in by_diff:
                d_totals = [r.total_ms for r in by_diff[diff]]
                print(f"  {diff:8s} ({len(by_diff[diff])} runs) median={statistics.median(d_totals):6.0f}ms")


def write_results(runs: list[Run], hardware: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "runs": [dc.asdict(r) for r in runs],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Hybrid model router benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gateway",
        default="http://localhost:8001",
        help="Gateway base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="API key for paid providers (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--prompts",
        default=str(Path(__file__).with_name("prompts.yaml")),
        help="Path to prompts YAML",
    )
    parser.add_argument(
        "--config",
        default=str(repo_root / "config.yaml"),
        help="Path to gateway config.yaml (for cost lookup)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Models to test using --runs iterations (mixed list)",
    )
    parser.add_argument(
        "--local-models",
        nargs="*",
        default=[],
        help="Local model keys (use --local-runs iterations)",
    )
    parser.add_argument(
        "--frontier-models",
        nargs="*",
        default=[],
        help="Frontier model keys (use --frontier-runs iterations, cost control)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Iterations per prompt for --models")
    parser.add_argument(
        "--local-runs", type=int, default=3, help="Iterations per prompt for --local-models"
    )
    parser.add_argument(
        "--frontier-runs",
        type=int,
        default=1,
        help="Iterations per prompt for --frontier-models",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Filter prompts by category (e.g. code-generation tool-use)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=None,
        help="Filter prompts by difficulty (trivial/easy/medium/hard)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: benchmark/results/<timestamp>.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of prompts (useful for smoke tests)",
    )

    args = parser.parse_args()

    if not (args.models or args.local_models or args.frontier_models):
        parser.error("Specify at least one of --models, --local-models, --frontier-models")

    prompts = load_prompts(Path(args.prompts))
    if args.categories:
        prompts = [p for p in prompts if p["category"] in args.categories]
    if args.difficulties:
        prompts = [p for p in prompts if p.get("difficulty") in args.difficulties]
    if args.limit:
        prompts = prompts[: args.limit]

    if not prompts:
        parser.error("No prompts matched filters")

    cost_table = load_cost_table(Path(args.config))
    hardware = collect_hardware()

    print(f"Prompts:  {len(prompts)}")
    print(f"Hardware: {hardware.get('cpu_model', '?')}")
    print(f"GPU:      {hardware.get('gpu', 'unknown')}")
    print(f"Gateway:  {args.gateway}")
    print()

    client = GatewayClient(args.gateway, args.api_key)
    all_runs: list[Run] = []
    try:
        for model in args.models:
            print(f"=== {model} ({args.runs} runs) ===")
            all_runs.extend(run_benchmark(client, model, prompts, args.runs, cost_table))
            # Save incrementally after each model
            if args.output:
                write_results(all_runs, hardware, Path(args.output))
        for model in args.local_models:
            print(f"=== {model} (local, {args.local_runs} runs) ===")
            all_runs.extend(run_benchmark(client, model, prompts, args.local_runs, cost_table))
            if args.output:
                write_results(all_runs, hardware, Path(args.output))
        for model in args.frontier_models:
            print(f"=== {model} (frontier, {args.frontier_runs} runs) ===")
            all_runs.extend(
                run_benchmark(client, model, prompts, args.frontier_runs, cost_table)
            )
            if args.output:
                write_results(all_runs, hardware, Path(args.output))
    finally:
        client.close()

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(__file__).parent / "results" / f"{ts}.json"

    write_results(all_runs, hardware, output_path)
    summarize(all_runs)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
