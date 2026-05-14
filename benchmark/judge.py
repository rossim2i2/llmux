#!/usr/bin/env python3
"""
judge.py — Optional LLM-judge quality scoring for benchmark results.

Reads a benchmark output JSON (from bench.py), sends each (prompt, response, rubric)
triple to a frontier model, and asks it to score the response 1-5 across:
    - correctness  (does it solve the task?)
    - completeness (does it cover what was asked?)
    - clarity      (is it well-structured and readable?)

Writes a new JSON with scores attached. Original responses are preserved.

Usage:

    # Score with Claude Sonnet via OpenRouter (configured in gateway)
    python benchmark/judge.py \\
        --results benchmark/results/20260513-093000.json \\
        --judge-model claude-sonnet-4-6

    # Limit to a sample for cost control
    python benchmark/judge.py --results <file> --judge-model gpt-4.1-nano --sample 20

Cost:
    Each judged run is one extra API call to the judge model. With ~200 runs
    and gpt-4.1-nano ($0.10/$0.40 per 1M), full scoring is well under $1.
    Sonnet ($3/$15) is more expensive but typically more reliable as a judge.

Notes:
    - Skips runs without a rubric in the original prompt set.
    - Skips runs that errored.
    - Judge sees the prompt, the response, and the rubric — not the model name.
    - Output JSON adds a `scores` field per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

JUDGE_SYSTEM_PROMPT = """You are an expert technical reviewer grading model outputs against a rubric.

You will receive:
- The original prompt
- The model's response
- A rubric describing what a correct, complete answer should contain

Your job: return ONLY a JSON object (no prose, no markdown fences) with three integer scores 1-5 and a one-sentence reason:

{"correctness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "reason": "<one sentence>"}

Scoring guide:
  5 = fully correct and complete
  4 = correct but missing minor details
  3 = mostly correct with notable gaps or one significant error
  2 = partially correct, significant issues
  1 = wrong, off-topic, or empty

Be strict. If the rubric specifies a constraint (e.g., "one line", "no off-by-one errors") and the response violates it, lower the score accordingly. Do not award points for prose that doesn't actually answer the question.
"""


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_prompt_lookup(prompts_path: Path) -> dict[str, dict]:
    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    return {p["id"]: p for p in data["prompts"]}


def judge_one(
    client: httpx.Client,
    gateway: str,
    api_key: str | None,
    judge_model: str,
    prompt_text: str,
    response_text: str,
    rubric: str,
) -> dict:
    user_msg = (
        f"=== PROMPT ===\n{prompt_text}\n\n"
        f"=== RESPONSE ===\n{response_text}\n\n"
        f"=== RUBRIC ===\n{rubric}\n\n"
        f"Return only the JSON object."
    )
    body = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{gateway.rstrip('/')}/chat/completions?route={judge_model}"
    resp = client.post(url, json=body, headers=headers, timeout=120.0)
    resp.raise_for_status()
    obj = resp.json()
    text = obj["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if the judge added them despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"Judge returned non-JSON: {text[:200]}"}

    # Tolerate judge returning duplicate keys (e.g. "correctness" twice instead of "completeness")
    if "completeness" not in scores and "correctness" in scores:
        # If correctness appears but completeness is missing, assume the judge
        # duplicated correctness. Use the same value for completeness.
        scores["completeness"] = scores["correctness"]

    # Sanity-check shape
    for key in ("correctness", "completeness", "clarity"):
        if key not in scores:
            return {"error": f"Missing key '{key}' in judge response: {text[:200]}"}
    return scores


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="LLM-judge quality scoring")
    parser.add_argument("--results", required=True, help="Path to benchmark results JSON")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Model key (as configured in gateway) to use as judge",
    )
    parser.add_argument(
        "--gateway", default="http://localhost:8001", help="Gateway base URL"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="API key for paid providers (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--prompts",
        default=str(Path(__file__).with_name("prompts.yaml")),
        help="Prompt library path (for rubrics)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Score only the first N runs (cost control)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <results>.judged.json)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (rate-limit safety)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    data = load_results(results_path)
    prompts = load_prompt_lookup(Path(args.prompts))

    runs = data["runs"]
    if args.sample:
        runs = runs[: args.sample]

    judgeable = [
        r
        for r in runs
        if not r.get("error") and r.get("response") and prompts.get(r["prompt_id"], {}).get("rubric")
    ]
    print(f"Total runs: {len(data['runs'])}")
    print(f"Judging:    {len(judgeable)} (sampled to {len(runs)})")
    print(f"Judge:      {args.judge_model}")
    print()

    client = httpx.Client()
    scored = 0
    try:
        for r in judgeable:
            prompt = prompts.get(r["prompt_id"])
            if not prompt:
                continue
            rubric = prompt.get("rubric", "")
            scores = judge_one(
                client,
                args.gateway,
                args.api_key,
                args.judge_model,
                prompt["prompt"],
                r["response"],
                rubric,
            )
            r["scores"] = scores
            scored += 1
            if "error" in scores:
                print(
                    f"  [{scored}/{len(judgeable)}] {r['model']:25s} {r['prompt_id']:30s} "
                    f"JUDGE ERROR: {scores['error']}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [{scored}/{len(judgeable)}] {r['model']:25s} {r['prompt_id']:30s} "
                    f"correctness={scores['correctness']} completeness={scores['completeness']} "
                    f"clarity={scores['clarity']}"
                )
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        client.close()

    data["judge_model"] = args.judge_model
    data["judged_at"] = datetime.now(timezone.utc).isoformat()

    output_path = (
        Path(args.output)
        if args.output
        else results_path.with_suffix(".judged.json")
    )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Summary by model
    print("\n" + "=" * 80)
    print("JUDGE SUMMARY (median scores)")
    print("=" * 80)
    by_model: dict[str, dict[str, list[int]]] = {}
    for r in data["runs"]:
        if not r.get("scores") or "error" in r["scores"]:
            continue
        m = by_model.setdefault(r["model"], {"correctness": [], "completeness": [], "clarity": []})
        for key in m:
            m[key].append(r["scores"][key])

    for model, scores in by_model.items():
        if not scores["correctness"]:
            continue
        median = lambda lst: sorted(lst)[len(lst) // 2]
        print(
            f"  {model:30s} correctness={median(scores['correctness'])} "
            f"completeness={median(scores['completeness'])} "
            f"clarity={median(scores['clarity'])}  (n={len(scores['correctness'])})"
        )

    print(f"\nScored results: {output_path}")


if __name__ == "__main__":
    main()
