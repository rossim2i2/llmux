#!/usr/bin/env python3
"""Validate classify_prompt() against the benchmark dataset.

Runs the production classifier on every prompt in prompts.yaml, maps each
result to a routing tier (local/mid/frontier), and compares to the ideal
tier derived from the prompt's difficulty label.

Reports two accuracy metrics:
  - exact_match: classifier picked the same tier as ideal
  - no_worse_than_ideal: classifier picked a tier at least as capable as ideal
    (e.g., easy prompt sent to mid or frontier is acceptable, just expensive)
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Make the gateway module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import classify_prompt  # noqa: E402

# Difficulty → ideal routing tier
IDEAL_TIER = {
    "trivial": "local",
    "easy": "local",
    "medium": "mid",
    "hard": "frontier",
}

# Tier capability ordering (higher = more capable)
TIER_RANK = {"local": 0, "mid": 1, "frontier": 2}


def score_to_tier(score: int) -> str:
    """Replicate the routing logic in choose_model()."""
    if score <= 0:
        return "local"
    if score <= 3:
        return "mid"
    return "frontier"


def main() -> int:
    prompts_path = Path(__file__).resolve().parent / "prompts.yaml"
    with prompts_path.open() as f:
        data = yaml.safe_load(f)

    prompts = data.get("prompts", [])

    exact_matches = 0
    no_worse = 0
    rows = []

    for p in prompts:
        pid = p.get("id", "?")
        difficulty = p.get("difficulty", "?")
        category = p.get("category", "?")
        text = p.get("prompt", "")

        score, signals = classify_prompt(text)
        predicted = score_to_tier(score)
        ideal = IDEAL_TIER.get(difficulty, "frontier")

        is_exact = predicted == ideal
        is_no_worse = TIER_RANK[predicted] >= TIER_RANK[ideal]

        if is_exact:
            exact_matches += 1
        if is_no_worse:
            no_worse += 1

        status = "  " if is_exact else ("ok" if is_no_worse else "FAIL")
        rows.append({
            "status": status,
            "id": pid,
            "category": category,
            "difficulty": difficulty,
            "score": score,
            "predicted": predicted,
            "ideal": ideal,
            "signals": signals,
        })

    # Print results
    total = len(prompts)
    print(f"{'='*100}")
    print(f"Classifier validation against {total} benchmark prompts")
    print(f"{'='*100}")
    print(f"{'STATUS':<6} {'ID':<28} {'DIFF':<8} {'SCORE':>5}  {'PRED':<9} {'IDEAL':<9} SIGNALS")
    print(f"{'-'*100}")
    for r in rows:
        signals_str = ",".join(r["signals"]) if r["signals"] else "-"
        print(f"{r['status']:<6} {r['id']:<28} {r['difficulty']:<8} {r['score']:>5}  {r['predicted']:<9} {r['ideal']:<9} {signals_str}")

    print(f"{'-'*100}")
    exact_pct = 100 * exact_matches / total
    no_worse_pct = 100 * no_worse / total
    print(f"Exact match:         {exact_matches}/{total}  ({exact_pct:.1f}%)")
    print(f"No worse than ideal: {no_worse}/{total}  ({no_worse_pct:.1f}%)")

    # By difficulty breakdown
    print()
    print("By difficulty:")
    by_diff: dict[str, list[dict]] = {}
    for r in rows:
        by_diff.setdefault(r["difficulty"], []).append(r)
    for diff in ("trivial", "easy", "medium", "hard"):
        items = by_diff.get(diff, [])
        if not items:
            continue
        exact = sum(1 for r in items if r["predicted"] == r["ideal"])
        no_w = sum(1 for r in items if TIER_RANK[r["predicted"]] >= TIER_RANK[r["ideal"]])
        print(f"  {diff:<8}  exact={exact}/{len(items)}  no_worse={no_w}/{len(items)}")

    # Failures (worse than ideal)
    failures = [r for r in rows if TIER_RANK[r["predicted"]] < TIER_RANK[r["ideal"]]]
    if failures:
        print()
        print(f"FAILURES (routed below ideal tier): {len(failures)}")
        for r in failures:
            signals_str = ",".join(r["signals"]) if r["signals"] else "-"
            print(f"  {r['id']} ({r['difficulty']}): score={r['score']} predicted={r['predicted']} ideal={r['ideal']} signals=[{signals_str}]")

    # Return non-zero exit if accuracy thresholds not met
    if exact_pct < 70 or no_worse_pct < 95:
        print()
        print(f"FAIL: targets are exact >= 70% and no_worse >= 95%")
        return 1
    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
