#!/usr/bin/env python3
"""LLM-as-judge for recent unjudged requests.

Samples unjudged requests from the database, sends prompt+response to a judge
model (gpt-5.4-nano by default), and updates judged_* columns with results.

Designed to run as a daily cron. Target: ~50 requests/day, ~$0.50/day cost.

Usage:
    python scripts/judge-recent.py [--limit 50] [--model gpt-5.4-nano] [--dry-run]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. pip install httpx", file=sys.stderr)
    sys.exit(1)

DB_PATH = Path(__file__).parent.parent / "router.db"
JUDGE_MODEL_DEFAULT = "gpt-5-nano"
SAMPLE_LIMIT_DEFAULT = 50

JUDGE_SYSTEM_PROMPT = """You are a prompt complexity evaluator. Determine the minimum model tier needed.

Tiers: local (simple), mid (code/debug), frontier (architecture/security/complex).

Respond in JSON only, no markdown:
{"complexity": "local"|"mid"|"frontier", "quality": 1-5, "reasoning": "brief"}

Quality: 1=wrong, 2=partial, 3=ok, 4=good, 5=excellent."""

JUDGE_USER_TEMPLATE = """PROMPT: {prompt}

RESPONSE: {response}

JSON:"""


def get_openai_client():
    """Get httpx client and API key for OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return httpx.Client(timeout=60.0), api_key


def fetch_unjudged(limit: int, prioritize_flags: bool = True) -> list[dict]:
    """Fetch unjudged requests, prioritizing truncated/short responses."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if prioritize_flags:
        # Prioritize rows with response flags set
        query = """
            SELECT id, request_id, prompt_text, response_text, chosen_tier, score, signals
            FROM requests
            WHERE judged_quality IS NULL
              AND prompt_text IS NOT NULL
              AND response_text IS NOT NULL
              AND response_text != ''
            ORDER BY response_truncated DESC, response_short DESC, timestamp DESC
            LIMIT ?
        """
    else:
        query = """
            SELECT id, request_id, prompt_text, response_text, chosen_tier, score, signals
            FROM requests
            WHERE judged_quality IS NULL
              AND prompt_text IS NOT NULL
              AND response_text IS NOT NULL
              AND response_text != ''
            ORDER BY timestamp DESC
            LIMIT ?
        """

    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def call_judge(client: httpx.Client, api_key: str, model: str, prompt: str, response: str) -> dict:
    """Call the judge model and parse JSON response."""
    user_msg = JUDGE_USER_TEMPLATE.format(prompt=prompt[:4000], response=response[:4000])

    resp = client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_completion_tokens": 1000,
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()

    content = data["choices"][0]["message"].get("content") or ""
    finish = data["choices"][0].get("finish_reason")
    if not content or finish == "length":
        raise RuntimeError(f"Judge response empty or truncated (finish={finish})")

    # Parse JSON from response (handle markdown fences)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned non-JSON: {content[:200]}") from e


def update_judgment(row_id: int, result: dict, model: str):
    """Update database with judge result."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE requests
        SET judged_quality = ?,
            judged_at = ?,
            judged_by = ?,
            judged_reasoning = ?,
            ideal_tier = COALESCE(?, ideal_tier)
        WHERE id = ?
        """,
        (
            result.get("quality"),
            time.time(),
            model,
            result.get("reasoning", "")[:500],
            result.get("complexity"),
            row_id,
        ),
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Judge recent unjudged requests")
    parser.add_argument("--limit", type=int, default=SAMPLE_LIMIT_DEFAULT, help="Max requests to judge")
    parser.add_argument("--model", default=JUDGE_MODEL_DEFAULT, help="Judge model")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each judgment")
    args = parser.parse_args()

    print(f"[JUDGE] Fetching up to {args.limit} unjudged requests...")
    rows = fetch_unjudged(args.limit)
    print(f"[JUDGE] Found {len(rows)} candidates")

    if not rows:
        print("[JUDGE] Nothing to judge. Done.")
        return

    client, api_key = get_openai_client()

    judged = 0
    errors = 0

    for row in rows:
        rid = row["request_id"][:8]
        try:
            if args.verbose:
                print(f"[JUDGE] {rid}: prompt='{row['prompt_text'][:40]}...' chosen={row['chosen_tier']}")

            result = call_judge(client, api_key, args.model, row["prompt_text"], row["response_text"])

            if args.verbose:
                print(f"        → complexity={result.get('complexity')} quality={result.get('quality')}")

            if not args.dry_run:
                update_judgment(row["id"], result, args.model)

            judged += 1

        except Exception as e:
            print(f"[JUDGE] {rid}: ERROR - {e}", file=sys.stderr)
            errors += 1
            if errors > 5:
                print("[JUDGE] Too many errors, aborting", file=sys.stderr)
                break

    print(f"[JUDGE] Done. Judged={judged} Errors={errors}")


if __name__ == "__main__":
    main()
