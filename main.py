#!/usr/bin/env python3
"""FastAPI model-router gateway: OpenAI-compatible /chat/completions with pluggable routing."""

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DB_PATH = Path(__file__).with_name("router.db")

# Privacy gate: set LLMUX_CAPTURE_BODIES=0 to disable prompt/response storage.
# Default is on — the feedback loop depends on raw data.
CAPTURE_BODIES = os.environ.get("LLMUX_CAPTURE_BODIES", "1") != "0"

# Provider API keys from environment — used when client doesn't supply one
# or supplies a placeholder (e.g., "sk-placeholder" from proxy configs).
PROVIDER_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
    "google": os.environ.get("GOOGLE_API_KEY", ""),
}

# ── config ──────────────────────────────────────────────────────────

class Config:
    models: dict
    routing: dict
    aliases: dict
    proxy_models: set

    def __init__(self, path: Path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        self.models = raw["models"]
        self.routing = raw.get("routing", {})
        # Build alias map from config + per-model aliases field
        self.aliases = {}
        for key, cfg in self.models.items():
            if "aliases" in cfg:
                for alias in cfg["aliases"]:
                    self.aliases[alias] = key
        # Model names from proxies that should be ignored for routing
        self.proxy_models = set(self.routing.get("proxy_models", []))

    # Provider prefixes that clients may send (e.g. "lmstudio/qwen3:4b-local").
    # Stripped before lookup so the gateway works with any OpenAI-compatible client.
    KNOWN_PREFIXES = ("lmstudio/", "ollama/", "openai/", "openrouter/", "anthropic/", "google/")

    def resolve_model(self, name: str) -> str:
        """Resolve a model name (possibly an alias or provider-prefixed) to a config key."""
        # Strip known provider prefixes (lmstudio/qwen3:4b-local → qwen3:4b-local)
        for prefix in self.KNOWN_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name in self.models:
            return name
        if name in self.aliases:
            return self.aliases[name]
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")

    def get_model(self, name: str):
        key = self.resolve_model(name)
        return self.models[key]

    @property
    def default_model(self) -> str:
        return self.routing.get("default", next(iter(self.models)))

    @property
    def force_param(self) -> str:
        return self.routing.get("force_param", "route")


config = Config(CONFIG_PATH)

# ── database ──────────────────────────────────────────────────────────


def init_db():
    """Initialize schema and run idempotent migrations.

    New columns are added via ALTER TABLE wrapped in try/except so the
    function is safe to call on every startup.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            prompt_hash TEXT    NOT NULL,
            model       TEXT    NOT NULL,
            provider    TEXT    NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms  REAL    NOT NULL,
            route_reason TEXT   NOT NULL,
            cost_estimate REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON requests(timestamp)")

    # Migration: add feedback-loop columns. ALTER TABLE ADD COLUMN is
    # idempotent only at the schema level — we wrap each in try/except so
    # re-running on an already-migrated DB is a no-op.
    migrations = [
        "ALTER TABLE requests ADD COLUMN request_id TEXT",
        "ALTER TABLE requests ADD COLUMN client_ip TEXT",
        "ALTER TABLE requests ADD COLUMN prompt_text TEXT",
        "ALTER TABLE requests ADD COLUMN response_text TEXT",
        "ALTER TABLE requests ADD COLUMN score INTEGER",
        "ALTER TABLE requests ADD COLUMN signals TEXT",
        "ALTER TABLE requests ADD COLUMN chosen_tier TEXT",
        "ALTER TABLE requests ADD COLUMN forced_tier INTEGER DEFAULT 0",
        "ALTER TABLE requests ADD COLUMN escalation_signal INTEGER DEFAULT 0",
        "ALTER TABLE requests ADD COLUMN response_truncated INTEGER DEFAULT 0",
        "ALTER TABLE requests ADD COLUMN response_short INTEGER DEFAULT 0",
        "ALTER TABLE requests ADD COLUMN response_error INTEGER DEFAULT 0",
        "ALTER TABLE requests ADD COLUMN feedback_rating INTEGER",
        "ALTER TABLE requests ADD COLUMN feedback_at REAL",
        "ALTER TABLE requests ADD COLUMN feedback_comment TEXT",
        "ALTER TABLE requests ADD COLUMN judged_quality INTEGER",
        "ALTER TABLE requests ADD COLUMN judged_at REAL",
        "ALTER TABLE requests ADD COLUMN judged_by TEXT",
        "ALTER TABLE requests ADD COLUMN judged_reasoning TEXT",
        "ALTER TABLE requests ADD COLUMN ideal_tier TEXT",
    ]
    for stmt in migrations:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            # "duplicate column name" — column already exists
            if "duplicate column" not in str(e).lower():
                raise

    # Indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON requests(request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_unjudged ON requests(timestamp) WHERE judged_quality IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_client_ip ON requests(client_ip, timestamp)")

    # View: misroutes — labeled training data for classifier improvement.
    # A request enters this view as soon as it has an ideal_tier set, which
    # happens via (a) explicit feedback API, (b) judge process, or (c)
    # correlation from a subsequent user escalation.
    conn.execute("DROP VIEW IF EXISTS misroutes")
    conn.execute(
        """
        CREATE VIEW misroutes AS
        SELECT
            request_id,
            timestamp,
            prompt_text,
            response_text,
            score,
            signals,
            chosen_tier,
            ideal_tier,
            CASE
                WHEN ideal_tier IS NULL THEN NULL
                WHEN chosen_tier = ideal_tier THEN 'correct'
                WHEN (chosen_tier = 'local' AND ideal_tier IN ('mid', 'frontier'))
                  OR (chosen_tier = 'mid' AND ideal_tier = 'frontier') THEN 'under_routed'
                ELSE 'over_routed'
            END AS routing_outcome,
            CASE
                WHEN feedback_rating IS NOT NULL THEN 'user_feedback'
                WHEN judged_quality IS NOT NULL THEN 'llm_judge'
                WHEN escalation_signal = 1 THEN 'user_escalate'
                ELSE 'correlated_escalation'
            END AS label_source
        FROM requests
        WHERE ideal_tier IS NOT NULL
        """
    )

    conn.commit()
    conn.close()


def log_request(
    request_id: str,
    prompt_hash: str,
    model: str,
    provider: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float,
    route_reason: str,
    cost_estimate: float | None,
    *,
    client_ip: str | None = None,
    prompt_text: str | None = None,
    response_text: str | None = None,
    score: int | None = None,
    signals: list[str] | None = None,
    chosen_tier: str | None = None,
    forced_tier: bool = False,
    escalation_signal: bool = False,
    response_truncated: bool = False,
    response_short: bool = False,
    response_error: bool = False,
) -> int:
    """Insert a request row and return the row ID.

    Body capture is controlled by CAPTURE_BODIES — when off, prompt_text
    and response_text are stored as NULL even if provided.
    """
    captured_prompt = prompt_text if CAPTURE_BODIES else None
    captured_response = response_text if CAPTURE_BODIES else None
    signals_json = json.dumps(signals) if signals is not None else None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        INSERT INTO requests
            (request_id, timestamp, client_ip, prompt_hash, model, provider,
             input_tokens, output_tokens, latency_ms, route_reason, cost_estimate,
             prompt_text, response_text, score, signals, chosen_tier,
             forced_tier, escalation_signal,
             response_truncated, response_short, response_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            time.time(),
            client_ip,
            prompt_hash,
            model,
            provider,
            input_tokens,
            output_tokens,
            latency_ms,
            route_reason,
            cost_estimate,
            captured_prompt,
            captured_response,
            score,
            signals_json,
            chosen_tier,
            1 if forced_tier else 0,
            1 if escalation_signal else 0,
            1 if response_truncated else 0,
            1 if response_short else 0,
            1 if response_error else 0,
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_response(request_id: str, response_text: str, *,
                    response_truncated: bool = False,
                    response_short: bool = False,
                    response_error: bool = False,
                    output_tokens: int | None = None,
                    cost_estimate: float | None = None):
    """Update a logged request with the captured response text and heuristics.

    Used after streaming completes to backfill the response data.
    """
    captured = response_text if CAPTURE_BODIES else None
    conn = sqlite3.connect(DB_PATH)
    # Build update dynamically so optional fields don't overwrite with None
    fields = ["response_text = ?", "response_truncated = ?",
              "response_short = ?", "response_error = ?"]
    params: list = [captured, 1 if response_truncated else 0,
                    1 if response_short else 0, 1 if response_error else 0]
    if output_tokens is not None:
        fields.append("output_tokens = ?")
        params.append(output_tokens)
    if cost_estimate is not None:
        fields.append("cost_estimate = ?")
        params.append(cost_estimate)
    params.append(request_id)
    conn.execute(
        f"UPDATE requests SET {', '.join(fields)} WHERE request_id = ?",
        params,
    )
    conn.commit()
    conn.close()


def _bearer(authorization: str) -> str:
    """Extract bearer token from an Authorization header value."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return ""


def _client_ip(request: Request) -> str | None:
    """Resolve client IP from X-Forwarded-For or socket address."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        # First IP in the chain is the original client
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _resolve_api_key(provider: str, client_key: str) -> str:
    """Use the client's API key if it's real, otherwise fall back to the gateway's own."""
    if client_key and client_key not in ("sk-placeholder", "sk-xxx", "not-needed", ""):
        return client_key
    gateway_key = PROVIDER_KEYS.get(provider, "")
    if not gateway_key:
        raise HTTPException(status_code=401, detail=f"No API key for {provider} (set {provider.upper()}_API_KEY)")
    return gateway_key


# ── routing ─────────────────────────────────────────────────────────


def _extract_user_intent(content) -> str:
    """Extract the user's actual text intent from a message content field.

    Handles both plain string content and OpenAI content-block arrays.
    For content blocks, the LAST text block is the user's actual intent
    — agent frameworks (Letta, etc.) prepend system reminders, agent
    info, and permission context as earlier text blocks, then append
    the user's message as the final text block.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Content blocks — last text block is the user's actual intent
        last_text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                last_text = block.get("text", "")
        return last_text
    return str(content)


# ── escalation prefix parsing ────────────────────────────────────────
# Users can prefix their message with `!escalate`, `!frontier`, `!mid`,
# or `!local` to override the classifier and signal feedback. The prefix
# is stripped before the message is forwarded to the model.

_ESCALATION_RE = re.compile(r"^\s*!(escalate|frontier|mid|local)\b\s*", re.IGNORECASE)


def _parse_escalation_prefix(text: str) -> tuple[str, str | None]:
    """Look for an escalation prefix at the start of the user's message.

    Returns (cleaned_text, directive). Directive is one of:
    - 'escalate' (bump one tier higher than classifier)
    - 'frontier' / 'mid' / 'local' (force specific tier)
    - None (no prefix; classifier decides)
    """
    if not text:
        return text, None
    match = _ESCALATION_RE.match(text)
    if not match:
        return text, None
    directive = match.group(1).lower()
    cleaned = text[match.end():]
    return cleaned, directive


def _strip_escalation_from_body(body: dict) -> str | None:
    """Mutate body in place to strip escalation prefix from the last user message.

    Handles both string and content-block formats. Returns the directive (or None).
    The cleaned message is what gets forwarded to the model.
    """
    messages = body.get("messages", [])
    if not messages:
        return None
    # Find the LAST user message
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_idx = i
            break
    if last_idx is None:
        return None
    msg = messages[last_idx]
    content = msg.get("content", "")

    if isinstance(content, str):
        cleaned, directive = _parse_escalation_prefix(content)
        if directive:
            msg["content"] = cleaned
        return directive
    if isinstance(content, list):
        # Content blocks — strip the LAST text block (where user intent lives)
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                cleaned, directive = _parse_escalation_prefix(block.get("text", ""))
                if directive:
                    block["text"] = cleaned
                return directive
        return None
    return None


def _apply_directive(decision: "RouteDecision", directive: str) -> "RouteDecision":
    """Override a routing decision based on a user-supplied directive.

    !escalate -> bump one tier higher than classifier picked
    !frontier / !mid / !local -> force that tier
    """
    tier_order = ["local", "mid", "frontier"]

    if directive == "escalate":
        # Find current tier, go one up. If already frontier, stay there.
        current = decision.chosen_tier or "local"
        try:
            idx = tier_order.index(current)
        except ValueError:
            idx = 0
        target = tier_order[min(idx + 1, len(tier_order) - 1)]
    elif directive in tier_order:
        target = directive
    else:
        return decision

    tier_to_route = {
        "local": config.routing.get("default", config.default_model),
        "mid": config.routing.get("mid", config.routing.get("frontier", config.default_model)),
        "frontier": config.routing.get("frontier", config.default_model),
    }
    new_model = tier_to_route[target]

    reason_extra = f"|directive=!{directive}"
    return RouteDecision(
        model_key=new_model,
        reason=decision.reason + reason_extra,
        score=decision.score,
        signals=decision.signals,
        chosen_tier=target,
        forced_tier=True,
        escalation_signal=(directive in ("escalate", "frontier")),
        user_intent=decision.user_intent,
    )


def _tokens_for_similarity(text: str) -> set[str]:
    """Lowercase word tokens for Jaccard similarity comparison."""
    return set(re.findall(r"\w+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def correlate_misroute(current_prompt: str, current_tier: str | None,
                       client_ip: str | None) -> str | None:
    """Try to label a recent request as under-routed.

    When the user escalates, find the most recent prior request from the same
    client within the last 5 minutes whose prompt is similar (Jaccard > 0.5)
    and mark its `ideal_tier` to the current (escalated) tier.

    Returns the request_id that was labeled, or None.
    """
    if not current_tier or not client_ip:
        return None
    if not current_prompt:
        return None

    window_start = time.time() - 5 * 60
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT request_id, prompt_text, chosen_tier
            FROM requests
            WHERE client_ip = ?
              AND timestamp > ?
              AND ideal_tier IS NULL
              AND escalation_signal = 0
              AND prompt_text IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            (client_ip, window_start),
        ).fetchall()
        current_tokens = _tokens_for_similarity(current_prompt)
        for row in rows:
            sim = _jaccard(current_tokens, _tokens_for_similarity(row["prompt_text"]))
            if sim > 0.5:
                conn.execute(
                    "UPDATE requests SET ideal_tier = ? WHERE request_id = ?",
                    (current_tier, row["request_id"]),
                )
                conn.commit()
                print(f"[MISROUTE] labeled {row['request_id'][:8]} as under_routed (was {row['chosen_tier']}, ideal {current_tier}, sim={sim:.2f})")
                return row["request_id"]
    finally:
        conn.close()
    return None


# ── classifier signal patterns ───────────────────────────────────────
# Precompiled for performance. Each pattern contributes points to a
# complexity score. Score thresholds map to routing tier in classify_prompt().

_CODE_BLOCK_RE = re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|\bfunction \w+\(|\basync def \b|=>\s*[{(]", re.MULTILINE)
# Traceback / structured error indicators — does NOT fire on bare "exception" in prose
_ERROR_RE = re.compile(r"\bTraceback\b|\bError:|\b\w+Error\b|\bat line \d+|\bpanic:|^\s*at \w+\.", re.MULTILINE)
# Code reference patterns — backtick-wrapped calls or "Write a <thing>" for code/config artifacts
_CODE_REF_RE = re.compile(
    r"`\w+\([^`]*\)`"
    r"|\bWrite a (?:minimal |simple |basic )?(?:Python|JavaScript|TypeScript|Go|Rust|Java|JS|TS|Kubernetes|Docker|nginx|bash|shell|YAML|JSON|SQL)\b"
    r"|\bWrite a (?:minimal |simple |basic )?(?:function|decorator|class|server block|deployment|manifest|config|script|compose file|one-liner)",
    re.IGNORECASE,
)
_ARCH_TERMS = (
    "architecture", "design pattern", "system design", "distributed",
    "microservice", "scalability", "tradeoff", "consistency", "latency",
    "throughput", "partition", "consensus", "concurrency", "race condition",
    "deadlock", "transaction", "atomic", "event-driven", "kubernetes",
    "deployment manifest", "exponential backoff",
)
_REASONING_TERMS = (
    "compare", "evaluate", "review this", "audit", "analyze", "why does",
    "why is", "which is better", "should i use", "tradeoffs between",
    "pros and cons", "what's the difference", "explain why", "when should i",
)
_SECURITY_TERMS = (
    "security", "vulnerability", "injection", "exploit", "cve",
    "find the bug", "fix this bug", "find it and fix", "has a bug",
    "is this correct", "if and only if", "o(n²)", "o(n^2)", "o(n*n)",
    "performance issue", "memory leak", "race", "thread", "thread-safe",
    "threading", "off-by-one", "wrong totals", "produces wrong",
)
# Multi-step: ordered/sequential task
_MULTISTEP_RE = re.compile(
    r"\b(?:first|step\s*1)\b.{0,200}\b(?:then|next|step\s*2)\b"
    r"|\bthen\s+(?:write|run|call|execute|return|print|create|save)"
    r"|\blist the (?:tool calls|steps|commands)\b"
    r"|\bbe exact about\b"
    r"|\bin order\b(?!\s+to\b)",  # avoid "in order to"
    re.IGNORECASE | re.DOTALL,
)
_NUMBERED_LIST_RE = re.compile(r"(?:^|\n)\s*[1-9]\.\s+\S.+\n\s*[2-9]\.\s+\S", re.MULTILINE)
# Multi-constraint task: numeric specs OR multiple "with X" clauses
_MULTI_CONSTRAINT_RE = re.compile(
    r"\b\d+[,]?\d*\s*(?:replicas|requests?/?sec|transactions?/?sec|tps|tokens|ms|seconds|kb|mb|gb)\b"
    r"|\bexposing port \d+\b"
    r"|\bwith (?:a |an |the )?\w+(?: \w+){0,3} mounted\b"
    r"|\bmounted as (?:environment )?variables?\b",
    re.IGNORECASE,
)
# Specific tool/command request — "exact X command" implies precision needed
_PRECISION_RE = re.compile(
    r"\b(?:exact|exactly) (?:the )?(?:command|syntax|arguments|tool calls)\b"
    r"|\bgive me the exact\b",
    re.IGNORECASE,
)
# "find every/all X that Y" — multi-clause search/filter query
_SEARCH_QUERY_RE = re.compile(r"\bfind (?:all|every|each) \w+\b", re.IGNORECASE)
_SIMPLICITY_TERMS = (
    "one-liner", "one line", "single command", "just the command",
    "in one sentence", "in two sentences", "in one or two sentences",
    "briefly", "quick question", "in a sentence", "no explanation",
)
_SIMPLE_START_RE = re.compile(r"^\s*(?:what(?:'s| is) the\b|how do i\b|how to\b)", re.IGNORECASE)


def classify_prompt(user_intent: str) -> tuple[int, list[str]]:
    """Score a prompt by complexity using multiple weighted signals.

    Returns (score, fired_signals). Higher score = more complex.
    Used by choose_model() to pick local/mid/frontier tier.
    """
    text = user_intent or ""
    score = 0
    fired: list[str] = []

    # Length signal — only long prompts contribute positively. Short prompts
    # are NOT penalized because "short ≠ simple" — a 50-char architecture
    # question is still complex. The simplicity signals handle genuinely
    # simple prompts ("one-liner", "what's the X").
    n = len(text)
    if n < 800:
        # Neutral band — no contribution
        pass
    elif n < 1500:
        score += 1
        fired.append("long")
    else:
        score += 2
        fired.append("very_long")

    lowered = text.lower()

    # Code block / multi-line code structure — strong complexity signal
    if _CODE_BLOCK_RE.search(text):
        score += 2
        fired.append("code_block")

    # Backtick function calls or "Write a Python/Kubernetes/etc."
    if _CODE_REF_RE.search(text):
        score += 1
        fired.append("code_reference")

    # Error message / traceback (structured, not bare "exception")
    if _ERROR_RE.search(text):
        score += 1
        fired.append("error_traceback")

    # Architecture / systems concepts
    if any(term in lowered for term in _ARCH_TERMS):
        score += 2
        fired.append("architecture")

    # Reasoning / comparison verbs
    if any(term in lowered for term in _REASONING_TERMS):
        score += 2
        fired.append("reasoning_verb")

    # Security / correctness analysis
    if any(term in lowered for term in _SECURITY_TERMS):
        score += 2
        fired.append("security_correctness")

    # Multi-step indicators
    if _MULTISTEP_RE.search(text) or _NUMBERED_LIST_RE.search(text):
        score += 3
        fired.append("multi_step")

    # Multi-constraint task (config/spec with multiple requirements)
    if _MULTI_CONSTRAINT_RE.search(text):
        score += 2
        fired.append("multi_constraint")

    # Precision/exactness requirement
    if _PRECISION_RE.search(text):
        score += 1
        fired.append("precision_required")

    # Multi-clause search query ("find every X that Y")
    if _SEARCH_QUERY_RE.search(text):
        score += 1
        fired.append("search_query")

    # Simplicity signals (negative contribution)
    if any(term in lowered for term in _SIMPLICITY_TERMS):
        score -= 2
        fired.append("simplicity_hint")
    elif _SIMPLE_START_RE.search(text) and n < 200:
        # Short "what's the X" / "how do I X" questions
        score -= 1
        fired.append("simple_lookup")

    return score, fired


@dataclass
class RouteDecision:
    """The full routing decision for one request.

    Carries both the model selection and the classifier's reasoning so
    callers can log structured data for the feedback loop.
    """
    model_key: str
    reason: str
    score: int | None = None            # None when not classifier-driven
    signals: list[str] = field(default_factory=list)
    chosen_tier: str | None = None       # 'local' | 'mid' | 'frontier' | None
    forced_tier: bool = False            # Explicit override (param or prefix)
    escalation_signal: bool = False      # User asked for higher tier
    user_intent: str = ""                # The text the classifier saw


def _tier_for_model(model_key: str) -> str | None:
    """Map a model key to its routing tier (local/mid/frontier)."""
    if model_key == config.routing.get("default"):
        return "local"
    if model_key == config.routing.get("mid"):
        return "mid"
    if model_key == config.routing.get("frontier"):
        return "frontier"
    # Aliases / unknown — return None rather than guessing
    return None


def choose_model(body: dict, force: str | None) -> RouteDecision:
    """Decide which model handles this request.

    Resolution order:
    1. `?route=` query param (forced)
    2. `body.model` if it resolves to a real model and is not a proxy alias
    3. Classifier v3 score → three-tier routing
    """
    if force and force in config.models:
        return RouteDecision(
            model_key=force,
            reason=f"forced_by_param:{force}",
            chosen_tier=_tier_for_model(force),
            forced_tier=True,
        )
    if force and force in config.aliases:
        resolved = config.aliases[force]
        return RouteDecision(
            model_key=resolved,
            reason=f"forced_by_param(alias):{force}->{resolved}",
            chosen_tier=_tier_for_model(resolved),
            forced_tier=True,
        )

    # If the request body specifies a model, try to resolve it
    # — but skip proxy model names (let the heuristic decide instead)
    body_model = body.get("model", "")
    if body_model and not force and body_model not in config.proxy_models:
        try:
            resolved = config.resolve_model(body_model)
            if resolved != config.default_model:
                return RouteDecision(
                    model_key=resolved,
                    reason=f"body_model:{body_model}->{resolved}",
                    chosen_tier=_tier_for_model(resolved),
                    forced_tier=True,
                )
        except HTTPException:
            pass  # Unknown model name, fall through to heuristic

    # Classifier v3: extract user intent, score by complexity signals,
    # map to three-tier routing (local / mid / frontier).
    messages = body.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_user = user_messages[-1].get("content", "") if user_messages else ""
    user_intent = _extract_user_intent(last_user)
    score, signals = classify_prompt(user_intent)

    signals_str = ",".join(signals) if signals else "none"

    if score <= 0:
        return RouteDecision(
            model_key=config.default_model,
            reason=f"classifier:simple(score={score},signals=[{signals_str}])",
            score=score,
            signals=signals,
            chosen_tier="local",
            user_intent=user_intent,
        )
    elif score <= 3:
        mid = config.routing.get("mid", config.routing.get("frontier", config.default_model))
        return RouteDecision(
            model_key=mid,
            reason=f"classifier:medium(score={score},signals=[{signals_str}])",
            score=score,
            signals=signals,
            chosen_tier="mid",
            user_intent=user_intent,
        )
    frontier = config.routing.get("frontier", config.default_model)
    return RouteDecision(
        model_key=frontier,
        reason=f"classifier:complex(score={score},signals=[{signals_str}])",
        score=score,
        signals=signals,
        chosen_tier="frontier",
        user_intent=user_intent,
    )


# ── providers ───────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = "You are a terse assistant. Answer directly. Never explain your reasoning."


def _inject_system_prompt(messages: list, provider: str, model_name: str) -> list:
    """Prepend default system prompt if none exists."""
    if not messages:
        return [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}]
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and model_name.startswith("qwen3"):
        return [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}, *messages]
    return list(messages)


def _flatten_content_blocks(messages: list) -> list:
    """Flatten content-block arrays to plain strings for providers that don't support them.

    Ollama and some other providers only accept string content, not the
    OpenAI content-block array format. This converts:
      {"role": "user", "content": [{"type": "text", "text": "hello"}, ...]}
    to:
      {"role": "user", "content": "hello\n..."}
    """
    result = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            # Flatten content blocks to text
            parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        # Include tool result content (it's context the model needs)
                        rc = block.get("content", "")
                        parts.append(rc if isinstance(rc, str) else str(rc))
                    elif btype == "tool_use":
                        # Include tool call name + args for context
                        parts.append(f"Tool call: {block.get('name', '')}({block.get('input', '')})")
                    elif btype in ("input_text", "output_text"):
                        parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)
        result.append({**m, "content": content})
    return result


def _hash_prompt(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _capture_stream_and_log(upstream, log_kwargs: dict, start: float, cfg: dict):
    """Wrap an SSE generator to capture response telemetry, then log on completion.

    Yields each chunk to the client immediately (no buffering). In parallel
    accumulates response_text, finish_reason, and usage tokens from the
    stream's events. On completion (or error) writes one row via log_request.

    All providers in this gateway emit OpenAI-compatible Chat Completions
    SSE chunks (`data: {"choices":[{"delta":{"content":"..."}}]}`), so a
    single parser handles all of them.
    """
    response_parts: list[str] = []
    finish_reason: str | None = None
    input_tokens = 0
    output_tokens = 0
    error_occurred = False

    try:
        async for chunk in upstream:
            # Yield to the client first — never block streaming on logging
            yield chunk
            # Then parse the SSE event for telemetry
            if not isinstance(chunk, str) or not chunk.startswith("data: "):
                continue
            data = chunk[6:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            for choice in event.get("choices", []) or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    response_parts.append(text)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
            usage = event.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", input_tokens) or input_tokens
                output_tokens = usage.get("completion_tokens", output_tokens) or output_tokens
    except Exception:
        error_occurred = True
        raise
    finally:
        response_text = "".join(response_parts)
        latency_ms = (time.perf_counter() - start) * 1000
        cost = None
        if "cost_per_1m_input" in cfg:
            cost = (
                input_tokens * cfg["cost_per_1m_input"]
                + output_tokens * cfg["cost_per_1m_output"]
            ) / 1_000_000
        try:
            log_request(
                **log_kwargs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_estimate=cost,
                response_text=response_text,
                response_truncated=(finish_reason == "length"),
                response_short=(len(response_text) < 50 and not error_occurred),
                response_error=error_occurred,
            )
        except Exception as log_err:  # never let logging break the stream
            print(f"[LOG ERROR] {log_err}")


async def _stream_ollama(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
):
    """Stream Ollama /api/chat format, yield SSE chunks."""
    messages = _flatten_content_blocks(
        _inject_system_prompt(
            body.get("messages", []), cfg["provider"], cfg["model_name"]
        )
    )
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "stream": True,
        "options": {
            **cfg.get("default_options", {}),
            **body.get("options", {}),
        },
    }
    # Thinking models (qwen3, etc.) split output between `thinking` and `content`.
    # When think=false, Ollama merges the thinking into content so we can stream
    # it like any other response. Set per-model in config.yaml.
    if "think" in cfg:
        payload["think"] = cfg["think"]
    async with client.stream(
        "POST", cfg["endpoint"], json=payload, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("done"):
                # Final chunk with metrics
                done_chunk = {
                    "id": "chatcmpl-ollama",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg["model_name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": data.get("prompt_eval_count", 0)
                        + data.get("eval_count", 0),
                    },
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                break
            content = data.get("message", {}).get("content", "")
            if content:
                chunk = {
                    "id": "chatcmpl-ollama",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg["model_name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"


async def _stream_openrouter(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    """Stream OpenRouter / OpenAI format, yield SSE chunks."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model_name"],
        "messages": _flatten_content_blocks(body.get("messages", [])),
        "stream": True,
    }
    # Some models (GPT-5.5) require max_completion_tokens instead of max_tokens
    if cfg.get("use_max_completion_tokens"):
        payload["max_completion_tokens"] = body.get("max_completion_tokens", cfg.get("default_options", {}).get("max_completion_tokens", 1024))
    else:
        payload["temperature"] = body.get("temperature", cfg.get("default_options", {}).get("temperature", 0.7))
        if "max_tokens" in body:
            payload["max_tokens"] = body["max_tokens"]
    # Request usage in final chunk for OpenAI-compatible providers
    if cfg.get("provider") in ("openai", "openrouter"):
        payload["stream_options"] = {"include_usage": True}

    async with client.stream(
        "POST", cfg["endpoint"], json=payload, headers=headers, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {data}\n\n"


async def _stream_anthropic(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    """Stream Anthropic Messages API, translate to OpenAI SSE chunks."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    # Convert OpenAI messages to Anthropic format
    messages = body.get("messages", [])
    system_text = None
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            anthropic_msgs.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": cfg["model_name"],
        "messages": anthropic_msgs,
        "max_tokens": body.get("max_tokens", cfg.get("default_options", {}).get("max_tokens", 1024)),
        "stream": True,
    }
    if system_text:
        payload["system"] = system_text
    if not cfg.get("no_temperature"):
        if "temperature" in body:
            payload["temperature"] = body["temperature"]
        elif "temperature" in cfg.get("default_options", {}):
            payload["temperature"] = cfg["default_options"]["temperature"]

    input_tokens = 0
    output_tokens = 0

    async with client.stream(
        "POST", cfg["endpoint"], json=payload, headers=headers, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "message_start":
                input_tokens = event.get("message", {}).get("usage", {}).get("input_tokens", 0)
            elif event_type == "content_block_delta":
                delta_text = event.get("delta", {}).get("text", "")
                if delta_text:
                    output_tokens += 1  # approximate; real count comes in message_delta
                    chunk = {
                        "id": "chatcmpl-anthropic",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cfg["model_name"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            elif event_type == "message_delta":
                output_tokens = event.get("usage", {}).get("output_tokens", output_tokens)
                finish_chunk = {
                    "id": "chatcmpl-anthropic",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg["model_name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
            elif event_type == "message_stop":
                yield "data: [DONE]\n\n"
                break


async def _nonstream_ollama(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
):
    messages = _flatten_content_blocks(
        _inject_system_prompt(
            body.get("messages", []), cfg["provider"], cfg["model_name"]
        )
    )
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "stream": False,
        "options": {
            **cfg.get("default_options", {}),
            **body.get("options", {}),
        },
    }
    if "think" in cfg:
        payload["think"] = cfg["think"]
    resp = await client.post(cfg["endpoint"], json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("message", {}).get("content", "")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return {
        "id": "chatcmpl-ollama",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": cfg["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _nonstream_openrouter(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model_name"],
        "messages": _flatten_content_blocks(body.get("messages", [])),
        "stream": False,
    }
    if cfg.get("use_max_completion_tokens"):
        payload["max_completion_tokens"] = body.get("max_completion_tokens", cfg.get("default_options", {}).get("max_completion_tokens", 1024))
    else:
        payload["temperature"] = body.get("temperature", cfg.get("default_options", {}).get("temperature", 0.7))
        if "max_tokens" in body:
            payload["max_tokens"] = body["max_tokens"]

    resp = await client.post(cfg["endpoint"], json=payload, headers=headers, timeout=120.0)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    return resp.json()


async def _nonstream_anthropic(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    messages = body.get("messages", [])
    system_text = None
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            anthropic_msgs.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": cfg["model_name"],
        "messages": anthropic_msgs,
        "max_tokens": body.get("max_tokens", cfg.get("default_options", {}).get("max_tokens", 1024)),
        "stream": False,
    }
    if system_text:
        payload["system"] = system_text
    # Some models (e.g., Opus 4.7 with adaptive thinking) reject temperature.
    if not cfg.get("no_temperature"):
        if "temperature" in body:
            payload["temperature"] = body["temperature"]
        elif "temperature" in cfg.get("default_options", {}):
            payload["temperature"] = cfg["default_options"]["temperature"]

    resp = await client.post(cfg["endpoint"], json=payload, headers=headers, timeout=120.0)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    data = resp.json()

    content = data.get("content", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)

    return {
        "id": "chatcmpl-anthropic",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": cfg["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        },
    }


async def _stream_gemini(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    """Stream Gemini API, translate to OpenAI SSE chunks."""
    # Convert OpenAI messages to Gemini format
    messages = body.get("messages", [])
    contents = []
    for m in messages:
        if m["role"] == "system":
            # Gemini doesn't have system role; prepend as user with special formatting
            contents.append({"role": "user", "parts": [{"text": f"System: {m['content']}"}]})
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": m["content"]}]})
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": body.get("max_tokens", cfg.get("default_options", {}).get("max_tokens", 2048)),
        },
    }
    if "temperature" in body:
        payload["generationConfig"]["temperature"] = body["temperature"]
    elif "temperature" in cfg.get("default_options", {}):
        payload["generationConfig"]["temperature"] = cfg["default_options"]["temperature"]
    
    # Gemini uses API key in URL, not header
    endpoint = f"{cfg['endpoint']}?key={api_key}&alt=sse"
    
    async with client.stream(
        "POST", endpoint, json=payload, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            
            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                continue
            
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            
            if text:
                chunk = {
                    "id": "chatcmpl-gemini",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg["model_name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Check for finish
            if candidates[0].get("finishReason"):
                usage_meta = data.get("usageMetadata", {})
                finish_chunk = {
                    "id": "chatcmpl-gemini",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cfg["model_name"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                        "total_tokens": usage_meta.get("totalTokenCount", 0),
                    },
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break


async def _nonstream_gemini(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
    api_key: str,
):
    """Non-streaming Gemini API call."""
    messages = body.get("messages", [])
    contents = []
    for m in messages:
        if m["role"] == "system":
            contents.append({"role": "user", "parts": [{"text": f"System: {m['content']}"}]})
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": m["content"]}]})
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": body.get("max_tokens", cfg.get("default_options", {}).get("max_tokens", 2048)),
        },
    }
    if "temperature" in body:
        payload["generationConfig"]["temperature"] = body["temperature"]
    elif "temperature" in cfg.get("default_options", {}):
        payload["generationConfig"]["temperature"] = cfg["default_options"]["temperature"]
    
    endpoint = f"{cfg['endpoint']}?key={api_key}"
    
    resp = await client.post(endpoint, json=payload, timeout=120.0)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    data = resp.json()
    
    candidates = data.get("candidates", [{}])
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    
    usage_meta = data.get("usageMetadata", {})
    inp = usage_meta.get("promptTokenCount", 0)
    out = usage_meta.get("candidatesTokenCount", 0)
    
    return {
        "id": "chatcmpl-gemini",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": cfg["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        },
    }


# ── FastAPI app ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.http = httpx.AsyncClient(timeout=30.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="model-router", lifespan=lifespan)


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(default=""),
    route: str | None = Query(default=None),
):
    body = await request.json()
    stream = body.get("stream", False)
    body_model = body.get("model", "")

    request_id = str(uuid.uuid4())
    client_ip = _client_ip(request)

    # Parse !escalate / !frontier / !mid / !local prefix BEFORE routing.
    # This strips the prefix from the message that gets forwarded.
    directive = _strip_escalation_from_body(body)

    decision = choose_model(body, route)
    if directive:
        decision = _apply_directive(decision, directive)
    cfg = config.get_model(decision.model_key)
    provider = cfg["provider"]

    # Extract prompt details for logging
    last_user_raw = ""
    last_user_intent = ""
    user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
    if user_msgs:
        c = user_msgs[-1].get("content", "")
        last_user_raw = c if isinstance(c, str) else str(c)
        last_user_intent = _extract_user_intent(c)
    total_len = len(" ".join(str(m.get("content", "")) for m in body.get("messages", [])))
    print(
        f"[ROUTE] rid={request_id[:8]} body_model={body_model!r} -> key={decision.model_key} "
        f"provider={provider} reason={decision.reason} user_msg_len={len(last_user_raw)} "
        f"user_intent_len={len(last_user_intent)} total_len={total_len} "
        f"directive={directive or 'none'}"
    )

    # If the user escalated, try to label the previous request as under-routed.
    # Fire and forget — never block the actual request on this.
    if decision.escalation_signal:
        try:
            correlate_misroute(last_user_intent, decision.chosen_tier, client_ip)
        except Exception as e:
            print(f"[MISROUTE ERR] {e}")

    full_prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
    prompt_hash = _hash_prompt(full_prompt)

    log_kwargs = {
        "request_id": request_id,
        "prompt_hash": prompt_hash,
        "model": decision.model_key,
        "provider": provider,
        "route_reason": decision.reason,
        "client_ip": client_ip,
        # Store the extracted user intent (what the classifier saw), not the
        # full message blob which includes tool definitions and system context.
        "prompt_text": last_user_intent or full_prompt,
        "score": decision.score,
        "signals": decision.signals,
        "chosen_tier": decision.chosen_tier,
        "forced_tier": decision.forced_tier,
        "escalation_signal": decision.escalation_signal,
    }

    start = time.perf_counter()
    response_headers = {"X-LLMux-Request-Id": request_id}

    # Streaming branch — wrap each provider's stream with the capture/logger
    if stream:
        if provider == "ollama":
            upstream = _stream_ollama(request.app.state.http, cfg, body)
        elif provider in ("openrouter", "openai"):
            client_key = _bearer(authorization)
            api_key = _resolve_api_key(provider, client_key)
            upstream = _stream_openrouter(request.app.state.http, cfg, body, api_key)
        elif provider == "anthropic":
            client_key = _bearer(authorization)
            api_key = _resolve_api_key(provider, client_key)
            upstream = _stream_anthropic(request.app.state.http, cfg, body, api_key)
        elif provider == "google":
            client_key = _bearer(authorization)
            api_key = _resolve_api_key(provider, client_key)
            upstream = _stream_gemini(request.app.state.http, cfg, body, api_key)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")
        return StreamingResponse(
            _capture_stream_and_log(upstream, log_kwargs, start, cfg),
            media_type="text/event-stream",
            headers=response_headers,
        )

    # Non-streaming branch
    if provider == "ollama":
        result = await _nonstream_ollama(request.app.state.http, cfg, body)
    elif provider in ("openrouter", "openai"):
        client_key = _bearer(authorization)
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_openrouter(request.app.state.http, cfg, body, api_key)
    elif provider == "anthropic":
        client_key = _bearer(authorization)
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_anthropic(request.app.state.http, cfg, body, api_key)
    elif provider == "google":
        client_key = _bearer(authorization)
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_gemini(request.app.state.http, cfg, body, api_key)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

    latency_ms = (time.perf_counter() - start) * 1000

    usage = result.get("usage", {})
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = None
    if "cost_per_1m_input" in cfg:
        cost = (inp * cfg["cost_per_1m_input"] + out * cfg["cost_per_1m_output"]) / 1_000_000

    # Extract response text + finish_reason for passive heuristics
    response_text = ""
    finish_reason = None
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {}) or {}
        response_text = msg.get("content") or ""
        finish_reason = choices[0].get("finish_reason")

    log_request(
        **log_kwargs,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
        cost_estimate=cost,
        response_text=response_text,
        response_truncated=(finish_reason == "length"),
        response_short=(len(response_text) < 50),
        response_error=("error" in result or not choices),
    )

    # Wrap in JSONResponse to attach the request_id header
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, headers=response_headers)


def _responses_to_chat(body: dict) -> dict:
    """Convert OpenAI Responses API request to Chat Completions format."""
    messages = []
    # Responses API uses 'input' which can be a string or list of items
    inp = body.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    # Content blocks — extract text from input_text or output_text
                    text_parts = [
                        b.get("text", "")
                        for b in content
                        if b.get("type") in ("input_text", "output_text")
                    ]
                    content = "\n".join(text_parts)
                messages.append({"role": role, "content": str(content)})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})

    chat_body = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": body.get("stream", False),
    }
    # Pass through relevant params
    for key in ("temperature", "max_tokens", "max_completion_tokens", "tools", "tool_choice"):
        if key in body:
            chat_body[key] = body[key]
    return chat_body


def _chat_to_responses(chat_result: dict, model: str) -> dict:
    """Convert Chat Completions response to Responses API format."""
    content_text = ""
    tool_calls = []

    if "choices" in chat_result and chat_result["choices"]:
        msg = chat_result["choices"][0].get("message", {})
        content_text = msg.get("content", "") or ""
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append({
                    "type": "function_call",
                    "id": tc.get("id", ""),
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                })

    output_items = []
    if content_text:
        output_items.append({
            "type": "output_text",
            "text": content_text,
        })
    for tc in tool_calls:
        output_items.append(tc)

    usage = chat_result.get("usage", {})
    return {
        "id": chat_result.get("id", f"resp-{model}"),
        "object": "response",
        "model": model,
        "status": "completed",
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


@app.post("/v1/responses")
async def responses(
    request: Request,
    authorization: str = Header(default=""),
    route: str | None = Query(default=None),
):
    """OpenAI Responses API — convert to Chat Completions, route, convert back."""
    body = await request.json()

    # Debug: log the actual input format
    print(f"[RESP-INPUT] model={body.get('model','')!r} stream={body.get('stream',False)} input_type={type(body.get('input','')).__name__} input_preview={str(body.get('input',''))[:200]}")

    # Convert to chat completions format
    chat_body = _responses_to_chat(body)

    # Parse escalation prefix BEFORE routing
    directive = _strip_escalation_from_body(chat_body)

    # Route using the same logic
    decision = choose_model(chat_body, route)
    if directive:
        decision = _apply_directive(decision, directive)
    cfg = config.get_model(decision.model_key)
    provider = cfg["provider"]

    request_id = str(uuid.uuid4())
    client_ip = _client_ip(request)

    # Log
    last_user_raw = ""
    last_user_intent = ""
    user_msgs = [m for m in chat_body.get("messages", []) if m.get("role") == "user"]
    if user_msgs:
        c = user_msgs[-1].get("content", "")
        last_user_raw = c if isinstance(c, str) else str(c)
        last_user_intent = _extract_user_intent(c)
    total_len = len(" ".join(str(m.get("content", "")) for m in chat_body.get("messages", [])))
    print(
        f"[ROUTE-RESP] rid={request_id[:8]} body_model={body.get('model','')!r} -> "
        f"key={decision.model_key} provider={provider} reason={decision.reason} "
        f"user_msg_len={len(last_user_raw)} user_intent_len={len(last_user_intent)} "
        f"total_len={total_len} directive={directive or 'none'}"
    )

    if decision.escalation_signal:
        try:
            correlate_misroute(last_user_intent, decision.chosen_tier, client_ip)
        except Exception as e:
            print(f"[MISROUTE ERR] {e}")

    log_kwargs = {
        "request_id": request_id,
        "prompt_hash": _hash_prompt(str(body.get("input", ""))),
        "model": decision.model_key,
        "provider": provider,
        "route_reason": decision.reason,
        "client_ip": client_ip,
        "prompt_text": last_user_intent or str(body.get("input", "")),
        "score": decision.score,
        "signals": decision.signals,
        "chosen_tier": decision.chosen_tier,
        "forced_tier": decision.forced_tier,
        "escalation_signal": decision.escalation_signal,
    }

    stream = body.get("stream", False)
    start = time.perf_counter()
    response_headers = {"X-LLMux-Request-Id": request_id}

    if stream:
        resp_id = f"resp-{uuid.uuid4().hex[:24]}"

        # Get the Chat Completions SSE stream from the provider
        if provider == "ollama":
            chat_stream = _stream_ollama(request.app.state.http, cfg, chat_body)
        elif provider in ("openrouter", "openai"):
            api_key = _resolve_api_key(provider, _bearer(authorization))
            chat_stream = _stream_openrouter(request.app.state.http, cfg, chat_body, api_key)
        elif provider == "anthropic":
            api_key = _resolve_api_key(provider, _bearer(authorization))
            chat_stream = _stream_anthropic(request.app.state.http, cfg, chat_body, api_key)
        elif provider == "google":
            # Google streaming not yet supported for Responses API — fall back to non-streaming
            api_key = _resolve_api_key(provider, _bearer(authorization))
            result = await _nonstream_gemini(request.app.state.http, cfg, chat_body, api_key)
            _log_nonstream_result(result, cfg, log_kwargs, start)
            from fastapi.responses import JSONResponse
            return JSONResponse(content=_chat_to_responses(result, decision.model_key), headers=response_headers)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

        # Wrap chat_stream with capture+log, then convert to Responses SSE format
        captured_chat_stream = _capture_stream_and_log(chat_stream, log_kwargs, start, cfg)
        return StreamingResponse(
            _stream_responses_from_chat(captured_chat_stream, decision.model_key, resp_id),
            media_type="text/event-stream",
            headers=response_headers,
        )

    # Non-streaming path
    if provider == "ollama":
        result = await _nonstream_ollama(request.app.state.http, cfg, chat_body)
    elif provider in ("openrouter", "openai"):
        api_key = _resolve_api_key(provider, _bearer(authorization))
        result = await _nonstream_openrouter(request.app.state.http, cfg, chat_body, api_key)
    elif provider == "anthropic":
        api_key = _resolve_api_key(provider, _bearer(authorization))
        result = await _nonstream_anthropic(request.app.state.http, cfg, chat_body, api_key)
    elif provider == "google":
        api_key = _resolve_api_key(provider, _bearer(authorization))
        result = await _nonstream_gemini(request.app.state.http, cfg, chat_body, api_key)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

    _log_nonstream_result(result, cfg, log_kwargs, start)

    from fastapi.responses import JSONResponse
    return JSONResponse(content=_chat_to_responses(result, decision.model_key), headers=response_headers)


def _log_nonstream_result(result: dict, cfg: dict, log_kwargs: dict, start: float) -> None:
    """Extract telemetry from a non-streaming Chat Completions result and log it."""
    latency_ms = (time.perf_counter() - start) * 1000
    usage = result.get("usage", {})
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = None
    if "cost_per_1m_input" in cfg:
        cost = (inp * cfg["cost_per_1m_input"] + out * cfg["cost_per_1m_output"]) / 1_000_000

    response_text = ""
    finish_reason = None
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {}) or {}
        response_text = msg.get("content") or ""
        finish_reason = choices[0].get("finish_reason")

    log_request(
        **log_kwargs,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
        cost_estimate=cost,
        response_text=response_text,
        response_truncated=(finish_reason == "length"),
        response_short=(len(response_text) < 50),
        response_error=("error" in result or not choices),
    )


async def _stream_responses_from_chat(
    chat_stream,
    model_key: str,
    resp_id: str,
):
    """Convert Chat Completions SSE stream into Responses API SSE events.

    The Responses API streaming format is a sequence of typed SSE events:
    response.created → response.in_progress → response.output_item.added →
    response.content_part.added → response.output_text.delta (repeated) →
    response.output_text.done → response.content_part.done →
    response.output_item.done → response.completed → done
    """
    import uuid

    msg_id = f"msg_{uuid.uuid4().hex}"
    seq = 0
    full_text = ""
    usage_data = {}

    # 1. response.created
    seq += 1
    created_event = {
        "type": "response.created",
        "sequence_number": seq,
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model_key,
            "output": [],
            "usage": None,
        },
    }
    yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

    # 2. response.in_progress
    seq += 1
    in_progress_event = {
        "type": "response.in_progress",
        "sequence_number": seq,
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model_key,
            "output": [],
            "usage": None,
        },
    }
    yield f"event: response.in_progress\ndata: {json.dumps(in_progress_event)}\n\n"

    # 3. response.output_item.added
    seq += 1
    output_item_added = {
        "type": "response.output_item.added",
        "sequence_number": seq,
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "content": [],
            "role": "assistant",
        },
    }
    yield f"event: response.output_item.added\ndata: {json.dumps(output_item_added)}\n\n"

    # 4. response.content_part.added
    seq += 1
    content_part_added = {
        "type": "response.content_part.added",
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": [],
        },
    }
    yield f"event: response.content_part.added\ndata: {json.dumps(content_part_added)}\n\n"

    # 5. Stream text deltas from the Chat Completions stream
    async for chunk_str in chat_stream:
        if not chunk_str.startswith("data: "):
            continue
        data = chunk_str[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Extract text delta from Chat Completions chunk
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_text += content
                seq += 1
                text_delta = {
                    "type": "response.output_text.delta",
                    "sequence_number": seq,
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content,
                }
                yield f"event: response.output_text.delta\ndata: {json.dumps(text_delta)}\n\n"

            # Check for tool calls
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                # Emit function_call events
                tc_id = tc.get("id", f"fc_{uuid.uuid4().hex[:24]}")
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", "")

                if fn_name:
                    seq += 1
                    fc_item = {
                        "type": "response.output_item.added",
                        "sequence_number": seq,
                        "output_index": 1,
                        "item": {
                            "id": tc_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": tc_id,
                            "name": fn_name,
                            "arguments": "",
                        },
                    }
                    yield f"event: response.output_item.added\ndata: {json.dumps(fc_item)}\n\n"

                if fn_args:
                    seq += 1
                    args_delta = {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": seq,
                        "item_id": tc_id,
                        "output_index": 1,
                        "delta": fn_args,
                    }
                    yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(args_delta)}\n\n"

        # Capture usage from final chunk
        if chunk.get("usage"):
            usage_data = chunk["usage"]

    # 6. response.output_text.done
    seq += 1
    text_done = {
        "type": "response.output_text.done",
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text,
        "annotations": [],
    }
    yield f"event: response.output_text.done\ndata: {json.dumps(text_done)}\n\n"

    # 7. response.content_part.done
    seq += 1
    content_part_done = {
        "type": "response.content_part.done",
        "sequence_number": seq,
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": full_text,
            "annotations": [],
        },
    }
    yield f"event: response.content_part.done\ndata: {json.dumps(content_part_done)}\n\n"

    # 8. response.output_item.done
    seq += 1
    output_item_done = {
        "type": "response.output_item.done",
        "sequence_number": seq,
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": full_text,
                    "annotations": [],
                }
            ],
            "role": "assistant",
        },
    }
    yield f"event: response.output_item.done\ndata: {json.dumps(output_item_done)}\n\n"

    # 9. response.completed
    seq += 1
    completed_event = {
        "type": "response.completed",
        "sequence_number": seq,
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model_key,
            "output": [
                {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        }
                    ],
                    "role": "assistant",
                }
            ],
            "usage": {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"

    # 10. done
    yield "event: done\ndata: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": key,
                "object": "model",
                "owned_by": cfg["provider"],
            }
            for key, cfg in config.models.items()
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "default_model": config.default_model}


@app.post("/v1/feedback")
async def feedback(request: Request):
    """Explicit feedback on a previous request.

    Body: {
        "request_id": "<uuid>",
        "rating": -1 | +1,
        "ideal_tier": "local" | "mid" | "frontier",  // optional
        "comment": "..."  // optional
    }

    Updates the matching row with feedback signals. Used by the upcoming
    `llmux feedback` CLI and by anyone wanting to label routing quality.
    """
    body = await request.json()
    rid = body.get("request_id")
    rating = body.get("rating")
    ideal_tier = body.get("ideal_tier")
    comment = body.get("comment")

    if not rid or rating not in (-1, 1):
        raise HTTPException(
            status_code=400,
            detail="Body must include 'request_id' and 'rating' in {-1, 1}",
        )
    if ideal_tier and ideal_tier not in ("local", "mid", "frontier"):
        raise HTTPException(
            status_code=400,
            detail="ideal_tier must be one of: local, mid, frontier",
        )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        UPDATE requests
        SET feedback_rating = ?,
            feedback_at = ?,
            feedback_comment = ?,
            ideal_tier = COALESCE(?, ideal_tier)
        WHERE request_id = ?
        """,
        (rating, time.time(), comment, ideal_tier, rid),
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()

    if affected == 0:
        raise HTTPException(status_code=404, detail=f"No request found with id {rid}")
    return {"status": "ok", "request_id": rid, "rating": rating, "ideal_tier": ideal_tier}


@app.get("/stats/weekly")
async def weekly_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    week_ago = time.time() - 7 * 24 * 3600
    rows = conn.execute(
        """
        SELECT
            model,
            COUNT(*) as requests,
            AVG(latency_ms) as avg_latency_ms,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(cost_estimate) as total_cost
        FROM requests
        WHERE timestamp > ?
        GROUP BY model
        """,
        (week_ago,),
    ).fetchall()
    conn.close()
    total = sum(r["requests"] for r in rows)
    local = sum(r["requests"] for r in rows if config.models[r["model"]]["cost_per_1m_input"] == 0)
    return {
        "total_requests": total,
        "local_requests": local,
        "local_pct": round(local / total * 100, 1) if total else 0,
        "by_model": [dict(r) for r in rows],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
