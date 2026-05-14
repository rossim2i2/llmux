#!/usr/bin/env python3
"""FastAPI model-router gateway: OpenAI-compatible /chat/completions with pluggable routing."""

import json
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DB_PATH = Path(__file__).with_name("router.db")

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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ts ON requests(timestamp)
        """
    )
    conn.commit()
    conn.close()


def log_request(
    prompt_hash: str,
    model: str,
    provider: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float,
    route_reason: str,
    cost_estimate: float | None,
):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO requests
            (timestamp, prompt_hash, model, provider, input_tokens, output_tokens,
             latency_ms, route_reason, cost_estimate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(),
            prompt_hash,
            model,
            provider,
            input_tokens,
            output_tokens,
            latency_ms,
            route_reason,
            cost_estimate,
        ),
    )
    conn.commit()
    conn.close()


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


def choose_model(body: dict, force: str | None) -> tuple[str, str]:
    """Return (model_key, route_reason)."""
    if force and force in config.models:
        return force, f"forced_by_param:{force}"
    if force and force in config.aliases:
        resolved = config.aliases[force]
        return resolved, f"forced_by_param(alias):{force}->{resolved}"

    # If the request body specifies a model, try to resolve it
    # — but skip proxy model names (let the heuristic decide instead)
    body_model = body.get("model", "")
    if body_model and not force and body_model not in config.proxy_models:
        try:
            resolved = config.resolve_model(body_model)
            if resolved != config.default_model:
                return resolved, f"body_model:{body_model}->{resolved}"
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
        return config.default_model, f"classifier:simple(score={score},signals=[{signals_str}])"
    elif score <= 3:
        mid = config.routing.get("mid", config.routing.get("frontier", config.default_model))
        return mid, f"classifier:medium(score={score},signals=[{signals_str}])"
    frontier = config.routing.get("frontier", config.default_model)
    return frontier, f"classifier:complex(score={score},signals=[{signals_str}])"


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

    model_key, reason = choose_model(body, route)
    cfg = config.get_model(model_key)
    provider = cfg["provider"]

    # Log the request routing decision
    last_user_raw = ""
    last_user_intent = ""
    user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
    if user_msgs:
        c = user_msgs[-1].get("content", "")
        last_user_raw = c if isinstance(c, str) else str(c)
        last_user_intent = _extract_user_intent(c)
    total_len = len(" ".join(str(m.get("content", "")) for m in body.get("messages", [])))
    print(f"[ROUTE] body_model={body_model!r} -> key={model_key} provider={provider} reason={reason} user_msg_len={len(last_user_raw)} user_intent_len={len(last_user_intent)} total_len={total_len}")

    prompt_text = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
    prompt_hash = _hash_prompt(prompt_text)

    start = time.perf_counter()

    if provider == "ollama":
        if stream:
            return StreamingResponse(
                _stream_ollama(request.app.state.http, cfg, body),
                media_type="text/event-stream",
            )
        result = await _nonstream_ollama(request.app.state.http, cfg, body)
    elif provider in ("openrouter", "openai"):
        # Extract API key from Authorization header
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        if stream:
            return StreamingResponse(
                _stream_openrouter(request.app.state.http, cfg, body, api_key),
                media_type="text/event-stream",
            )
        result = await _nonstream_openrouter(request.app.state.http, cfg, body, api_key)
    elif provider == "anthropic":
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        if stream:
            return StreamingResponse(
                _stream_anthropic(request.app.state.http, cfg, body, api_key),
                media_type="text/event-stream",
            )
        result = await _nonstream_anthropic(request.app.state.http, cfg, body, api_key)
    elif provider == "google":
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        if stream:
            return StreamingResponse(
                _stream_gemini(request.app.state.http, cfg, body, api_key),
                media_type="text/event-stream",
            )
        result = await _nonstream_gemini(request.app.state.http, cfg, body, api_key)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Estimate cost
    usage = result.get("usage", {})
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = None
    if "cost_per_1m_input" in cfg:
        cost = (inp * cfg["cost_per_1m_input"] + out * cfg["cost_per_1m_output"]) / 1_000_000

    log_request(
        prompt_hash=prompt_hash,
        model=model_key,
        provider=provider,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
        route_reason=reason,
        cost_estimate=cost,
    )

    return result


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

    # Route using the same logic
    model_key, reason = choose_model(chat_body, route)
    cfg = config.get_model(model_key)
    provider = cfg["provider"]

    # Log
    last_user_raw = ""
    last_user_intent = ""
    user_msgs = [m for m in chat_body.get("messages", []) if m.get("role") == "user"]
    if user_msgs:
        c = user_msgs[-1].get("content", "")
        last_user_raw = c if isinstance(c, str) else str(c)
        last_user_intent = _extract_user_intent(c)
    total_len = len(" ".join(str(m.get("content", "")) for m in chat_body.get("messages", [])))
    print(f"[ROUTE-RESP] body_model={body.get('model','')!r} -> key={model_key} provider={provider} reason={reason} user_msg_len={len(last_user_raw)} user_intent_len={len(last_user_intent)} total_len={total_len}")

    stream = body.get("stream", False)
    start = time.perf_counter()

    if stream:
        import uuid
        resp_id = f"resp-{uuid.uuid4().hex[:24]}"

        # Get the Chat Completions SSE stream from the provider
        if provider == "ollama":
            chat_stream = _stream_ollama(request.app.state.http, cfg, chat_body)
        elif provider in ("openrouter", "openai"):
            client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
            api_key = _resolve_api_key(provider, client_key)
            chat_stream = _stream_openrouter(request.app.state.http, cfg, chat_body, api_key)
        elif provider == "anthropic":
            client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
            api_key = _resolve_api_key(provider, client_key)
            chat_stream = _stream_anthropic(request.app.state.http, cfg, chat_body, api_key)
        elif provider == "google":
            # Google streaming not yet supported for Responses API
            client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
            api_key = _resolve_api_key(provider, client_key)
            result = await _nonstream_gemini(request.app.state.http, cfg, chat_body, api_key)
            return _chat_to_responses(result, model_key)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

        return StreamingResponse(
            _stream_responses_from_chat(chat_stream, model_key, resp_id),
            media_type="text/event-stream",
        )

    # Non-streaming path
    if provider == "ollama":
        result = await _nonstream_ollama(request.app.state.http, cfg, chat_body)
    elif provider in ("openrouter", "openai"):
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_openrouter(request.app.state.http, cfg, chat_body, api_key)
    elif provider == "anthropic":
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_anthropic(request.app.state.http, cfg, chat_body, api_key)
    elif provider == "google":
        client_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        api_key = _resolve_api_key(provider, client_key)
        result = await _nonstream_gemini(request.app.state.http, cfg, chat_body, api_key)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

    latency_ms = (time.perf_counter() - start) * 1000
    usage = result.get("usage", {})
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = None
    if "cost_per_1m_input" in cfg:
        cost = (inp * cfg["cost_per_1m_input"] + out * cfg["cost_per_1m_output"]) / 1_000_000

    log_request(
        prompt_hash=_hash_prompt(str(body.get("input", ""))),
        model=model_key,
        provider=provider,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
        route_reason=reason,
        cost_estimate=cost,
    )

    return _chat_to_responses(result, model_key)


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
