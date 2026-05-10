#!/usr/bin/env python3
"""FastAPI model-router gateway: OpenAI-compatible /chat/completions with pluggable routing."""

import json
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

# ── config ──────────────────────────────────────────────────────────

class Config:
    models: dict
    routing: dict

    def __init__(self, path: Path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        self.models = raw["models"]
        self.routing = raw.get("routing", {})

    def get_model(self, name: str):
        if name not in self.models:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
        return self.models[name]

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


# ── routing ─────────────────────────────────────────────────────────


def choose_model(body: dict, force: str | None) -> tuple[str, str]:
    """Return (model_key, route_reason)."""
    if force and force in config.models:
        return force, f"forced_by_param:{force}"

    # TODO: replace with real classifier
    # Heuristic v1: if prompt is short (<500 chars) and no architecture keywords, route local
    messages = body.get("messages", [])
    prompt_text = " ".join(str(m.get("content", "")) for m in messages)
    keywords = ["architecture", "design", "refactor", "plan", "system design"]
    if len(prompt_text) < 500 and not any(k in prompt_text.lower() for k in keywords):
        return config.default_model, "heuristic:short_simple"

    # Default to local for now (we'll add frontier fallback later)
    return config.default_model, "heuristic:default"


# ── providers ───────────────────────────────────────────────────────


def _hash_prompt(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _stream_ollama(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
):
    """Stream Ollama /api/chat format, yield SSE chunks."""
    messages = body.get("messages", [])
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "stream": True,
        "options": {
            **cfg.get("default_options", {}),
            **body.get("options", {}),
        },
    }
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
        "messages": body.get("messages", []),
        "stream": True,
        "temperature": body.get("temperature", cfg.get("default_options", {}).get("temperature", 0.7)),
    }
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]

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


async def _nonstream_ollama(
    client: httpx.AsyncClient,
    cfg: dict,
    body: dict,
):
    messages = body.get("messages", [])
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "stream": False,
        "options": {
            **cfg.get("default_options", {}),
            **body.get("options", {}),
        },
    }
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
        "messages": body.get("messages", []),
        "stream": False,
        "temperature": body.get("temperature", cfg.get("default_options", {}).get("temperature", 0.7)),
    }
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]

    resp = await client.post(cfg["endpoint"], json=payload, headers=headers, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


# ── FastAPI app ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.http = httpx.AsyncClient(timeout=30.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="model-router", lifespan=lifespan)


@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(default=""),
    route: str | None = Query(default=None),
):
    body = await request.json()
    stream = body.get("stream", False)

    model_key, reason = choose_model(body, route)
    cfg = config.get_model(model_key)
    provider = cfg["provider"]

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
    elif provider == "openrouter":
        # Extract API key from Authorization header
        api_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else ""
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing Authorization header for OpenRouter")
        if stream:
            return StreamingResponse(
                _stream_openrouter(request.app.state.http, cfg, body, api_key),
                media_type="text/event-stream",
            )
        result = await _nonstream_openrouter(request.app.state.http, cfg, body, api_key)
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
