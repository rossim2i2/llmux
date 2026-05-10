# Model Router — Project Status

**Repo:** `rossim2i2/model-router`  
**Date:** 2026-05-10  
**Hardware:** AMD Ryzen 9 9950X3D + Radeon RX 7900 XTX (24 GB VRAM)

---

## What We Built

A FastAPI gateway (`localhost:8001`) that exposes an **OpenAI-compatible** `/chat/completions` endpoint. It routes prompts between multiple model backends with pluggable logic, SQLite cost tracking, and a `/stats/weekly` dashboard.

### Components

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — routing, providers, logging |
| `config.yaml` | Model definitions (Ollama, OpenRouter) + routing rules |
| `requirements.txt` | `fastapi`, `uvicorn`, `httpx`, `pyyaml` |
| `benchmark.py` | Full 10-prompt benchmark script |
| `benchmark_quick.py` | Quick 8-prompt benchmark (used) |
| `router.db` | SQLite — per-request cost/latency log |

### Supported Providers

- **Ollama** (`qwen3:4b-local`) — GPU inference via `/api/chat`
- **OpenRouter** (`kimi-k2.6`, `gpt-4.1-nano`) — wired but untested (needs API key)

### Routing Logic (Current)

Stub heuristic: if prompt < 500 chars and no keywords ("architecture", "design", "refactor"), route local. Otherwise also local (frontier fallback is stubbed).

### Endpoints

- `POST /chat/completions` — OpenAI-compatible, streaming + non-streaming
- `GET /v1/models` — List configured models
- `GET /health` — Gateway status
- `GET /stats/weekly` — Cost/latency summary by model

---

## Hardware Discovery

| Phase | Ollama Package | GPU Used? | qwen3:4b Speed | Usable? |
|-------|---------------|-----------|----------------|---------|
| Initial | `pacman/ollama` (CPU-only) | No | ~15 tok/s, 60s TTFT | No |
| After reinstall | Official binary (0.23.2) | **Yes** (ROCm) | **~140 tok/s** | Yes |

**Key finding:** The Arch `ollama` package is CPU-only. The official install script bundles ROCm libraries and detects the RX 7900 XTX correctly (gfx1100).

---

## Benchmark Results (GPU)

8/8 prompts succeeded. Median throughput: **141 tok/s**.

| Prompt Type | TTFT | Total | Tok/s | Output |
|-------------|------|-------|-------|--------|
| explain (6s) | 748ms | 983ms | 144 | 127 |
| search-analysis | 3.8s | 6.8s | 142 | 935 |
| git-op | 5.2s | 9.8s | 141 | 1344 |
| code-short | 16.9s | 17.0s | 139 | 2302 |
| refactor | 40.9s | 41.0s | 134 | 5371 |

**Problem:** qwen3:4b is extremely verbose. It generated 2302 tokens for "write a one-liner" and 5371 tokens for a refactor prompt. The model emits extensive internal `thinking` content even when the visible answer is short.

---

## Critical Discovery: Letta Code's API

**The original goal** was to place this gateway between Letta Code (the desktop app) and its upstream API, so 90% of prompts hit the local GPU instead of Letta's cloud.

**What we learned:** Letta Code speaks a **proprietary protocol** to `https://api.letta.com`. It is not OpenAI-compatible. The binary hardcodes endpoints like `/v1/environments`, `/v1/agents/{id}/messages`, etc. There is no `OPENAI_BASE_URL` or similar override.

**Implication:** An OpenAI-compatible gateway cannot intercept Letta Code traffic without a full Letta-protocol reverse proxy — a substantially larger and more fragile project.

---

## The Fork in the Road

| Option | What it means | Effort | Risk |
|--------|--------------|--------|------|
| **A. Repurpose gateway** | Keep it as a general OpenAI-compatible router for other tools (scripts, Cursor, Claude Code, custom clients). Add OpenRouter key, make it usable by anything with a `base_url` setting. | Low | None |
| **B. Letta-protocol proxy** | Reverse-engineer Letta's API format and build a translation layer. Gateway would speak Letta on the client side and OpenAI on the upstream side. | Very high | Fragile — breaks when Letta updates |
| **C. Replace Letta Code** | Switch to a generic OpenAI-compatible client (aider, claude code, custom TUI). You'd lose Letta's persistent memory, agent scheduler, and memory blocks. | Medium | Lose ecosystem |
| **D. Abandon routing for Letta** | Accept that Letta Code → Letta cloud is a closed pipe. Use the gateway for other projects only. | None | Project scope shrinks |

**Default recommendation:** Option A. The gateway is already useful for non-Letta tools. Option B is a rabbit hole. Option C sacrifices too much. Option D is honest but wastes working code.

---

## Immediate Next Steps (pending decision)

1. **OpenRouter API key** — needed to test frontier routing
2. **qwen3 verbosity fix** — cap `num_predict` or inject a system prompt to suppress `thinking` bloat
3. **Classifier v2** — replace stub heuristic with something real (prompt-length + keyword is too crude)
4. **Feedback loop** — detect retry patterns, flag possible bad routes
5. **Systemd service** — auto-start gateway on boot

---

## Open Questions

- Does Letta Code have an undocumented `base_url` or proxy setting? (Searched binary — none found)
- Would Letta's local runtime mode (`letta run`) bypass the cloud API entirely? (Untested)
- Can qwen3:4b be tuned via Ollama `system` prompt or `OPTIONS` to suppress reasoning output?
