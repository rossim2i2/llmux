# Model Router

A local FastAPI gateway that routes prompts between your GPU and frontier APIs. OpenAI-compatible `/chat/completions` endpoint with cost tracking.

**Status:** On hold pending Michael's decision on direction (see [Fork in the Road](#fork-in-the-road)).

---

## What Works Today

| Component | State |
|-----------|-------|
| Gateway API | ✅ `localhost:8001` — OpenAI-compatible `/chat/completions` |
| GPU inference | ✅ `qwen3:4b` on RX 7900 XTX at ~140 tok/s |
| Verbosity fix | ✅ Auto-injects terse system prompt (6,126 → 229 tokens for code gen) |
| Cost tracking | ✅ SQLite `router.db` + `/stats/weekly` dashboard |
| Systemd service | ✅ Auto-starts on boot, after Ollama |
| Streaming | ✅ SSE format for real-time responses |
| Frontier routing | 🔧 Wired for OpenRouter but untested (no API key) |
| Letta integration | ❌ Blocked — Letta uses proprietary API, not OpenAI-compatible |

### Quick Test

```bash
curl http://localhost:8001/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b-local","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

---

## Architecture

```
Your OpenAI-compatible client
           |
           v
    localhost:8001
    (FastAPI gateway)
           |
    Router decides:
           |
    ┌──────┴──────┐
    |             |
    v             v
  Ollama      OpenRouter
  (GPU)       (frontier APIs)
qwen3:4b    kimi-k2.6, etc.
```

---

## The Fork in the Road

The original goal was to intercept Letta Code traffic and route 90% to the local GPU. We discovered Letta Code speaks a **proprietary protocol** to `https://api.letta.com` — not OpenAI's API. The gateway cannot sit between Letta Code and Letta's cloud without reverse-engineering Letta's entire protocol.

| Option | Description | Effort | Risk |
|--------|-------------|--------|------|
| **A. Repurpose gateway** | Use for other OpenAI-compatible tools (scripts, editors, CLI). Keep local-only, no APIs. | Low | None |
| **B. Letta-protocol proxy** | Reverse-engineer Letta's API and build a translation layer | Very high | Fragile — breaks on Letta updates |
| **C. Replace Letta Code** | Switch to generic client (aider, claude code, custom TUI). Lose persistent memory, scheduler, ecosystem. | Medium | Lose Letta features |
| **D. Park project** | Accept the gateway is only useful for non-Letta work | None | Code exists but unused for main goal |

**Michael's preference:** No API costs (already pays for ChatGPT, Claude, Letta subscriptions). Wants to think before deciding.

---

## Completed Work

### 1. GPU Setup
- Replaced CPU-only Arch `ollama` with official binary (bundles ROCm)
- RX 7900 XTX detected: gfx1100, 24 GB VRAM
- qwen3:4b inference: **15 tok/s (CPU) → 140 tok/s (GPU)**

### 2. Verbosity Fix
qwen3:4b emits extensive internal `thinking` content. The gateway now auto-injects a terse system prompt for any qwen3 request without one:

> "You are a terse assistant. Answer directly. Never explain your reasoning."

| Prompt | Before | After |
|--------|--------|-------|
| Code generation (JSON keys function) | 6,126 tokens | **229 tokens** |
| Simple math (2+2) | 318 tokens | **117 tokens** |

### 3. Gateway Features
- **Config-driven routing** (`config.yaml`): add models without code changes
- **Pluggable providers**: Ollama (working), OpenRouter (wired, needs key)
- **Stub heuristic**: <500 chars + no architecture keywords → local
- **SQLite logging**: timestamp, model, latency, tokens, cost estimate
- **Systemd service**: `model-router` — starts after Ollama, restarts on failure

---

## Pending (when project resumes)

1. **Michael's direction decision** — A, B, C, or D above
2. **Classifier v2** — Replace stub heuristic with real routing logic
3. **OpenRouter test** — Only if Michael wants frontier fallback (needs API key)
4. **Feedback loop** — Detect retry patterns, flag bad routes
5. **Client integration** — Point an actual tool at `http://localhost:8001`

---

## Repo Structure

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — routing, providers, logging |
| `config.yaml` | Model definitions + routing rules |
| `requirements.txt` | Python dependencies |
| `STATUS.md` | Detailed session history and analysis |
| `benchmark.py` / `benchmark_quick.py` | Performance testing scripts |
| `router.db` | SQLite request log (auto-created) |
| `gateway.log` | Uvicorn logs |

---

## Commands

```bash
# Start manually (systemd handles this on boot)
cd ~/Repos/github.com/rossim2i2/model-router
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001

# Check service
sudo systemctl status model-router

# View weekly stats
curl http://localhost:8001/stats/weekly

# Test health
curl http://localhost:8001/health
```
