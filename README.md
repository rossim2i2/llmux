# LLMux

A local FastAPI gateway that routes prompts between your GPU and frontier APIs. OpenAI-compatible `/v1/chat/completions` and `/v1/responses` endpoints with heuristic routing, cost tracking, a closed-loop feedback pipeline, and integration support for Claude Code, Codex CLI, and Letta Code.

**Status:** Phases 1–3 shipped — data capture, in-prompt feedback, LLM-as-judge. Article in progress.

> **A note on naming.** This repo is `rossim2i2/llmux` on GitHub but the working directory and systemd service are still `model-router` on the author's machine. LLMux is the project name going forward; `model-router` references reflect the local install path and have not been renamed yet.

----

## What Works Today

| Component | State |
|-----------|-------|
| Gateway API | ✅ `localhost:8001` — `/v1/chat/completions` + `/v1/responses` |
| Local inference | ✅ 4 Ollama models on RX 7900 XTX |
| Frontier providers | ✅ OpenAI, Anthropic, Google Gemini, OpenRouter (16 models total) |
| Streaming | ✅ SSE for Chat Completions + Responses API |
| Classifier v3 | ✅ Multi-signal scoring, three-tier routing, 100% no-worse accuracy on 25-prompt benchmark |
| Feedback loop | ✅ Full prompt/response capture, request IDs, passive heuristics |
| In-prompt directives | ✅ `!escalate` / `!frontier` / `!mid` / `!local` — strip and override |
| Misroute correlation | ✅ 5-min window + Jaccard similarity retroactively labels under-routed requests |
| Feedback API | ✅ `POST /v1/feedback` — explicit rating + ideal_tier |
| LLM-as-judge | ✅ Daily cron, gpt-5-nano, ~50 requests/day, ~$0.50/day |
| Proxy model bypass | ✅ Ignores proxy-sent model names, applies gateway heuristic |
| API key fallback | ✅ Gateway uses its own keys when client sends placeholder |
| Model aliases | ✅ Config-driven alias resolution (e.g. `gpt-5-mini` → `gpt-5.4-mini`) |
| Cost tracking | ✅ SQLite `router.db` |
| Systemd service | ✅ User service, auto-restarts |
| Claude Code integration | ✅ Via claude-code-proxy + gateway |
| Codex CLI integration | ✅ Via `model_providers` config + `/v1/responses` |
| Letta Code integration | ✅ Via local backend + `LMSTUDIO_BASE_URL` |

----

## Architecture

```
Claude Code ──→ claude-code-proxy:8082 ──┐
                                        │
Codex CLI ──→ ~/.codex/config.toml ─────┤
                                        ├──→ Gateway :8001
Letta Code ──→ local backend (LMSTUDIO_BASE_URL) ┘        │
                                           Router decides:
                                                  │
                                          ┌───────┴───────┐
                                          │               │
                                          v               v
                                        Ollama       Frontier APIs
                                        (GPU)    (OpenAI/Anthropic/Google)
                                     4 models       12 models
```

----

## Integration Guides

### Claude Code

Claude Code speaks the Anthropic Messages API. The gateway speaks OpenAI Chat Completions. A translation proxy bridges the gap.

**Setup:**

1. Install [claude-code-proxy](https://github.com/nielspeter/claude-code-proxy) (Go binary)
2. Configure proxy to point at gateway:

```bash
# ~/.claude/proxy.env
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=sk-placeholder
```

3. Start proxy: `claude-code-proxy`
4. Launch Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8082 claude`

**How routing works:**

The proxy sends a fixed model name (e.g. `gpt-5`) for every request. The gateway's `proxy_models` config lists these names — when it sees one, it ignores it and applies the heuristic instead. This means:
- Simple prompts → local Ollama
- Complex prompts → frontier (gpt-5.4-mini)
- The user never has to switch models manually

**Gotchas:**
- The proxy's model name mapping overrides the gateway's body-model resolution. Without `proxy_models`, every request routes to the same model.
- Claude Code wraps user prompts in large system messages (40-50K chars). The heuristic must look at the *last user message* length, not total prompt length.
- The proxy uses `sk-placeholder` as the API key. The gateway falls back to its own environment-configured keys via `_resolve_api_key()`.

### Codex CLI

Codex uses the OpenAI Responses API (`/v1/responses`), not Chat Completions. The gateway implements a Responses API endpoint that converts to/from Chat Completions internally.

**Setup:**

1. Switch from ChatGPT auth to API key auth:

```bash
export $(grep -v '^#' ~/Repos/github.com/rossim2i2/model-router/.env | xargs)
echo "$OPENAI_API_KEY" | codex login --with-api-key
```

ChatGPT auth forces all requests through OpenAI's API and ignores custom base URLs. API key auth is required for custom routing.

2. Configure `~/.codex/config.toml`:

```toml
[model_providers.model-router]
name = "Model Router"
type = "openai"
base_url = "http://localhost:8001/v1"
api_key = "sk-placeholder"
wire_api = "responses"

[profiles.routed]
model_provider = "model-router"
model = "gpt-5"
```

3. Launch: `codex -p routed`

**How routing works:**

Same as Claude Code — the profile sends `gpt-5` as the model name, which is in `proxy_models`, so the gateway applies the heuristic.

**Gotchas:**
- `wire_api = "chat"` is no longer supported in Codex v0.130+. Must use `wire_api = "responses"`.
- The `type = "openai"` field is required in the model provider config.
- A `profiles` section is needed to select the provider — Codex doesn't use `model_providers` without a profile.
- The Responses API requires streaming (SSE). The gateway implements the full event sequence: `response.created` → `response.in_progress` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta` (repeated) → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.completed` → `done`.
- The Responses API input format uses `type: "message"` items with `input_text` content blocks (not `output_text`).

### Letta Code

Letta Code v0.25.8+ supports local provider connections via `--backend local`, which routes LLM inference directly from the CLI — no Docker or self-hosted server required.

**Setup:**

1. Set the LM Studio base URL to point at the gateway (add to `.bashrc`):

```bash
export LMSTUDIO_BASE_URL=http://localhost:8001/v1
```

2. Connect the local provider:

```bash
letta --backend local connect lmstudio
```

3. Run with a local model:

```bash
letta --backend local -p "What is 2+2?" --model lmstudio/qwen2.5-coder-7b-local
```

**How routing works:**

The local backend sends model names prefixed with `lmstudio/` (e.g., `lmstudio/qwen2.5-coder-7b-local`). The gateway strips the prefix and applies the classifier. The local backend bundles tool definitions into the user message as content-block arrays (~1.3K chars), but the gateway's `_extract_user_intent()` parses the content blocks and extracts only the last `type: "text"` block — the user's actual message. This means:
- "What is 2+2?" (12 chars extracted) → local
- "Design a distributed system..." (150 chars, architecture signal) → frontier

**Gotchas:**
- `--backend local` is required — without it, `connect` fails with "Settings not initialized"
- The LMStudio provider sends `"not-needed"` as its API key — the gateway treats this as a placeholder and falls back to its own keys
- Local backend agents are separate from cloud agents — your cloud agent (api.letta.com) is untouched
- `LMSTUDIO_BASE_URL` is not stored in the provider config — must be set in the environment
- OpenAI-compatible proxy endpoints are "not officially supported" per Letta docs

**Self-hosted server path (alternative):**

The self-hosted Letta server (Docker) also supports `OPENAI_API_BASE` to redirect inference, but has a known `openai-proxy/*` handle prefix bug (issue #476). The local backend path avoids this entirely.

----

## Routing Logic

The gateway routes requests using a **multi-signal prompt classifier** that scores the user's actual intent (not framework scaffolding) and maps to a three-tier routing table.

### How it works

1. **Extract user intent** — Parse the last user message. For content-block arrays (agent frameworks like Letta), extract only the last `type: "text"` block (the user's actual message, not system reminders or tool definitions).
2. **Score with 11 signals** — Each signal contributes integer points to a complexity score.
3. **Map score to tier** — Score ≤ 0 → local, 1–3 → mid, ≥ 4 → frontier.

### Signals

| Signal | Score | Detects |
|--------|-------|---------|
| `code_block` | +2 | Multi-line code (`def`, `class`, `async def`, ` ``` `) |
| `code_reference` | +1 | Backtick function calls, "Write a Python/K8s/..." |
| `error_traceback` | +1 | `Traceback`, `XError`, `at line N` |
| `architecture` | +2 | distributed, k8s, consistency, latency, throughput, transaction... |
| `reasoning_verb` | +2 | compare, evaluate, why does, should I use, tradeoffs... |
| `security_correctness` | +2 | bug, race, threading, off-by-one, O(n²), vulnerability... |
| `multi_step` | +3 | "then write", "in order", "list tool calls", "be exact about" |
| `multi_constraint` | +2 | "3 replicas", "exposing port", numeric specs |
| `precision_required` | +1 | "exact command", "exact arguments" |
| `search_query` | +1 | "find every/all X that Y" |
| `simplicity_hint` | -2 | "one-liner", "single command", "briefly", "in one sentence" |
| `simple_lookup` | -1 | Short "what's the X" / "how do I" questions |
| `long` / `very_long` | +1/+2 | 800+ / 1500+ chars (positive only — short ≠ simple) |

### Three-tier routing

| Tier | Score | Model | Cost/1K input | When |
|------|-------|-------|---------------|------|
| **Local** | ≤ 0 | `qwen2.5-coder-7b-local` (Ollama) | $0.00 | Simple lookups, trivial code, factual questions |
| **Mid** | 1–3 | `gpt-5.4-nano` (OpenAI) | $0.11 | Code writing, debugging, config generation |
| **Frontier** | ≥ 4 | `gpt-5.4-mini` (OpenAI) | $0.69 | Architecture, security review, multi-step reasoning |

### Validation results

The classifier is validated against the 25-prompt benchmark suite with known difficulty labels:

| Metric | Result |
|--------|--------|
| Exact tier match | 19/25 (76%) |
| No worse than ideal | 25/25 (100%) |
| Hard prompts → frontier | 8/8 (100%) |
| Medium prompts → mid+ | 8/8 (100%) |
| Easy/trivial → local | 4/9 (44% — rest over-routed to mid, acceptable) |

"No worse than ideal" means the prompt is sent to a tier at least as capable as the ideal — over-routing (easy → mid) costs more but doesn't sacrifice quality.

Run the validator: `python3 benchmark/validate_classifier.py`

### Proxy model bypass

Model names in `proxy_models` (e.g. `gpt-5`, `gpt-5-mini`) are ignored by the body-model resolver. This prevents proxy layers from overriding the gateway's routing decisions.

### API key fallback

When the client sends a placeholder key (`sk-placeholder`, `sk-xxx`), the gateway falls back to its own environment-configured keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

----

## Feedback Loop

The classifier learns from real-world usage through an explicit feedback channel. Every request gets logged with the full prompt + response + routing decision, and users can flag misroutes inline.

### In-prompt directives

Prefix any user message with one of these tokens to override or correct routing. The prefix is stripped before the model sees it.

| Prefix | Effect |
|--------|--------|
| `!escalate` | Route this message one tier higher than the classifier picked, AND label the previous request as under-routed |
| `!frontier` | Force frontier tier |
| `!mid` | Force mid tier |
| `!local` | Force local tier |

Example: if you ask a question and the local model gives a weak answer, just re-ask it as `!escalate explain how a database index works`. The gateway:

1. Strips `!escalate` before sending to the frontier model
2. Marks this request with `escalation_signal=1`
3. Finds your previous request (same client IP, last 5 minutes, similar prompt — Jaccard > 0.5) and labels it as under-routed (`ideal_tier=mid`)

That gives the classifier a labeled training example without any explicit feedback step.

### Feedback API

For explicit feedback (e.g. from a CLI), POST to `/v1/feedback`:

```bash
curl -X POST http://localhost:8001/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "1e3b11b2-a61c-4702-9dbd-0a251526e41c",
    "rating": -1,
    "ideal_tier": "frontier",
    "comment": "response was incomplete"
  }'
```

The `request_id` comes from the `X-LLMux-Request-Id` response header that every chat completions / responses call returns.

| Field | Required | Notes |
|-------|----------|-------|
| `request_id` | yes | UUID from `X-LLMux-Request-Id` header |
| `rating` | yes | `-1` (bad) or `+1` (good) |
| `ideal_tier` | no | `local` / `mid` / `frontier` — what the request should have routed to |
| `comment` | no | Free-form text |

### What gets logged

Every request stores:

- **Routing decision**: chosen tier, classifier score, fired signals (JSON array), forced/escalation flags
- **Bodies**: prompt text + response text (set `LLMUX_CAPTURE_BODIES=0` to disable)
- **Passive heuristics**: `response_truncated` (finish_reason=length), `response_short` (<50 chars), `response_error`
- **Labels**: `ideal_tier`, source (`user_feedback` / `user_escalate` / `correlated_escalation` / `llm_judge`)

Use the `misroutes` view for training data:

```bash
sqlite3 router.db "SELECT chosen_tier, ideal_tier, routing_outcome, label_source, score, signals, substr(prompt_text,1,60) FROM misroutes ORDER BY timestamp DESC LIMIT 20;"
```

### Privacy

Body capture is enabled by default — the feedback loop needs the raw text to identify what's misclassified. Disable with `LLMUX_CAPTURE_BODIES=0` in the environment if you don't want prompts/responses persisted.

### LLM-as-judge (Phase 3)

A daily cron samples ~50 unjudged requests and sends prompt+response to a judge model (gpt-5-nano). The judge evaluates:

- **Complexity**: What tier was actually required? (local/mid/frontier)
- **Quality**: How good was the response? (1-5)
- **Reasoning**: One-sentence explanation

Results update `judged_quality`, `judged_at`, `judged_by`, `judged_reasoning`, and `ideal_tier` columns.

Run manually:
```bash
cd ~/Repos/github.com/rossim2i2/model-router
export $(grep -v '^#' .env | xargs)
./.venv/bin/python scripts/judge-recent.py --limit 50 --verbose
```

Cost: ~$0.50/day at 50 requests/day with gpt-5-nano.

----

## Quick Start

```bash
# Start gateway (systemd handles this normally)
systemctl --user start model-router

# Test health
curl http://localhost:8001/health

# Test Chat Completions
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b-local","messages":[{"role":"user","content":"What is 2+2?"}]}'

# Test Responses API
curl http://localhost:8001/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4-mini","input":"What is 2+2?"}'

# Force a specific route
curl "http://localhost:8001/v1/chat/completions?route=gpt-5.5" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b-local","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

----

## Config

Models and routing are defined in `config.yaml`:

```yaml
models:
  qwen2.5-coder-7b-local:
    provider: ollama
    endpoint: http://localhost:11434/api/chat
    model_name: qwen2.5-coder:7b
    default_options:
      temperature: 0.2

  gpt-5.4-mini:
    provider: openai
    endpoint: https://api.openai.com/v1/chat/completions
    model_name: gpt-5.4-mini
    use_max_completion_tokens: true
    default_options:
      max_completion_tokens: 1024
    cost_per_1m_input: 0.75
    cost_per_1m_output: 4.50
    aliases: [gpt-5-mini]

routing:
  default: qwen2.5-coder-7b-local
  mid: gpt-5.4-nano
  frontier: gpt-5.4-mini
  force_param: route
  proxy_models: [gpt-5, gpt-5-mini]
```

**Key config options:**
- `aliases` — map client model names to configured backends
- `proxy_models` — model names to ignore for routing (let heuristic decide)
- `use_max_completion_tokens` — for OpenAI models that reject `max_tokens`
- `think: false` — for Ollama thinking models (merge thinking into content)
- `no_temperature` — for models that reject temperature parameter

----

## API Keys

The gateway needs API keys for frontier providers. Store them in `.env` (gitignored):

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
```

The systemd service loads these via `EnvironmentFile`.

----

## Repo Structure

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — routing, providers, Responses API, feedback loop, logging |
| `config.yaml` | Model definitions, aliases, routing rules |
| `scripts/judge-recent.py` | LLM-as-judge daily cron — scores unjudged requests |
| `benchmark/` | Benchmark runner, judge, prompts, results, classifier validator |
| `.env` | API keys (gitignored) |
| `requirements.txt` | Python dependencies |
| `router.db` | SQLite request log (auto-created) |

----

## Commands

```bash
# Systemd (preferred)
systemctl --user start model-router
systemctl --user status model-router
journalctl --user -u model-router -f

# Manual start
cd ~/Repos/github.com/rossim2i2/model-router
export $(grep -v '^#' .env | xargs)
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8001

# Check health
curl http://localhost:8001/health

# List available models
curl http://localhost:8001/v1/models
```

----

## Evidence

- **Benchmark narrative:** [`benchmark/BENCHMARK-RESULTS.md`](benchmark/BENCHMARK-RESULTS.md) — full write-up of the 16-model evaluation
- **Aggregated metrics (sanitized):** [`benchmark/results/summary.json`](benchmark/results/summary.json) — per-model run counts, median latency/cost, judge scores. No raw prompts or responses.
- **Integration sessions:** captured runs through real CLI tools
  - [`benchmark/results/integration-claude-code-20260514-111038.md`](benchmark/results/integration-claude-code-20260514-111038.md)
  - [`benchmark/results/integration-claude-code-20260514-115405.md`](benchmark/results/integration-claude-code-20260514-115405.md)
  - [`benchmark/results/integration-codex-20260514-134802.md`](benchmark/results/integration-codex-20260514-134802.md)
