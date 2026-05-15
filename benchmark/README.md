# model-router benchmark

A reusable benchmark for measuring how well a hybrid LLM routing setup actually works on *your* hardware with *your* models.

The point of this suite is not to crown a winner. It's to map where small local models start to fall behind larger ones, so you know what to route locally and what to send to a frontier API.

## What it measures

| Metric | What it tells you |
|---|---|
| **TTFT** (time to first token) | How long the user waits before *anything* appears |
| **Total latency** | Wall-clock time for the full response |
| **Throughput** (tok/s) | How fast the model generates after TTFT |
| **Input / output tokens** | Real token counts from the model |
| **Cost** | Per-call cost in USD, computed from gateway config rates |
| **Quality** (optional, via `judge.py`) | 1-5 ratings on correctness, completeness, clarity from a frontier judge model |

## What it does *not* measure

- **Tool-calling protocol compliance.** The `tool-use` prompts test whether the model can *reason* about tool calls; they don't exercise actual function-calling format. Add that benchmark when your gateway passes tool-call payloads through.
- **Multi-turn behavior.** Single-turn prompts only.
- **Long context.** Prompts top out around 200 tokens. If your use case is multi-file reasoning, add longer prompts.

## How to run

### Prerequisites

1. Gateway running (`python main.py` from the repo root, default port 8001)
2. At least one model configured in `config.yaml`
3. For frontier models: `OPENROUTER_API_KEY` (or equivalent) in env

### Quick smoke test

```bash
# 5 prompts, qwen2.5-coder-7b-local only — finishes in a few seconds
python benchmark/bench.py --models qwen2.5-coder-7b-local --limit 5
```

### Full local-only run

```bash
# All 25 prompts × 3 iterations each
python benchmark/bench.py --local-models qwen2.5-coder-7b-local
```

### Local vs. frontier comparison

```bash
# Local runs 3x per prompt (warm-up tolerance); frontier runs 1x (cost control)
python benchmark/bench.py \
    --local-models qwen2.5-coder-7b-local deepseek-coder-v2-16b-local \
    --frontier-models gpt-5.4-nano
```

### Filter by category or difficulty

```bash
python benchmark/bench.py --models qwen2.5-coder-7b-local \
    --categories code-generation tool-use \
    --difficulties medium hard
```

### Score quality with a judge model

```bash
# After bench.py writes results/<timestamp>.json
python benchmark/judge.py \
    --results benchmark/results/20260513-093000.json \
    --judge-model claude-sonnet-4-6
```

Output: `<results>.judged.json` with `scores` attached per run, plus a median-score summary by model.

## Cost expectations

- **Local only:** $0
- **Local + gpt-4.1-nano frontier comparison (1 run each):** under $0.10
- **Local + claude-sonnet-4-6 frontier (1 run each):** ~$1
- **Full judge pass with claude-sonnet-4-6:** ~$1-2
- **Full judge pass with gpt-4.1-nano:** under $0.20

All cost estimates assume the default 25 prompts and modest response lengths.

## Output format

`benchmark/results/<timestamp>.json`:

```json
{
  "schema_version": 1,
  "timestamp": "2026-05-13T13:00:00+00:00",
  "hardware": {
    "platform": "...",
    "cpu_model": "...",
    "ram_gb": 128.0,
    "gpu": ["AMD Radeon RX 7900 XTX"]
  },
  "runs": [
    {
      "model": "qwen2.5-coder-7b-local",
      "prompt_id": "codegen-binary-search",
      "category": "code-generation",
      "difficulty": "medium",
      "iteration": 1,
      "cold": true,
      "ttft_ms": 320.5,
      "total_ms": 1840.2,
      "tokens_per_sec": 138.7,
      "input_tokens": 38,
      "output_tokens": 256,
      "cost_usd": 0.0,
      "response": "def binary_search(arr, target):\n    ...",
      "error": null,
      "timestamp": "2026-05-13T13:00:01+00:00"
    }
  ]
}
```

After `judge.py`, each run also gets:

```json
{
  "scores": {
    "correctness": 5,
    "completeness": 4,
    "clarity": 5,
    "reason": "Correct iterative implementation, returns -1 on miss, but doesn't handle empty list edge case."
  }
}
```

## Cold vs. warm runs

The **first run** of each prompt is marked `cold: true`. On Ollama, the first request loads the model into VRAM (slow). Subsequent runs are warm. The summary excludes cold runs from latency stats when warm runs are available, since you're more likely to experience warm latency in real use.

If you want to characterize cold-start cost specifically, look at the raw JSON for runs where `cold: true`.

## Adding your own prompts

Edit `prompts.yaml`. Each entry needs:

```yaml
- id: unique-slug
  category: code-generation  # or code-editing, debugging, explanation, config, tool-use, hard-reasoning
  difficulty: easy           # trivial | easy | medium | hard
  prompt: "What you want the model to answer."
  rubric: "What a correct, complete answer should contain. Used by judge.py."
```

The `rubric` field is what makes `judge.py` work. Without it, that prompt gets skipped during judging.

## A note on what "good" looks like

A local 4B model crushing trivial and easy prompts at 140 tok/s, then producing plausible-but-incorrect answers to hard prompts, is exactly the result that justifies a hybrid setup. Don't expect the local model to match a frontier model on everything — expect it to handle the long tail of routine prompts, and route the rest.

If your local model scores 5/5 on every prompt in this suite, the suite isn't hard enough.
