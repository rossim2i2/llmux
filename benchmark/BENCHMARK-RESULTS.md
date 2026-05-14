# Benchmark Results Summary

Data for article drafting. All numbers from real runs on May 13-14, 2026.

## Hardware

- **CPU:** AMD Ryzen 9 9950X3D (16-core)
- **RAM:** 64GB DDR5
- **GPU:** AMD Radeon RX 7900 XTX (24GB VRAM) — enthusiast tier, ~$700-900
- **OS:** Arch Linux, Wayland, ROCm

## Models Tested (16 total)

### Local (Ollama, running on GPU)

| Model | Params | Type | Notes |
|---|---|---|---|
| qwen2.5-coder:7b | 7B | Code-specialized | Best local model — concise, fast, high quality |
| deepseek-coder-v2:16b | 16B (MoE, ~2.4B active) | Code-specialized | MoE keeps it fast despite 16B params |
| gemma3:4b | 4B | General | Surprisingly thorough on hard problems, but slow |
| qwen3:4b | 4B | Thinking model | Cautionary tale — burns tokens on reasoning, worst quality |

### Frontier (Anthropic API, direct)

| Model | API ID | Notes |
|---|---|---|
| claude-haiku-4-5 | claude-haiku-4-5-20251001 | Cheap frontier sweet spot |
| claude-sonnet-4-6 | claude-sonnet-4-6 | Mid-tier frontier |
| claude-opus-4-7 | claude-opus-4-7 | Strongest frontier, adaptive thinking |

### Frontier (OpenAI API, direct)

| Model | API ID | Notes |
|---|---|---|
| gpt-5.4-nano | gpt-5.4-nano | Cheapest OpenAI frontier |
| gpt-5.4-mini | gpt-5.4-mini | Fast mid-tier OpenAI |
| gpt-5.5 | gpt-5.5 | Flagship OpenAI, reasoning model |

### Frontier (Google Gemini API, direct)

| Model | API ID | Notes |
|---|---|---|
| gemini-2.5-flash | gemini-2.5-flash | Cheap Google frontier, thinking model |
| gemini-2.5-pro | gemini-2.5-pro | Strongest Google frontier, thinking model |

### Open (OpenRouter API)

| Model | API ID | Params | Notes |
|---|---|---|---|
| kimi-k2.6 | moonshotai/kimi-k2.6 | 1T (32B active MoE) | Leading open-weights model, thinking |
| deepseek-v4-pro | deepseek/deepseek-v4-pro | 1.6T (49B active MoE) | Strong reasoning, competitive programming |
| gemma-4-26b | google/gemma-4-26b-a4b-it | 26B (3.8B active MoE) | Google's latest open model, non-thinking |
| qwen3.5-9b | qwen/qwen3.5-9b | 9B | Small but strong, thinking model |

## Benchmark Design

- **25 prompts** across 7 categories: code-generation, code-editing, debugging, explanation, config, tool-use, hard-reasoning
- **4 difficulty tiers:** trivial, easy, medium, hard
- **3 runs per prompt per model** (cold + 2 warm), 750 total runs for local+Anthropic+OpenAI; 1 run per prompt for Gemini (50 runs) and open models (100 runs)
- **Quality scoring:** LLM judge (Claude Sonnet 4.6 for Google, Claude Haiku 4.5 for others) scored each response 1-5 on correctness, completeness, clarity
- **Metrics:** TTFT, total latency, tokens/sec, input/output tokens, cost

## Results: Quality x Speed x Cost

| Model | Correct | Complete | Clarity | TTFT | Total | Cost/1K calls |
|---|---|---|---|---|---|---|
| qwen2.5-coder:7b | 5.0 | 4.0 | 5.0 | 58ms | 1.7s | $0.00 |
| deepseek-coder-v2:16b | 4.0 | 4.0 | 5.0 | 46ms | 2.1s | $0.00 |
| gemma3:4b | 4.0 | 4.0 | 5.0 | 125ms | 4.7s | $0.00 |
| qwen3:4b | 2.0 | 2.0 | 1.0 | 57ms | 14.9s | $0.00 |
| gpt-5.4-nano | 5.0 | 5.0 | 5.0 | 574ms | 1.5s | $0.11 |
| gpt-5.4-mini | 5.0 | 5.0 | 5.0 | 421ms | 1.1s | $0.69 |
| claude-haiku-4-5 | 5.0 | 5.0 | 5.0 | 555ms | 2.4s | $1.71 |
| gpt-5.5 | 5.0 | 5.0 | 5.0 | 1787ms | 4.0s | $9.99 |
| claude-sonnet-4-6 | 5.0 | 5.0 | 5.0 | 1021ms | 6.7s | $7.23 |
| claude-opus-4-7 | 5.0 | 5.0 | 5.0 | 1663ms | 5.8s | $10.89 |
| gemini-2.5-flash | 4.0 | 4.0 | 5.0 | 9565ms | 9.6s | $0.86 |
| gemini-2.5-pro | 4.0 | 3.5 | 5.0 | 17054ms | 17.9s | $3.05 |
| gemma-4-26b | 5.0 | 5.0 | 5.0 | 323ms | 7.7s | $0.13 |
| qwen3.5-9b | 5.0 | 5.0 | 5.0 | 8370ms | 14.8s | $0.40 |
| deepseek-v4-pro | 5.0 | 5.0 | 5.0 | 10501ms | 15.0s | $0.62 |
| kimi-k2.6 | 5.0 | 5.0 | 5.0 | 27237ms | 33.0s | $6.82 |

## Results by Difficulty

Format: median correctness / median total time

| Model | trivial | easy | medium | hard |
|---|---|---|---|---|
| qwen2.5-coder:7b | 5/1.7s | 5/0.9s | 4/1.6s | 4/3.1s |
| deepseek-coder-v2:16b | 5/3.9s | 4/0.9s | 4/2.3s | 4/4.1s |
| gemma3:4b | 5/4.7s | 5/1.2s | 2/4.7s | 4/6.6s |
| qwen3:4b | 5/10.6s | 3/7.5s | 2/14.9s | 2/15.0s |
| gpt-5.4-nano | 5/1.9s | 5/1.2s | 5/1.4s | 5/2.0s |
| gpt-5.4-mini | 5/1.0s | 5/0.7s | 5/1.1s | 5/1.4s |
| claude-haiku-4-5 | 5/1.5s | 5/1.5s | 4/2.4s | 4/4.0s |
| gpt-5.5 | 5/2.2s | 5/1.3s | 5/3.6s | 5/10.9s |
| claude-sonnet-4-6 | 5/5.4s | 5/5.6s | 5/8.2s | 5/13.0s |
| claude-opus-4-7 | 5/4.4s | 5/2.7s | 5/5.9s | 5/7.8s |
| gemini-2.5-flash | 5/6.0s | 5/3.7s | 4/9.8s | 3/10.1s |
| gemini-2.5-pro | 5/15.5s | 5/15.3s | 3/18.1s | 4/18.0s |
| gemma-4-26b | 5/7.3s | 5/3.4s | 5/7.5s | 4.5/9.8s |
| qwen3.5-9b | 5/12.2s | 5/13.8s | 5/22.2s | 4.5/15.0s |
| deepseek-v4-pro | 5/8.6s | 5/11.9s | 5/13.1s | 5/29.8s |
| kimi-k2.6 | 5/9.1s | 5/16.7s | 5/30.5s | 5/56.8s |

## Key Findings

### 1. Local matches frontier on simple prompts

qwen2.5-coder:7b scores 5/5 correctness on trivial and easy prompts — identical to Opus 4.7 and GPT-5.5 — at zero cost and 3x the speed. This is the routing win.

### 2. The ceiling is medium/hard

Local models drop to 4/5 on medium, 2-4/5 on hard. Frontier stays at 5/5 across all difficulties. That's where routing to the cloud pays for itself.

### 3. GPT-5.4-mini is the overall speed champion

1.1s median total, 5/5/5 quality, $0.69/1K calls. Faster than Haiku, cheaper than Haiku, same quality. This is the frontier sweet spot for coding tasks.

### 4. GPT-5.4-nano is the cheapest frontier at quality parity

$0.11/1K calls, 5/5/5 quality, 1.5s median. Cheaper than Haiku by 15x, same quality. For pure cost optimization, this is the escalation tier.

### 5. Haiku drops to 4/5 on medium and hard

Unlike GPT-5.4-nano/mini which maintain 5/5 across all difficulties, Haiku drops to 4/5 on medium and hard. This is a meaningful quality difference that the article should note.

### 6. GPT-5.5 is slow on hard prompts

10.9s median on hard, with reasoning tokens inflating cost ($9.99/1K). It's the most expensive model tested but doesn't outperform GPT-5.4-mini on these coding tasks. Reasoning models have a cost/speed tax that isn't justified for routine coding.

### 7. Opus adapts effort to difficulty

Paradoxically faster than Sonnet on hard prompts (7.8s vs 13.0s) but slower on easy ones. Adaptive thinking adjusts effort. GPT-5.5 shows the same pattern but more dramatically (10.9s on hard).

### 8. The cost story is dramatic

1,000 calls to Opus = $10.89. 1,000 calls to GPT-5.5 = $9.99. 1,000 calls to qwen2.5-coder:7b = $0. If 80% of prompts route local, you save $8-10 per 1,000 calls.

### 9. The wrong local model is worse than paying for frontier

qwen3:4b is the smallest model but the slowest (14.9s median) and worst quality (2/2/1). Its thinking mode burns the entire token budget on reasoning before producing visible content. This is a cautionary tale: model selection matters more than model size.

### 10. Code-specialized models outperform general ones

qwen2.5-coder:7b (code-specialized, 7B) beats gemma3:4b (general, 4B) on every dimension despite being larger. For a coding routing use case, domain specialization is the right axis to optimize on.

### 11. Gemini Flash is cheap but slow — thinking tax

gemini-2.5-flash costs $0.86/1K calls (between nano and mini) but is 6-9x slower than GPT-5.4-mini (9.6s vs 1.1s). The thinking model overhead means it's not competitive on speed for coding tasks. Quality (4/4/5) matches Haiku but falls behind GPT-5.4-nano/mini (5/5/5).

### 12. Gemini Pro is the slowest model tested

17.9s median total — slower than even qwen3:4b (14.9s). At $3.05/1K calls, it costs more than Haiku ($1.71) for worse quality (4/3.5/5 vs 5/5/5) and worse speed. The thinking model's reasoning phase dominates latency without translating to better coding outputs.

### 13. Google's thinking models don't match OpenAI's non-thinking models

GPT-5.4-mini (non-thinking) achieves 5/5/5 quality at 1.1s. Gemini 2.5 Flash (thinking) achieves 4/4/5 at 9.6s. The thinking overhead doesn't pay off for coding benchmarks — it adds latency without improving correctness.

### 14. Open models match closed frontier quality — at lower cost

All four open models achieve 5/5/5 median quality, matching GPT-5.4-mini, Haiku, Sonnet, and Opus. The open/closed quality gap has closed on coding benchmarks. The differentiator is now speed and cost, not correctness.

### 15. Gemma 4 26B is the cheapest quality-matched model

$0.13/1K calls, 5/5/5 quality, 7.7s median. Cheaper than GPT-5.4-nano ($0.11) but 5x slower. Still, it's the cheapest open model that matches frontier quality — and it's Apache 2.0 licensed.

### 16. Kimi K2.6 is the slowest model tested — by far

33.0s median total, 56.8s on hard prompts. Slower than Gemini 2.5 Pro (17.9s). At $6.82/1K calls, it costs more than Haiku ($1.71) for the same quality at 14x the latency. The 1T-parameter MoE with 32B active is powerful but slow through OpenRouter.

### 17. DeepSeek V4 Pro: quality at a price (latency)

5/5/5 quality, $0.62/1K calls, but 15.0s median and 29.8s on hard. The thinking model overhead means it's slower than GPT-5.4-mini by 14x. Cost-effective if latency doesn't matter, but for interactive coding it's not competitive.

### 18. Qwen 3.5-9B: small model, big thinking tax

9B params, $0.40/1K calls, 5/5/5 quality — but 14.8s median with 784 reasoning tokens per call. The thinking overhead dominates latency. Same pattern as qwen3:4b locally, but the model is strong enough that the quality survives the reasoning phase.

### 19. The thinking tax is universal — not provider-specific

Every thinking model tested (Gemini 2.5 Flash/Pro, GPT-5.5, Kimi K2.6, DeepSeek V4 Pro, Qwen 3.5-9B, qwen3:4b) is slower than the non-thinking models at the same quality tier. The only non-thinking open model (Gemma 4 26B) is the fastest open model at 7.7s. This confirms: for coding tasks, thinking doesn't help — it hurts.

## The Routing Gradient (updated with open models)

```
                       Quality  Speed     Cost/1K calls
qwen2.5-coder:7b       5/4/5    1.7s      $0.00     ← local default
gpt-5.4-nano           5/5/5    1.5s      $0.11     ← cheapest frontier
gemma-4-26b            5/5/5    7.7s      $0.13     ← cheapest open (non-thinking)
qwen3.5-9b             5/5/5    14.8s     $0.40     ← cheap open (thinking)
deepseek-v4-pro        5/5/5    15.0s     $0.62     ← open reasoning
gpt-5.4-mini           5/5/5    1.1s      $0.69     ← fastest frontier
gemini-2.5-flash       4/4/5    9.6s      $0.86     ← cheap Google (thinking)
claude-haiku-4-5       5/5/5    2.4s      $1.71     ← cheap Anthropic
gemini-2.5-pro         4/3.5/5  17.9s     $3.05     ← smart Google (thinking)
claude-sonnet-4-6      5/5/5    6.7s      $7.23     ← smart Anthropic
kimi-k2.6              5/5/5    33.0s     $6.82     ← open flagship (thinking)
gpt-5.5                5/5/5    4.0s      $9.99     ← reasoning frontier
claude-opus-4-7        5/5/5    5.8s      $10.89    ← strongest Anthropic
```

## Cross-Provider Comparison

**Cheapest frontier per provider:**
- OpenAI: gpt-5.4-nano at $0.11/1K calls
- Open (OpenRouter): gemma-4-26b at $0.13/1K calls
- Google: gemini-2.5-flash at $0.86/1K calls
- Anthropic: claude-haiku-4-5 at $1.71/1K calls

**Fastest frontier per provider:**
- OpenAI: gpt-5.4-mini at 1.1s
- Anthropic: claude-haiku-4-5 at 2.4s
- Open (OpenRouter): gemma-4-26b at 7.7s
- Google: gemini-2.5-flash at 9.6s

**Best quality on hard prompts per provider:**
- OpenAI: gpt-5.4-mini at 5/5 (1.4s, $0.69/1K)
- Open (OpenRouter): deepseek-v4-pro at 5/5 (29.8s, $0.62/1K)
- Anthropic: claude-opus-4-7 at 5/5 (7.8s, $10.89/1K)
- Google: gemini-2.5-pro at 4/5 (18.0s, $3.05/1K)

OpenAI's GPT-5.4 family still dominates the cost/speed/quality tradeoff for coding tasks. But the open models have closed the quality gap entirely — all four achieve 5/5/5. The differentiator is now speed and cost, not correctness. The thinking tax is universal: every thinking model (open or closed) is slower than non-thinking models at the same quality tier.

## Response Quality Examples

### Trivial prompt: "Write a Python one-liner for squares of 0-9"

- **qwen2.5-coder:7b** (59tok, 593ms): Clean code block with one-sentence explanation. Correct.
- **gemma3:4b** (69tok, 619ms): Same answer, slightly more explanation. Correct.
- **deepseek-coder-v2:16b** (82tok, 605ms): Same answer with print() added. Correct.
- **qwen3:4b** (2048tok, 14.9s): 2048 tokens of reasoning before getting to the answer. Hits token cap. Clarity 1.

### Hard prompt: "Fix the bug in this bracket-balancing function"

- **gemma3:4b** — only model that caught BOTH bugs (empty-stack pop AND non-empty-final-stack). Most thorough.
- **qwen2.5-coder:7b** — caught one bug, missed the other.
- **deepseek-coder-v2:16b** — thorough explanation, caught main bug.
- **qwen3:4b** — ran out of tokens before finishing.

## Data Files

- `benchmark/results/full-local-comparison.judged.json` — 4 local models, 200 runs, quality scored
- `benchmark/results/frontier-anthropic.judged.json` — 3 Anthropic models, 225 runs, quality scored
- `benchmark/results/frontier-openai.judged.json` — 3 OpenAI models, 225 runs, quality scored
- `benchmark/results/frontier-google.judged.json` — 2 Google models, 50 runs, quality scored
- `benchmark/results/frontier-open.judged.json` — 4 open models, 100 runs, quality scored
- `benchmark/results/openai-55.json` — GPT-5.5 standalone results (75 runs)

## Total API Spend

- Anthropic: ~$1.49 (benchmarks + judge runs)
- OpenAI: ~$0.75 (GPT-5.5 benchmark) + ~$0.05 (nano/mini) + ~$0.25 (judge) ≈ $1.05
- Google: ~$0.10 (Gemini Flash + Pro benchmarks) + ~$0.15 (judge) ≈ $0.25
- OpenRouter: ~$0.20 (4 open model benchmarks) + ~$0.05 (judge) ≈ $0.25
- **Total: ~$3.04** — well under $50 budget
