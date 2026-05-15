# CLAUDE.md

Notes for AI assistants and contributors working on this repo.

## Ground Truth

**`README.md` is canonical.** If anything in this file conflicts with the README, the README wins. Update the README before changing behavior; update this file only when the assistant-facing guidance needs to change.

## What This Is

LLMux is a reference implementation of a local LLM routing gateway, written as the companion code for an article on hybrid local + frontier inference. It is not a production-grade product. Treat it as:

- A working example you can run on your own hardware
- A starting point to fork and adapt
- A demonstration that the routing patterns described in the article actually work

The project name on GitHub is `rossim2i2/llmux`. The local install path is still `model-router` on the author's machine. See the README's naming note.

## Key Commands

```bash
# Start the gateway (manual)
cd ~/Repos/github.com/rossim2i2/model-router
export $(grep -v '^#' .env | xargs)
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001

# Start via systemd
systemctl --user start model-router

# Health check
curl http://localhost:8001/health

# Run benchmark (quick smoke test)
python benchmark/bench.py --models qwen2.5-coder-7b-local --limit 5

# Validate classifier against benchmark
.venv/bin/python benchmark/validate_classifier.py

# Run the LLM-as-judge over recent unjudged requests
./.venv/bin/python scripts/judge-recent.py --limit 50 --verbose
```

## Important Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — routing, providers, Responses API, feedback |
| `config.yaml` | Model definitions, aliases, routing rules, classifier config |
| `benchmark/bench.py` | Benchmark runner |
| `benchmark/judge.py` | LLM-as-judge for benchmark runs |
| `benchmark/prompts.yaml` | 25 benchmark prompts across categories/difficulties |
| `benchmark/validate_classifier.py` | Validates classifier against benchmark expectations |
| `scripts/judge-recent.py` | Daily cron — judges live production requests |
| `benchmark/BENCHMARK-RESULTS.md` | Article-ready benchmark write-up |
| `benchmark/results/summary.json` | Sanitized aggregated metrics |
| `router.db` | SQLite request log (auto-created, gitignored) |

## Conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- Stage individual files; never `git add -A` or `git add .`
- No AI attribution in commit messages
- Python 3.11+, `.venv/` for the local virtualenv (gitignored)
- API keys live in `.env` (gitignored, `chmod 600`)

## What Not To Do

- Don't commit `router.db`, `.env`, `gateway.log`, or raw benchmark JSON
- Don't reintroduce `qwen3:4b-local` as the default — it scored 2/2/1 on the benchmark and is documented as a cautionary tale. Default is `qwen2.5-coder-7b-local`.
- Don't add features without updating the README's "What Works Today" table
- Don't break the OpenAI compatibility of `/v1/chat/completions` — Claude Code, Codex CLI, and Letta Code all depend on it

## Article Context

The repo backs an article on hybrid LLM routing. The article references:
- The 16-model benchmark in `BENCHMARK-RESULTS.md`
- The aggregated metrics in `results/summary.json`
- The integration sessions in `results/integration-*.md`

If you change benchmark methodology, defaults, or model lineup, update the article evidence accordingly.
