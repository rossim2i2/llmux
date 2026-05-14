# Integration Test: claude-code

**Date:** 2026-05-14
**Client:** claude-code
**Gateway:** http://localhost:8001
**Tester:** Michael Rossi

## Setup

- Gateway running: (yes/no)
- Client configured: (yes/no)
- Client version: (fill in)

## Automated Checks

| # | Check | Prompt | Pass/Fail | Notes |
|---|-------|--------|-----------|-------|
| 1 | Basic routing | "What is 2+2?" | | |
| 2 | Streaming | "Explain git rebase in 3 sentences" | | |
| 3 | Model switching | Same prompt, different models | | |
| 4 | Tool use — file read | "Read main.py" | | |
| 5 | Tool use — shell | "Run git log --oneline -5" | | |
| 6 | Multi-turn | Reverse string → add type hints | | |
| 7 | Error recovery | Bad model key | | |

## Extended-Use Tasks

| # | Task | Completed | Issues | Notes |
|---|------|-----------|--------|-------|
| 1 | Code generation (/v1/responses) | | | |
| 2 | Code editing (tools/tool_choice passthrough) | | | |
| 3 | Debugging (429 retry-after) | | | |
| 4 | Multi-file (azure provider) | | | |
| 5 | Search + edit (max_tokens config) | | | |
| 6 | Test writing (test_routing.py) | | | |
| 7 | Documentation (Gemini provider) | | | |
| 8 | Complex tool use (benchmark run) | | | |
| 9 | Error scenario (missing API key) | | | |
| 10 | Architecture (classifier design) | | | |

## Issues Found

(Record anything that breaks, stalls, or produces unexpected results)

| Time | Task | Issue | Severity | Reproducible? |
|------|------|-------|----------|---------------|
| | | | | |

## Summary

- Automated checks passed: X/7
- Extended tasks completed: X/10
- Issues found: X
- Overall verdict: (usable / usable with caveats / broken)

## Notes

(Any additional observations about the client's behavior through the gateway)
