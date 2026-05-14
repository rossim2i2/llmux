#!/usr/bin/env bash
# integration-test.sh — Standardized client integration test for model-router.
#
# Usage:
#   ./benchmark/integration-test.sh claude-code    # Test Claude Code via proxy
#   ./benchmark/integration-test.sh codex          # Test Codex CLI
#   ./benchmark/integration-test.sh letta           # Test Letta Code (self-hosted)
#
# This script:
#   1. Verifies the gateway is running and the client is configured
#   2. Prints the 7 automated checks with exact prompts and what to look for
#   3. Prints the 10 extended-use prompts for a day of real work
#   4. Generates a blank results log template
#
# The actual prompt execution is manual — you type them into the real client.
# The script just sets up, instructs, and records.

set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8001}"
CLIENT="${1:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${REPO_ROOT}/benchmark/results/integration-${CLIENT:-unknown}-${TIMESTAMP}.md"

# ---- colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "  ${GREEN}PASS${RESET}"; }
fail() { echo -e "  ${RED}FAIL${RESET}"; }
warn() { echo -e "  ${YELLOW}WARN${RESET}"; }
info() { echo -e "  ${CYAN}$1${RESET}"; }
header() { echo -e "\n${BOLD}$1${RESET}"; }

# ---- usage ----

if [[ -z "$CLIENT" ]]; then
    echo "Usage: $0 <claude-code|codex|letta>"
    echo ""
    echo "Runs the standardized integration test for a specific client."
    exit 1
fi

# ---- setup verification ----

header "==== Setup Verification ===="

# Check gateway
info "Checking gateway at ${GATEWAY_URL}..."
if curl -sf "${GATEWAY_URL}/health" > /dev/null 2>&1; then
    pass
    GATEWAY_HEALTH=$(curl -sf "${GATEWAY_URL}/health")
    echo "    ${GATEWAY_HEALTH}"
else
    fail
    echo "    Gateway not running. Start it first:"
    echo "    cd ${REPO_ROOT} && .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001"
    exit 1
fi

# Check models endpoint
info "Checking /v1/models..."
if curl -sf "${GATEWAY_URL}/v1/models" > /dev/null 2>&1; then
    MODEL_COUNT=$(curl -sf "${GATEWAY_URL}/v1/models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo "?")
    echo -e "  ${GREEN}PASS${RESET} — ${MODEL_COUNT} models available"
else
    fail
    echo "    /v1/models not responding"
fi

# Client-specific setup
header "==== Client Setup: ${CLIENT} ===="

case "$CLIENT" in
    claude-code)
        echo ""
        echo "Claude Code connects via claude-code-proxy (Go):"
        echo ""
        echo "  1. Install proxy:"
        echo "     go install github.com/nielspeter/claude-code-proxy@latest"
        echo ""
        echo "  2. Start proxy (pointing at gateway):"
        echo "     OPENAI_BASE_URL=http://localhost:8001/v1 claude-code-proxy --port 8082 &"
        echo ""
        echo "  3. Launch Claude Code:"
        echo "     ANTHROPIC_BASE_URL=http://localhost:8082 claude"
        echo ""
        echo "  4. Verify: send a message and check gateway logs for the request"
        echo ""
        info "Checking claude-code-proxy..."
        if command -v claude-code-proxy &> /dev/null; then
            pass
        else
            warn
            echo "    claude-code-proxy not found in PATH. Install it first."
        fi
        ;;
    codex)
        echo ""
        echo "Codex CLI connects directly via config:"
        echo ""
        echo "  1. Add to ~/.codex/config.toml:"
        echo ""
        echo '     [model_providers.model-router]'
        echo '     name = "Model Router"'
        echo '     base_url = "http://localhost:8001/v1"'
        echo '     wire_api = "chat"'
        echo ""
        echo "  2. Launch Codex:"
        echo "     codex --provider model-router"
        echo ""
        echo "  3. Verify: send a message and check gateway logs for the request"
        echo ""
        info "Checking codex..."
        if command -v codex &> /dev/null; then
            pass
        else
            warn
            echo "    codex not found in PATH. Install it first."
        fi
        ;;
    letta)
        echo ""
        echo "Letta Code connects via self-hosted server:"
        echo ""
        echo "  1. Start self-hosted Letta server (Docker):"
        echo ""
        echo "     docker run -d \\"
        echo "       -p 8283:8283 \\"
        echo "       -e OPENAI_API_BASE=http://host.docker.internal:8001/v1 \\"
        echo "       -e OPENAI_API_KEY=sk-placeholder \\"
        echo "       --name letta-server \\"
        echo "       letta/letta:latest"
        echo ""
        echo "  2. Launch Letta CLI pointing at self-hosted server:"
        echo "     LETTA_BASE_URL=http://localhost:8283 letta ..."
        echo ""
        echo "  3. Verify: send a message and check gateway logs for the request"
        echo ""
        echo "  NOTE: OPENAI_API_BASE is for the Letta server's inference calls,"
        echo "  not the Letta API protocol. The gateway only needs to handle"
        echo "  /v1/chat/completions with tool-calling support."
        echo ""
        info "Checking docker..."
        if command -v docker &> /dev/null; then
            pass
        else
            warn
            echo "    docker not found. Install Docker first."
        fi
        ;;
    *)
        echo "Unknown client: ${CLIENT}"
        echo "Supported: claude-code, codex, letta"
        exit 1
        ;;
esac

# ---- automated checks ----

header "==== Automated Checks (7 checks, ~15 min) ===="
echo ""
echo "Run each check through the client. Record PASS/FAIL in the log."
echo ""

echo -e "${BOLD}Check 1: Basic Routing${RESET}"
echo "  Prompt:  What is 2+2?"
echo "  Expect:  Correct answer (4), gateway logs the request"
echo "  Verify:  Check gateway log shows the request with model and latency"
echo ""

echo -e "${BOLD}Check 2: Streaming${RESET}"
echo "  Prompt:  Explain what git rebase does in 3 sentences."
echo "  Expect:  Tokens stream in real-time, no buffering or stalling"
echo "  Verify:  Response appears incrementally, not all at once"
echo ""

echo -e "${BOLD}Check 3: Model Switching${RESET}"
echo "  Prompt A:  What is 2+2? (should route to local default)"
echo "  Prompt B:  What is 2+2? (force frontier via client config or gateway ?route= param)"
echo "  Expect:  Different models respond — check gateway logs for model key"
echo "  Verify:  Gateway log shows different model_key for A vs B"
echo ""

echo -e "${BOLD}Check 4: Tool Use — File Read${RESET}"
echo "  Prompt:  Read the file main.py in this repo and tell me what it does in one sentence."
echo "  Expect:  Client invokes a file-read tool, returns file content summary"
echo "  Verify:  Tool call appears in client UI, response references actual file content"
echo ""

echo -e "${BOLD}Check 5: Tool Use — Shell Command${RESET}"
echo "  Prompt:  Run \`git log --oneline -5\` and summarize the recent commits."
echo "  Expect:  Client invokes a bash/shell tool, returns command output"
echo "  Verify:  Tool call appears in client UI, response references actual git log"
echo ""

echo -e "${BOLD}Check 6: Multi-Turn${RESET}"
echo "  Turn 1:  Write a Python function that reverses a string."
echo "  Turn 2:  Now add type hints and a docstring."
echo "  Expect:  Second response modifies the first function, not a new one"
echo "  Verify:  Response includes the original function with additions, not a rewrite"
echo ""

echo -e "${BOLD}Check 7: Error Recovery${RESET}"
echo "  Prompt:  Use a nonexistent model or force a bad route (e.g., ?route=nonexistent-model)"
echo "  Expect:  Client shows an error message, does not crash or hang"
echo "  Verify:  Client remains usable after the error"
echo ""

# ---- extended-use prompts ----

header "==== Extended-Use Prompts (10 tasks, ~1 day) ===="
echo ""
echo "These are real coding tasks in the model-router repo."
echo "Use them during a full day of actual work through the client."
echo "Record any issues (breaks, stalls, wrong tool calls, etc.) in the log."
echo ""

echo -e "${BOLD}Task 1: Code Generation${RESET}"
echo "  Add a /v1/responses endpoint to the gateway that proxies to the same"
echo "  handler as /v1/chat/completions. Include it in the OpenAPI spec."
echo ""

echo -e "${BOLD}Task 2: Code Editing${RESET}"
echo "  Refactor the _stream_openrouter function to also handle the tools and"
echo "  tool_choice fields from the request body, passing them through to the"
echo "  upstream API."
echo ""

echo -e "${BOLD}Task 3: Debugging${RESET}"
echo "  I'm getting a 429 from the gateway when I send requests too fast."
echo "  Find the rate limiting code and add a retry-after header to the response."
echo ""

echo -e "${BOLD}Task 4: Multi-File${RESET}"
echo "  Add a new provider 'azure' that calls Azure OpenAI endpoints. Create"
echo "  the provider handler, add config entries, and wire it into the routing"
echo "  logic."
echo ""

echo -e "${BOLD}Task 5: Search + Edit${RESET}"
echo "  Find all places in the codebase where we hardcode max_tokens: 1024"
echo "  and make it configurable per-model in config.yaml."
echo ""

echo -e "${BOLD}Task 6: Test Writing${RESET}"
echo "  Write a test for choose_model() that verifies the heuristic routes"
echo "  short prompts local and long prompts to the default. Put it in"
echo "  tests/test_routing.py."
echo ""

echo -e "${BOLD}Task 7: Documentation${RESET}"
echo "  Update the README to document the Google Gemini provider, including"
echo "  the environment variable needed and an example config entry."
echo ""

echo -e "${BOLD}Task 8: Complex Tool Use${RESET}"
echo "  Run the benchmark against qwen2.5-coder:7b with only the"
echo "  code-generation prompts, then show me the results and tell me which"
echo "  prompts scored below 5."
echo ""

echo -e "${BOLD}Task 9: Error Scenario${RESET}"
echo "  What happens if I configure a model in config.yaml but the upstream"
echo "  API key is missing? Trace the error path and add a better error message."
echo ""

echo -e "${BOLD}Task 10: Architecture${RESET}"
echo "  Design a classifier that replaces the stub heuristic. It should look"
echo "  at prompt length, keyword presence, and estimated complexity. Write"
echo "  the design as a docstring in main.py — don't implement it yet."
echo ""

# ---- generate log template ----

header "==== Generating Log Template ===="

mkdir -p "${REPO_ROOT}/benchmark/results"

cat > "$LOG_FILE" <<TEMPLATE
# Integration Test: ${CLIENT}

**Date:** $(date +%Y-%m-%d)
**Client:** ${CLIENT}
**Gateway:** ${GATEWAY_URL}
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
TEMPLATE

echo "  Log template written to: ${LOG_FILE}"
echo ""
echo "Fill in the log as you run through the checks and extended-use tasks."
echo ""

# ---- done ----

header "==== Ready ===="
echo ""
echo "1. Make sure the gateway is running at ${GATEWAY_URL}"
echo "2. Configure and launch the ${CLIENT} client (see setup above)"
echo "3. Run the 7 automated checks (~15 min)"
echo "4. Use the 10 extended-use prompts during a day of real work"
echo "5. Record results in: ${LOG_FILE}"
echo ""
