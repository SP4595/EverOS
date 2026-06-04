#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
DATA_PATH="${DATA_PATH:-/media/ubuntu/E/locomo/data/locomo10.json}"
SAMPLE_ID="${SAMPLE_ID:-conv-26}"
METHODS="${METHODS:-hybrid}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-50}"
POST_FLUSH_WAIT="${POST_FLUSH_WAIT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
EVAL_OWNER="${EVAL_OWNER:-speaker_a}"
CONCURRENCY="${CONCURRENCY:-1}"
SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-4}"
QA_LOG="${QA_LOG:-true}"

if [[ -z "${CORPUS_PATH:-}" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]]; then
    env_root="$(grep -E '^EVEROS_MEMORY__ROOT=' "$REPO_ROOT/.env" | tail -n1 | cut -d= -f2- || true)"
    CORPUS_PATH="${env_root:-}"
  fi
fi
CORPUS_PATH="${CORPUS_PATH:-${EVEROS_MEMORY__ROOT:-$REPO_ROOT/.everos-local}}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "❌ python not found at $PYTHON_BIN"
  echo "   run: uv sync"
  exit 1
fi

if ! curl -fsS -o /dev/null "$BASE_URL/health" 2>/dev/null; then
  echo "❌ server at $BASE_URL is not responding"
  echo "   start with:"
  echo "   cd $REPO_ROOT"
  echo "   EVEROS_LLM__TIMEOUT_SECONDS=300 PYTHONPATH=src $PYTHON_BIN -m everos.entrypoints.cli.main server start --port 8000"
  exit 1
fi

echo "✓ server healthy: $BASE_URL"

if [[ -z "${LLM_MODEL:-}" || -z "${LLM_BASE_URL:-}" || -z "${LLM_API_KEY:-}" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source <(grep -E '^EVEROS_LLM__' "$REPO_ROOT/.env" | sed 's/^EVEROS_LLM__/LLM_/')
    set +a
  fi
fi

LLM_MODEL="${LLM_MODEL:-qwen3.6:35b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
LLM_API_KEY="${LLM_API_KEY:-ollama}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/benchmark_results/locomo_conv26_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints"
OUTPUT_JSON="$OUTPUT_ROOT/result.json"
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_DIR"

CMD=(
  "$PYTHON_BIN"
  tests/test_locomo.py
  --base-url "$BASE_URL"
  --request-timeout "$REQUEST_TIMEOUT"
  --data-path "$DATA_PATH"
  --sample-id "$SAMPLE_ID"
  --methods "$METHODS"
  --top-k "$TOP_K"
  --batch-size "$BATCH_SIZE"
  --post-flush-wait "$POST_FLUSH_WAIT"
  --corpus-path "$CORPUS_PATH"
  --judge-runs "$JUDGE_RUNS"
  --eval-owner "$EVAL_OWNER"
  --concurrency "$CONCURRENCY"
  --search-concurrency "$SEARCH_CONCURRENCY"
  --answer-model "$LLM_MODEL"
  --answer-base-url "$LLM_BASE_URL"
  --answer-api-key "$LLM_API_KEY"
  --judge-model "$LLM_MODEL"
  --judge-base-url "$LLM_BASE_URL"
  --judge-api-key "$LLM_API_KEY"
  --output "$OUTPUT_JSON"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --quiet
)

if [[ "$QA_LOG" == "true" ]]; then
  CMD+=(--qa-log)
fi

echo "═════════════════════════════════════════════════════════════════"
echo "  LoCoMo conv-26 via Ollama"
echo "═════════════════════════════════════════════════════════════════"
printf "  base_url          : %s\n" "$BASE_URL"
printf "  data_path         : %s\n" "$DATA_PATH"
printf "  sample_id         : %s\n" "$SAMPLE_ID"
printf "  methods           : %s\n" "$METHODS"
printf "  corpus_path       : %s\n" "$CORPUS_PATH"
printf "  request_timeout   : %s\n" "$REQUEST_TIMEOUT"
printf "  llm_model         : %s\n" "$LLM_MODEL"
printf "  llm_base_url      : %s\n" "$LLM_BASE_URL"
printf "  output_root       : %s\n" "$OUTPUT_ROOT"
echo

cd "$REPO_ROOT"
PYTHONPATH=src "${CMD[@]}"
