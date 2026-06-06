#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

dotenv_value() {
  local key="$1"
  local env_file="$REPO_ROOT/.env"
  if [[ ! -f "$env_file" ]]; then
    return 1
  fi
  grep -E "^${key}=" "$env_file" | tail -n1 | cut -d= -f2-
}

port_pids() {
  local port="$1"
  ss -ltnp "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]\+' | cut -d= -f2 | sort -u
}

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"
DATA_PATH="${DATA_PATH:-/media/ubuntu/E/locomo/data/locomo10.json}"
SAMPLE_ID="${SAMPLE_ID:-conv-26}"
METHODS="${METHODS:-hybrid}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-50}"
POST_FLUSH_WAIT="${POST_FLUSH_WAIT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-900}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
EVAL_OWNER="${EVAL_OWNER:-speaker_a}"
CONCURRENCY="${CONCURRENCY:-1}"
SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-4}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-256}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-96}"
OLLAMA_REPEAT_PENALTY="${OLLAMA_REPEAT_PENALTY:-1.1}"
QA_LOG="${QA_LOG:-true}"
KEEP_SERVER="${KEEP_SERVER:-false}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-$LLM_TIMEOUT_SECONDS}"
SESSION_LOCK_TIMEOUT_SECONDS="${SESSION_LOCK_TIMEOUT_SECONDS:-900}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "❌ python not found at $PYTHON_BIN"
  echo "   run: uv sync"
  exit 1
fi

llm_model_default="$(dotenv_value 'EVEROS_LLM__MODEL' || true)"
llm_base_url_default="$(dotenv_value 'EVEROS_LLM__BASE_URL' || true)"
llm_api_key_default="$(dotenv_value 'EVEROS_LLM__API_KEY' || true)"
llm_temperature_default="$(dotenv_value 'EVEROS_LLM__TEMPERATURE' || true)"
embedding_model_default="$(dotenv_value 'EVEROS_EMBEDDING__MODEL' || true)"
embedding_base_url_default="$(dotenv_value 'EVEROS_EMBEDDING__BASE_URL' || true)"
embedding_api_key_default="$(dotenv_value 'EVEROS_EMBEDDING__API_KEY' || true)"

# LoCoMo runner uses its own benchmark model default instead of inheriting the
# general-purpose app model from .env.
LLM_MODEL="${LLM_MODEL:-qwen3.6:35b}"
LLM_BASE_URL="${LLM_BASE_URL:-${llm_base_url_default:-http://127.0.0.1:11434/v1}}"
LLM_API_KEY="${LLM_API_KEY:-${llm_api_key_default:-ollama}}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-${llm_temperature_default:-0.6}}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-${embedding_model_default:-qwen3-embedding:8b}}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-${embedding_base_url_default:-http://127.0.0.1:11434/v1}}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-${embedding_api_key_default:-ollama}}"

ANSWER_MODEL="${ANSWER_MODEL:-$LLM_MODEL}"
ANSWER_BASE_URL="${ANSWER_BASE_URL:-$LLM_BASE_URL}"
ANSWER_API_KEY="${ANSWER_API_KEY:-$LLM_API_KEY}"
JUDGE_MODEL="${JUDGE_MODEL:-$LLM_MODEL}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$LLM_BASE_URL}"
JUDGE_API_KEY="${JUDGE_API_KEY:-$LLM_API_KEY}"

SERVER_LLM_MODEL="${SERVER_LLM_MODEL:-$ANSWER_MODEL}"
SERVER_LLM_BASE_URL="${SERVER_LLM_BASE_URL:-$ANSWER_BASE_URL}"
SERVER_LLM_API_KEY="${SERVER_LLM_API_KEY:-$ANSWER_API_KEY}"
SERVER_LLM_TEMPERATURE="${SERVER_LLM_TEMPERATURE:-$LLM_TEMPERATURE}"
SERVER_EMBEDDING_MODEL="${SERVER_EMBEDDING_MODEL:-$EMBEDDING_MODEL}"
SERVER_EMBEDDING_BASE_URL="${SERVER_EMBEDDING_BASE_URL:-$EMBEDDING_BASE_URL}"
SERVER_EMBEDDING_API_KEY="${SERVER_EMBEDDING_API_KEY:-$EMBEDDING_API_KEY}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/benchmark_results/locomo_conv26_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints"
OUTPUT_JSON="$OUTPUT_ROOT/result.json"
RUN_LOG="$OUTPUT_ROOT/benchmark.log"
SERVER_LOG="$OUTPUT_ROOT/server.log"
SERVER_PID_FILE="$OUTPUT_ROOT/server.pid"
CORPUS_PATH="${CORPUS_PATH:-$OUTPUT_ROOT/memory}"
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_DIR"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "${KEEP_SERVER}" != "true" && -f "$SERVER_PID_FILE" ]]; then
    local pid
    pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo
      echo "→ stopping everos server pid=$pid"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  fi
  exit "$exit_code"
}

stop_existing_server() {
  local pids
  pids="$(port_pids "$SERVER_PORT" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "→ stopping existing listeners on :$SERVER_PORT ($pids)"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    if [[ -z "$(port_pids "$SERVER_PORT" || true)" ]]; then
      return 0
    fi
    sleep 1
  done

  pids="$(port_pids "$SERVER_PORT" || true)"
  if [[ -n "$pids" ]]; then
    echo "→ forcing listeners on :$SERVER_PORT down ($pids)"
    kill -9 $pids 2>/dev/null || true
  fi
}

check_ollama() {
  local model_url="${SERVER_LLM_BASE_URL%/}/models"
  if ! curl -fsS -o /dev/null "$model_url" 2>/dev/null; then
    echo "❌ Ollama OpenAI endpoint is not responding: $model_url"
    exit 1
  fi
}

start_server() {
  echo "→ starting everos server on $BASE_URL"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH=src
    export EVEROS_LLM__MODEL="$SERVER_LLM_MODEL"
    export EVEROS_LLM__API_KEY="$SERVER_LLM_API_KEY"
    export EVEROS_LLM__BASE_URL="$SERVER_LLM_BASE_URL"
    export EVEROS_LLM__TEMPERATURE="$SERVER_LLM_TEMPERATURE"
    export EVEROS_LLM__TIMEOUT_SECONDS="$SERVER_TIMEOUT_SECONDS"
    export EVEROS_EMBEDDING__MODEL="$SERVER_EMBEDDING_MODEL"
    export EVEROS_EMBEDDING__API_KEY="$SERVER_EMBEDDING_API_KEY"
    export EVEROS_EMBEDDING__BASE_URL="$SERVER_EMBEDDING_BASE_URL"
    export EVEROS_MEMORY__ROOT="$CORPUS_PATH"
    export EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS="$SESSION_LOCK_TIMEOUT_SECONDS"
    export EVEROS_LOG_LEVEL="${EVEROS_LOG_LEVEL:-INFO}"
    export EVEROS_LOG_FORMAT="${EVEROS_LOG_FORMAT:-text}"
    exec "$PYTHON_BIN" -m everos.entrypoints.cli.main server start --host "$SERVER_HOST" --port "$SERVER_PORT"
  ) >"$SERVER_LOG" 2>&1 &

  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$SERVER_PID_FILE"

  for _ in $(seq 1 120); do
    if curl -fsS -o /dev/null "$BASE_URL/health" 2>/dev/null; then
      echo "✓ server healthy: $BASE_URL (pid=$SERVER_PID)"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "❌ everos exited before becoming healthy"
      tail -n 40 "$SERVER_LOG" || true
      return 1
    fi
    sleep 1
  done

  echo "❌ everos did not become healthy in time"
  tail -n 40 "$SERVER_LOG" || true
  return 1
}

trap cleanup EXIT INT TERM

check_ollama
stop_existing_server
start_server

CMD=(
  "$PYTHON_BIN"
  tests/test_locomo.py
  --base-url "$BASE_URL"
  --request-timeout "$REQUEST_TIMEOUT"
  --llm-timeout "$LLM_TIMEOUT_SECONDS"
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
  --answer-max-tokens "$ANSWER_MAX_TOKENS"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --ollama-repeat-penalty "$OLLAMA_REPEAT_PENALTY"
  --answer-model "$ANSWER_MODEL"
  --answer-base-url "$ANSWER_BASE_URL"
  --answer-api-key "$ANSWER_API_KEY"
  --judge-model "$JUDGE_MODEL"
  --judge-base-url "$JUDGE_BASE_URL"
  --judge-api-key "$JUDGE_API_KEY"
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
printf "  server_port       : %s\n" "$SERVER_PORT"
printf "  data_path         : %s\n" "$DATA_PATH"
printf "  sample_id         : %s\n" "$SAMPLE_ID"
printf "  methods           : %s\n" "$METHODS"
printf "  corpus_path       : %s\n" "$CORPUS_PATH"
printf "  request_timeout   : %s\n" "$REQUEST_TIMEOUT"
printf "  llm_timeout       : %s\n" "$LLM_TIMEOUT_SECONDS"
printf "  answer_max_tokens : %s\n" "$ANSWER_MAX_TOKENS"
printf "  judge_max_tokens  : %s\n" "$JUDGE_MAX_TOKENS"
printf "  repeat_penalty    : %s\n" "$OLLAMA_REPEAT_PENALTY"
printf "  server_timeout    : %s\n" "$SERVER_TIMEOUT_SECONDS"
printf "  server_llm_model  : %s\n" "$SERVER_LLM_MODEL"
printf "  server_llm_temp   : %s\n" "$SERVER_LLM_TEMPERATURE"
printf "  answer_model      : %s\n" "$ANSWER_MODEL"
printf "  judge_model       : %s\n" "$JUDGE_MODEL"
printf "  llm_base_url      : %s\n" "$SERVER_LLM_BASE_URL"
printf "  embedding_model   : %s\n" "$SERVER_EMBEDDING_MODEL"
printf "  output_root       : %s\n" "$OUTPUT_ROOT"
printf "  benchmark_log     : %s\n" "$RUN_LOG"
printf "  server_log        : %s\n" "$SERVER_LOG"
echo

cd "$REPO_ROOT"
set +e
PYTHONPATH=src "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
benchmark_status=${PIPESTATUS[0]}
set -e

echo
if [[ "$benchmark_status" -eq 0 ]]; then
  echo "✓ LoCoMo run finished"
else
  echo "❌ LoCoMo run failed with exit code $benchmark_status"
  echo "   benchmark log: $RUN_LOG"
  echo "   server log   : $SERVER_LOG"
fi
echo "   result json  : $OUTPUT_JSON"
echo "   checkpoints  : $CHECKPOINT_DIR"

exit "$benchmark_status"
