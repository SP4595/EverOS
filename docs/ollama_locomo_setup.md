# Ollama + LoCoMo (`conv-26`) Setup

This note captures the local shell and benchmark settings that worked for running EverOS against Ollama and evaluating LoCoMo `conv-26`.

## 1. Add Ollama shell defaults to `~/.bashrc`

Add this block once:

```bash
# ollama settings
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=4
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=32768
```

Reload the shell:

```bash
source ~/.bashrc
```

Verify:

```bash
env | grep '^OLLAMA_'
```

Note: on this machine the block already exists in `~/.bashrc`, so do not add a duplicate copy.

## 2. EverOS `.env` values for Ollama

Use Ollama's OpenAI-compatible endpoint for both chat and embeddings:

```bash
EVEROS_LLM__MODEL=qwen3.6:35b
EVEROS_LLM__API_KEY=ollama
EVEROS_LLM__BASE_URL=http://127.0.0.1:11434/v1
EVEROS_LLM__TIMEOUT_SECONDS=300

EVEROS_EMBEDDING__MODEL=qwen3-embedding:8b
EVEROS_EMBEDDING__API_KEY=ollama
EVEROS_EMBEDDING__BASE_URL=http://127.0.0.1:11434/v1

EVEROS_MEMORY__ROOT=/home/ubuntu/Desktop/EverMemOS/.everos-local
EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS=900
```

The important additions for slow local models are:

- `EVEROS_LLM__TIMEOUT_SECONDS=300`
- `EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS=900`

The first prevents Ollama-backed chat completions from being cut off after 60 seconds. The second gives one `add` or `flush` request enough wall-clock budget to finish the full memory pipeline.

## 3. Start EverOS

Run the server in one terminal:

```bash
cd /home/ubuntu/Desktop/EverMemOS
PYTHONPATH=src .venv/bin/python -m everos.entrypoints.cli.main server start --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

## 4. Run LoCoMo `conv-26`

Use the dedicated wrapper:

```bash
cd /home/ubuntu/Desktop/EverMemOS
bash tests/run_locomo_conv26_ollama.sh
```

Useful overrides:

```bash
BASE_URL=http://127.0.0.1:8001 bash tests/run_locomo_conv26_ollama.sh
DATA_PATH=/media/ubuntu/E/locomo/data/locomo10.json bash tests/run_locomo_conv26_ollama.sh
CORPUS_PATH=/home/ubuntu/Desktop/EverMemOS/.everos-local-conv26 bash tests/run_locomo_conv26_ollama.sh
METHODS=hybrid REQUEST_TIMEOUT=900 bash tests/run_locomo_conv26_ollama.sh
```

## 5. What the wrapper does

The wrapper in [tests/run_locomo_conv26_ollama.sh](../tests/run_locomo_conv26_ollama.sh) runs:

- LoCoMo sample id `conv-26`
- `hybrid` retrieval by default
- per-QA logging with running `f1`, `bleu1`, and `judge`
- answer + judge model = `qwen3.6:35b`
- benchmark HTTP request timeout = `900` seconds
- corpus polling after flush using `--corpus-path`

Outputs land under `benchmark_results/locomo_conv26_<timestamp>/`.

## 6. Dataset location used here

This setup assumes the LoCoMo subset repo is available at:

```bash
/media/ubuntu/E/locomo/data/locomo10.json
```

In that file, `conv-26` is the first sample's `sample_id`, not array index 26. The benchmark driver now supports `--sample-id conv-26` directly.
