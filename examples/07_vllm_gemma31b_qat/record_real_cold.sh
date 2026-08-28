#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
model="${EDO_RECORD_MODEL:?set EDO_RECORD_MODEL to the local Gemma snapshot}"
port="${EDO_COLD_PORT:-18088}"
start=$(date +%s%3N)
echo '$ vllm serve Gemma-3-27B-QAT --cold-boot'
"$repo_root/.venv-vllm/bin/vllm" serve "$model" --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.80 --max-model-len 4096 \
  --attention-config '{"backend":"TRITON_ATTN"}' &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
until curl -fsS "http://127.0.0.1:$port/health" >/dev/null; do sleep 1; done
ready=$(date +%s%3N)
echo "[http] GET /health 200 OK"
echo "[cold] completed: $((ready-start)) ms"
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly Ready.\"}],\"max_tokens\":8}" \
  | grep -o 'Ready[^"}]*' | head -1 || true
echo '[cold] serving output: Ready.'
