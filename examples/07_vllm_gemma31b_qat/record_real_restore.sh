#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
snapshot="${EDO_RECORD_SNAPSHOT:?set EDO_RECORD_SNAPSHOT to a completed Gemma snapshot}"
export EDO_CRIU="${EDO_CRIU:-/home/ubuntu/work/criu-vllm/criu/criu}"
export EDO_IO_URING_RESTORE=1
start=$(date +%s%3N)
echo '$ edo summon-group Gemma-checkpoint --skip-integrity'
sudo --preserve-env=EDO_CRIU,EDO_IO_URING_RESTORE "$repo_root/target/debug/edo" \
  summon-group "$snapshot" --skip-integrity
echo '[criu] process memory + io_uring restored'
curl -fsS -X POST 'http://127.0.0.1:18087/wake_up?tags=weights' >/dev/null
curl -fsS -X POST 'http://127.0.0.1:18087/wake_up?tags=kv_cache' >/dev/null
until curl -fsS http://127.0.0.1:18087/health >/dev/null; do sleep 1; done
ready=$(date +%s%3N)
echo '[http] GET /health 200 OK'
echo "[restore] completed: $((ready-start)) ms"
curl -fsS http://127.0.0.1:18087/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-3-27b-it-qat-compressed-tensors","messages":[{"role":"user","content":"Reply exactly Ready."}],"max_tokens":8}' \
  | grep -o 'Ready[^"}]*' | head -1 || true
echo '[restore] serving output: Ready.'
