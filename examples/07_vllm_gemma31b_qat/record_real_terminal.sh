#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

echo "=== REAL vLLM COLD BOOT -> CRIU RESTORE ==="
echo "=== cold boot, snapshot, restore, and serving probe are real ==="

export EDO_CRIU="${EDO_CRIU:-/usr/local/sbin/criu}"
export EDO_BIN="${EDO_BIN:-$repo_root/target/debug/edo}"
export EDO_VLLM_COMMAND="${EDO_VLLM_COMMAND:-$repo_root/.venv-vllm/bin/vllm}"
export EDO_SHOW_STARTUP_PROGRESS=1

"$repo_root/.venv-vllm/bin/python" examples/04_vllm_resume/vllm_adapter.py \
  --model "${EDO_RECORD_MODEL:-gaunernst/gemma-3-27b-it-qat-compressed-tensors}" \
  --vllm-command "$EDO_VLLM_COMMAND" \
  --edo "$EDO_BIN" \
  --host 127.0.0.1 \
  --port "${EDO_RECORD_PORT:-18087}" \
  --gpu-memory-utilization "${EDO_RECORD_GPU_UTILIZATION:-0.80}" \
  --max-model-len "${EDO_RECORD_MAX_MODEL_LEN:-4096}" \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  --release-kv-cache \
  --fast-restore \
  --io-uring-restore \
  --full-snapshot

echo "=== REAL RUN COMPLETE ==="
