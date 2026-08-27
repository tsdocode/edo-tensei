#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

# Gemma's published QAT checkpoint is a 27B model (the 31B-class workload
# requested by this example). Keep the model and vLLM launcher configurable so
# the same harness can be used with a local mirror or a different QAT export.
model="${EDO_GEMMA_QAT_MODEL:-gaunernst/gemma-3-27b-it-qat-compressed-tensors}"
gpu_memory_utilization="${EDO_GEMMA_GPU_MEMORY_UTILIZATION:-0.90}"
max_model_len="${EDO_GEMMA_MAX_MODEL_LEN:-4096}"

exec python3 examples/04_vllm_resume/vllm_adapter.py \
  --model "$model" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-model-len "$max_model_len" \
  --full-snapshot \
  "$@"
