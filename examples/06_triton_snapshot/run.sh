#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec docker run --rm --gpus all --ipc=host --network=host \
  -v "$repo_root/examples/06_triton_snapshot:/models" \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
