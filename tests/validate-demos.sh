#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for directory in examples/00_hello_checkpoint examples/01_stateful_process \
    examples/02_fastapi_resume examples/03_pytorch_warm_start \
    examples/04_vllm_resume examples/05_cuda_restore \
    examples/07_vllm_gemma31b_qat \
    examples/06_triton_snapshot examples/08_kubernetes_migration; do
    test -f "$directory/README.md"
done

for script in examples/*/run.sh; do
    bash -n "$script"
done

cargo check --example resume
echo "Demo structure and shell entry points are valid."
