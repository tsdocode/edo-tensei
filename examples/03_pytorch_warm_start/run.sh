#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/examples/modal-gpu-snapshot/run-qwen-demo.sh" "$@"
