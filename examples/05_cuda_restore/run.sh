#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/examples/cuda-checkpoint/run-combined-demo.sh" "$@"
