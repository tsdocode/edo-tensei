#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EDO_DEMO_NAME=stateful-process EDO_DEMO_STARTUP_SECONDS="${EDO_DEMO_STARTUP_SECONDS:-4}" \
  exec "$repo_root/examples/00_hello_checkpoint/run.sh" "$@"
