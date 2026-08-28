#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kubectl apply -f "$repo_root/examples/08_kubernetes_migration/namespace-rbac.yaml" \
  -f "$repo_root/examples/08_kubernetes_migration/gpu-smoke-test.yaml"
