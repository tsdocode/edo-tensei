#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kubectl apply -f "$repo_root/kubernetes/namespace-rbac.yaml" \
  -f "$repo_root/kubernetes/gpu-smoke-test.yaml"
