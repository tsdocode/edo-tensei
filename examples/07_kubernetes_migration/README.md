# 07 — Kubernetes migration

Use the node-local snapshot agent to restore a GPU workload into a same-node placeholder Pod.

## Prerequisites

k3s or Kubernetes, NVIDIA device plugin, privileged Pod permissions, and a GPU node.

## Run

```bash
kubectl apply -f ../../kubernetes/namespace-rbac.yaml -f ../../kubernetes/gpu-smoke-test.yaml
```

For the validated vLLM path, follow [the Kubernetes guide](../../kubernetes/README.md).

## Expected output

The destination Pod becomes ready and serves after the snapshot agent restores the process and CUDA state.

## What is checkpointed?

A node-local process/container boundary plus CUDA state and the application checkpoint image.

## Limitations

The current path is same-node and placeholder-Pod based; controller-managed discovery and cross-node storage remain future work.

## Cleanup

```bash
kubectl delete namespace edo-system --ignore-not-found
```

This is the final progressive example; see [architecture](../../docs/concepts/architecture.md) for internals.
