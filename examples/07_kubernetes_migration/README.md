# 07 · Kubernetes migration

Restore a GPU workload into a same-node placeholder Pod using the node-local Edo snapshot agent.

## Run

Prerequisites: k3s/Kubernetes, `kubectl`, a GPU node, NVIDIA device plugin, privileged workload permissions, and the manifests in `kubernetes/`.

```bash
./examples/07_kubernetes_migration/run.sh
```

For the validated vLLM flow, follow [the Kubernetes integration guide](../../docs/integrations/kubernetes.md) and [the full operations README](../../kubernetes/README.md).

## Architecture, step by step

```text
source Pod + GPU
      ↓ node-local snapshot agent enters runtime namespaces
      ↓ CUDA checkpoint + CRIU image + endpoint metadata
      ↓ source failure / destination placeholder Pod
      ↓ namespace join + CRIU restore + CUDA restore
destination Pod → readiness → real inference request
```

1. Kubernetes schedules a GPU workload and the device plugin exposes the NVIDIA device.
2. The privileged node-local agent discovers the host-visible process and coordinates the application snapshot.
3. The source Pod is quiesced; CRIU captures the process/container state and Edo records network endpoint metadata.
4. A placeholder Pod supplies PID, mount, network, and UTS namespaces for restore.
5. Edo remaps the old Pod IP to the destination Pod IP, restores CUDA, waits for readiness, and verifies inference.

## Result

Validated same-node Qwen3-0.6B CNI restore:

| Metric | Source | Destination after restore |
| --- | --- | --- |
| Pod readiness | healthy | healthy |
| Endpoint | `10.42.0.42` | `10.42.0.44` remapped |
| Snapshot size | 9.4 GiB | restored |
| End-to-end restore | — | 6.5 s |
| Inference | valid | valid, 41 ms |
| Model reload | initial startup only | no |

The validated path is same-node and placeholder-Pod based; the script above is a smoke manifest installer, not a destructive production migration command.

## What is checkpointed?

The process/container boundary, CUDA state, io_uring images, network endpoint metadata, and application memory required by the tested workload. Compile/model caches are mounted as shared rebuildable resources.

## Limitations and cleanup

Controller-managed PID discovery, persistent checkpoint storage, cross-node transfer, and node-drain orchestration remain future work. Remove the smoke resources with:

```bash
kubectl delete namespace edo-system --ignore-not-found
```

This is the final progressive example; return to [the README](../../README.md) or inspect the [architecture](../../docs/concepts/architecture.md).
