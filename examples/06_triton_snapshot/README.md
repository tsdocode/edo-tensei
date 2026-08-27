# 06 · Triton snapshot

Run an official Triton container with a GPU-resident Python-backend model and compare server readiness/inference before and after the container-aware snapshot work.

## Run

Prerequisites: Docker, NVIDIA Container Toolkit, one NVIDIA GPU, access to `nvcr.io`, and the Triton 25.05 image.

```bash
./examples/06_triton_snapshot/run.sh
```

The model repository is mounted automatically from this directory. Triton serves readiness at `http://127.0.0.1:8000/v2/health/ready` and the warmup model at `/v2/models/gpu_warmup/infer`.

## Architecture, step by step

```text
Docker → Triton server → Python backend → CuPy GPU weights
              ↓ readiness + inference
       future: Edo/CRIU dump → restore placeholder → CUDA restore
```

1. Triton loads `gpu_warmup` from the local model repository.
2. The Python backend allocates GPU weights and performs 32 warm-up iterations.
3. Readiness and inference establish the before state.
4. Container-aware Edo integration is the next restore boundary because Docker mount and namespace ownership must be recreated before CUDA restore.

## Result

| Check | Before snapshot | After restore |
| --- | --- | --- |
| Triton `/v2/health/ready` | validated | pending native container restore |
| GPU warmup inference | validated | pending native container restore |
| CUDA state | allocated in backend | pending |
| Current status | snapshot creation validated | restore is an open limitation |

This README intentionally does not claim a restore latency or post-restore result until the Docker mount-namespace blocker is resolved.

## What is checkpointed?

The intended boundary is the Triton server plus Python backend process and CUDA state. The current demo validates startup and inference, not the final restore path.

## Limitations and cleanup

Native Docker namespace restoration is not complete. Stop the container with `Ctrl-C` or:

```bash
docker ps --filter ancestor=nvcr.io/nvidia/tritonserver:25.05-py3 --format '{{.ID}}' | xargs -r docker stop
```

Next: [07 · Kubernetes migration](../07_kubernetes_migration/README.md).
