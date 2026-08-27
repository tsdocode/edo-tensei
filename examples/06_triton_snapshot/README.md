# 06 — Triton snapshot

Run the official Triton Python-backend GPU workload and inspect its snapshot boundary.

## Prerequisites

Docker, NVIDIA Container Toolkit, one GPU, and the Triton image.

## Run

```bash
./run.sh
```

If Docker namespace restoration is unavailable, use the direct command in the [legacy Triton guide](../../examples/triton-snapshot/README.md).

## Expected output

Triton reaches `/v2/health/ready` and responds to the warmup inference endpoint.

## What is checkpointed?

The Triton server and Python backend process, subject to container namespace compatibility.

## Limitations

Native Edo restore from a Docker mount namespace is not complete yet.

## Cleanup

```bash
docker ps --filter ancestor=nvcr.io/nvidia/tritonserver:25.05-py3 --format '{{.ID}}' | xargs -r docker stop
```

Next: [07 — Kubernetes migration](../07_kubernetes_migration/README.md).
