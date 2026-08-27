# 02 — Resume a FastAPI service

This example reuses the existing heavy-startup FastAPI workload with readiness withdrawal and request draining.

## Prerequisites

Python dependencies from `examples/fastapi-heavy-startup/README.md`, Linux, and CRIU.

## Run

```bash
./run.sh
```

## Expected output

The server becomes ready, is checkpointed while quiesced, then returns to `/ready` and serves after restore. Timings are saved in `.edo/runs/`.

## What is checkpointed?

The Python server process, loaded CPU model state, readiness state, and local listening socket.

## Limitations

This is a CPU/local-loopback demo. Requests in flight and external supervisors are not preserved.

## Cleanup

The script terminates the restored server and removes its temporary checkpoint.

Next: [03 — PyTorch warm start](../03_pytorch_warm_start/README.md).
