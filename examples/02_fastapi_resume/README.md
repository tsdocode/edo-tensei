# 02 · Resume a FastAPI service

Checkpoint a real HTTP service after expensive startup, withdraw readiness, restore it, and serve again without rebuilding the model.

## Run

Prerequisites: Linux, Python dependencies from [the FastAPI notes](../../docs/integrations/fastapi.md), Rust, CRIU, and `sudo` permission.

```bash
./examples/02_fastapi_resume/run.sh
```

Use a larger model or startup delay when benchmarking:

```bash
EDO_HF_MODEL=distilbert-base-uncased EDO_STARTUP_SECONDS=8 ./examples/02_fastapi_resume/run.sh
```

## Architecture, step by step

```text
client → FastAPI /ready, /infer
             ↓ lifespan startup: model load + warm-up
             ↓ POST /quiesce (readiness off, requests drained)
             ↓ CRIU dump
             ↓ CRIU restore
             ↓ POST /resume (readiness on)
client → FastAPI /ready, /infer
```

1. `server.py` loads a Hugging Face model during lifespan startup and exposes health, readiness, state, and inference endpoints.
2. The script records a pre-checkpoint state and inference response.
3. `/quiesce` rejects new work and waits for active requests to drain before CRIU captures the process.
4. CRIU restores the process and listening socket; `/health` is checked before `/resume` reopens traffic.
5. The post-restore model checksum and inference path are compared with the pre-checkpoint values.

## Result

| Signal | Before checkpoint | After restore |
| --- | --- | --- |
| `/health` | 200 | 200 |
| `/ready` | ready | ready after `/resume` |
| Model checksum | recorded | must match |
| Inference | succeeds | succeeds |
| Cold/restore timing | measured by run | measured by run |

The current script prints the before/after state and checksum. CPU model startup time depends on the selected model and cache; no illustrative latency is presented as a benchmark.

## What is checkpointed?

The Python server, loaded CPU model, readiness state, local socket, and process memory. External clients and active requests are not part of the snapshot.

## Limitations and cleanup

This is a local CPU/loopback integration. Use a dedicated port and do not checkpoint production traffic. The script terminates the restored server and removes temporary images.

Next: [03 · PyTorch warm start](../03_pytorch_warm_start/README.md).
