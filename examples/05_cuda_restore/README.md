# 05 — Restore CUDA state

Exercise the native CUDA Driver checkpoint API and verify a deterministic GPU allocation after restore.

## Prerequisites

Linux, NVIDIA driver with checkpoint symbols, CUDA toolkit, CRIU, and `source env.sh`.

## Run

```bash
./run.sh
```

## Expected output

CUDA transitions through `RUNNING → LOCKED → CHECKPOINTED → RESTORED` and the checksum remains identical.

## What is checkpointed?

CUDA context and device allocation, coordinated with the native process checkpoint.

## Limitations

Single GPU and same compatible host; this does not test a framework or container.

## Cleanup

The fixture and temporary checkpoint are removed by the script.

Next: [06 — Triton snapshot](../06_triton_snapshot/README.md).
