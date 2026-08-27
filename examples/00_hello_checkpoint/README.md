# 00 — Hello checkpoint

Checkpoint a live CPU workload, let CRIU remove it, restore it, and send a request to the restored process.

## Prerequisites

- Linux
- Rust toolchain
- CRIU installed and usable through `sudo`

## Run

From the repository root:

```bash
./examples/00_hello_checkpoint/run.sh
```

The same demo is available through the Rust example:

```bash
cargo run --example resume
```

## Expected output

```text
Edo Tensei — Resume a warm workload
✓ Warm-up completed
✓ Checkpoint created
✓ Original process disappeared
✓ Restored successfully
✓ Request completed after restore
Cold start:   measured
Restore time: measured
Run report:   .edo/runs/...
```

The exact timings are measured on each machine. The demo stores a JSON report in `.edo/runs/`.

## What is checkpointed?

The Python process, its counter, signal handlers, open report file, working directory, and CRIU process image are checkpointed. No model or GPU is required.

## Limitations

This is a local CPU proof. It requires CRIU permissions, does not migrate between hosts, and does not preserve an in-flight request.

## Cleanup

The script removes its temporary checkpoint and terminates the restored process before exiting. Reports remain under `.edo/runs/`.

Next: [01 — Stateful process](../01_stateful_process/README.md).
