# 01 — Stateful process

The hello demo checkpointed a live process; this version makes the state transition visible through a persistent event stream.

## Prerequisites

Linux, Rust, Python 3, and CRIU with the permissions described by `edo doctor`.

## Run

```bash
../00_hello_checkpoint/run.sh
```

## Expected output

The report contains warmup and pre/post-restore request events, and the second request is handled by the restored PID.

## What is checkpointed?

Python memory, signal handlers, open files, and the process identity are restored by CRIU.

## Limitations

The example is single-host and CPU-only; active requests are intentionally drained before checkpointing.

## Cleanup

The linked script removes its temporary checkpoint. Reports remain in `.edo/runs/`.

Next: [02 — FastAPI resume](../02_fastapi_resume/README.md).
