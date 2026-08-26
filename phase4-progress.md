# Phase 4 progress — CUDA + CRIU resurrection

Status: CORE COMPLETE; framework quiescence remains an integration contract

Implemented:

- `edo freeze <target> <snapshot>` performs CUDA lock and GPU checkpoint before CRIU dump.
- CUDA rollback is attempted if lock, checkpoint, CRIU dump, or manifest writing fails.
- CUDA snapshots use a distinct `kind: "cuda-criu"` manifest.
- `edo summon <snapshot>` performs CRIU restore first, then CUDA restore on the restored PID, followed by unlock.
- Bounded CUDA state polling is used at every transition.
- Native fixture verifies the GPU allocation checksum before and after restore.
- Repeatable demo: `examples/cuda-checkpoint/run-combined-demo.sh`.

Verified on the H100 host:

```text
CUDA RUNNING → LOCKED → CHECKPOINTED
CRIU dump
CRIU restore
CUDA RESTORE → RUNNING
GPU pattern verified before and after restore
```

Failure recovery verified: an intentionally invalid CRIU snapshot target
returned an error and restored the CUDA process to `RUNNING`.

Validation passed:

- `cargo fmt --check`
- `cargo check`
- `cargo clippy -- -D warnings`
- `cargo test`
- Combined CUDA+CRIU demo
- Failure-injection recovery test

Remaining integration contract:

- A framework adapter must stop accepting work and drain in-flight requests
  before calling `edo freeze`. Edo cannot infer application-level quiescence
  from an arbitrary PID.
