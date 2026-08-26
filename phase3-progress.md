# Phase 3 progress — CUDA checkpoint FFI

Status: COMPLETE on the validated host

Completed:

- Implemented CUDA 12.8-compatible dynamic FFI for all six process checkpoint functions.
- Added exact 64-byte Rust representations for CUDA checkpoint argument structs.
- Added process-state decoding: `RUNNING`, `LOCKED`, `CHECKPOINTED`, and `FAILED`.
- Added descriptive CUDA error-name reporting.
- Added bounded polling with timeout for state transitions.
- Added `edo cuda-state <pid>` for non-mutating state inspection.
- Added `edo cuda-roundtrip <pid>` for lock → checkpoint → restore → unlock.
- Added a native CUDA fixture that creates a CUDA context, allocates device
  memory, and writes a deterministic pattern.
- Verified on the H100 host with CUDA 12.8 / driver 570.211.01:

  ```text
  RUNNING
  LOCKED
  CHECKPOINTED
  RESTORED and LOCKED
  UNLOCKED and RUNNING
  ```

Validation:

- `cargo fmt --check` passed.
- `cargo check` passed.
- `cargo clippy -- -D warnings` passed.
- `cargo test` passed with 3 tests.

Next phase:

- Integrate CUDA state transitions with the existing CRIU snapshot state machine.
- Add application quiescing and CUDA-before-CRIU ordering.
- Verify GPU allocation contents after a combined CUDA + CRIU resurrection.
