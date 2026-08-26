# Roadmap update — Phase 3 complete

Phase 3 is complete on the validated Ubuntu 24.04 / NVIDIA H100 host.

Completed roadmap items:

- CUDA 12.8-compatible Rust FFI for all six process checkpoint functions.
- Exact ABI layouts for CUDA checkpoint argument structs.
- Dynamic `libcuda.so` loading with descriptive CUDA error names.
- Bounded CUDA process-state polling.
- Lock → checkpoint → state verification.
- Restore → state verification → unlock.
- Native CUDA fixture with a context, device allocation, and deterministic data.
- `edo cuda-state <pid>` and `edo cuda-roundtrip <pid>` commands.

Verified state sequence:

```text
RUNNING
LOCKED
CHECKPOINTED
RESTORED and LOCKED
UNLOCKED and RUNNING
```

Validation passed:

- `cargo fmt --check`
- `cargo check`
- `cargo clippy -- -D warnings`
- `cargo test`

Next roadmap phase: integrate CUDA transitions with application quiescing and
the existing CRIU snapshot state machine, preserving CUDA-before-CRIU ordering.
