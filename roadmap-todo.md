# Edo Tensei Roadmap and TODO

This document turns the project plan into an execution roadmap. The priority is to prove the smallest complete resurrection path before adding framework integrations or persistent VRAM.

## North-star milestone

Restore one small Linux process that owns CUDA state, after:

1. the process is warmed up;
2. CUDA state is locked and checkpointed;
3. the process is destroyed from the user's perspective;
4. CRIU restores the CPU process;
5. CUDA restores the GPU state;
6. the process continues and produces a verified result.

The first implementation should target Linux x86_64, one NVIDIA GPU, one process, and a native CUDA test program. PyTorch and FastAPI come only after the raw CUDA path works.

## Priority and status

- **P0** — blocks the first working resurrection path.
- **P1** — required for a usable v0.1 release.
- **P2** — useful hardening or developer experience.
- **Future** — deliberately outside v0.1.

Status values used below: `TODO`, `IN PROGRESS`, `BLOCKED`, and `DONE`.

## Current progress

- **Phase 0 — DONE:** Rust crate, CLI scaffold, toolchain, CI workflow, CUDA environment, and CRIU environment are complete.
- Verified host: Ubuntu 24.04 x86_64 with NVIDIA H100 80GB, driver 570.211.01, CUDA 12.8, and exported CUDA checkpoint symbols.
- CRIU 4.2.1 is installed from source; `sudo criu check` passes with `Looks good.`
- Phase 0 verification passed: formatting, compilation, strict Clippy, tests, and the initial `edo doctor` command.
- **Phase 1 — DONE:** `edo doctor` performs CRIU, CUDA, process-permission, ptrace, namespace, and capability discovery with human and JSON output. The host still requires an explicit CRIU capability configuration for unprivileged dumps.
- **Phase 2 — DONE:** CPU counter and resource-rich fixtures survived CRIU dump, process disappearance, restore, continued execution, and resource validation.

## Phase 0 — Repository and development environment

Goal: make the project buildable and make unsupported environments fail clearly.

### TODO

- [x] **P0 / DONE** Create a single Rust binary crate named `edo`.
- [x] **P0 / DONE** Add `Cargo.toml`, `README.md`, `LICENSE`, and a basic CI workflow.
- [x] **P0 / DONE** Add Linux-only feature guards around CRIU and CUDA functionality.
- [x] **P0 / DONE** Define a typed error model for unsupported platform, missing binary, missing library, permission failure, incompatible snapshot, and invalid state transition.
- [x] **P1 / DONE** Add formatting, linting, and test commands to the README.
- [x] **P1 / DONE** Add a documented Linux test host in `phase0-environment.md`.
- [x] **P1 / DONE** Record the tested kernel/CRIU/NVIDIA driver/CUDA environment in `phase0-environment.md`.

### Exit criteria

- `cargo build` succeeds on Linux.
- `edo --help` works.
- `edo doctor` runs on any platform and explains missing capabilities without panicking.
- macOS and non-NVIDIA systems receive a clear unsupported-platform result.

## Phase 1 — Capability discovery and `edo doctor`

Goal: discover compatibility before attempting a destructive operation.

### TODO

- [x] **P0 / DONE** Detect OS and CPU architecture.
- [x] **P0 / DONE** Locate and execute `criu --version`.
- [x] **P0 / DONE** Detect `libcuda.so` and load it dynamically.
- [x] **P0 / DONE** Call CUDA initialization and query the driver version.
- [x] **P0 / DONE** Query GPU count, UUID, chip identity, total memory, and compute capability.
- [x] **P0 / DONE** Detect whether CUDA checkpoint symbols are available.
- [x] **P1 / DONE** Check process permissions, ptrace restrictions, namespaces, and required CRIU capabilities. `edo doctor` reports effective UID, ptrace scope, namespace readability, and the current CRIU capability result.
- [ ] **P1 / TODO** Check persistence mode / CUDA initialization prerequisites for restore.
- [x] **P1 / DONE** Emit both human-readable output and machine-readable JSON.
- [x] **P1 / DONE** Add `edo doctor --json` for automation.

### Exit criteria

- A supported host reports every required capability.
- An unsupported host identifies the exact missing prerequisite and remediation.
- Doctor output can be saved into a snapshot compatibility manifest.

## Phase 2 — CPU-only CRIU proof

Goal: prove that the process manager and CRIU wrapper work independently of CUDA.

### Test fixture

Create a small fixture process that:

- increments a counter;
- writes its PID and counter to a status file;
- keeps a file descriptor open;
- optionally listens on a local socket;
- records a monotonic timestamp after restore.

### TODO

- [x] **P0 / DONE** Implement process launch with a stable project-owned run directory.
- [x] **P0 / DONE** Implement process identity checks using `/proc`, executable path, start time, and command line—not PID alone.
- [x] **P0 / DONE** Implement `edo run --name <name> -- <command> ...`.
- [x] **P0 / DONE** Implement `edo cpu-dump <name-or-pid> <snapshot-dir>`.
- [x] **P0 / DONE** Implement `edo cpu-restore <snapshot-dir>`.
- [x] **P0 / DONE** Capture CRIU stdout, stderr, exit status, and image directory.
- [x] **P0 / DONE** Write a CPU snapshot manifest.
- [x] **P1 / DONE** Validate that the restored process is alive and has resumed the counter.
- [x] **P1 / DONE** Test open files, working directory, environment, signals, and local sockets with `cpu-resource-fixture.py`.
- [x] **P1 / DONE** Add cleanup for failed or partial CRIU dumps.

### Exit criteria

- A CPU counter survives dump, process termination, and restore.
- A failed dump never appears as a usable snapshot.
- Restore reports the restored PID and process identity.

## Phase 3 — Minimal CUDA FFI

Goal: expose only the CUDA checkpoint API required for the proof.

### Required API surface

- `cuInit`
- `cuCheckpointProcessGetState`
- `cuCheckpointProcessLock`
- `cuCheckpointProcessCheckpoint`
- `cuCheckpointProcessGetRestoreThreadId`
- `cuCheckpointProcessRestore`
- `cuCheckpointProcessUnlock`

### TODO

- [ ] **P0 / TODO** Define exact Rust FFI structs and ABI versions from the installed CUDA headers.
- [ ] **P0 / TODO** Dynamically load `libcuda.so` and report missing symbols precisely.
- [ ] **P0 / TODO** Map CUDA return codes into typed Rust errors.
- [ ] **P0 / TODO** Implement state polling with timeout and bounded retry.
- [ ] **P0 / TODO** Implement lock → checkpoint → state verification.
- [ ] **P0 / TODO** Implement restore → state verification → unlock.
- [ ] **P0 / TODO** Test lock timeout and rollback behavior.
- [ ] **P1 / TODO** Add a small native CUDA fixture that allocates memory, writes a known pattern, and performs a kernel operation.
- [ ] **P1 / TODO** Verify that CUDA restore is only attempted when the restored process is in the expected state.

### Exit criteria

- A native CUDA process can be locked and checkpointed.
- The process reaches the expected checkpoint state.
- A documented raw CUDA checkpoint/restore round trip succeeds without CRIU.

## Phase 4 — Combined CUDA + CRIU resurrection

Goal: prove the central product claim.

### Required freeze protocol

Before locking CUDA:

1. stop accepting new application work;
2. drain or reject in-flight requests;
3. synchronize relevant CUDA streams;
4. record the application-level quiescent state;
5. verify the target process identity;
6. acquire an exclusive snapshot lock.

### TODO

- [ ] **P0 / TODO** Implement the state machine: `RUNNING → QUIESCING → READY_TO_FREEZE → CUDA_LOCKED → CUDA_CHECKPOINTED → CRIU_DUMPING → SNAPSHOT_READY`.
- [ ] **P0 / TODO** Persist a state transition journal so interrupted operations are diagnosable.
- [ ] **P0 / TODO** Implement `edo freeze <name>` with the exact CUDA-before-CRIU ordering.
- [ ] **P0 / TODO** Ensure a failed CRIU dump leaves the CUDA process recoverable or clearly marks it unsafe.
- [ ] **P0 / TODO** Implement `edo summon <snapshot>` with explicit restore coordination.
- [ ] **P0 / TODO** Handle restored PID/TID discovery and CUDA restore-thread coordination.
- [ ] **P0 / TODO** Restore CUDA state before unlocking the process.
- [ ] **P0 / TODO** Verify the restored GPU allocation contents with a checksum or known output.
- [ ] **P0 / TODO** Verify the CPU counter and GPU result both continue from pre-freeze state.
- [ ] **P1 / TODO** Add a health-check command and configurable post-restore timeout.
- [ ] **P1 / TODO** Test failure at every state transition and document recovery behavior.

### Exit criteria

- A complete native CUDA process survives freeze, process disappearance, CRIU restore, CUDA restore, and resume.
- The result is verified independently on both CPU and GPU.
- A failed operation cannot silently produce a snapshot marked ready.

## Phase 5 — Snapshot format, compatibility, and security

Goal: make snapshots diagnosable, safe, and reject incompatible restores.

### Manifest requirements

- [ ] **P0 / TODO** Add schema version and Edo version.
- [ ] **P0 / TODO** Record source hostname and namespace/container context.
- [ ] **P0 / TODO** Record kernel version and architecture.
- [ ] **P0 / TODO** Record CRIU version and configuration.
- [ ] **P0 / TODO** Record NVIDIA driver and CUDA driver API versions.
- [ ] **P0 / TODO** Record GPU UUID, chip type, compute capability, and memory capacity.
- [ ] **P0 / TODO** Record process executable, arguments, working directory, environment policy, and process tree.
- [ ] **P0 / TODO** Record checkpoint state, timestamps, sizes, checksums, and restore requirements.
- [ ] **P1 / TODO** Define strict, compatible, and unsafe override modes.
- [ ] **P1 / TODO** Reject mismatched chip type, insufficient GPU memory, incompatible driver, and missing images by default.

### Security requirements

- [ ] **P0 / TODO** Restrict snapshot directory permissions.
- [ ] **P0 / TODO** Document that snapshots contain process memory and may contain secrets.
- [ ] **P1 / TODO** Add optional encryption at rest.
- [ ] **P1 / TODO** Add manifest and image integrity checks.
- [ ] **P1 / TODO** Add retention and secure cleanup commands.
- [ ] **P2 / TODO** Support an external key provider without storing keys in snapshots.

### Exit criteria

- An incompatible host is rejected before CRIU or CUDA restore begins.
- Snapshot integrity failures are detected.
- The security model is documented before public release.

## Phase 6 — PyTorch demonstration

Goal: demonstrate that a warmed Python/PyTorch process can resume without model construction.

### TODO

- [ ] **P0 / TODO** Build a small single-process PyTorch fixture with deterministic weights.
- [ ] **P0 / TODO** Warm it up and record initialization time, first inference time, and steady-state latency.
- [ ] **P0 / TODO** Freeze only when no inference is in flight.
- [ ] **P0 / TODO** Restore and prove that model initialization code did not run again.
- [ ] **P0 / TODO** Verify output checksum and model/device state after restore.
- [ ] **P1 / TODO** Test CUDA graphs only after ordinary eager execution succeeds.
- [ ] **P1 / TODO** Document unsupported Python resources and extension modules.

### Exit criteria

- The PyTorch process resumes and serves a valid inference.
- The demo is explicitly documented as single-process and single-GPU.
- The benchmark compares cold start, warm start, checkpoint time, restore time, and storage size.

## Phase 7 — FastAPI inference demo

Goal: prove the operational user experience without pretending in-flight requests are preserved.

### TODO

- [ ] **P1 / TODO** Add a small FastAPI server with `/health`, `/ready`, and `/infer`.
- [ ] **P1 / TODO** Stop readiness before freeze and restore readiness only after health validation.
- [ ] **P1 / TODO** Drain or reject requests during the quiesce window.
- [ ] **P1 / TODO** Define socket behavior for local and external clients.
- [ ] **P1 / TODO** Test process restart behind a simple supervisor.
- [ ] **P1 / TODO** Document that active requests are not preserved in v0.1.

### Exit criteria

- The server resumes with the warmed model and passes health checks.
- No request is falsely reported as successful across a freeze boundary.
- Cold-start and restore-start behavior are benchmarked.

## Phase 8 — v0.1 hardening and release

Goal: release a narrow, honest, reproducible tool.

### TODO

- [ ] **P1 / TODO** Add integration tests for every state transition.
- [ ] **P1 / TODO** Add tests for partial snapshots and interrupted restores.
- [ ] **P1 / TODO** Add tests for insufficient GPU memory and wrong GPU chip type.
- [ ] **P1 / TODO** Add tests for CRIU permission and namespace failures.
- [ ] **P1 / TODO** Add structured logs and timing metrics.
- [ ] **P1 / TODO** Publish a tested compatibility matrix.
- [ ] **P1 / TODO** Document operational recovery and cleanup procedures.
- [ ] **P1 / TODO** Add reproducible benchmark scripts.
- [ ] **P1 / TODO** Add release binaries only for tested Linux targets.
- [ ] **P2 / TODO** Add shell completion and better progress output.

### v0.1 release criteria

- Native CUDA resurrection works on the supported matrix.
- A PyTorch demo works on the same narrow matrix.
- FastAPI demo works with drained requests.
- Incompatible restores fail safely and clearly.
- Snapshot permissions and security warnings are documented.
- Benchmark results are published with hardware and software details.

## Suggested 14-day execution schedule

### Days 1–2: foundation

- [ ] Create crate, CLI skeleton, error types, logging, and CI.
- [ ] Implement `edo doctor`.
- [ ] Capture the first compatibility report.

### Days 3–4: CRIU proof

- [ ] Implement process launch and identity checks.
- [ ] Implement CPU dump and restore.
- [ ] Pass the CPU counter integration test.

### Days 5–7: CUDA proof

- [ ] Read installed CUDA headers and define FFI.
- [ ] Implement dynamic loading and state inspection.
- [ ] Pass raw CUDA lock/checkpoint/restore/unlock tests.

### Days 8–10: combined resurrection

- [ ] Implement quiesce protocol and state machine.
- [ ] Implement freeze ordering.
- [ ] Implement summon coordination and restored process discovery.
- [ ] Pass the native CUDA + CRIU resurrection test.

### Days 11–12: manifest and safety

- [ ] Add compatibility manifest, checksums, permissions, and partial-snapshot handling.
- [ ] Add failure-injection tests.

### Days 13–14: demo and release decision

- [ ] Add PyTorch fixture.
- [ ] Add FastAPI fixture if the PyTorch path is stable.
- [ ] Benchmark and publish results.
- [ ] Decide whether the project is ready for a v0.1 tag or needs another engineering cycle.

## GitHub issue backlog

### P0 issues

1. Linux platform and capability detection
2. `edo doctor` implementation
3. CRIU command wrapper
4. Process identity and lifecycle manager
5. CPU counter dump/restore integration test
6. Minimal CUDA dynamic FFI
7. CUDA state and error mapping
8. Native CUDA checkpoint round-trip fixture
9. Quiesce protocol and state machine
10. Combined CUDA-before-CRIU freeze
11. Restored PID/TID and CUDA restore coordination
12. Combined native CUDA resurrection test
13. Snapshot manifest and compatibility validation
14. Partial snapshot and failure recovery

### P1 issues

15. Snapshot permissions and security documentation
16. Snapshot integrity checks
17. Compatibility matrix
18. PyTorch single-process fixture
19. FastAPI readiness and request draining
20. Benchmark framework
21. Structured logs and timing metrics
22. v0.1 documentation and release packaging

## Explicit non-goals for v0.1

- Multi-GPU, multi-node, NCCL, or distributed process trees.
- vLLM or SGLang multi-process workers.
- In-flight request preservation.
- KV-cache preservation guarantees.
- Arbitrary container migration.
- Cross-chip GPU migration.
- Persistent VRAM or a separate GPU memory service.
- Sub-second restore promises before benchmarks demonstrate them.

## Decision gates

### Gate 1 — after Phase 2

If CPU-only CRIU restore is unreliable, stop and fix process/resource handling before touching CUDA.

### Gate 2 — after Phase 3

If raw CUDA checkpoint/restore is unavailable or too driver-sensitive on the test host, narrow the supported matrix before building higher layers.

### Gate 3 — after Phase 4

If combined resurrection cannot pass with a native fixture, do not add PyTorch or FastAPI complexity.

### Gate 4 — before v0.1

Release only if failures are safe, compatibility is explicit, and benchmarks compare against cold startup.

## First coding session

Start with these tasks, in order:

1. `cargo new edo-tensei`
2. Add the CLI commands: `doctor`, `run`, `cpu-dump`, `cpu-restore`, `freeze`, and `summon`.
3. Implement typed errors and structured logging.
4. Implement `edo doctor` without CUDA FFI first.
5. Create the CPU counter fixture.
6. Implement and test `edo cpu-dump`.
7. Implement and test `edo cpu-restore`.

Do not start with PyTorch, FastAPI, persistent VRAM, or a multi-crate workspace.







