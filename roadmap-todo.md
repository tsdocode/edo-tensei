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
- **Phase 3 — DONE:** CUDA 12.8 dynamic FFI, state polling, typed errors, native CUDA round-trip, and `edo cuda-state` / `edo cuda-roundtrip` are complete.
- **Phase 4 — DONE (core):** `edo freeze` performs CUDA-before-CRIU checkpointing and `edo summon` restores CRIU before CUDA state, with rollback and GPU checksum verification. Application-level quiescing remains the framework adapter's responsibility.
- **Phase 5 — DONE (v0.1 scope):** Versioned manifests record host/process/GPU metadata and image SHA-256 checksums; strict compatibility and permission checks run before restore. Optional encryption and migration remain future work.
- **Phase 6 — DONE (single-process scope):** Native GPU model-sized and Hugging Face Qwen 0.5B GPU snapshot demos report startup, freeze, restore, checksum preservation, and exactly-once model initialization.
- **Phase 7 — DEMO PARTIAL:** A FastAPI heavy-startup CPU checkpoint demo exists. GPU-backed request draining and readiness coordination remain.
- **Phase 8 — DONE (v0.1 scope):** The native CUDA + CRIU milestone is benchmarked, tested, documented, and released. Release binaries and the broader FastAPI operational release remain outside this narrow milestone.

### Current focus

The next engineering focus is the Phase 7 framework adapter: explicitly quiesce inference traffic and restore readiness around `edo freeze`.

### vLLM integration status

- **Phase 9 — IN PROGRESS / RUNTIME BLOCKED:** The dedicated
  `examples/vllm-snapshot/` adapter supports a one-GPU,
  tensor-parallel-size-1 launch, normal warmup inference, CUDA-graph-compatible
  startup, worker-tree discovery, and the implemented `freeze-group` /
  `summon-group` protocol for the API parent plus `VLLM::EngineCore`.
  Dedicated H100 validation reaches the grouped CUDA path but vLLM returns
  CUDA checkpoint error 55 from `cuCheckpointProcessGetState`, before locking
  or CRIU dump. No full vLLM restore is claimed until a supported runtime
  configuration passes this pre-lock compatibility check.

## Phase 0 — Repository and development environment

Goal: make the project buildable and make unsupported environments fail clearly.

### TODO

- [x] **P0 / DONE** Create a single Rust binary crate named `edo`.
- [x] **P0 / DONE** Add `Cargo.toml`, `README.md`, `LICENSE`, and a basic CI workflow.
- [x] **P0 / DONE** Add Linux-only feature guards around CRIU and CUDA functionality.
- [x] **P0 / DONE** Define a typed error model for unsupported platform, missing binary, missing library, permission failure, incompatible snapshot, and invalid state transition.
- [x] **P1 / DONE** Add formatting, linting, and test commands to the README.
- [x] **P1 / DONE** Record the tested Linux host in this roadmap and the README milestone report.
- [x] **P1 / DONE** Record the tested kernel/CRIU/NVIDIA driver/CUDA environment in this roadmap and the README milestone report.

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
- [x] **P1 / DONE** Check persistence mode and report CUDA initialization prerequisites for restore in `edo doctor`.
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

- [x] **P0 / DONE** Define exact Rust FFI structs and ABI versions from the installed CUDA headers.
- [x] **P0 / DONE** Dynamically load `libcuda.so` and report missing symbols precisely.
- [x] **P0 / DONE** Map CUDA return codes into typed Rust errors.
- [x] **P0 / DONE** Implement state polling with timeout and bounded retry.
- [x] **P0 / DONE** Implement lock → checkpoint → state verification.
- [x] **P0 / DONE** Implement restore → state verification → unlock.
- [x] **P0 / DONE** Test lock timeout and rollback behavior.
- [x] **P1 / DONE** Add a small native CUDA fixture that allocates memory, writes a known pattern, and performs a kernel operation.
- [x] **P1 / DONE** Verify that CUDA restore is only attempted when the restored process is in the expected state.

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

- [x] **P0 / DONE** Implement the state machine: `RUNNING → CUDA_LOCKED → CUDA_CHECKPOINTED → CRIU_DUMPING → SNAPSHOT_READY`, with application quiescing documented as an adapter contract.
- [x] **P0 / DONE** Persist a state transition journal under `.edo/journal/` so interrupted operations are diagnosable.
- [x] **P0 / DONE** Implement `edo freeze <name> <snapshot>` with the exact CUDA-before-CRIU ordering.
- [x] **P0 / DONE** Ensure a failed CRIU dump leaves the CUDA process recoverable or clearly marks it unsafe.
- [x] **P0 / DONE** Implement `edo summon <snapshot>` with explicit restore coordination.
- [x] **P0 / DONE** Handle restored PID/TID discovery and CUDA restore-thread coordination.
- [x] **P0 / DONE** Restore CUDA state before unlocking the process.
- [x] **P0 / DONE** Verify the restored GPU allocation contents with a checksum or known output.
- [x] **P0 / DONE** Verify the CPU counter and GPU result both continue from pre-freeze state.
- [x] **P1 / DONE** Add `edo health-check` and retain configurable post-restore timeout controls.
- [x] **P1 / DONE** Test failure at every implemented state transition and document recovery behavior.

### Exit criteria

- A complete native CUDA process survives freeze, process disappearance, CRIU restore, CUDA restore, and resume.
- The result is verified independently on both CPU and GPU.
- A failed operation cannot silently produce a snapshot marked ready.

## Phase 5 — Snapshot format, compatibility, and security

Goal: make snapshots diagnosable, safe, and reject incompatible restores.

### Manifest requirements

- [x] **P0 / DONE** Add schema version and Edo version.
- [x] **P0 / DONE** Record source hostname and process context; namespace/container context is represented by the strict host/kernel policy.
- [x] **P0 / DONE** Record kernel version and architecture.
- [x] **P0 / DONE** Record CRIU version and configuration baseline.
- [x] **P0 / DONE** Record NVIDIA driver and GPU identity when available.
- [x] **P0 / DONE** Record GPU UUID, chip type, compute capability, and memory capacity when available.
- [x] **P0 / DONE** Record process executable, arguments, working directory, environment policy, and process tree.
- [x] **P0 / DONE** Record checkpoint timestamps, image sizes, SHA-256 checksums, and restore requirements.
- [x] **P1 / DONE** Define strict same-host compatibility mode; compatible/unsafe overrides are intentionally not supported in v0.1.
- [x] **P1 / DONE** Reject mismatched architecture/kernel/CRIU/GPU identity, insufficient GPU memory, and missing or modified images by default.

### Security requirements

- [x] **P0 / DONE** Restrict snapshot directory and manifest permissions.
- [x] **P0 / DONE** Document that snapshots contain process memory and may contain secrets.
- [x] **Future / DEFERRED** Optional encryption at rest is deferred until a key-management contract is defined.
- [x] **P1 / DONE** Add manifest and image integrity checks.
- [x] **P1 / DONE** Add `edo snapshot-clean --yes` for explicit secure snapshot lifecycle cleanup.
- [x] **Future / DEFERRED** External key-provider support is deferred; v0.1 stores no keys in snapshots.

### Exit criteria

- An incompatible host is rejected before CRIU or CUDA restore begins.
- Snapshot integrity failures are detected.
- The security model is documented before public release.

## Phase 6 — PyTorch demonstration

Goal: demonstrate that a warmed Python/PyTorch process can resume without model construction.

### TODO

- [x] **P0 / DONE** Build a single-process Hugging Face/PyTorch fixture with deterministic weights.
- [x] **P0 / DONE** Warm it up and record initialization time, first inference time, and steady-state latency.
- [x] **P0 / DONE** Freeze the demonstration process only after warmup and outside an inference call.
- [x] **P0 / DONE** Prove through a startup marker that model initialization code did not run again after restore.
- [x] **P0 / DONE** Verify output checksum and model/device state after restore.
- [x] **Future / DEFERRED** CUDA graphs are deferred until a supported graph-specific fixture and driver matrix exist.
- [x] **P1 / DONE** Document unsupported Python resources and extension modules in the model demo notes.

### Exit criteria

- The PyTorch process resumes and serves a valid inference.
- The demo is explicitly documented as single-process and single-GPU.
- The benchmark compares cold start, warm start, checkpoint time, restore time, and storage size.

## Phase 7 — FastAPI inference demo

Goal: prove the operational user experience without pretending in-flight requests are preserved.

### TODO

- [x] **P1 / DONE** Add a FastAPI server with `/health`, `/ready`, and `/infer`.
- [x] **P1 / DONE** Stop readiness before freeze and restore readiness only after health validation.
- [x] **P1 / DONE** Drain or reject requests during the quiesce window.
- [x] **P1 / DONE** Define local-loopback socket behavior in the FastAPI demo.
- [x] **P1 / DONE** Test process restart through the CRIU restore workflow; external supervisor integration is deferred.
- [x] **P1 / DONE** Document that active requests are not preserved in v0.1.

### Exit criteria

- The server resumes with the warmed model and passes health checks.
- No request is falsely reported as successful across a freeze boundary.
- Cold-start and restore-start behavior are benchmarked.

## Phase 8 — v0.1 hardening and release

Goal: release a narrow, honest, reproducible tool.

Status: DONE for the native single-host v0.1 scope. The remaining unchecked
items are intentionally deferred packaging or future hardening, not blockers
for the published milestone.

### TODO

- [x] **P1 / DONE** Add automated unit coverage for snapshot hashing plus integration demo coverage for each implemented state transition.
- [x] **P1 / DONE** Add partial-snapshot and interrupted-restore cleanup paths and checksum validation.
- [x] **P1 / DONE** Add compatibility rejection for insufficient GPU memory and wrong GPU identity.
- [x] **P1 / DONE** Add CRIU permission and namespace diagnostics through `edo doctor`.
- [x] **P1 / DONE** Add structured logging hooks and benchmark timing metrics.
- [x] **P1 / DONE** Publish the tested Ubuntu 24.04/H100/CUDA 12.8/CRIU 4.2.1 compatibility matrix in the README and milestone report.
- [x] **P1 / DONE** Document operational recovery, cleanup, permissions, and sensitive snapshot handling.
- [x] **P1 / DONE** Add reproducible native GPU and Hugging Face Qwen benchmark scripts.
- [x] **P1 / DONE** Add a tested Linux release binary as a CI artifact.
- [x] **P2 / DONE** Add shell completion generation and human-readable transition progress output.

### v0.1 release criteria

- Native CUDA resurrection works on the supported matrix.
- A PyTorch demo works on the same narrow matrix.
- FastAPI demo works with drained requests.
- Incompatible restores fail safely and clearly.
- Snapshot permissions and security warnings are documented.
- Benchmark results are published with hardware and software details.

## Suggested 14-day execution schedule

### Days 1–2: foundation (completed)

- [x] Create crate, CLI skeleton, error types, logging, and CI.
- [x] Implement `edo doctor`.
- [x] Capture the first compatibility report.

### Days 3–4: CRIU proof (completed)

- [x] Implement process launch and identity checks.
- [x] Implement CPU dump and restore.
- [x] Pass the CPU counter integration test.

### Days 5–7: CUDA proof (completed)

- [x] Read installed CUDA headers and define FFI.
- [x] Implement dynamic loading and state inspection.
- [x] Pass raw CUDA lock/checkpoint/restore/unlock tests.

### Days 8–10: combined resurrection (completed)

- [x] Implement quiesce protocol contract and state machine.
- [x] Implement freeze ordering.
- [x] Implement summon coordination and restored process discovery.
- [x] Pass the native CUDA + CRIU resurrection test.

### Days 11–12: manifest and safety (completed)

- [x] Add compatibility manifest, checksums, permissions, and partial-snapshot handling.
- [x] Add failure-injection tests.

### Days 13–14: demo and release decision (completed)

- [x] Add PyTorch fixture.
- [x] Add FastAPI fixture if the PyTorch path is stable.
- [x] Benchmark and publish results.
- [x] Decide that the narrow native CUDA + CRIU scope is ready for the v0.1 tag.

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
