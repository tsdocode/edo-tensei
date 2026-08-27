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

The current engineering focus is reducing the real vLLM restore-to-serving
latency below three seconds. The working baseline is a verified grouped
vLLM restore; the remaining gap is dominated by CRIU page restoration and
large CUDA/process mappings rather than model loading.

### vLLM integration status

- **Phase 9 — DONE (single-GPU proof):** The dedicated
  `examples/vllm-snapshot/` adapter supports a one-GPU,
  tensor-parallel-size-1 launch, normal warmup inference, CUDA-graph-compatible
  startup, worker-tree discovery, and the implemented `freeze-group` /
  `summon-group` protocol for the API parent plus `VLLM::EngineCore`.
  The CUDA 12.8 environment and patched CRIU fork now pass a real dump,
  restore, readiness check, and post-restore chat completion without model
  reload, recompilation, or graph recapture.

- **Phase 10 — DONE (minimum io_uring support):** The CRIU fork/prototype lives at
  `/home/ubuntu/work/criu-vllm` on branch `port-io-uring`. Commit
  `port-io-uring` and supports the vLLM ring shape, ring data images, and
  shared-memory remaps needed by the tested one-GPU workflow. The historical
  upstream PR #1597 was evaluated as a base, but its direct import causes
  substantial source/API drift on current CRIU; the implementation was ported
  selectively. Broader ring features remain future hardening.

- **Phase 11 — IN PROGRESS / PERFORMANCE:** The adapter now supports explicit
  vLLM KV-cache release/wake and an explicit `--kv-cache-memory-bytes` budget.
  With a 2 GiB KV budget plus trusted fast restore, the latest real Qwen
  0.5B run produced a 7.7 GiB checkpoint and restored to serving in 5.046 s
  (first request at 5.063 s), down from 6.524 s with the 9.8 GiB artifact.
  The sub-3-second target is still open. Remaining work is parallel native
  CRIU I/O and a separate GPU-weight artifact/restore path. A real native-AIO
  trial (`--image-io-mode direct`, 16 workers) reached serving in 34.934 s on
  this host versus 5.046 s for buffered restore, so direct I/O is not enabled
  by default. The next target is parallel memfd/anonymous-page restoration
  without depending on O_DIRECT.
  With a 256 MiB KV budget, the artifact fell further to 5.77 GiB and the
  real restore-to-serving measurement was 3.707 s (first request at 3.724 s).
  CUDA restore transitions are now issued concurrently for independent
  process owners; this changed the result by only about 20 ms.
  A 16 MiB KV budget reduced the artifact to about 5.6 GiB and reached serving
  in 3.498 s (first request at 3.515 s), but KV reduction alone does not meet
  the target. The opt-in buffered io_uring reader restored successfully at
  3.725 s, slightly slower than the default buffered path on this host.
  For a batch-1 profile, one captured shape plus synchronous scheduling
  reached a best 3.011 s to health and 3.033 s to the first warm completion.
  Reducing the KV budget to 8 MiB produced a best 2.967 s to health and
  2.988 s to the first warm completion. A repeat measured 3.052 s and 3.076 s,
  so sub-3 is currently a best-case result rather than a stable bound. This
  profile keeps CUDA graph capture enabled but trades away graph-shape coverage
  and scheduler concurrency. Measurements are for Qwen2.5-0.5B-Instruct and do
  not represent Qwen3-0.6B. The exact Qwen3-0.6B profile requires 64 MiB KV
  for a 512-token context and restored successfully in 3.125–3.212 s to health
  and 3.163–3.249 s to the first warm inference. Phase timing attributes about
  2.3–2.6 s to CRIU private-page materialization and about 0.62 s to CUDA
  restore; the stable under-3-second target therefore remains open.
  A production-capacity test with 2 GiB runtime KV successfully released and
  woke the full cache, but produced an approximately 7.1 GiB snapshot and
  restored in 4.647 s to health (4.685 s to the first warm inference). The
  current vLLM sleep/wake hook preserves capacity but does not yet separate KV
  backing from the checkpoint; a CUDA VMM/GMS-style artifact remains required.

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

## Experimental CRIU io_uring work

This is a post-v0.1 research track for vLLM/SGLang compatibility. The isolated
CRIU fork at `../criu-vllm` currently builds and recognizes stock Linux
io_uring VMAs and fdinfo; the focused idle-ring round trip is complete, while
broader application compatibility is not.

- [x] Build a minimal idle io_uring fixture without liburing.
- [x] Confirm baseline CRIU fails on the io_uring VMA/fd.
- [x] Port the historical CRIU io_uring design far enough to compile on the
  current CRIU tree and parse current stock fdinfo.
- [x] Confirm the patched CRIU reaches descriptor collection.
- [x] Implement a CRIU-side duplicate and placeholder fd transfer so dump
  completes despite the kernel rejecting io_uring `SCM_RIGHTS` transfer.
- [x] Restore the io_uring object and replay its captured ring data.
- [x] Bypass parasite `SCM_RIGHTS` for io_uring descriptors using a CRIU-side
  duplicate and placeholder fd during dump.
- [ ] Clean up that bypass and validate descriptor behavior across broader
  io_uring feature combinations.
- [x] Add `IORING_SETUP_SQPOLL` handling. CRIU omits the kernel-owned
  `iou-sqp-*` helper from userspace thread enumeration and recreates it during
  restore; the SQPOLL harness round trip passes.
- [x] Restore a simple registered-file set by preserving fdinfo registration
  order and resolving restored file paths.
- [x] Restore registered buffers by serializing a flat, restorer-safe buffer
  list in the io_uring image and issuing `IORING_REGISTER_BUFFERS` after
  restored mappings exist; the `--buffers` round-trip restores two buffers
  sharing one VMA.
- [ ] Make registered-file resolution robust for
  duplicate paths, deleted files, and non-path-backed descriptors.
- [x] Accept duplicate fixed-file slots that reference the same open-file
  description using `kcmp(KCMP_FILE)` while continuing to reject ambiguous
  independent descriptors; the `--duplicate-files` round-trip passes.
- [x] Restore a memfd-backed registered file by matching its preserved
  `/memfd:*` descriptor identity; the `--memfd-file` round-trip passes.
- [x] Normalize basename and ` (deleted)` fdinfo/proc-link differences for
  deleted-but-open registered files; the `--deleted-file` round-trip passes.
- [x] Restore sparse fixed-file tables from `UserFiles` entries containing
  `<none>` by using `IORING_REGISTER_FILES2` plus slot updates; the
  `--sparse-files` round-trip passes.
- [x] Apply registered-file restoration consistently to both stock and
  extended fdinfo parser formats.
- [x] Add conditional restore for registered eventfds and async eventfds when
  fdinfo exposes their descriptor numbers.
- [ ] Validate eventfd/async-eventfd restore on a kernel that exposes those
  lines.
- [x] Restore an attached shared workqueue by correlating the shared
  `SqThread` identity when stock fdinfo omits `WqFd`; explicit `WqFd` parser
  plumbing and fail-closed descriptor validation are also present. The
  `--attach-wq` fixture covers an un-mapped parent ring and verifies that both
  restored rings share one SQPOLL thread.
- [x] Restore ring mappings at their original virtual addresses. The prototype
  now reserves page-aligned user memory and restores the ring with
  `IORING_SETUP_NO_MMAP`, avoiding the kernel's rejected address-directed
  io_uring `mmap` path.
- [x] Complete dump and restore round-trip for an idle io_uring process,
  including repeated runs, ring/SQE VMA verification, fdinfo verification,
  and replayed SQE data.
- [x] Verify `SQE128`/`CQE32` formats with enlarged fixture mappings. Since
  stock Linux fdinfo does not expose the setup flags, CRIU infers them from
  page-rounded VMA sizes; the `--wide` dump/restore and widened-SQE sentinel
  test pass.
- [x] Test the design against vLLM, including CUDA graph warmup and serving
  readiness after restore.
- [x] Build a project-local uv vLLM environment on the H100 host with CUDA
  PyTorch cu128, matching TorchAudio/TorchVision, Python development headers,
  Ninja, and the editable vLLM tree; a real Qwen 0.5B server reaches healthy
  warm serving with async scheduling and both piecewise/full CUDA graphs.
- [x] Exercise the real vLLM group freeze path far enough to expose the
  application-specific `rw-s`, mode-0600 `anon_inode:[io_uring]` VMA form and
  add an explicit `EDO_CRIU` selector so tests can use the patched CRIU fork.
- [x] Complete the real vLLM group dump/restore. With other GPU workloads
  stopped, Qwen2.5-0.5B was warmed through vLLM's async scheduler, torch.compile,
  FlashInfer, and piecewise/full CUDA graph capture; Edo froze both the API and
  `VLLM::EngineCore`, CRIU dumped/restored them, and a post-restore chat request
  succeeded. The large checkpoint is about 12 GiB; image hashing adds several
  minutes before dump and restore.
- [x] Validate the shipped `vllm_adapter.py` workflow end-to-end after the
  PID-reaping fix; the adapter reports `vLLM group restore passed` and a
  post-restore warmup response of `Ready.`. The measured run reached health
  after 30.046 s cold and 254.678 s from restore command start (including
  checksum verification), with a 0.018 s post-restore inference request. An
  opt-in trusted-local fast restore skips rereading page images and reached
  health in 7.978 s, with the first post-restore inference in 7.996 s.
- [ ] Support `IORING_SETUP_NO_MMAP` rings whose SQ/CQ memory is anonymous
  application memory; current Linux fdinfo provides no user-address metadata,
  so generic association is not yet possible.

### Dynamo-inspired restore performance

- [x] Add a trusted-local fast restore that skips redundant SHA-256 page-image
  reads while retaining metadata, host/GPU, presence, and size validation.
- [x] Add an explicit vLLM quiesce/resume hook that releases unused KV-cache
  backing while preserving serving state; the H100 test freed 2.11 GiB,
  produced a 9.8 GiB artifact, and woke the cache in 0.005 s.
- [x] Profile the patched CRIU restore path and validate parallel restore
  candidates on the H100. Buffered page restoration remains the fastest safe
  path measured: native O_DIRECT/AIO reached 34.934 s versus 5.046 s buffered,
  while parallel CUDA transitions changed the 256 MiB result by only about
  20 ms. An experimental zero-copy VMA mapping path was rejected because it
  reduced CRIU copy time but did not reliably complete the vLLM CUDA handoff.
- [ ] Prototype a separate GPU-memory weight artifact and overlap its restore
  with CRIU process restoration; this is the GMS-equivalent track.
- [ ] Decouple the production KV-cache capacity from checkpoint backing so a
  large runtime cache can be recreated at the original virtual addresses after
  restore without including its physical pages in the process snapshot.
- [x] Exercise the exact staged vLLM lifecycle: `sleep(level=1)` followed by
  `wake_up(tags=["weights"])` and `wake_up(tags=["kv_cache"])`. The API supports
  reinitializing the cache this way, but the Qwen3-0.6B / 2 GiB checkpoint was
  still about 7.1 GiB before restore, proving that vLLM-level KV discard alone
  does not exclude the backing from CRIU's CUDA image. The run was stopped after
  the image completed but Edo's freeze wrapper hung; no serving result is
  claimed for this staged variant.
- [x] Fix CRIU fork VMA recognition for kernel-reported
  `anon_inode:[io_uring]` mappings (including synthetic mode `0600`). The
  standalone io_uring round-trip now passes, and the vLLM dump passes this
  parser stage.
- [x] Diagnose the apparent post-image `freeze-group` hang: CRIU had already
  exited successfully, while Edo was hashing the large CUDA image for the
  integrity manifest. Manifest generation now prints per-file progress and
  elapsed time so this work is observable rather than looking stalled.
- [x] Validate real vLLM async scheduling end to end after the parser fix. The
  Qwen3-0.6B group dumped/restored successfully, staged weight/KV wake passed,
  and post-restore inference succeeded: 3.368 s to `/health`, 3.400 s to the
  first warm inference.
- [x] Compare streaming TTFT before and after async restore. For the same
  Qwen3-0.6B prompt, cold/warm TTFT was 0.040 s and post-restore TTFT was
  0.017 s (delta -0.024 s); no torch.compile or CUDA-graph recapture occurred
  during restore, so the restored compiled/graph state remained usable.
- [x] Validate SGLang group snapshot/restore. SGLang 0.5.9 with Qwen3-0.6B,
  async scheduling, torch.compile, and CUDA graph dumped and restored
  successfully. The CRIU fork now handles SGLang's `/dev/nvidia*` character
  device FDs and defers driver-private `-w-s` mappings to CUDA restore. The
  restored `/model_info` endpoint returned 200 and `/generate` returned valid
  JSON in 0.03 s; restore timing was CRIU 3.082 s, CUDA restore 3.776 s.
- [x] Test vLLM sleep level 2 as a possible backing-release shortcut. It freed
  about 2.08 GiB, but level 2 discards model weights as well; the attempted
  CRIU dump also failed on an `anon_inode:[io_uring]` mapping. It is therefore
  not a no-reload solution and is retained only as a negative experiment.

Current result: the patched CRIU binary builds, ordinary processes still dump,
and idle io_uring processes pass repeated dump/restore round trips for default,
SQPOLL, registered-file, sparse-file, deleted-file, memfd, duplicate-slot,
registered-buffer, SQE128/CQE32, and attached shared-workqueue cases. The
fixture also covers an un-mapped workqueue parent and verifies restored ring
addresses, fdinfo state, and replayed SQE data. CRIU unit tests and an
ordinary-process dump also pass. The implementation is still a focused
prototype: eventfd discovery, NO_MMAP rings, and independent same-path
identity remain unvalidated. The real vLLM environment, warmup path, full
group restore, and post-restore serving request are now validated; no vLLM
source process was modified.

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

## Triton container experiment

- [x] Add a reproducible official Triton 25.05 Python-backend GPU demo with
  a warmed CuPy model and HTTP inference validation.
- [x] Confirm Edo/CRIU can dump the Triton server plus Python backend stub;
  the snapshot was created successfully.
- [ ] Restore Triton from the native Edo path. The current blocker is Docker
  and NVIDIA Container Toolkit mount-namespace restoration (`criu/mount.c:48`),
  before CUDA restore is reached. The next implementation should use a
  container-aware checkpoint path or run Triton directly in the host namespace.
- [x] Add the first Kubernetes container-aware MVP: privileged node-local
  snapshot agent, CRD, RBAC, namespace entry, PID translation, and Triton
  deployment notes. Same-node restore with a placeholder Pod is the supported
  scope; controller/CRI discovery and cross-node storage remain open.
- [x] Install a local k3s cluster and validate NVIDIA runtime integration with
  the device plugin and a CUDA smoke-test Pod (H100, CUDA 12.8).
- [x] Run the Edo snapshot agent inside k3s against a real GPU workload. Live
  dump reached CUDA `CHECKPOINTED`, created the manifest, and resumed the
  source process; restore into a PID-1-only placeholder recreated and unlocked
  the CUDA process. Restore was 17.35s for the 256 MiB fixture (CUDA init
  16.59s, CRIU restore 0.47s, CUDA restore/unlock 0.29s).
- [ ] **P1 / NEXT** Add controller-managed Pod/CRI PID discovery and readiness
  probes around this validated same-node snapshot/restore path.
- [x] Investigate the apparent CUDA initialization latency. Instrumentation
  showed `cuInit()` at about 11ms; the apparent 16.8s cost was integrity
  hashing of the 832 MiB CRIU pages image.
- [x] Optimize snapshot hashing with the native SHA-256 implementation. The
  same fixture's snapshot creation fell from 17.70s to 2.11s; pages hashing
  fell to about 0.7s.
- [ ] **P1 / NEXT** Integrate a persistent restore worker only if model-server
  benchmarks show a remaining CUDA initialization bottleneck, then validate
  the target model-server readiness path.

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
