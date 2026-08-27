# Edo Tensei

![Edo Tensei Project](assets/edo-tensei-project.png)

GPU process snapshot and fast-resume runtime.

## Build

Requires Rust 1.80 or newer.

```bash
cargo build
cargo test
```

For this development host, load the CUDA toolkit environment before CUDA work:

```bash
source ./env.sh
```

## CLI

```bash
edo --help
edo doctor
edo doctor --json
```

`edo doctor` reports Linux and architecture support, CRIU version and kernel checks, CUDA driver loading, CUDA checkpoint symbols, GPU identity, memory, and compute capability.

```text
edo run --name <name> -- <command> [args...]
edo cpu-dump <name-or-pid> <snapshot-dir>
edo cpu-restore <snapshot-dir>
edo snapshot-check <snapshot-dir>
edo freeze <name-or-pid> <snapshot-dir>
edo summon <snapshot-dir>
```

## v0.1 milestone results

Validated on Ubuntu 24.04 x86_64 with an NVIDIA H100 80GB, driver 570.211.01,
CUDA 12.8, and CRIU 4.2.1.

```text
CUDA RUNNING → LOCKED → CHECKPOINTED
→ CRIU DUMP → CRIU RESTORE
→ CUDA RESTORE → UNLOCKED and RUNNING
```

### Native GPU model-sized allocation

For a 512 MiB CUDA-resident weight tensor:

| Metric | Result |
|---|---:|
| Freeze time | 2.043 s |
| Restore time | 1.534 s |
| Checksum before | `17193905543863665539` |
| Checksum after | `17193905543863665539` |

```bash
EDO_MODEL_MB=512 ./examples/gpu-model-snapshot/run-demo.sh
```

### Hugging Face Qwen 0.5B Instruct on GPU

Loaded directly into H100 VRAM with CUDA-enabled PyTorch:

| Metric | Result |
|---|---:|
| Parameters | 494,032,768 |
| Cold startup + warmup | 18.178 s |
| Freeze time | 3.674 s |
| Restore time | 3.068 s |
| Checksum before/after | identical |

```bash
./examples/modal-gpu-snapshot/run-qwen-demo.sh
```

The model was loaded, warmed up with generation, frozen, restored, and
verified with the same GPU model checksum before and after restore.

### Real vLLM before and after

The full vLLM adapter was validated on the same H100 with the cached
Qwen2.5-0.5B-Instruct model, vLLM `0.23.1rc1`, CUDA 12.8, and the patched
CRIU fork. The test used vLLM's normal asynchronous scheduler, torch.compile,
FlashInfer, and piecewise/full CUDA graph capture.

| Checkpoint boundary | Before snapshot | After restore |
|---|---|---|
| API readiness | `/health` and `/v1/models` succeed | `/health` succeeds |
| Model state | Qwen 0.5B loaded on H100 | CUDA state restored in-place |
| Runtime state | CUDA graphs captured; warmup returns `Ready.` | CUDA owners return to `RUNNING`; warmup returns `Ready.` |
| Process group | API parent + `VLLM::EngineCore` | Same recorded process group restored by CRIU |
| Serving | Chat completion succeeds | Post-restore chat completion succeeds |

Measured timing from the end-to-end adapter run:

| Metric | Measured latency |
|---|---:|
| Cold launch to `/health` | 30.046 s |
| Cold launch to first warm inference | 30.080 s |
| Integrity-verified restore to `/health` | 254.678 s |
| Integrity-verified restore to first warm inference | 254.697 s |
| Trusted fast restore to `/health` | 7.978 s |
| Trusted fast restore to first warm inference | 7.996 s |
| Fast restore with Dynamo-style KV release to `/health` | 6.524 s |
| Fast restore with Dynamo-style KV release to first warm inference | 6.542 s |
| Fast restore with 2 GiB KV budget + KV release to `/health` | 5.046 s |
| Fast restore with 2 GiB KV budget + KV release to first warm inference | 5.063 s |
| Fast restore with 16 MiB KV budget + KV release to `/health` | 3.498 s |
| Fast restore with 16 MiB KV budget + KV release to first warm inference | 3.515 s |
| Best fast restore with 16 MiB KV + one captured shape + synchronous scheduling to `/health` | 3.011 s |
| Best fast restore with 16 MiB KV + one captured shape + synchronous scheduling to first warm inference | 3.033 s |
| Best fast restore with 8 MiB KV + one captured shape + synchronous scheduling to `/health` | 2.967 s |
| Best fast restore with 8 MiB KV + one captured shape + synchronous scheduling to first warm inference | 2.988 s |
| Repeat 8 MiB profile to `/health` | 3.052 s |
| Repeat 8 MiB profile to first warm inference | 3.076 s |
| Qwen3-0.6B, 64 MiB KV + one captured shape to `/health` | 3.125–3.212 s |
| Qwen3-0.6B, 64 MiB KV + one captured shape to first warm inference | 3.163–3.249 s |
| Qwen3-0.6B, 2 GiB runtime KV + release/wake to `/health` | 4.647 s |
| Qwen3-0.6B, 2 GiB runtime KV + release/wake to first warm inference | 4.685 s |
| Post-restore inference request | 0.018 s |

The CRIU image set is approximately 12 GiB. Dump and restore also perform
SHA-256 verification, which takes several minutes on this host. This is a
checkpoint/resume result, not a claim of zero downtime or a benchmark of
request latency. The test drains requests and uses one GPU with tensor
parallelism disabled.

For a trusted local snapshot, `summon-group --skip-integrity` (or the adapter's
`--fast-restore`) skips rereading the large page images. It still validates
snapshot metadata, host/GPU compatibility, file presence, and file sizes.
Keep full integrity verification enabled when snapshots may have been copied,
modified, or exposed to an untrusted filesystem.

The KV-cache release experiment freed 2.11 GiB before checkpointing and produced
a 9.8 GiB artifact. The restored engine woke its KV cache in 0.005 s and served
again without model reload. The remaining artifact is process/CUDA shadow
state, so reaching the sub-3-second target requires the parallel CRIU and
separate GPU-weight paths described below.

An explicit vLLM KV budget is available for bounded deployments. With
`--kv-cache-memory-bytes 2147483648 --release-kv-cache --fast-restore`, the
Qwen 0.5B artifact fell to 7.7 GiB and restore-to-serving fell to 5.046 s
(first request at 5.063 s). The remaining gap to three seconds requires
parallel CRIU I/O and a separate GPU-weight artifact.

With a 256 MiB KV budget, the artifact reached 5.77 GiB and restore-to-serving
reached 3.707 s (first request at 3.724 s). CUDA restore transitions are
parallelized for independent process owners, but that saves only about 20 ms;
the remaining gap is CRIU page materialization.

Reducing the explicit KV budget to 16 MiB reduced the artifact to about 5.6 GiB
and reached `/health` in 3.498 s (first warm inference at 3.515 s). vLLM still
completed normal warmup and CUDA graph capture before checkpointing. Lowering
the KV budget alone is therefore not sufficient to reach three seconds.

For the batch-1 demo profile, limiting vLLM to one captured shape and disabling
async scheduling reduced the best run to 3.011 s to `/health` and 3.033 s to
the first warm completion. This still captures both the piecewise and full
CUDA graph for the selected shape; it is a latency/throughput tradeoff and is
not the default production configuration.

An aggressively bounded 8 MiB KV profile reached a best 2.967 s to `/health`
and 2.988 s to the first warm completion. A repeat reached 3.052 s and
3.076 s respectively, so the sub-3 result is currently a best-case measurement
and not a guaranteed service-level bound. Both runs used the local
Qwen2.5-0.5B-Instruct model; this result does not represent Qwen3-0.6B.

The exact Qwen3-0.6B model requires at least about 50 MiB of KV cache for a
512-token serving profile, so the smallest tested budget was 64 MiB. It
restored successfully with captured CUDA graphs and no model reload, but
measured 3.125–3.212 s to `/health` and 3.163–3.249 s to the first warm
inference. CRIU phase timing attributes about 2.3–2.6 s to private-page
materialization and about 0.62 s to CUDA restore. This is the remaining gap to
the under-3-second target.

The production-capacity experiment configured 2 GiB of KV cache, released it
before checkpointing, and successfully woke the same 2 GiB runtime after
restore. However, the snapshot was still approximately 7.1 GiB and restore
took 4.647 s to health (4.685 s to the first warm inference). vLLM sleep/wake
therefore preserves capacity but does not yet create a small independent KV
artifact; a CUDA VMM/GMS-style backing split is still required.

Native AIO was trialed with the patched CRIU fork, but O_DIRECT reached
serving in 34.934 s on this host versus 5.046 s for buffered restore. Edo
therefore keeps buffered restore as the default; the remaining optimization
target is parallel memfd/anonymous-page restoration without O_DIRECT.

An opt-in buffered `io_uring` reader was also tested against the 256 MiB
configuration. It restored successfully and served, but measured 3.725 s to
`/health`, slightly slower than the 3.707 s buffered baseline on this host.

### Dynamo-inspired optimization track

The current fast path applies the first practical optimization for this local
prototype: remove the redundant page-image hashing from the serving-critical
restore path. NVIDIA's Dynamo Snapshot design identifies three deeper
optimizations for inference workers: release unused KV-cache backing while
preserving virtual addresses, restore shared memory and anonymous pages with
parallel/asynchronous I/O, and decouple large model weights into a separate
GPU-memory artifact. Edo does not claim those latter mechanisms yet; they are
the next performance track because they require vLLM quiesce/resume hooks and
additional CRIU/GPU-memory integration.

## Development status

- Phase 0: environment and crate scaffold complete.
- Phase 1: capability discovery and `edo doctor` complete.
- Phase 2: CPU-only CRIU dump/restore proof complete.
- Phase 3: CUDA checkpoint FFI and native CUDA round trip complete.
- Phase 4 core: CUDA + CRIU freeze/summon, checksum verification, and failure recovery complete.

## CPU-only proof

```bash
cargo build
target/debug/edo run --name counter -- setsid python3 examples/cpu-counter.py
sudo target/debug/edo cpu-dump counter /tmp/edo-snapshot
sudo target/debug/edo cpu-restore /tmp/edo-snapshot
```

See [the v0.1 milestone report](V0.1-MILESTONE.md) and the [GPU model demo](examples/gpu-model-snapshot/README.md) for scope, limitations, and reproduction commands.

## Snapshot safety

Snapshot manifests use schema version 2 and record the Edo version, source
host, kernel, architecture, CRIU version, available GPU identity/capacity,
process identity, restore requirements, and SHA-256 checksums for CRIU image
files. `edo cpu-restore` and `edo summon` validate this metadata and the image
checksums before invoking CRIU. `edo snapshot-check <snapshot-dir>` performs
the same non-mutating validation.

Snapshots are restricted to mode `0700`, and manifests to `0600`. Snapshot
directories contain process memory and may contain credentials or other
secrets; store them as sensitive data and remove them securely when no longer
needed. v0.1 uses strict same-host compatibility and does not provide
encryption-at-rest or cross-GPU migration.
