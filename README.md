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
edo freeze <name-or-pid> <snapshot-dir>
edo summon <snapshot-dir>
```

## v0.1 milestone results

Validated on Ubuntu 24.04 x86_64 with an NVIDIA H100 80GB, driver 570.211.01,
CUDA 12.8, and CRIU 4.2.1.

The validated core transition is:

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

Run it with:

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

Run it with the dedicated `edo-models` Conda/uv environment:

```bash
./examples/modal-gpu-snapshot/run-qwen-demo.sh
```

The model was loaded, warmed up with generation, frozen, restored, and
verified with the same GPU model checksum before and after restore.

## Development status

- Phase 0: environment and crate scaffold complete.
- Phase 1: capability discovery and `edo doctor` complete.
- Phase 2: CPU-only CRIU dump/restore proof complete.
- Phase 3: CUDA checkpoint FFI and native CUDA round trip complete.
- Phase 4 core: CUDA + CRIU freeze/summon, checksum verification, and failure recovery complete.

## CPU-only proof

Run the bundled fixture:

```bash
cargo build
target/debug/edo run --name counter -- setsid python3 examples/cpu-counter.py
sudo target/debug/edo cpu-dump counter /tmp/edo-snapshot
sudo target/debug/edo cpu-restore /tmp/edo-snapshot
```

See [the v0.1 milestone report](V0.1-MILESTONE.md) and the [GPU model demo](examples/gpu-model-snapshot/README.md) for detailed scope, limitations, and reproduction commands.
