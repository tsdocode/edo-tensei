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

The process-management commands are scaffolded for later phases:

```text
edo run --name <name> -- <command> [args...]
edo cpu-dump <name-or-pid> <snapshot-dir>
edo cpu-restore <snapshot-dir>
edo freeze <name-or-pid>
edo summon <snapshot-dir>
```

## Development status

- Phase 0: environment and crate scaffold complete.
- Phase 1: capability discovery and `edo doctor` core complete; CRIU privilege hardening remains documented.
- Phase 2: CPU-only CRIU dump/restore proof complete.
- FastAPI demo: Hugging Face CPU model loading and Torch CRIU round trip verified with Qwen 0.5B.

## CPU-only proof

Run the bundled fixture:

```bash
cargo build
target/debug/edo run --name counter -- setsid python3 examples/cpu-counter.py
sudo target/debug/edo cpu-dump counter /tmp/edo-snapshot
sudo target/debug/edo cpu-restore /tmp/edo-snapshot
```

The dump command writes CRIU images and a `manifest.json`. Restore writes `restored.pid` and reports the restored process. The current proof targets one simple process and does not yet promise arbitrary sockets, containers, process trees, or request preservation.

## FastAPI and Hugging Face demo

Load a 494M-parameter Qwen model on CPU and checkpoint the running service:

```bash
EDO_HF_MODEL=Qwen/Qwen2.5-0.5B \
  ./examples/fastapi-heavy-startup/run-demo.sh
```

The demo verifies that model state and inference survive CRIU restore. See [the demo README](examples/fastapi-heavy-startup/README.md) for dependency installation and model options.
