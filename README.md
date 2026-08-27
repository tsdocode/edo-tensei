# Edo Tensei

Edo Tensei checkpoints live AI workloads and restores them with their runtime state intact — from Python processes to PyTorch and vLLM servers.

> Warm it once. Freeze it. Let it disappear. Bring it back ready to serve.

<img width="1774" height="887" alt="19388b01-0b92-4b7c-89ae-f46379f7da58" src="https://github.com/user-attachments/assets/b58ac69a-e245-4e60-a3c6-8ce691e0ab48" />


## The wow moment: restore beats cold start

These are measured runs, not promises. The CPU row is produced by the first-run demo on this repository's development host; the vLLM row is the validated Qwen3-0.6B single-GPU run documented in the roadmap.

| Workload | Cold start → ready | Restore → ready | Observed improvement |
| --- | ---: | ---: | ---: |
| Stateful CPU process | 2.102 s | 0.164 s | **12.8× faster** |
| Qwen3-0.6B + vLLM | 30.046 s | 3.368 s | **8.9× faster** |

```mermaid
xychart-beta
    title "Time to ready (seconds; lower is better)"
    x-axis [CPU, vLLM]
    y-axis "seconds" 0 --> 32
    bar [2.102, 30.046]
    bar [0.164, 3.368]
```

The first bar in each pair is cold start; the second is restore. Exact numbers vary with the host, model, image storage, integrity verification, and runtime configuration. See the [full benchmark report](V0.1-MILESTONE.md) for scope and limitations.

## Run your first restore in 60 seconds

This CPU-only demo starts a warm stateful process, checkpoints it, removes the original process, restores it, sends another request, and records measured timings.

```bash
cargo run --example resume
```

You should see:

```text
Edo Tensei — Resume a warm workload
✓ Warm-up completed
✓ Checkpoint created
✓ Original process disappeared
✓ Restored successfully
✓ Request completed after restore
Cold start:    <measured on your host>
Restore time:  <measured on your host>
Run report:    .edo/runs/<timestamp>-resume.json
```

The first demo needs Linux, Rust, Python 3, CRIU, and permission to execute CRIU through `sudo`. It does not need a GPU or model download.

For the equivalent CLI vocabulary:

```bash
cargo run -- demo resume
```

## What just happened?

```mermaid
flowchart LR
    A[Warm workload] --> B[Checkpoint]
    B --> C[Process failure]
    C --> D[Restore]
    D --> E[Request succeeds]
```

CRIU restored the process memory, counter, signal handlers, open report file, and execution context. The JSON report is local and disposable; checkpoint images are never committed.

## Progressive demos

| Example | Demonstrates | Requirements |
| --- | --- | --- |
| [00 — Hello checkpoint](examples/00_hello_checkpoint/) | One-command CPU resurrection | Linux + CRIU |
| [01 — Stateful process](examples/01_stateful_process/) | State and event continuity | Linux + CRIU |
| [02 — FastAPI resume](examples/02_fastapi_resume/) | Readiness and request draining | Python + FastAPI |
| [03 — Warm-start PyTorch](examples/03_pytorch_warm_start/) | Model initialization once | PyTorch + CUDA for full path |
| [04 — Resume vLLM](examples/04_vllm_resume/) | Compiled engine process group | CUDA + vLLM + patched CRIU |
| [05 — Restore CUDA](examples/05_cuda_restore/) | Native CUDA state | NVIDIA checkpoint API |
| [06 — Triton snapshot](examples/06_triton_snapshot/) | Container inference server | Docker + NVIDIA Toolkit |
| [08 — Kubernetes migration](examples/08_kubernetes_migration/) | DaemonSet agent and same-node Pod restore | k3s/Kubernetes + GPU |

## Project structure

The repository remains a single Rust crate for v0.1, with logical boundaries around the product experience:

```text
src/          Rust CLI and runtime modules
examples/     progressive, user-oriented demos
docs/         getting started, concepts, compatibility, operations
examples/08_kubernetes_migration/   node-local agent and Pod restore manifests
assets/       project artwork
.edo/         local reports and ignored checkpoints
```

The former implementation-oriented demos are now merged into the progressive paths. See [the migration guide](docs/migration.md) for the old-to-new mapping.

## Capabilities

```bash
cargo run -- doctor
cargo run -- doctor --json
```

The CLI can launch managed processes, perform CPU CRIU snapshots, coordinate CUDA-before-CRIU freeze/summon, validate manifests, and emit structured diagnostics. See the [CLI reference](docs/cli.md).

## Validated scope

The narrow v0.1 path is Linux x86_64, one compatible NVIDIA GPU, and same-host restore. The H100/CUDA 12.8 environment has validated native CUDA, PyTorch, FastAPI, vLLM, SGLang, Kubernetes GPU, and snapshot reporting experiments. Triton snapshot creation works; native Docker namespace restore remains open.

Multi-GPU, distributed workers, cross-node container migration, in-flight requests, guaranteed KV-cache preservation, and persistent VRAM are not promised.

See the [compatibility matrix](docs/compatibility.md), [architecture](docs/concepts/architecture.md), and [troubleshooting guide](docs/troubleshooting.md).

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Read [CONTRIBUTING.md](CONTRIBUTING.md), [the v0.1 report](V0.1-MILESTONE.md), and [the roadmap](roadmap-todo.md). `demo.md` is the product-experience design brief that defines this progressive structure.
