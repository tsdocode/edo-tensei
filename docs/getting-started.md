# Getting started

Edo Tensei checkpoints a live AI workload, removes the process, and restores it with its runtime state intact.

## First restore in 60 seconds

```bash
cargo run --example resume
```

The first demo is CPU-only and needs Linux, CRIU, and permission to run the CRIU commands through `sudo`. It creates a disposable checkpoint and a JSON report under `.edo/runs/`.

## Choose a path

| Goal | Start here | Requirements |
| --- | --- | --- |
| Learn the core idea | [`00_hello_checkpoint`](../examples/00_hello_checkpoint/) | Linux + CRIU |
| Resume an HTTP service | [`02_fastapi_resume`](../examples/02_fastapi_resume/) | Python + FastAPI |
| Warm-start a model | [`03_pytorch_warm_start`](../examples/03_pytorch_warm_start/) | PyTorch + NVIDIA GPU for full path |
| Resume vLLM | [`04_vllm_resume`](../examples/04_vllm_resume/) | CUDA + vLLM + patched CRIU |
| Understand CUDA | [`05_cuda_restore`](../examples/05_cuda_restore/) | CUDA + NVIDIA checkpoint API |
| Try containers | [`08_kubernetes_migration`](../examples/08_kubernetes_migration/) | k3s/Kubernetes + GPU |

## Before troubleshooting

```bash
cargo run -- doctor
cargo test
```

See [compatibility](compatibility.md), [CLI](cli.md), and [troubleshooting](troubleshooting.md) for details.
