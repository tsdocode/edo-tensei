# Example migration guide

The old examples remain intact so existing commands do not suddenly break. New users should use the progressive names below.

| Old path | New path |
| --- | --- |
| `examples/cpu-checkpoint` | `examples/00_hello_checkpoint` |
| `examples/cpu-counter.py` | `examples/01_stateful_process` |
| `examples/fastapi-heavy-startup` | `examples/02_fastapi_resume` |
| `examples/modal-gpu-snapshot` | `examples/03_pytorch_warm_start` |
| `examples/vllm-snapshot` | `examples/04_vllm_resume` |
| `examples/cuda-checkpoint` | `examples/05_cuda_restore` |
| `examples/triton-snapshot` | `examples/06_triton_snapshot` |
| `kubernetes/*` | `examples/07_kubernetes_migration` + `kubernetes/` |

The old paths are implementation-oriented compatibility entry points. New demos use the product vocabulary: resume a process, resume a service, warm-start a model, and migrate a workload.
