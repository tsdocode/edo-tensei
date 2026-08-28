# Compatibility

## Validated

| Component | Validated configuration |
| --- | --- |
| OS/CPU | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA H100 80 GB, one GPU |
| Driver/CUDA | Driver 570.211.01, CUDA 12.8 |
| CRIU | 4.2.1 with required privileges |
| vLLM | 0.23.1rc1, tensor parallel size 1 |
| SGLang | 0.5.9, one GPU |
| Triton | 25.05 Python backend, snapshot creation only |
| Kubernetes | Local k3s, NVIDIA device plugin, same-node placeholder restore |

## Scope limits

AMD, Windows, macOS, multi-GPU/NCCL, distributed workers, arbitrary cross-node container migration, in-flight requests, and guaranteed KV-cache preservation are not v0.1 targets.

CUDA restore requires a compatible NVIDIA driver and enough device memory. CRIU also needs the host permissions reported by `edo doctor`.
