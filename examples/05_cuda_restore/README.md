# 05 · Restore CUDA state

Start with the smallest native CUDA proof: lock a CUDA process, checkpoint its GPU allocation, let CRIU capture the CPU process, then restore and verify the same device data.

## Run

Prerequisites: Linux x86_64, CUDA toolkit, NVIDIA driver with checkpoint symbols, CRIU, GCC, and `sudo` permission. See [CUDA integration notes](../../docs/integrations/cuda.md).

```bash
./examples/05_cuda_restore/run.sh
```

For a larger model-shaped allocation:

```bash
EDO_MODEL_MB=512 ./examples/05_cuda_restore/run-model-demo.sh
```

## Architecture, step by step

```text
native CUDA fixture
       ↓ cuMemAlloc + deterministic pattern
       ↓ signal verifies GPU pattern
edo freeze
       ├─ CUDA lock → checkpoint (GPU state held by driver)
       └─ CRIU dump (CPU process state)
edo summon
       ├─ CRIU restore
       └─ CUDA restore → unlock → signal verifies pattern
```

1. The fixture creates a CUDA context and writes a deterministic allocation.
2. A signal-triggered check establishes the before value.
3. `edo freeze` runs CUDA checkpoint before CRIU dump and records a manifest.
4. `edo summon` restores the CPU process, asks CUDA to restore the GPU state, and unlocks it only after the expected state is reached.
5. The second signal verifies the device checksum after restore.

## Result

Validated 512 MiB model-sized allocation on H100/CUDA 12.8:

| Metric | Before | After |
| --- | ---: | ---: |
| GPU allocation | 512 MiB | 512 MiB restored |
| Checksum | `17193905543863665539` | identical |
| Freeze | — | 2.043 s |
| Restore | — | 1.534 s |
| Verification | pattern valid | pattern valid |

The scripts print the exact values for the current host. The model-sized fixture is a CUDA allocation proof, not a PyTorch or vLLM benchmark.

## What is checkpointed?

The CUDA context/device allocation and the native process state, coordinated in the required CUDA-before-CRIU order.

## Limitations and cleanup

Single GPU and same compatible host. The fixture does not represent multi-process CUDA IPC, NCCL, UVM, or cross-GPU migration. Compiled fixtures and temporary snapshots are removed automatically.

Next: [06 · Triton snapshot](../06_triton_snapshot/README.md).
