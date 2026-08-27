# 04 — Resume vLLM

Run the real one-GPU vLLM adapter after warmup, CUDA graph capture, and request draining.

## Prerequisites

The project vLLM environment, one NVIDIA GPU, the patched CRIU fork, and the model cache. See the original [vLLM guide](../../examples/vllm-snapshot/README.md).

## Run

```bash
./run.sh
```

## Expected output

The API and `VLLM::EngineCore` are restored as a group and a post-restore completion succeeds without model reload or graph recapture.

## What is checkpointed?

The vLLM process group, compiled runtime state, CUDA state, io_uring descriptors, and model memory covered by the tested snapshot.

## Limitations

One GPU and tensor-parallel size 1 only. KV-cache backing remains a known optimization area.

## Cleanup

Run only against a dedicated server; the full snapshot flow is destructive to that process.

Next: [05 — CUDA restore](../05_cuda_restore/README.md).
