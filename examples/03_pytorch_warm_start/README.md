# 03 — Warm-start PyTorch

Load a Hugging Face model once, warm it up, and use the existing model snapshot fixture to verify initialization is not repeated after restore.

## Prerequisites

Python with PyTorch/Transformers and a CUDA-capable NVIDIA host for the full GPU path.

## Run

```bash
./run.sh
```

## Expected output

The report shows model initialization, warmup, freeze, restore, matching checksum, and no second model initialization.

## What is checkpointed?

The process memory and CUDA-resident model state when CUDA checkpointing is available.

## Limitations

Single process, single GPU, same-host compatibility. KV-cache and in-flight requests are not guaranteed.

## Cleanup

The script removes temporary images; model caches are left for reuse.

Next: [04 — vLLM resume](../04_vllm_resume/README.md).
