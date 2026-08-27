# 03 · Warm-start PyTorch

Load a Hugging Face model once, warm inference, checkpoint the CUDA-backed process, and verify that restore does not execute model initialization again.

## Run

Prerequisites: Linux, Python with CUDA-enabled PyTorch and Transformers, an NVIDIA GPU exposing the CUDA checkpoint API, CRIU, and the model cache. See [PyTorch integration notes](../../docs/integrations/pytorch.md).

```bash
./examples/03_pytorch_warm_start/run.sh
```

Override the model or interpreter:

```bash
EDO_HF_MODEL=Qwen/Qwen3-0.6B \
EDO_MODEL_PYTHON=/path/to/python \
./examples/03_pytorch_warm_start/run.sh
```

## Architecture, step by step

```text
Python + Transformers
       ↓ model load (once)
       ↓ CUDA transfer + warm-up
       ↓ checksum before
edo freeze: CUDA lock → CUDA checkpoint → CRIU dump
edo summon: CRIU restore → CUDA restore → unlock
       ↓ checksum after + inference
```

1. `model.py` loads the selected model and writes a startup marker.
2. A warm-up generation initializes the framework and device allocations.
3. Edo freezes CUDA before asking CRIU to dump the process.
4. Edo restores CRIU first, restores CUDA state, and unlocks the process.
5. The script sends a post-restore signal and checks checksum plus exactly-once initialization.

## Result

Validated Qwen 0.5B GPU run on the H100 host:

| Metric | Before | After |
| --- | ---: | ---: |
| Model initialization | once | not repeated |
| GPU checksum | recorded | identical |
| Warm inference | succeeds | succeeds |
| Restore | — | measured by script |

The native model-sized CUDA fixture in example 05 reports its exact freeze and restore timings; this model demo reports timings from each run rather than claiming a universal number.

## What is checkpointed?

The Python process, model object graph, CUDA context, and device-resident allocations covered by the driver checkpoint. Framework-specific external resources need separate validation.

## Limitations and cleanup

Single GPU and same compatible host. KV-cache portability, multi-process serving, and cross-GPU migration are not guaranteed. Use a dedicated process; cleanup removes temporary images but leaves model caches.

Next: [04 · vLLM resume](../04_vllm_resume/README.md).
