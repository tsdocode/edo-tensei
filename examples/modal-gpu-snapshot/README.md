# Modal-style GPU memory snapshot demo

This reproduces the core workflow described in Modal's GPU Memory Snapshots
article: load a Hugging Face model directly onto CUDA, run a warmup generation,
checkpoint GPU state before serving, restore it, and compare model state and
latency before/after.

Run it from the repository root:

```bash
./examples/modal-gpu-snapshot/run-qwen-demo.sh
```

The dedicated environment is expected at:

```text
/home/ubuntu/miniconda3/envs/edo-models
```

Override the model or Python interpreter when needed:

```bash
EDO_HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
EDO_MODEL_PYTHON=/home/ubuntu/miniconda3/envs/edo-models/bin/python \
  ./examples/modal-gpu-snapshot/run-qwen-demo.sh
```

Unlike Modal's managed gVisor/container snapshot service, this local demo uses
Edo's CUDA checkpoint API plus CRIU directly. It reports the model identifier,
parameter count, warmup output, checksum before/after restore, freeze time, and
restore time.
