# Triton GPU snapshot demo

This demo runs NVIDIA Triton Server's Python backend with a GPU-resident
PyTorch model, warms it up, and provides a target for Edo CUDA/CRIU snapshot
and restore testing.

The model repository is mounted into the official Triton container:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD/examples/triton-snapshot:/models" \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
```

The HTTP endpoint is `http://127.0.0.1:8000/v2/health/ready` and inference
uses `/v2/models/gpu_warmup/infer`.
