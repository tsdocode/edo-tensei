# GPU model snapshot demo

This example loads a deterministic model-sized weight tensor into CUDA device
memory, snapshots it with `edo freeze`, restores it with `edo summon`, and
compares the GPU checksum before and after restore.

Run it from the repository root:

```bash
./examples/gpu-model-snapshot/run-demo.sh
```

Use a larger model-sized allocation when desired:

```bash
EDO_MODEL_MB=512 ./examples/gpu-model-snapshot/run-demo.sh
```

The fixture uses the CUDA driver API directly so it works with the installed
CUDA 12.8 driver without replacing the CPU-only PyTorch environment. It models
the critical property of a framework model: persistent device-resident weights
with a verifiable output checksum.

The report includes model bytes, checksum before/after, freeze time, and
restore time.
