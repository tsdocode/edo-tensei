# Architecture

Edo is an orchestration layer over two checkpoint systems:

```text
edo CLI
  └── runtime coordinator
       ├── process manager ── /proc and CRIU
       ├── CUDA backend ───── libcuda.so checkpoint API
       └── snapshot manifest ─ metadata, integrity, compatibility
```

The freeze order is important:

```text
quiesce application → lock CUDA → checkpoint CUDA → CRIU dump → write manifest
```

Restore reverses the ownership handoff:

```text
CRIU restore → CUDA restore → unlock CUDA → application readiness
```

Framework adapters own request draining and readiness. Edo does not guess whether arbitrary application work is safe to freeze.

## State ownership

- CPU process state belongs to CRIU.
- CUDA context and device allocations belong to the CUDA checkpoint backend.
- Model weights, compiled kernels, graphs, and KV cache are application state. Their portability must be validated by each integration.

The v0.1 implementation intentionally targets one Linux process or tested process group, one NVIDIA GPU, and same-host restore. Persistent VRAM, separate immutable weight artifacts, and cross-node migration are later architecture tracks.
