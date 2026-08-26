# vLLM snapshot adapter probe

This is the first vLLM integration layer for Edo. It starts a dedicated
single-GPU vLLM server, performs a real warmup inference with the normal CUDA
graph path enabled, and discovers its worker process group:

```bash
./examples/vllm-snapshot/run-demo.sh
```

Use a different model, port, or launcher with environment variables:

```bash
EDO_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
EDO_VLLM_COMMAND=/path/to/vllm \
  ./examples/vllm-snapshot/run-demo.sh --port 18080
```

If vLLM is installed in another mount namespace, use a command wrapper such
as `EDO_VLLM_COMMAND='sudo nsenter -t <existing-vllm-pid> -m --
/usr/local/bin/vllm'`.

The adapter does not use Sleep Mode. It waits for the server, sends a warmup
request, and declares the process snapshot-ready only after warmup. With
`--full-snapshot`, it calls `freeze-group` and `summon-group` for the API
process plus `VLLM::EngineCore`, then sends a post-restore request. vLLM has
an API
parent plus engine worker processes, IPC, and potentially distributed state;
the full Edo snapshot must freeze and restore that group together.

The safe implementation target is one GPU and tensor parallel size 1 first.
Requests must be drained before freeze. The target snapshot should preserve
the warmed weights, CUDA context, compiled kernels, CUDA graphs, and any
other validated engine state. In-flight requests and KV-cache portability are
not promised until separately tested.

For a non-mutating view of an already running server:

```bash
python3 examples/vllm-snapshot/vllm_adapter.py \
  --port 8000 --pid 3669047 --inspect-only
```

Inspection never launches or mutates a server. Replace the PID with the API
parent process you want to inspect.

## Host test result

On the test H100 host, the adapter successfully launched Qwen 0.5B with vLLM
`0.23.1rc1` and `0.26.0`, loaded the model, served `/health` and `/v1/models`,
and discovered the API parent plus `VLLM::EngineCore`. The normal startup path
captured CUDA graphs and served a warmup request successfully.

The grouped freeze/restore protocol is implemented. vLLM 0.26.0 in the
available CUDA 13 environment is not compatible with this host's CUDA process
checkpoint driver: `cuCheckpointProcessGetState` returns undocumented driver
values before locking. The tested CUDA 12.8 vLLM environment uses the host
570 driver and passes the CUDA state phase for both the API parent and engine
worker. Use that matching environment while validating the remaining CRIU
IPC behavior:

```bash
EDO_VLLM_COMMAND='sudo nsenter -t <cuda-12.8-vllm-pid> -m -p -- /usr/local/bin/vllm' \
  ./examples/vllm-snapshot/run-demo.sh --port 18080 --full-snapshot
```

The CUDA 12.8 run reached CRIU, but the H100 test still needs a clean,
dedicated GPU because CRIU rejected the vLLM process group. No full vLLM
restore is claimed until that CRIU phase passes.

Run the destructive group test only on a dedicated vLLM server/GPU:

```bash
EDO_VLLM_COMMAND='sudo nsenter -t <vllm-pid> -m -p -- /usr/local/bin/vllm' \
  ./examples/vllm-snapshot/run-demo.sh --port 18080 --full-snapshot
```

The test is intentionally destructive to the dedicated server: it freezes the
API parent and `VLLM::EngineCore`, restores the CRIU group, waits for both CUDA
owners to return to `RUNNING`, and sends a post-restore inference request.
