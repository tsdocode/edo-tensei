# vLLM snapshot adapter probe

This is the first vLLM integration layer for Edo. It starts a dedicated
single-GPU vLLM server, discovers its worker process group, and validates the
pause boundary using vLLM Sleep Mode:

```bash
./examples/vllm-snapshot/run-demo.sh
```

Use a different model, port, or launcher with environment variables:

```bash
EDO_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
EDO_VLLM_BIN=/path/to/vllm \
  ./examples/vllm-snapshot/run-demo.sh --port 18080
```

The adapter sets `VLLM_SERVER_DEV_MODE=1`, enables Sleep Mode, pauses the
engine, and wakes it again. It does not call `edo freeze` yet. vLLM has an API
parent plus engine worker processes, IPC, and potentially distributed state;
the full Edo snapshot must freeze and restore that group together.

The safe implementation target is one GPU and tensor parallel size 1 first.
Requests must be drained before sleep or freeze, and KV-cache preservation is
not promised. vLLM's Sleep Mode itself discards KV cache at level 1 while
keeping weights available for wake-up. See the [vLLM Sleep Mode
documentation](https://docs.vllm.ai/en/latest/features/sleep_mode/).

For a non-mutating view of an already running server:

```bash
python3 examples/vllm-snapshot/vllm_adapter.py \
  --port 8000 --pid 3669047 --inspect-only
```

Inspection never launches or mutates a server. Replace the PID with the API
parent process you want to inspect.
