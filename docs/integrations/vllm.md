# vLLM integration

The one-GPU process-group adapter is [`examples/04_vllm_resume`](../../examples/04_vllm_resume/). It covers the API parent and `VLLM::EngineCore`, normal warmup, CUDA graph state, and post-restore serving. Use only with a dedicated server because the full snapshot test is destructive.

For the large-model path, see [`examples/07_vllm_gemma31b_qat`](../../examples/07_vllm_gemma31b_qat/). It applies the same adapter to Gemma 3 27B QAT (a 31B-class workload), including the optional release-and-recreate KV-cache boundary.

## Async scheduler requirement

When vLLM asynchronous scheduling is enabled, use the Edo CRIU fork rather
than stock CRIU. The async worker owns `io_uring` descriptors and mappings;
the tested restore path depends on the fork's `port-io-uring` branch:

```bash
git clone --branch port-io-uring https://github.com/tsdocode/criu.git edo-criu
cd edo-criu
make -C criu -j"$(nproc)"
export EDO_CRIU="$PWD/criu/criu"
```

Without this fork, vLLM async snapshots can fail during CRIU's `io_uring` VMA
or descriptor handling. The fork is an experimental compatibility branch and
is not a claim that all Linux `io_uring` feature combinations are supported.

```bash
./examples/04_vllm_resume/run.sh --help
```
