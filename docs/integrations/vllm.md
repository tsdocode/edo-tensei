# vLLM integration

The one-GPU process-group adapter is [`examples/04_vllm_resume`](../../examples/04_vllm_resume/). It covers the API parent and `VLLM::EngineCore`, normal warmup, CUDA graph state, and post-restore serving. Use only with a dedicated server because the full snapshot test is destructive.

For the large-model path, see [`examples/07_vllm_gemma31b_qat`](../../examples/07_vllm_gemma31b_qat/). It applies the same adapter to Gemma 3 27B QAT (a 31B-class workload), including the optional release-and-recreate KV-cache boundary.

```bash
./examples/04_vllm_resume/run.sh --help
```
