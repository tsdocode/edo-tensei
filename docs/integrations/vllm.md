# vLLM integration

The one-GPU process-group adapter is [`examples/04_vllm_resume`](../../examples/04_vllm_resume/). It covers the API parent and `VLLM::EngineCore`, normal warmup, CUDA graph state, and post-restore serving. Use only with a dedicated server because the full snapshot test is destructive.

```bash
./examples/04_vllm_resume/run.sh --help
```
