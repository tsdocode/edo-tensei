# FastAPI integration

The runnable workload is [`examples/02_fastapi_resume`](../../examples/02_fastapi_resume/). It loads a CPU model during lifespan startup and exposes `/health`, `/ready`, `/state`, and `/infer`. Call `/quiesce` before checkpointing and `/resume` after restore.

```bash
./examples/02_fastapi_resume/run.sh
```
