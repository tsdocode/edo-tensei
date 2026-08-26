# FastAPI heavy-startup demo

This server loads a real Hugging Face Transformer onto CPU during FastAPI
lifespan startup, then exposes `/health`, `/ready`, `/state`, and `/infer`.
`POST /quiesce` stops new inference and waits for active requests to drain;
`POST /resume` reopens readiness after restore.

Run the checkpoint demo from the repository root:

```bash
./examples/fastapi-heavy-startup/run-demo.sh
```

The default model is `sshleifer/tiny-distilbert-base-cased`. Select a larger
model to make cold startup more visible:

```bash
EDO_HF_MODEL=distilbert-base-uncased EDO_STARTUP_SECONDS=8 \
  ./examples/fastapi-heavy-startup/run-demo.sh
```

The Python dependencies are CPU-only PyTorch, Transformers, and safetensors.
On Python 3.14, install PyTorch from the CPU index before the other packages:

```bash
python3 -m pip install --user torch==2.13.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user 'transformers>=5.15' 'safetensors>=0.8'
```

Useful controls:

```bash
EDO_HF_MODEL=sshleifer/tiny-distilbert-base-cased
EDO_USE_HF_MODEL=1              # set to 0 for the CRIU-safe byte-buffer fallback
EDO_STARTUP_SECONDS=8           # artificial delay, in addition to model loading
EDO_MODEL_MB=64                 # only used by the fallback
```

The demo verifies readiness is withdrawn before checkpoint, restored before
inference resumes, and model checksum/inference before and after restore. The
Hugging Face model load is proven checkpointable only if the CRIU round trip
passes on the host; the fallback remains available for the CRIU CPU fixture.
