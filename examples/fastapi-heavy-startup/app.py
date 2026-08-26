#!/usr/bin/env python3
"""FastAPI process with deliberately expensive, checkpointable startup state."""

import hashlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

STARTUP_SECONDS = float(os.environ.get("EDO_STARTUP_SECONDS", "8"))
MODEL_MB = int(os.environ.get("EDO_MODEL_MB", "64"))
MODEL_SEED = int(os.environ.get("EDO_MODEL_SEED", "20260826"))
HF_MODEL_ID = os.environ.get("EDO_HF_MODEL", "sshleifer/tiny-distilbert-base-cased")
USE_HF_MODEL = os.environ.get("EDO_USE_HF_MODEL", "1") == "1"

state: dict[str, Any] = {
    "ready": False,
    "startup_started_at": None,
    "startup_finished_at": None,
    "startup_duration_seconds": None,
    "model_bytes": 0,
    "model_checksum": None,
    "model_name": None,
    "model_parameters": 0,
    "request_count": 0,
    "model": None,
}


def build_model() -> Any:
    started = time.monotonic()
    state["startup_started_at"] = time.time()
    time.sleep(STARTUP_SECONDS)

    if USE_HF_MODEL:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModel.from_pretrained(HF_MODEL_ID).to("cpu").eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        checksum = hashlib.sha256()
        checksum.update(HF_MODEL_ID.encode())
        for name, parameter in model.named_parameters():
            checksum.update(f"{name}:{tuple(parameter.shape)}".encode())
            checksum.update(repr(parameter.detach().flatten()[:64].tolist()).encode())
        state["model_name"] = HF_MODEL_ID
        state["model_parameters"] = parameter_count
        state["model_bytes"] = model_bytes
        state["model_checksum"] = checksum.hexdigest()
        model = {"model": model, "tokenizer": tokenizer}
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    else:
        # Keep a deterministic resident model without NumPy's io_uring-backed
        # allocator, which CRIU cannot dump on this host.
        target_bytes = max(MODEL_MB, 1) * 1024 * 1024
        block = hashlib.sha256(f"edo-model-{MODEL_SEED}".encode()).digest()
        model = bytearray((block * ((target_bytes // len(block)) + 1))[:target_bytes])
        state["model_name"] = "deterministic-byte-buffer"
        state["model_parameters"] = 0
        state["model_bytes"] = len(model)
        state["model_checksum"] = hashlib.sha256(model).hexdigest()

    state["startup_finished_at"] = time.time()
    state["startup_duration_seconds"] = round(time.monotonic() - started, 3)
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["model"] = build_model()
    state["ready"] = True
    yield
    state["ready"] = False
    state["model"] = None


app = FastAPI(title="Edo Tensei FastAPI Heavy Startup Demo", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="startup is still running")
    return {"status": "ready"}


@app.get("/state")
def snapshot_state() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "ready": state["ready"],
        "startup_started_at": state["startup_started_at"],
        "startup_finished_at": state["startup_finished_at"],
        "startup_duration_seconds": state["startup_duration_seconds"],
        "model_bytes": state["model_bytes"],
        "model_checksum": state["model_checksum"],
        "model_name": state["model_name"],
        "model_parameters": state["model_parameters"],
        "request_count": state["request_count"],
    }


@app.get("/infer")
def infer(value: float = 1.0) -> dict[str, Any]:
    if not state["ready"] or state["model"] is None:
        raise HTTPException(status_code=503, detail="model is not ready")

    state["request_count"] += 1
    if USE_HF_MODEL:
        import torch

        model_bundle = state["model"]
        tokens = model_bundle["tokenizer"](str(value), return_tensors="pt")
        with torch.inference_mode():
            output = model_bundle["model"](**tokens).last_hidden_state
        result = float(output.mean().item())
    else:
        model = state["model"]
        sample = bytes(model[:4096]) + repr(value).encode()
        digest = hashlib.blake2b(sample, digest_size=8).hexdigest()
        result = int(digest, 16) / float(2**64)
    return {"value": value, "result": result, "request_count": state["request_count"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("EDO_HOST", "127.0.0.1"),
        port=int(os.environ.get("EDO_PORT", "8000")),
        log_level=os.environ.get("EDO_LOG_LEVEL", "info"),
        loop="asyncio",
        http="h11",
        ws="none",
    )
