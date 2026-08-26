#!/usr/bin/env python3
"""Warm a Hugging Face causal LM on CUDA and verify it across Edo restore."""

import hashlib
import os
import signal
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("EDO_HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = torch.float16
STARTUP_MARKER = os.environ.get("EDO_STARTUP_MARKER")
verify_requested = True


def request_verify(_signum, _frame):
    global verify_requested
    verify_requested = True


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_ID.encode())
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        sample = parameter.detach().view(torch.uint8).flatten()[:4096].cpu().numpy().tobytes()
        digest.update(sample)
    return digest.hexdigest()


def main():
    global verify_requested
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to("cuda").eval()
    prompt = os.environ.get("EDO_PROMPT", "Explain GPU memory snapshots in one sentence.")
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**tokens, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    if STARTUP_MARKER:
        with open(STARTUP_MARKER, "a", encoding="utf-8") as marker:
            marker.write("model-initialized\n")
    warmup_text = tokenizer.decode(output[0], skip_special_tokens=True)
    checksum = model_checksum(model)
    print(f"model-ready pid={os.getpid()} model={MODEL_ID} device={torch.cuda.get_device_name()} parameters={sum(p.numel() for p in model.parameters())} startup_seconds={time.monotonic() - started:.3f}", flush=True)
    print(f"warmup-output={warmup_text!r}", flush=True)
    signal.signal(signal.SIGUSR1, request_verify)

    while True:
        if verify_requested:
            current = model_checksum(model)
            if current != checksum:
                raise RuntimeError(f"model checksum mismatch: expected {checksum}, got {current}")
            print(f"gpu-model-checksum pid={os.getpid()} model={MODEL_ID} checksum={current}", flush=True)
            verify_requested = False
        time.sleep(1)


if __name__ == "__main__":
    main()
