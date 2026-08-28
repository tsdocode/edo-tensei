# 07 · Large vLLM Gemma QAT snapshot

This example applies Edo Tensei to a large, quantization-aware-trained Gemma
3 model served by vLLM. It is the heavyweight progression after the small
Qwen3/vLLM proof: the model remains loaded, compiled, CUDA-warmed, and ready
for restore, while the KV-cache can be released and recreated separately.

> Naming note: the published Gemma 3 instruction QAT checkpoint is a 27B
> model, not literally 31B. This example calls it “31B-class” to match the
> requested workload size and keeps the exact model ID configurable.

## What this proves

- vLLM serves a large QAT model through the OpenAI-compatible API;
- model load, torch compilation, CUDA graph capture, and warm-up happen once;
- Edo snapshots the API parent and `VLLM::EngineCore` as one process group;
- the CRIU fork records the process memory and io_uring state;
- restore does not reload the model or recapture CUDA graphs; and
- vLLM can recreate a fresh, larger KV-cache after restore when the snapshot
  boundary releases KV backing.

This is a destructive test of a dedicated server. Do not run it against a
production vLLM process.

> **Required for this async run:** use the Edo CRIU fork built from
> [`tsdocode/criu:port-io-uring`](https://github.com/tsdocode/criu/tree/port-io-uring)
> and export `EDO_CRIU` to that binary. Stock CRIU does not restore the
> `io_uring` descriptors/mappings used by vLLM's asynchronous scheduler.

## Hardware and software gate

The QAT model is large even after quantization. Use a GPU/node with enough
VRAM for model weights, runtime workspace, CUDA graphs, and the desired KV
cache. Multi-GPU tensor parallelism is not implemented by this wrapper; the
default path is one GPU because the current Edo process-group mapping is
single-rank.

Required:

- Linux and a CUDA-compatible NVIDIA GPU;
- vLLM with the Gemma 3 architecture and QAT quantization support;
- a Hugging Face token if the model or license requires gated access;
- the patched CRIU fork with io_uring support;
- a built `target/debug/edo` or `EDO_BIN` path; and
- enough local disk for model weights plus the checkpoint pages image.

Install the model into the normal Hugging Face cache before measuring, or set
`HF_HOME` to a persistent high-capacity volume. Model download time is not
part of the restore measurement.

## Run

Inspect the exact command and environment without launching the server:

```bash
./examples/07_vllm_gemma31b_qat/run.sh --help
```

Run the complete cold-start → warm-up → snapshot → restore → serving flow:

```bash
export HF_TOKEN=...
export EDO_VLLM_COMMAND=/path/to/vllm
export EDO_BIN=target/debug/edo
./examples/07_vllm_gemma31b_qat/run.sh --port 18087
```

This example forces vLLM's `TRITON_ATTN` backend through
`--attention-config '{"backend":"TRITON_ATTN"}'`. Triton attention is useful
here because it avoids depending on the incompatible prebuilt FlashAttention
PTX in the current CUDA 12.8 environment while keeping the normal vLLM
attention/KV-cache path under test. It does not replace the Marlin kernel used
by compressed-tensors QAT weights; that kernel must also be compiled for the
installed CUDA/driver pair.

Use a local model mirror or another Gemma QAT export when needed:

```bash
EDO_GEMMA_QAT_MODEL=/models/gemma-3-27b-it-qat \
EDO_GEMMA_MAX_MODEL_LEN=8192 \
./examples/07_vllm_gemma31b_qat/run.sh --port 18087
```

For the small-snapshot experiment, release KV backing before checkpointing,
restore trusted local images without re-hashing, and explicitly opt into the
experimental buffered io_uring restore path:

```bash
./examples/07_vllm_gemma31b_qat/run.sh \
  --port 18087 \
  --release-kv-cache \
  --fast-restore \
  --io-uring-restore
```

`--fast-restore` skips integrity hashing and must only be used when the
snapshot directory is protected and authenticated by the deployment. The
default command keeps verification enabled.

The wrapper defaults to `--gpu-memory-utilization 0.80` so it leaves headroom
for existing CUDA allocations on the host. Increase it with
`EDO_GEMMA_GPU_MEMORY_UTILIZATION` only after checking `nvidia-smi`; vLLM
rejects startup when the requested fraction exceeds currently free VRAM.

## Architecture

```mermaid
flowchart TD
    A[Gemma 3 QAT weights] --> B[vLLM API parent]
    B --> C[VLLM::EngineCore]
    C --> D[Model load + torch.compile]
    D --> E[CUDA graph capture + warm-up]
    E --> F{Snapshot boundary}
    F -->|default| G[Weights + runtime + KV pages]
    F -->|--release-kv-cache| H[Weights + runtime only]
    G --> I[CRIU images + CUDA state]
    H --> I
    I --> J[Kill source process group]
    J --> K[CRIU restore + CUDA restore]
    K --> L{KV backing present?}
    L -->|yes| M[Health + inference]
    L -->|no| N[Wake weights]
    N --> O[Create fresh KV cache]
    O --> M
    M --> P[Ready to serve without model reload]
```

## Step-by-step lifecycle

1. The wrapper starts vLLM with the configured Gemma QAT model, one tensor
   parallel rank, bounded context length, and high GPU utilization.
2. The adapter waits for `/health`, then calls `/v1/models`, a warm completion,
   and a streaming completion to establish cold readiness, warm inference, and
   TTFT.
3. vLLM's model memory, compiled kernels, CUDA graphs, scheduler state, and
   KV allocator are now in the process group. This is the important snapshot
   point: the expensive work has already happened.
4. Edo discovers the API parent and `VLLM::EngineCore`, locks their CUDA
   owners, and invokes `freeze-group`. CRIU captures the process tree,
   descriptors, memory mappings, and io_uring rings.
5. The source process group is terminated so the restore is a real failure
   recovery rather than a second copy running beside the source.
6. `summon-group` restores CRIU first, restores CUDA state, and waits for the
   restored owners to run. With `--release-kv-cache`, the adapter then wakes
   weights and creates an empty KV cache before probing readiness.
7. The adapter sends a new completion and streaming request. It reports
   restore-to-health, restore-to-warm-inference, post-restore TTFT, and the
   TTFT delta.

## Before and after report

The script prints the measured values for the current GPU, vLLM build,
checkpoint storage, model configuration, and verification mode. Do not copy
small-model numbers into this large-model result.

| Signal | Cold start | After restore |
| --- | ---: | ---: |
| Model download | excluded when cache is warm | none |
| Model load | required | not repeated |
| torch.compile | required | not repeated |
| CUDA graph capture | required | not repeated |
| KV cache | allocated during startup | reused or recreated, depending on flag |
| `/health` | measured by script | measured by script |
| Warm completion | measured by script | measured by script |
| Streaming TTFT | measured by script | measured by script |
| Snapshot size | measured by script | same artifact |

The latest real recording reports the end-to-end **total time to ready** (the
moment the warm inference has completed, not merely when `/health` responds):

| Run | Total time to ready | Valid output |
| --- | ---: | --- |
| Cold boot | **104.158 s** | `Ready.` |
| CRIU + CUDA restore | **11.060 s** | `Ready.` |

The same run measured `/health` at 104.066 s cold and 11.007 s after restore.
The restored process kept its model/runtime state, recreated a fresh KV-cache,
and returned a post-restore TTFT of 0.034 s (cold TTFT: 0.025 s).

Re-run the command whenever the GPU, vLLM build, model, or storage changes;
the adapter prints fresh values and the checked-in numbers above should then
be treated as the reference run for this repository.

## Snapshot size and KV-cache interpretation

The full snapshot can be very large because process memory includes model
weights, allocator state, compiled runtime state, CUDA graph pools, and KV
pages. The `--release-kv-cache` path enables vLLM Sleep Mode and uses level 1
at the boundary:

1. model weights stay resident in the snapshot contract;
2. vLLM releases the reclaimable KV backing before CRIU hashes/captures the
   image;
3. after restore, weights are woken first; and
4. a fresh KV allocation is initialized before readiness is reported.

This can make the artifact much smaller, but it does not preserve active
conversations or token history in KV. On the tested vLLM build, most of the
40.18GiB KV allocation was released after the allocator was correctly enabled,
but the resulting image still contains CPU-backed weights and other CUDA
runtime/graph allocations. A CUDA VMM/GMS-style allocation export is still
needed for a true weights-only artifact with a large KV cache recreated after
restore.
Existing requests must be drained before the snapshot, and clients should
retry through the serving layer after restore.

## Failure diagnosis

| Symptom | First check |
| --- | --- |
| OOM during startup | lower `EDO_GEMMA_MAX_MODEL_LEN`, reserve less KV, or use more VRAM |
| Unsupported QAT format | confirm the vLLM version and override `EDO_GEMMA_QAT_MODEL` |
| No `VLLM::EngineCore` found | inspect the process tree and vLLM multiprocessing mode |
| Restore fails on `io_uring` | use the patched CRIU fork and `--io-uring-restore`; do not mix CRIU binaries |
| Restore succeeds but API is not ready | verify CUDA unlock, then `/health` and a real completion |
| Restore is dominated by hashing | use trusted local `--fast-restore` for diagnosis only |
| Post-restore OOM with large KV | recreate KV with a lower budget or increase available VRAM |

## Cleanup and limitations

The adapter terminates the server process group on exit, but inspect the GPU
and process table after an interrupted run:

```bash
nvidia-smi
ps -ef | grep -E 'vllm|VLLM::EngineCore' | grep -v grep || true
```

This example is single-GPU, same-host, and experimental for a large QAT model.
Cross-node restore, multi-rank tensor parallelism, active KV migration,
automatic Service cutover, and production checkpoint encryption remain outside
its scope.

### Current host validation

On the development H100 (driver 570.211.01, CUDA 12.8), vLLM selected
`TRITON_ATTN` successfully and loaded all four 16.83GiB checkpoint shards.
The CUDA 12.8-compatible vLLM extension and the patched CRIU fork completed a
real dump/restore with asynchronous vLLM scheduling and `io_uring` enabled.

| Signal | Cold start | After restore |
| --- | ---: | ---: |
| `/health` ready | 103.052s | 11.055s |
| Warm completion | 0.098s | 0.052s |
| TTFT | 0.027s | 0.025s |
| Snapshot artifact | — | ~32GiB |
| Model reload / compile / graph capture | required | not repeated |
| Output validation | `Ready.` | `Ready.` |

### Cold-boot interpretation

The initial Gemma experiment took approximately 180s for cold boot because it
also created the first torch.compile/autotune artifacts. After those artifacts
were cached, the latest measured cold boot was 103.052s. The restore number of
11.055s is from that same warm-cache run and includes CUDA/CRIU restore plus
fresh KV-cache wake-up.

The new allocator-enabled run used `EDO_GEMMA_GPU_MEMORY_UTILIZATION=0.68`
because unrelated services occupied about 9.5GiB on the H100. It measured
103.052s cold `/health`, 11.055s restore to `/health`, and a 32GiB snapshot.
vLLM reported 59.04GiB released before CRIU (40.23GiB discarded and 18.81GiB
backed up on CPU). The previous run had a ~71GiB image because the adapter did
not pass `--enable-sleep-mode`; this is a 39GiB / 55% reduction. The restored
completion was valid and no model reload, torch.compile, or CUDA-graph
recapture occurred. The adapter timeout was also increased for the multi-GiB
weight-to-host copy.

```mermaid
xychart-beta
    title "Gemma QAT: cold boot vs restore (H100)"
    x-axis [Cold boot, Restore]
    y-axis "Seconds" 0 --> 110
    bar [103.052, 11.055]
```

The earlier ~71GiB run remains useful as the “sleep requested but allocator not
enabled” baseline. The remaining size problem is GPU allocator/VMM ownership,
not Triton attention: CRIU still sees approximately 32GiB of resident process
state after KV release.

### Terminal comparison video

For a quick presentation-friendly replay of the measured comparison, run:

```bash
python3 examples/07_vllm_gemma31b_qat/make_terminal_video.py
```

This creates [`artifacts/gemma-cold-vs-restore.mp4`](../../artifacts/gemma-cold-vs-restore.mp4).
The equivalent lightweight GIF is available at
[`artifacts/gemma-cold-vs-restore.gif`](../../artifacts/gemma-cold-vs-restore.gif).
The 26-second video is time-compressed; the timings printed inside it are the
real benchmark values, not a second live model run.

Next: [08 · Kubernetes migration](../08_kubernetes_migration/README.md).
