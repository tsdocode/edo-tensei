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

On the test H100 host, the adapter launched Qwen2.5-0.5B-Instruct with vLLM
`0.23.1rc1`, loaded the model, captured CUDA graphs, and served a warmup
request. With other GPU workloads stopped, the complete grouped checkpoint
passed:

1. Before: `/health`, `/v1/models`, and warmup chat inference succeeded.
2. Edo locked and checkpointed the API parent and `VLLM::EngineCore`.
3. The patched CRIU fork dumped the process group, including its io_uring
   descriptors and IPC state.
4. The original process was reaped so CRIU could reuse its recorded PIDs.
5. After: CRIU restored the group, CUDA owners returned to `RUNNING`, and a
   post-restore warmup chat request returned `Ready.`.

The image set is approximately 12 GiB. SHA-256 manifest creation and
verification add several minutes to the operation; this is expected for the
current integrity-first prototype.

Measured on the successful adapter run:

| Metric | Latency |
|---|---:|
| Cold launch to `/health` | 30.046 s |
| Cold launch to first warm inference | 30.080 s |
| Integrity-verified restore to `/health` | 254.678 s |
| Integrity-verified restore to first warm inference | 254.697 s |
| Trusted fast restore to `/health` | 7.978 s |
| Trusted fast restore to first warm inference | 7.996 s |
| Fast restore with KV-cache release to `/health` | 6.524 s |
| Fast restore with KV-cache release to first warm inference | 6.542 s |
| Fast restore with 2 GiB KV budget + KV release to `/health` | 5.046 s |
| Fast restore with 2 GiB KV budget + KV release to first warm inference | 5.063 s |
| Post-restore inference request | 0.018 s |

The restore-to-health number includes checkpoint SHA-256 verification and
CRIU image restoration. It does not include model loading, torch compilation,
or CUDA graph capture after restore: those occurred before the snapshot and
were restored from checkpointed state.

Use `--fast-restore` for a trusted local snapshot to skip rereading the page
images for SHA-256. The same run reached health in 7.978 seconds and served
the first post-restore inference in 7.996 seconds. Metadata, host/GPU checks,
file existence, and file-size checks remain enabled; use full verification for
snapshots from an untrusted or mutable source.

With `--release-kv-cache`, vLLM freed 2.11 GiB before checkpointing, producing
a 9.8 GiB artifact. The restored KV cache woke in 0.005 seconds and the first
request succeeded without model reload. This is an intermediate optimization;
parallel CRIU I/O and separate GPU-weight storage are still needed for the
sub-3-second target.

The adapter also accepts `--kv-cache-memory-bytes` to bound the KV allocation
before checkpointing. A real Qwen 0.5B run with `--kv-cache-memory-bytes
2147483648 --release-kv-cache --fast-restore` produced a 7.7 GiB artifact and
restored to `/health` in 5.046 s, with the first request at 5.063 s. This is a
useful size reduction, but it is not yet the Dynamo-style separate weight
artifact needed to reach the sub-3-second target.

At a 256 MiB KV budget, a real run produced a 5.77 GiB artifact and restored
to `/health` in 3.707 s, with the first request at 3.724 s. The restored CUDA
owners were ready and the request succeeded without model reload.

At a 16 MiB KV budget, a real run produced an approximately 5.6 GiB artifact
and restored to `/health` in 3.498 s, with the first request at 3.515 s. Normal
vLLM warmup and CUDA graph capture still occurred before checkpointing; the
remaining gap is not model initialization.

For a batch-1 profile, adding `--cudagraph-capture-sizes 1` and
`--no-async-scheduling` produced a best run of 3.011 s to `/health` and 3.033 s to the first
warm completion. This preserves CUDA graph capture for the selected shape, but
reduces graph-shape coverage and scheduler concurrency, so it is an explicit
tradeoff rather than a universal default.

With the KV budget reduced to 8 MiB, the same profile reached a best 2.967 s
to `/health` and 2.988 s to the first warm completion. A repeat reached
3.052 s and 3.076 s, so this is a best-case sub-3 result rather than a stable
SLO. These measurements use the local Qwen2.5-0.5B-Instruct model and do not
represent Qwen3-0.6B.

The exact Qwen3-0.6B model requires about 50 MiB of KV cache for a 512-token
profile, so 64 MiB was the smallest valid tested budget. It restored with
CUDA graphs and without model reload in 3.125–3.212 s to `/health` and
3.163–3.249 s to the first warm inference. CRIU phase timing shows about
2.3–2.6 s of private-page materialization plus about 0.62 s of CUDA restore;
the under-3-second target is therefore not yet stable for this exact model.

With a production-sized 2 GiB KV budget, the adapter released the cache before
checkpointing and successfully woke it after restore. The resulting snapshot
was approximately 7.1 GiB and restore took 4.647 s to `/health` and 4.685 s to
the first warm inference. This confirms runtime capacity is preserved, but the
current vLLM sleep/wake path does not yet separate KV backing from the
checkpoint; a CUDA VMM/GMS-style memory artifact is still needed.

The adapter now also tests the exact staged wake sequence:
`wake_up(tags=["weights"])`, followed by `wake_up(tags=["kv_cache"])`. With
vLLM's asynchronous scheduler enabled, the CRIU fork successfully dumped and
restored the Qwen3-0.6B process group, recreated the empty KV cache, and served
a post-restore inference. The run completed in 3.368 s to `/health` and 3.400 s
to the first warm inference. This proves async vLLM compatibility, but the
checkpoint still contains the large CUDA image; KV backing is not yet excluded.

The same run measured streaming TTFT for the fixed prompt: 0.040 s before the
checkpoint and 0.017 s after restore (delta -0.024 s). Restore did not trigger
torch.compile or CUDA-graph recapture; the existing compiled and graph state
was usable immediately.

The patched CRIU fork's native AIO mode was benchmarked separately. On this
host, `--image-io-mode direct` restored to serving in 34.934 s, compared with
5.046 s using buffered I/O, so direct mode is not enabled by default.

The experimental `--io-uring-restore` path uses buffered `io_uring` reads. It
completed the same vLLM serving test, but measured 3.725 s on this host versus
3.707 s for the default buffered path, so it remains opt-in.

This fast path is the first Dynamo-inspired optimization in the adapter. The
next planned steps are a vLLM quiesce hook for releasing unused KV-cache
backing, profiling and enabling parallel/asynchronous CRIU page restore, and a
separate GPU-memory artifact for large weight buffers. Those are distinct from
vLLM's normal model startup and are not silently enabled by this prototype.

Run the destructive group test only on a dedicated vLLM server/GPU:

```bash
EDO_VLLM_COMMAND='sudo nsenter -t <vllm-pid> -m -p -- /usr/local/bin/vllm' \
  ./examples/vllm-snapshot/run-demo.sh --port 18080 --full-snapshot
```

The test is intentionally destructive to the dedicated server: it freezes the
API parent and `VLLM::EngineCore`, restores the CRIU group, waits for both CUDA
owners to return to `RUNNING`, and sends a post-restore inference request.
