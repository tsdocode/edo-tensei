# CLI reference

`edo` is a Linux command-line tool for inspecting, checkpointing, and
restoring a live process. The v0.1 product vocabulary is available alongside
the explicit low-level CPU/CUDA commands.

## Product commands

| Command | Purpose |
| --- | --- |
| `demo resume` | Run the guided CPU checkpoint/restore demo. |
| `status TARGET` | Report whether a managed process is alive; optionally probe `--url`. |
| `checkpoint TARGET SNAPSHOT` | Create a CPU CRIU snapshot. |
| `restore SNAPSHOT` | Restore a CPU CRIU snapshot. |
| `inspect SNAPSHOT` | Read snapshot metadata and integrity information. |
| `diff FIRST SECOND` | Compare high-level metadata from two snapshots. |

Show all flags and the current command surface with:

```bash
cargo run -- --help
cargo run -- demo --help
```

### First CPU workflow

```bash
cargo run --example resume

# Equivalent explicit vocabulary:
edo checkpoint <target> .edo/checkpoints/demo
edo inspect .edo/checkpoints/demo
edo restore .edo/checkpoints/demo
```

`checkpoint` and `restore` are aliases for the CPU-only `cpu-dump` and
`cpu-restore` operations. The first workflow requires Linux, CRIU, and
permission to invoke CRIU through `sudo`.

## Process and health commands

```text
edo run --name NAME -- COMMAND [ARGS...]
edo status TARGET [--url URL]
edo health-check TARGET [--url URL]
```

`run` starts a named managed process. `status` checks liveness and `health-check`
can additionally probe an HTTP endpoint.

## Snapshot inspection and cleanup

```text
edo inspect SNAPSHOT
edo diff FIRST SECOND
edo snapshot-check SNAPSHOT
edo snapshot-clean SNAPSHOT --yes
```

`snapshot-check` validates compatibility, permissions, and image checksums.
Use `snapshot-clean --yes` only for a snapshot directory you intend to remove.

## CUDA and CRIU coordination

These commands are used by the CUDA integration examples and require the
NVIDIA checkpoint API plus a compatible CRIU installation:

```text
edo cuda-state PID
edo cuda-init
edo cuda-roundtrip PID [--timeout-ms MS] [--lock-timeout-ms MS]
edo freeze TARGET SNAPSHOT [--timeout-ms MS] [--lock-timeout-ms MS]
edo summon SNAPSHOT [--timeout-ms MS] [--skip-integrity]
edo freeze-group ROOT CUDA_PIDS SNAPSHOT [--timeout-ms MS] [--lock-timeout-ms MS]
edo summon-group SNAPSHOT [--timeout-ms MS] [--skip-integrity]
```

`freeze`/`summon` coordinate one CUDA process with CRIU. The `*-group`
variants checkpoint and restore a process tree, such as a vLLM API parent and
`VLLM::EngineCore`. `--skip-integrity` is only for trusted local snapshots and
skips per-image SHA-256 verification; it does not make an image portable.

## Diagnostics and shell completion

```text
edo doctor [--json]
edo completions bash|elvish|fish|powershell|zsh
```

`doctor` reports host capabilities and permission blockers in human-readable
or JSON form. Generate shell completion output for the shell you use.

## Scope

The v0.1 CLI targets Linux x86_64 and same-host restore. Async vLLM requires
the Edo CRIU fork with io_uring support; see the [vLLM integration guide](integrations/vllm.md).
This release does not promise cross-node migration, in-flight request
preservation, universal GPU compatibility, or a production Kubernetes operator.
