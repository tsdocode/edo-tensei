# CLI reference

## Lifecycle commands

```text
edo doctor [--json]
edo run --name NAME -- COMMAND [ARGS...]
edo cpu-dump TARGET SNAPSHOT
edo cpu-restore SNAPSHOT
edo freeze TARGET SNAPSHOT
edo summon SNAPSHOT [--skip-integrity]
edo health-check TARGET [--url URL]
edo snapshot-check SNAPSHOT
```

The low-level `cpu-*` commands are useful for CPU onboarding. `freeze` and `summon` coordinate CUDA and CRIU and require the target process to own checkpointable CUDA state.

The planned product vocabulary is `checkpoint`, `restore`, `inspect`, and `demo`; the current command aliases remain explicit until the compatibility migration is complete.
