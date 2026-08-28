# Troubleshooting

Run the diagnostic first:

```bash
cargo run -- doctor
cargo run -- doctor --json
```

Common causes:

- `criu check` or ptrace failures: run the command with the required privilege or configure the documented CRIU capability policy.
- CUDA symbols missing: source `env.sh`, check `libcuda.so`, and verify that the driver exposes the CUDA checkpoint API.
- Restore rejects the snapshot: use the same host/GPU family and run `snapshot-check` to see the exact mismatch.
- HTTP service is not ready: quiesce before freeze and wait for the post-restore readiness hook before sending traffic.
- Model loads again: the process was not checkpointed after warmup, or the framework uses unsupported external/shared resources.

Snapshots contain process memory and may contain credentials. Store them as sensitive data and remove them with `edo snapshot-clean --yes` when finished.
