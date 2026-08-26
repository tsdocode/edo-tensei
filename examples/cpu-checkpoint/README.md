# CPU checkpoint demo

This demo launches a detached Python counter, checkpoints it with CRIU, restores it, and verifies the restored process.

From the repository root:

```bash
./examples/cpu-checkpoint/run-demo.sh
```

The demo uses `sudo` for the CRIU dump and restore commands because this host does not grant `CAP_CHECKPOINT_RESTORE` to the system-wide CRIU binary.

For manual control:

```bash
cargo build
edo_name=counter-demo
target/debug/edo run --name "$edo_name" -- setsid python3 examples/cpu-counter.py
sudo target/debug/edo cpu-dump "$edo_name" /tmp/edo-cpu-snapshot
sudo target/debug/edo cpu-restore /tmp/edo-cpu-snapshot
```

The snapshot contains CRIU image files, `manifest.json`, and restore logs. The demo cleans its temporary snapshot after completion.
