# Tests

The default CI path runs Rust unit tests, compiles all targets, validates demo shell syntax, and runs the CPU resume smoke test. GPU, model-server, Triton, and Kubernetes tests are opt-in because they require host-specific drivers, images, permissions, or a dedicated GPU.

The authoritative first-run test is:

```bash
cargo run --example resume
```

The lightweight structure check is:

```bash
bash tests/validate-demos.sh
```
