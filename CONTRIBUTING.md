# Contributing

Start with the CPU path:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo run --example resume
```

CUDA, vLLM, Triton, and Kubernetes demos are opt-in because they need host-specific drivers, images, privileges, and often a dedicated GPU. Do not commit `.edo/checkpoints/`, model weights, or generated virtual environments.

Changes should preserve the progressive examples and state clearly whether a timing is measured or illustrative.
