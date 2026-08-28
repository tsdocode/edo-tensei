You are the lead developer responsible for redesigning the Edo Tensei demo experience and project structure.

## Context

Edo Tensei is a Rust-based checkpoint/restore platform for AI workloads. The core flow already supports:

- CPU process checkpoint and restore
- CUDA state checkpoint and restore
- PyTorch model snapshots
- GPU model warm-start snapshots
- FastAPI workloads with expensive startup
- Triton snapshots
- vLLM snapshots
- Kubernetes GPU workloads and restore flows
- A CLI, doctor command, local run reports, and CRIU/CUDA integration

Current repository structure:

```text
edo-tensei/
├── .github/
│   └── workflows/ci.yml
├── .edo/
│   └── runs/
├── .models/
│   └── Qwen3-0.6B/
├── assets/
│   └── edo-tensei-project.png
├── examples/
│   ├── cpu-checkpoint/
│   ├── cuda-checkpoint/
│   ├── fastapi-heavy-startup/
│   ├── gpu-model-snapshot/
│   ├── modal-gpu-snapshot/
│   ├── triton-snapshot/
│   └── vllm-snapshot/
├── kubernetes/
│   ├── snapshot-agent/
│   ├── gpu-workload*
│   ├── vllm-qwen3-pod.yaml
│   └── *restore*.yaml
├── src/
│   ├── cli.rs
│   ├── criu.rs
│   ├── cuda.rs
│   ├── doctor.rs
│   ├── process.rs
│   ├── snapshot.rs
│   └── main.rs
├── Cargo.toml
├── README.md
├── V0.1-MILESTONE.md
├── edo-tensei-plan.md
├── roadmap-todo.md
├── env.sh
└── LICENSE
```

## Objective

Remake the demos and project structure so that a new user understands the value of Edo Tensei within 60 seconds.

The primary message should be:

> Checkpoint a live AI workload, kill it, restore it, and continue from the same state.

The onboarding should feel simple and polished, inspired by the clarity and progressive disclosure of Modal. Do not copy Modal’s branding, APIs, or implementation. Preserve Edo Tensei’s own identity and technical strengths.

## Product experience to create

The first demo must show a visible “wow moment”:

1. Start a small model or stateful service.
2. Perform warm-up work.
3. Show that the workload has expensive initialization.
4. Create a checkpoint.
5. Simulate process or node failure.
6. Restore from the checkpoint.
7. Send another request successfully.
8. Print cold-start time versus restore time.
9. Save a structured run report under `.edo/runs/`.

The first experience must not require Kubernetes, Triton, vLLM, or a large GPU. It should run locally on CPU where possible, with optional CUDA acceleration.

Suggested command:

```bash
cargo run --example resume
```

or, if the CLI supports it:

```bash
edo demo resume
```

Expected output:

```text
Edo Tensei — Resume a warm workload

✓ Started workload
✓ Warm-up completed
✓ Checkpoint created: .edo/checkpoints/demo
✓ Simulated process failure
✓ Restored successfully
✓ Request completed after restore

Cold start:    18.4s
Restore start: 1.7s
Improvement:   10.8x

Run report: .edo/runs/2026-08-28T-demo.json
```

## Proposed example hierarchy

Replace the current flat collection of examples with a progressive learning path:

```text
examples/
├── 00_hello_checkpoint/
│   ├── README.md
│   └── main.rs
├── 01_stateful_process/
│   ├── README.md
│   └── main.rs
├── 02_fastapi_resume/
│   ├── README.md
│   ├── server.py
│   └── run.sh
├── 03_pytorch_warm_start/
│   ├── README.md
│   ├── model.py
│   └── run.sh
├── 04_vllm_resume/
│   ├── README.md
│   ├── server.py
│   ├── client.py
│   └── run.sh
├── 05_cuda_restore/
│   ├── README.md
│   └── run.sh
├── 06_triton_snapshot/
│   ├── README.md
│   └── run.sh
└── 07_kubernetes_migration/
    ├── README.md
    ├── manifests/
    └── run.sh
```

Each example must have:

- A one-sentence explanation
- Prerequisites
- One copy-paste command
- Expected output
- What Edo Tensei is checkpointing
- Known limitations
- Cleanup instructions
- A link to the next example

The examples should progressively introduce complexity:

```text
Hello checkpoint
      ↓
Stateful process
      ↓
FastAPI service
      ↓
PyTorch model
      ↓
vLLM inference
      ↓
CUDA state
      ↓
Triton
      ↓
Kubernetes migration
```

## Demo naming

Use user-oriented names instead of implementation-oriented names.

Prefer:

- `resume-a-process`
- `resume-a-fastapi-service`
- `warm-start-pytorch`
- `resume-vllm`
- `migrate-a-gpu-workload`
- `restore-after-node-drain`

Avoid leading with terms such as:

- CRIU internals
- CUDA IPC
- snapshot agent
- memory image
- restore helper

Those technical terms belong in the advanced documentation.

## Recommended repository structure

Reorganize the repository around product experience, implementation, and operations:

```text
edo-tensei/
├── .github/
│   └── workflows/
├── .edo/
│   ├── checkpoints/
│   └── runs/
├── assets/
│   ├── hero-demo.gif
│   ├── architecture.svg
│   └── edo-tensei-project.png
├── crates/
│   ├── edo-cli/
│   ├── edo-core/
│   ├── edo-criu/
│   ├── edo-cuda/
│   ├── edo-process/
│   ├── edo-snapshot/
│   └── edo-doctor/
├── demos/
│   ├── resume/
│   ├── pytorch/
│   ├── vllm/
│   ├── cuda/
│   └── kubernetes/
├── examples/
│   └── progressive examples listed above
├── integrations/
│   ├── fastapi/
│   ├── pytorch/
│   ├── triton/
│   ├── vllm/
│   └── kubernetes/
├── kubernetes/
│   ├── base/
│   ├── snapshot-agent/
│   ├── gpu-workload/
│   └── restore/
├── docs/
│   ├── getting-started.md
│   ├── concepts/
│   ├── compatibility.md
│   ├── cli.md
│   ├── integrations/
│   ├── kubernetes/
│   └── troubleshooting.md
├── tests/
│   ├── cpu/
│   ├── cuda/
│   ├── pytorch/
│   └── kubernetes/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── COMPATIBILITY.md
├── V0.1-MILESTONE.md
└── roadmap-todo.md
```

If a full Cargo workspace migration is too large for this milestone, preserve the existing Rust layout temporarily but create the logical boundaries through modules and documentation. Do not perform a risky rewrite of working core code solely for aesthetics.

## CLI/API ergonomics

Review the CLI and introduce a coherent lifecycle:

```bash
edo doctor
edo run <target>
edo status
edo checkpoint <name>
edo restore <name>
edo inspect <name>
edo diff <checkpoint-a> <checkpoint-b>
edo demo <name>
```

The CLI should provide:

- Consistent success/error formatting
- Human-readable progress output
- `--json` output for automation
- `--verbose` diagnostics
- Stable exit codes
- Automatic run-report generation
- Clear remediation steps on failure

The `doctor` command should validate:

- Kernel and CRIU support
- Required permissions
- NVIDIA driver and CUDA compatibility
- PyTorch compatibility
- vLLM availability
- Kubernetes access
- Checkpoint directory health
- Required environment variables

## README redesign

Rewrite the README around the first successful experience.

Recommended order:

1. Project title and one-line value proposition
2. Hero GIF or terminal recording
3. “Run your first restore in 60 seconds”
4. One minimal command
5. Expected terminal output
6. Explanation of what happened
7. Demo catalogue
8. Supported runtimes matrix
9. Architecture overview
10. Production/Kubernetes usage
11. Troubleshooting
12. Development and contributing

The opening should communicate:

```text
Edo Tensei checkpoints live AI workloads and restores them with their
runtime state intact — from Python processes to PyTorch and vLLM servers.
```

The README should not begin with CRIU implementation details, a long feature inventory, or Kubernetes YAML.

## Technical requirements

- Keep all existing working core functionality.
- Do not fake benchmark numbers.
- Measure cold start and restore time during every demo.
- Clearly distinguish measured values from illustrative output.
- Make demos deterministic and safe to rerun.
- Avoid requiring users to download large models for the first demo.
- Reuse `.models/Qwen3-0.6B` when available.
- Add graceful fallback when the model or CUDA runtime is unavailable.
- Keep CPU-only onboarding functional.
- Ensure every demo can explain why it failed.
- Add CI smoke tests for the CPU demos.
- Add opt-in tests for CUDA, vLLM, Triton, and Kubernetes.
- Never commit local checkpoints or model weights.
- Add `.edo/checkpoints/` and `.edo/runs/` to the appropriate ignore rules.

## Deliverables

Produce the following:

1. Redesigned README
2. New first-run `resume` demo
3. Progressive example hierarchy
4. Updated CLI help and command naming
5. Run-report format under `.edo/runs/`
6. Compatibility matrix
7. Migration guide from the old example paths
8. Demo test/validation script
9. Terminal recording or GIF for the README
10. Short architecture diagram
11. Updated documentation navigation
12. Implementation plan divided into P0, P1, and P2

## Priority order

### P0 — Must have

- One-command CPU demo
- Warm process checkpoint/restore flow
- Cold-start versus restore measurement
- New README quickstart
- `edo doctor`
- Structured run reports
- CPU CI smoke test

### P1 — Should have

- FastAPI demo
- PyTorch demo
- vLLM demo
- CUDA compatibility documentation
- Hero terminal recording
- Improved CLI progress output

### P2 — Later

- Triton demo
- Kubernetes node-drain migration
- Web dashboard for run reports
- Remote checkpoint registry
- Multi-node restore orchestration

## Success criteria

A developer unfamiliar with Edo Tensei should be able to:

1. Clone the repository.
2. Run one command.
3. Understand what was checkpointed.
4. See a process fail and recover.
5. Compare cold start and restore time.
6. Find the next relevant example.
7. Know whether their CPU, CUDA, PyTorch, vLLM, or Kubernetes environment is supported.

Before implementation, first inspect the current source, CLI behavior, examples, and Kubernetes manifests. Preserve useful working code and migrate incrementally. At the end, provide a concise summary of:

- Files changed
- Commands tested
- Demos that work on CPU
- Demos requiring CUDA or Kubernetes
- Remaining limitations
- Recommended next milestone