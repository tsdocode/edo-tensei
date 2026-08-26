# Edo Tensei
## GPU Process Snapshot & Fast-Resume Runtime

**Working codename:** Edo Tensei  
**CLI name:** `edo`  
**Primary language:** Rust  
**Initial target:** Linux + NVIDIA CUDA + single process + single GPU  
**Document date:** 2026-08-26

> **Goal:** checkpoint a warmed CUDA application, terminate it, restore it, and continue from the same CPU + GPU execution state without re-running model initialization.

---

# 1. Why this project exists

GPU inference startup is expensive because a normal cold start may require:

```text
container/process startup
→ Python imports
→ framework initialization
→ model construction
→ weight loading
→ CPU → GPU transfer
→ CUDA context setup
→ compilation / CUDA Graph capture
→ warmup
→ API ready
```

For many GPU APIs, especially inference servers, most of this work is identical every time.

Edo Tensei explores a different lifecycle:

```text
warm application
      ↓
   freeze
      ↓
CPU checkpoint + GPU checkpoint
      ↓
process may disappear
      ↓
   summon
      ↓
resume warmed application
```

The first version is **not** trying to solve arbitrary GPU persistence, multi-node inference, or distributed recovery.

The first technical goal is deliberately small:

> **Checkpoint and restore one warmed CUDA process on one NVIDIA GPU.**

---

# 2. What NVIDIA Dynamo teaches us

NVIDIA Dynamo Snapshot is currently the most relevant reference architecture.

Its important ideas are:

1. **CRIU saves the CPU/Linux process state.**
2. **CUDA checkpoint saves CUDA device state.**
3. The CUDA process is checkpointed first, making its GPU memory/state representable on the host side.
4. CRIU then checkpoints the remaining Linux process state.
5. On restore, CRIU restores the CPU process first.
6. CUDA restore rebuilds the GPU side.
7. Large inference state should be separated by lifecycle:
   - model weights: long-lived / immutable
   - KV cache: ephemeral / disposable
   - other engine state: process-owned
8. Stable CUDA virtual addresses matter because restored Torch objects and CUDA Graphs may retain pointers.
9. A GPU memory owner can eventually decouple **VRAM lifetime** from **worker process lifetime**.

The current Dynamo direction calls this **Snapshot-coupled GPU Memory Service V1**.

The important architectural split is:

```text
Dynamo Snapshot
    owns:
    - process state
    - Python/Torch object graph
    - engine semantics
    - CUDA virtual-address reservations
    - restore ordering

GMS
    owns:
    - physical CUDA allocations
    - allocation identity
    - immutable model-weight backing
    - mapping/export lifecycle
```

Edo Tensei should learn from this split rather than trying to make a GPU memory service understand PyTorch model semantics.

---

# 3. Current NVIDIA reality we should design around

As of **2026-08-26**:

- CUDA checkpoint/restore is **Linux-only**.
- The public CUDA Driver API exposes process checkpoint functions.
- CUDA checkpoint moves GPU memory contents into host memory and releases underlying GPU references.
- CUDA restore can restore onto a compatible GPU with enough memory.
- Dynamo Snapshot is most mature on **single-GPU** inference workloads.
- vLLM and SGLang single-GPU snapshot paths are supported in current Dynamo documentation.
- Multi-GPU snapshot support is still experimental / in progress.
- Snapshot + GMS is still an **experimental path** and is disabled in normal Dynamo deployments.
- NVIDIA's current long-term GMS V1 proposal uses:
  - immutable weight backing,
  - ephemeral KV cache,
  - exact allocation identity,
  - exact CUDA VA restoration,
  - a separate GMS owner process/container.

This is useful for us because it means our scope should be:

```text
v0.1:
transparent CUDA + CRIU process resurrection

later:
persistent VRAM / VMM backend
```

We should **not** begin by rebuilding GMS.

---

# 4. Core design principle

Edo Tensei should initially be:

> **A framework-agnostic orchestration layer over Linux process checkpointing and CUDA process checkpointing.**

From Edo's perspective, it should not matter whether the process is:

```text
PyTorch
FastAPI + PyTorch
TensorRT
custom CUDA
small inference server
```

The target process should ideally need **zero code changes**.

User experience:

```bash
edo run --name demo python app.py

edo freeze demo

edo summon demo
```

---

# 5. v0.1 scope

## Supported

```text
OS              Linux
CPU arch        x86_64 first
GPU             NVIDIA
GPU count       1
CUDA process    1
Process tree    simple / single process
CPU snapshot    CRIU
GPU snapshot    CUDA Driver checkpoint API
Language        Rust
App frameworks  framework-agnostic
Official tests  raw CUDA / PyTorch / FastAPI
```

## Not supported initially

```text
AMD
macOS
Windows
multi-GPU
multi-node
NCCL
tensor parallel
distributed PyTorch
CUDA IPC-heavy workloads
Unified Memory / UVM
RDMA
MIG migration
arbitrary container migration
persistent VRAM
vLLM multi-process
SGLang multi-process
KV-cache preservation
in-flight request preservation
```

---

# 6. Architecture: v0.1

```text
                    user
                      │
                      ▼
                ┌──────────┐
                │ edo CLI  │
                └────┬─────┘
                     │
                     ▼
              ┌──────────────┐
              │ Edo runtime  │
              │ coordinator  │
              └──────┬───────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
 process manager  CUDA backend  CRIU backend
         │           │           │
         │           │           │
         ▼           ▼           ▼
 Linux /proc      libcuda.so     criu binary
                     │
                     ▼
                 NVIDIA GPU
```

---

# 7. Checkpoint lifecycle

The intended state machine:

```text
RUNNING
   │
   │ validate
   ▼
READY_TO_FREEZE
   │
   │ CUDA process lock
   ▼
CUDA_LOCKED
   │
   │ CUDA checkpoint
   ▼
CUDA_CHECKPOINTED
   │
   │ CRIU dump
   ▼
PROCESS_CHECKPOINTED
   │
   ▼
SNAPSHOT_READY
```

Detailed flow:

```text
1. Validate target PID
2. Validate Linux
3. Validate CRIU
4. Validate NVIDIA driver
5. Validate CUDA checkpoint support
6. Detect GPU usage
7. Lock CUDA process
8. Checkpoint CUDA state
9. CRIU dump CPU/process state
10. Write Edo manifest
11. Persist logs
12. Mark snapshot complete
```

Critical ordering:

```text
CUDA checkpoint
      ↓
CRIU dump
```

not the other way around.

---

# 8. Why CUDA must be checkpointed before CRIU

Before CUDA checkpoint:

```text
CPU process                         GPU
───────────                         ───
Python heap
Torch tensor metadata ────────────► VRAM weights
threads                            CUDA context
file descriptors                   streams
sockets                            mappings
```

CRIU understands Linux process state, but it cannot independently reconstruct arbitrary live NVIDIA GPU resources.

CUDA checkpoint transforms the process approximately into:

```text
CPU / host side
────────────────────────────
Python heap
Torch object metadata
CUDA checkpoint metadata
GPU memory contents in host memory
threads
FDs
sockets

GPU
────────────────────────────
underlying references released
```

Now CRIU can checkpoint the Linux process in a restorable form.

Conceptual save path:

```text
VRAM
 ↓
host checkpoint memory
 ↓
CRIU images
```

Conceptual restore path:

```text
CRIU images
 ↓
host checkpoint memory
 ↓
VRAM
```

---

# 9. Restore lifecycle

```text
SNAPSHOT_READY
      │
      │ validate host + GPU
      ▼
CRIU_RESTORE
      │
      ▼
CPU_PROCESS_RESTORED
      │
      │ CUDA restore
      ▼
CUDA_LOCKED
      │
      │ CUDA unlock
      ▼
RUNNING
```

Flow:

```text
1. Load manifest
2. Validate compatibility
3. CRIU restore Linux process
4. Resolve restored PID
5. Restore CUDA process
6. Unlock CUDA process
7. Resume serving
8. Optional health check
9. Record restore timing
```

---

# 10. Public CUDA APIs we care about

Initial Rust wrapper should target only the checkpoint API surface we actually need:

```text
cuCheckpointProcessGetState
cuCheckpointProcessLock
cuCheckpointProcessCheckpoint
cuCheckpointProcessRestore
cuCheckpointProcessUnlock
cuCheckpointProcessGetRestoreThreadId
```

Do **not** build a giant CUDA binding layer.

Expose a tiny safe Rust API around the unsafe FFI.

Example target interface:

```rust
pub struct CudaCheckpoint;

impl CudaCheckpoint {
    pub fn state(&self, pid: i32) -> Result<CudaProcessState>;
    pub fn lock(&self, pid: i32) -> Result<CudaProcessGuard>;
    pub fn checkpoint(&self, pid: i32) -> Result<()>;
    pub fn restore(&self, pid: i32) -> Result<()>;
}
```

Prefer RAII where possible.

For example, a lock guard should automatically unlock if control exits unexpectedly.

---

# 11. CRIU integration strategy

For v0.1, **do not link libcriu**.

Use:

```rust
std::process::Command
```

and invoke the `criu` executable.

Example concept:

```rust
Command::new("criu")
    .arg("dump")
    .arg("--tree")
    .arg(pid.to_string())
    .arg("--images-dir")
    .arg(snapshot_dir)
    .status()?;
```

Advantages:

- smallest implementation
- easiest to debug
- no extra Rust FFI
- easy to compare with manual CRIU commands
- easy to inspect CRIU logs

Later we can migrate to CRIU RPC if we need:

- structured responses
- daemon integration
- finer lifecycle control
- better progress reporting

---

# 12. Technology stack

## Core

```text
Rust
```

Recommended initial dependencies:

```toml
[dependencies]
anyhow = "1"
thiserror = "2"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
nix = { version = "0.30", features = ["signal", "process"] }
procfs = "0.17"
tracing = "0.1"
tracing-subscriber = "0.3"
```

Keep v0.1 **synchronous**.

Do not add Tokio until we actually need concurrency.

## GPU

```text
CUDA Driver API
Rust FFI
libcuda.so
```

## CPU process checkpoint

```text
CRIU executable
```

## Python

None initially.

Optional later:

```text
PyO3
maturin
```

## Persistent VRAM, later

```text
CUDA VMM
CUDA IPC
Rust daemon
Unix domain sockets
```

---

# 13. Minimal repository structure

Do **not** start with six crates.

Start with one Rust crate:

```text
edo-tensei/
├── Cargo.toml
├── README.md
├── LICENSE
├── src/
│   ├── main.rs
│   ├── cli.rs
│   ├── process.rs
│   ├── cuda.rs
│   ├── criu.rs
│   ├── snapshot.rs
│   ├── doctor.rs
│   └── error.rs
│
├── examples/
│   ├── cpu-counter.py
│   ├── torch-counter.py
│   └── fastapi-demo/
│
├── tests/
│   └── ...
│
└── docs/
    ├── architecture.md
    ├── compatibility.md
    └── internals.md
```

Only split into multiple crates after the PoC works.

---

# 14. Initial CLI

Start with only these commands:

```bash
edo doctor
edo cpu-dump <pid> <dir>
edo cpu-restore <dir>
edo gpu-state <pid>
edo gpu-lock <pid>
edo gpu-checkpoint <pid>
edo gpu-restore <pid>
edo gpu-unlock <pid>
edo freeze <pid> <dir>
edo summon <dir>
```

Later simplify public UX to:

```bash
edo run --name api ...
edo freeze api
edo summon api
edo status api
```

The low-level commands are useful during development because they let us debug CRIU and CUDA independently.

---

# 15. Phase 0 — development environment

Because CRIU and CUDA checkpoint are Linux-only, macOS is a development machine, not the final test machine.

Recommended workflow:

```text
MacBook
├── edit Rust
├── cargo fmt
├── cargo check
├── cargo test
└── git push
       ↓
Linux NVIDIA machine
├── NVIDIA driver
├── CUDA checkpoint support
├── CRIU
├── cargo build
└── integration tests
```

On macOS:

```rust
#[cfg(target_os = "linux")]
mod criu;

#[cfg(target_os = "linux")]
mod cuda_checkpoint;
```

Non-Linux builds should still compile enough for:

```text
manifest parsing
CLI help
unit tests
serialization
state-machine tests
```

but commands requiring CRIU/CUDA checkpoint should return a clear unsupported-platform error.

---

# 16. Phase 1 — CPU-only CRIU proof

Before touching CUDA, prove CRIU works.

Test workload:

```python
import time

counter = 0

while True:
    counter += 1
    print(counter, flush=True)
    time.sleep(1)
```

Goal:

```text
1
2
3
4
5
   ↓ CRIU dump
process gone
   ↓ CRIU restore
6
7
8
...
```

Tasks:

- [ ] install CRIU on Linux test host
- [ ] run `criu check`
- [ ] manually dump process
- [ ] manually restore process
- [ ] confirm counter continues
- [ ] implement `edo cpu-dump`
- [ ] implement `edo cpu-restore`

Expected Edo code:

```text
~150–300 LOC
```

---

# 17. Phase 2 — CUDA checkpoint proof

Do not combine CRIU yet.

Use a minimal CUDA/PyTorch process:

```python
import time
import torch

x = torch.randn(1024, 1024, device="cuda")
counter = 0

while True:
    y = x @ x
    torch.cuda.synchronize()

    counter += 1
    print(counter, y[0, 0].item(), flush=True)

    time.sleep(1)
```

Tasks:

- [ ] verify CUDA checkpoint support on host
- [ ] inspect CUDA process state
- [ ] lock CUDA process manually
- [ ] checkpoint CUDA process
- [ ] verify GPU memory is released / process is checkpointed
- [ ] restore CUDA process
- [ ] unlock CUDA process
- [ ] verify computation continues
- [ ] implement Rust bindings
- [ ] hide unsafe FFI behind safe wrapper

Expected code:

```text
~250–500 LOC
```

---

# 18. Phase 3 — first resurrection

Combine CUDA checkpoint + CRIU.

Target:

```bash
edo freeze <pid> ./snapshots/test-001
```

Internally:

```text
validate
↓
CUDA lock
↓
CUDA checkpoint
↓
CRIU dump
↓
write manifest
```

Then:

```bash
edo summon ./snapshots/test-001
```

Internally:

```text
read manifest
↓
CRIU restore
↓
resolve restored PID
↓
CUDA restore
↓
CUDA unlock
↓
health check
```

First major success criterion:

> The PyTorch process continues from the same Python counter and produces valid CUDA results after complete process destruction and restore.

This is the first meaningful Edo Tensei demo.

Expected cumulative code:

```text
~800–1,500 LOC
```

---

# 19. Snapshot manifest

Example:

```json
{
  "format_version": 1,
  "name": "torch-demo",
  "created_at": "2026-08-26T00:00:00Z",
  "original_pid": 18421,
  "command": [
    "python",
    "torch_counter.py"
  ],
  "host": {
    "kernel": "6.x",
    "architecture": "x86_64",
    "criu_version": "..."
  },
  "cuda": {
    "driver_version": "...",
    "device_index": 0,
    "gpu_uuid": "...",
    "gpu_name": "...",
    "process_state_at_capture": "checkpointed"
  }
}
```

Snapshot layout:

```text
snapshot/
├── manifest.json
├── cpu/
│   └── CRIU images
└── logs/
    ├── criu-dump.log
    └── edo.log
```

For v0.1 the CUDA checkpointed GPU contents are ultimately represented through the host-side process checkpoint path rather than a custom Edo GPU image format.

---

# 20. Phase 4 — FastAPI inference demo

Only after the PyTorch loop works.

Example application:

```python
from fastapi import FastAPI
import torch

app = FastAPI()

model = ...
model.cuda()
model.eval()

# warmup here

@app.post("/infer")
def infer(...):
    ...
```

Test:

```text
start API
↓
load model
↓
warm up
↓
request succeeds
↓
edo freeze
↓
process disappears
↓
edo summon
↓
request succeeds without model reload
```

Measure:

```text
normal cold-start TTFI
vs
Edo summon TTFI
```

Primary metric:

> **TTFI — Time To First Inference**

---

# 21. Benchmarking

Every benchmark should record:

```text
normal_startup_ms
cuda_checkpoint_ms
criu_dump_ms
snapshot_total_ms

criu_restore_ms
cuda_restore_ms
unlock_ms
healthcheck_ms
first_inference_ms
summon_total_ms

host_memory_before
host_memory_checkpointed
vram_before
vram_after_checkpoint
snapshot_disk_size
```

Example table:

| Workload | VRAM | Cold TTFI | Edo TTFI | Speedup |
|---|---:|---:|---:|---:|
| PyTorch matmul | | | | |
| ResNet | | | | |
| Small TTS | | | | |
| Small LLM | | | | |
| FastAPI model | | | | |

Do not market `<1s` until it is measured.

---

# 22. v0.1 completion criteria

Release `v0.1.0` only when:

- [ ] Linux-only build path works
- [ ] `edo doctor` works
- [ ] CRIU CPU-only round trip works
- [ ] CUDA state query works
- [ ] CUDA lock/checkpoint/restore/unlock works
- [ ] single PyTorch CUDA process round trip works
- [ ] same Python state is preserved
- [ ] same CUDA tensor contents are preserved
- [ ] FastAPI demo restores
- [ ] model does not reload on summon
- [ ] benchmark numbers are published
- [ ] errors are readable
- [ ] snapshot compatibility metadata is stored
- [ ] tests cover state-machine failures
- [ ] README includes a reproducible demo

---

# 23. Error-handling philosophy

Bad:

```text
checkpoint failed
```

Good:

```text
Edo could not checkpoint PID 18421.

CUDA process state:
RUNNING

Expected:
LOCKED

Action:
Run `edo doctor` and retry.
```

Suggested Rust errors:

```rust
enum EdoError {
    UnsupportedPlatform,
    ProcessNotFound,
    CriuUnavailable,
    CriuFailure,
    CudaDriverUnavailable,
    CudaCheckpointUnsupported,
    InvalidCudaState,
    CudaLockFailure,
    CudaCheckpointFailure,
    CudaRestoreFailure,
    SnapshotCorrupt,
    SnapshotIncompatible,
    InsufficientGpuMemory,
}
```

---

# 24. `edo doctor`

This command should become one of the most important debugging tools.

Example:

```text
$ edo doctor

Platform
  Linux                  ✓
  x86_64                 ✓

CRIU
  installed              ✓
  version                4.x
  criu check             ✓

NVIDIA
  driver                 ✓
  CUDA checkpoint API    ✓
  GPU count              1
  GPU                    NVIDIA ...

Edo
  single-GPU mode        ✓

Ready for Edo Tensei.
```

On Mac:

```text
$ edo doctor

Platform
  macOS                   ✗

CPU-only Edo development is supported.
Checkpoint/restore requires Linux + NVIDIA CUDA.
```

---

# 25. Rust learning plan while building

Do not learn all Rust first.

Learn in project order.

## First

```text
struct
enum
match
Option
Result
?
Vec
String
Path / PathBuf
modules
Cargo
```

## Then

```text
ownership
borrowing
&T
&mut T
traits basics
error types
```

## During CUDA FFI

```text
unsafe
extern "C"
raw pointers basics
C structs
FFI error translation
RAII / Drop
```

## Later

```text
Arc
Mutex
threads
Unix sockets
```

Ignore initially:

```text
advanced lifetime design
complex generics
procedural macro implementation
async internals
Pin
custom allocators
no_std
advanced pointer manipulation
```

---

# 26. Rust ownership patterns we should use

A key rule:

> **Borrow inputs by default; own long-lived resources.**

Example:

```rust
fn freeze(
    runtime: &EdoRuntime,
    process: &ProcessInfo,
    output: &Path,
) -> Result<Snapshot>
```

The function borrows runtime/process/path and returns an owned snapshot.

For CUDA locks, use RAII if the API semantics allow it:

```rust
let lock = CudaProcessLock::acquire(pid)?;

// checkpoint...

drop(lock); // or automatic Drop
```

If an error occurs, `Drop` should release resources where safe.

This is one of the main reasons Rust fits Edo well.

---

# 27. What NOT to build in v0.1

Do not start with:

```text
Kubernetes
Docker orchestration
vLLM
SGLang
Tensor parallel
multi-GPU
GMS clone
CUDA allocator interception
LD_PRELOAD
custom kernel module
CRIU modifications
async daemon
Python SDK
```

A successful 1,000 LOC resurrection PoC is more valuable than a 20,000 LOC architecture that has never restored one CUDA process.

---

# 28. v0.2 — process trees and containers

After v0.1:

```text
FastAPI master
   │
   ├── CUDA worker
   ├── helper process
   └── shared memory
```

Add:

- process tree discovery
- coordinated process freeze
- cgroups
- namespaces
- Docker / OCI integration
- restored TCP/socket testing
- multiple CUDA PIDs on one host

Likely complexity:

```text
~5k–10k cumulative LOC
```

---

# 29. v0.3 — CUDA IPC and multi-process workloads

Needed for serious inference engines.

Add:

```text
CUDA IPC
shared allocations
multi-process restore ordering
worker topology
shared-memory metadata
```

Potential targets:

```text
vLLM single GPU
SGLang single GPU
```

Do not promise NCCL or TP yet.

---

# 30. v0.4 — persistent VRAM backend

This is where Edo becomes much more interesting.

Problem with normal CUDA checkpoint + CRIU:

```text
SAVE:
VRAM → host RAM → snapshot storage

RESTORE:
snapshot storage → host RAM → VRAM
```

For a 40 GB model, memory bandwidth still dominates.

The next architecture should be GMS-like:

```text
                 edod
                  │
                  │ owns
                  ▼
         physical CUDA memory
                  │
             model weights
                  │
       ┌──────────┴──────────┐
       │                     │
   worker A              worker B
```

Worker lifetime becomes independent from model-weight lifetime.

---

# 31. Persistent VRAM architecture

New daemon:

```text
edod
```

Responsibilities:

```text
CUDA VMM allocation ownership
allocation IDs
shareable handles
virtual-address mapping metadata
lease lifecycle
weight-domain lifecycle
worker attach/detach
```

Core CUDA VMM APIs:

```text
cuMemCreate
cuMemRelease
cuMemAddressReserve
cuMemAddressFree
cuMemMap
cuMemUnmap
cuMemSetAccess
cuMemExportToShareableHandle
cuMemImportFromShareableHandle
```

Architecture:

```text
            edod
              │
        cuMemCreate()
              │
              ▼
       physical VRAM
              │
        model weights
              │
       ┌──────┴──────┐
       ▼             ▼
    worker A       worker B
    VA X           VA X
```

---

# 32. Learn directly from Dynamo GMS V1

The most important ideas to reuse conceptually:

## Weights

```text
long-lived
immutable after publication
shared read-only backing
stable allocation identity
```

## KV cache

```text
mutable
ephemeral
discard on sleep
recreate empty on wake
restore same expected VA if engine semantics require it
```

## Engine state

```text
Python/Torch semantics remain owned by restored process snapshot
```

## GMS / Edo VRAM daemon

```text
model-blind
owns memory, not model semantics
```

This prevents Edo from needing to know:

```text
model architecture
parameter names
quantization implementation
Torch module graph
tensor reconstruction rules
```

That is a critical design decision.

---

# 33. Why exact virtual addresses matter

A restored engine may contain:

```text
Torch TensorImpl pointers
CUDA Graph captured pointers
engine caches
internal workspace references
```

Therefore:

```text
same physical model bytes
```

is not necessarily enough.

We may need:

```text
same expected CUDA virtual addresses
```

for correctness.

Persistent-VRAM work should therefore treat stable VA restoration as a first-class requirement, not merely an optimization.

---

# 34. Long-term architecture

```text
                         Edo Tensei

                    ┌───────────────┐
                    │ edo runtime   │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
           CRIU       CUDA checkpoint    edod
              │             │             │
        CPU/process    GPU process     persistent
           state          state          VRAM
              │             │             │
              └─────────────┼─────────────┘
                            │
                            ▼
                         summon
```

Long-term target:

```text
process cold-start work eliminated
+
model loading eliminated
+
warm CUDA state preserved
+
large weight backing reused
```

That is the path toward genuinely sub-second warm GPU resurrection.

---

# 35. Relationship to NVIDIA Dynamo

Edo Tensei should not try to compete with Dynamo on the full serving stack.

Different target:

```text
Dynamo
────────────────────────
full inference serving platform
Kubernetes operator
routing
backend integrations
snapshot lifecycle
GMS
distributed serving

Edo Tensei
────────────────────────
small standalone systems primitive
checkpoint/restore arbitrary CUDA process
developer-friendly CLI/library
minimal dependencies
framework-agnostic first
```

Potential positioning:

> **CRIU for CUDA applications, with a path toward persistent VRAM.**

or:

> **Freeze and resurrect warmed GPU processes.**

---

# 36. First GitHub issues

Create these issues in order.

## Issue 1 — Linux platform guard

```text
Add cfg-gated Linux platform support and unsupported-platform errors.
```

## Issue 2 — `edo doctor`

```text
Check Linux, CRIU, NVIDIA driver, CUDA checkpoint API and GPU count.
```

## Issue 3 — CRIU command wrapper

```text
Implement cpu-dump and cpu-restore using std::process::Command.
```

## Issue 4 — CPU counter integration test

```text
Verify counter resumes after CRIU round trip.
```

## Issue 5 — minimal CUDA FFI

```text
Bind cuCheckpointProcessGetState and expose a safe Rust wrapper.
```

## Issue 6 — CUDA lock/unlock

```text
Implement lock state transition and safe cleanup.
```

## Issue 7 — CUDA checkpoint/restore

```text
Checkpoint and restore a simple CUDA/PyTorch process.
```

## Issue 8 — Edo snapshot manifest

```text
Add manifest.json and snapshot directory layout.
```

## Issue 9 — `edo freeze`

```text
Coordinate CUDA checkpoint followed by CRIU dump.
```

## Issue 10 — `edo summon`

```text
CRIU restore followed by CUDA restore and unlock.
```

## Issue 11 — PyTorch resurrection demo

```text
Preserve counter + CUDA tensor contents across full round trip.
```

## Issue 12 — benchmark framework

```text
Measure checkpoint, restore and TTFI timings.
```

## Issue 13 — FastAPI demo

```text
Restore a warmed API without model reload.
```

## Issue 14 — v0.1 release

```text
Docs, compatibility matrix, reproducible benchmark, release binary.
```

---

# 37. First coding session

The first coding session should **not touch CUDA**.

Create project:

```bash
cargo new edo-tensei
cd edo-tensei
```

Add CLI:

```bash
edo doctor
edo cpu-dump
edo cpu-restore
```

First files:

```text
src/main.rs
src/cli.rs
src/criu.rs
src/doctor.rs
src/error.rs
```

First milestone:

```text
edo doctor
```

must successfully answer:

```text
Linux?
CRIU installed?
CRIU check passes?
```

Then make this work:

```bash
edo cpu-dump <pid> ./snapshot
edo cpu-restore ./snapshot
```

Only after this passes do we add `cuda.rs`.

---

# 38. First real technical milestone

The project becomes technically interesting when this works:

```text
PyTorch CUDA process
      │
      │ counter = 42
      │ tensor lives on GPU
      ▼
   edo freeze
      │
      ▼
 process gone
      │
      ▼
  edo summon
      │
      ▼
 counter = 43
 tensor still correct
 CUDA computation continues
```

Call this milestone:

> **First Resurrection**

Tag:

```text
v0.0.1-poc
```

---

# 39. v0.1 target LOC

Do not optimize for LOC, but keep the implementation small.

Expected:

```text
CLI / argument parsing      200–300
process utilities           150–300
CRIU wrapper                150–300
CUDA FFI + safe wrapper     300–500
snapshot manifest           150–250
coordinator                 200–400
error handling              150–250
tests/examples              500–1,000
------------------------------------
total                       ~1.8k–3.3k LOC
```

A clean PoC may be even smaller.

---

# 40. Project risks

## Risk 1 — CUDA driver compatibility

Checkpoint/restore is driver-sensitive.

Mitigation:

```text
edo doctor
strict compatibility metadata
clear support matrix
```

## Risk 2 — CRIU resource incompatibility

Sockets, special files, namespaces and external resources may fail restoration.

Mitigation:

```text
start with single process
no complex networking
add resource validation
```

## Risk 3 — snapshot size

GPU memory copied to host can make snapshots huge.

Mitigation:

```text
benchmark first
later separate weight lifecycle
later persistent VRAM
```

## Risk 4 — large restore still bandwidth-bound

Normal checkpoint does not magically eliminate:

```text
storage → RAM → VRAM
```

Mitigation:

```text
persistent-VRAM/VMM backend in later versions
```

## Risk 5 — framework complexity

vLLM/SGLang use multiprocessing, CUDA IPC, KV caches, graphs and possibly NCCL.

Mitigation:

```text
do not support them until single-process correctness is proven
```

---

# 41. Success definition

Edo Tensei succeeds in stages.

## Stage A

> CRIU round trip through Rust.

## Stage B

> CUDA checkpoint round trip through Rust.

## Stage C

> Combined PyTorch CUDA resurrection.

## Stage D

> Warm FastAPI inference server resumes without model reload.

## Stage E

> Persistent VRAM separates model-weight lifetime from worker lifetime.

## Stage F

> vLLM/SGLang can use Edo as a generic warm-resume primitive.

---

# 42. Final roadmap

```text
v0.0
│
├── Rust CLI
├── edo doctor
├── CRIU wrapper
└── CPU checkpoint demo
│
v0.0.1
│
├── CUDA Driver FFI
├── CUDA state machine
└── basic CUDA checkpoint
│
v0.0.2
│
├── combined freeze
├── combined summon
└── First Resurrection
│
v0.1
│
├── snapshot manifest
├── PyTorch
├── FastAPI
├── benchmarks
└── public OSS release
│
v0.2
│
├── process trees
├── containers
└── lifecycle hardening
│
v0.3
│
├── CUDA IPC
├── multi-process
└── first vLLM/SGLang experiments
│
v0.4
│
├── edod
├── CUDA VMM
└── persistent VRAM
│
v0.5+
│
├── immutable shared weights
├── ephemeral KV policy
├── stable VA mapping
├── fast warm resume
└── serverless GPU experiments
```

---

# 43. The rule for the project

Whenever the architecture starts getting complicated, return to this test:

```text
Can Edo reliably:

1. freeze one CUDA process
2. destroy it
3. restore it
4. continue execution
```

If not, do not add another abstraction yet.

---

# 44. References

Primary references to read while implementing:

1. **NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes**  
   https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/

2. **CUDA Driver API — CUDA Checkpointing**  
   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html

3. **NVIDIA Dynamo — Snapshotting GPU Workers**  
   https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/snapshot

4. **Dynamo DEP #12521 — Snapshot-coupled GPU Memory Service V1**  
   https://github.com/ai-dynamo/dynamo/issues/12521

5. **Dynamo Shadow Engine / GMS documentation**  
   https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/shadow-engine-failover

6. **CRIU**  
   https://criu.org/

---

# 45. Immediate next action

Start with exactly this:

```text
Day 1
├── cargo new edo-tensei
├── implement clap CLI
├── implement `edo doctor`
└── manually verify CRIU

Day 2
├── implement `edo cpu-dump`
├── implement `edo cpu-restore`
└── restore CPU counter

Day 3+
├── study CUDA checkpoint structs
├── write minimal Rust FFI
├── implement `edo gpu-state`
└── proceed one state transition at a time
```

The first objective is **not** sub-second inference.

The first objective is:

> **Make resurrection real.**
