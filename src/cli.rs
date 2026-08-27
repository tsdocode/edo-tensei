use clap::{Parser, Subcommand};
use clap_complete::Shell;

#[derive(Debug, Parser)]
#[command(
    name = "edo",
    version,
    about = "GPU process snapshot and fast-resume runtime"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect host capabilities required by Edo.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Start a managed process.
    Run {
        #[arg(long)]
        name: String,
        #[arg(required = true, trailing_var_arg = true)]
        command: Vec<String>,
    },
    /// Create a CPU-only CRIU snapshot.
    CpuDump { target: String, snapshot: String },
    /// Restore a CPU-only CRIU snapshot.
    CpuRestore { snapshot: String },
    /// Validate snapshot compatibility, permissions, and image checksums.
    SnapshotCheck { snapshot: String },
    /// Check that a managed process is alive and optionally probe an HTTP health URL.
    HealthCheck {
        target: String,
        #[arg(long)]
        url: Option<String>,
    },
    /// Remove a snapshot directory after an explicit confirmation flag.
    SnapshotClean {
        snapshot: String,
        #[arg(long)]
        yes: bool,
    },
    /// Generate shell completions.
    Completions { shell: Shell },
    /// Query the CUDA checkpoint state of a process without changing it.
    CudaState { pid: i32 },
    /// Initialize the CUDA checkpoint driver before a restore request arrives.
    CudaInit,
    /// Run CUDA lock, checkpoint, restore, and unlock on a native CUDA process.
    CudaRoundtrip {
        pid: i32,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u64,
        #[arg(long, default_value_t = 5_000)]
        lock_timeout_ms: u32,
    },
    /// Lock CUDA, checkpoint GPU state, then dump the CPU process with CRIU.
    Freeze {
        /// Managed process name or PID.
        target: String,
        /// Destination CUDA+CRIU snapshot directory.
        snapshot: String,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u64,
        #[arg(long, default_value_t = 5_000)]
        lock_timeout_ms: u32,
    },
    /// Lock CUDA in a process group, then dump the complete CRIU descendant tree.
    FreezeGroup {
        /// API/root process name or PID.
        root: String,
        /// Comma-separated CUDA-owning process PIDs, including the root when applicable.
        cuda_pids: String,
        /// Destination CUDA+CRIU group snapshot directory.
        snapshot: String,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u64,
        #[arg(long, default_value_t = 5_000)]
        lock_timeout_ms: u32,
    },
    /// Restore a CUDA+CRIU snapshot and resume the GPU process.
    Summon {
        snapshot: String,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u64,
        /// Skip per-image SHA-256 verification for a trusted local snapshot.
        #[arg(long)]
        skip_integrity: bool,
    },
    /// Restore a CUDA+CRIU process group and resume every recorded CUDA worker.
    SummonGroup {
        snapshot: String,
        #[arg(long, default_value_t = 10_000)]
        timeout_ms: u64,
        /// Skip per-image SHA-256 verification for a trusted local snapshot.
        #[arg(long)]
        skip_integrity: bool,
    },
}
