use clap::{Parser, Subcommand};

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
        /// Emit machine-readable JSON instead of human-readable output.
        #[arg(long)]
        json: bool,
    },

    /// Start a managed process.
    Run {
        /// Stable Edo name used to resolve the process later.
        #[arg(long)]
        name: String,
        /// Command and arguments to execute.
        #[arg(required = true, trailing_var_arg = true)]
        command: Vec<String>,
    },

    /// Create a CPU-only CRIU snapshot.
    CpuDump {
        /// Managed process name or PID.
        target: String,
        /// Destination snapshot directory.
        snapshot: String,
    },

    /// Restore a CPU-only CRIU snapshot.
    CpuRestore {
        /// Snapshot directory.
        snapshot: String,
    },

    /// Freeze a managed CUDA process.
    Freeze {
        /// Managed process name or PID.
        target: String,
    },

    /// Restore a complete Edo snapshot.
    Summon {
        /// Snapshot directory.
        snapshot: String,
    },
}
