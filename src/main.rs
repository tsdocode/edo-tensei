mod cli;
mod criu;
mod doctor;
mod error;
mod process;
mod snapshot;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    init_logging();

    let cli = Cli::parse();
    match cli.command {
        Command::Doctor { json } => doctor::run(json),
        Command::Run { name, command } => {
            process::start(&name, &command)?;
            Ok(())
        }
        Command::CpuDump { target, snapshot } => cpu_dump(&target, &snapshot),
        Command::CpuRestore { snapshot } => cpu_restore(&snapshot),
        Command::Freeze { target } => {
            println!("freeze is not implemented yet: {target}");
            Ok(())
        }
        Command::Summon { snapshot } => {
            println!("summon is not implemented yet: {snapshot}");
            Ok(())
        }
    }
}

fn cpu_dump(target: &str, snapshot_directory: &str) -> Result<()> {
    let process = process::resolve(target)?;
    let directory = PathBuf::from(snapshot_directory);
    criu::dump(process.pid, &directory)?;
    snapshot::write(&directory, &process)?;
    println!(
        "CPU snapshot ready: {} (captured PID {})",
        directory.display(),
        process.pid
    );
    Ok(())
}

fn cpu_restore(snapshot_directory: &str) -> Result<()> {
    let directory = PathBuf::from(snapshot_directory);
    let manifest = snapshot::read(&directory)?;
    criu::restore(&directory)?;

    let restored_pid = fs::read_to_string(directory.join("restored.pid"))
        .context("CRIU restore succeeded but restored.pid was not written")?;
    println!(
        "CPU snapshot restored: {} (original PID {}, restored PID {})",
        manifest.process_name,
        manifest.captured_pid,
        restored_pid.trim()
    );
    Ok(())
}

fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "edo=info".into()),
        )
        .with_target(false)
        .init();
}
