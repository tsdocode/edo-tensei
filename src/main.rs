mod cli;
mod criu;
mod cuda;
mod doctor;
mod error;
mod process;
mod snapshot;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::generate;
use cli::{Cli, Command};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

fn main() -> Result<()> {
    init_logging();
    match Cli::parse().command {
        Command::Doctor { json } => doctor::run(json),
        Command::Run { name, command } => {
            process::start(&name, &command)?;
            Ok(())
        }
        Command::CpuDump { target, snapshot } => cpu_dump(&target, &snapshot),
        Command::CpuRestore { snapshot } => cpu_restore(&snapshot),
        Command::SnapshotCheck { snapshot } => snapshot_check(&snapshot),
        Command::HealthCheck { target, url } => health_check(&target, url.as_deref()),
        Command::SnapshotClean { snapshot, yes } => snapshot_clean(&snapshot, yes),
        Command::Completions { shell } => {
            let mut cli = Cli::command();
            generate(shell, &mut cli, "edo", &mut std::io::stdout());
            Ok(())
        }
        Command::CudaState { pid } => cuda_state(pid),
        Command::CudaRoundtrip {
            pid,
            timeout_ms,
            lock_timeout_ms,
        } => cuda_roundtrip(pid, timeout_ms, lock_timeout_ms),
        Command::Freeze {
            target,
            snapshot,
            timeout_ms,
            lock_timeout_ms,
        } => freeze(&target, &snapshot, timeout_ms, lock_timeout_ms),
        Command::Summon {
            snapshot,
            timeout_ms,
        } => summon(&snapshot, timeout_ms),
    }
}

fn cuda_state(pid: i32) -> Result<()> {
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()?;
    println!("CUDA process {pid}: {:?}", cuda.state(pid)?);
    if let Ok(thread_id) = cuda.restore_thread_id(pid) {
        println!("CUDA restore thread: {thread_id}");
    }
    Ok(())
}

fn cuda_roundtrip(pid: i32, timeout_ms: u64, lock_timeout_ms: u32) -> Result<()> {
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()?;
    let timeout = Duration::from_millis(timeout_ms);
    let poll = Duration::from_millis(25);
    cuda.wait_for_state(pid, cuda::ProcessState::Running, timeout, poll)?;
    println!("CUDA process {pid}: RUNNING");
    cuda.lock(pid, lock_timeout_ms)?;
    cuda.wait_for_state(pid, cuda::ProcessState::Locked, timeout, poll)?;
    println!("CUDA process {pid}: LOCKED");
    cuda.checkpoint(pid)?;
    cuda.wait_for_state(pid, cuda::ProcessState::Checkpointed, timeout, poll)?;
    println!("CUDA process {pid}: CHECKPOINTED");
    cuda.restore(pid)?;
    cuda.wait_for_state(pid, cuda::ProcessState::Locked, timeout, poll)?;
    println!("CUDA process {pid}: RESTORED and LOCKED");
    cuda.unlock(pid)?;
    cuda.wait_for_state(pid, cuda::ProcessState::Running, timeout, poll)?;
    println!("CUDA process {pid}: UNLOCKED and RUNNING");
    Ok(())
}

fn freeze(
    target: &str,
    snapshot_directory: &str,
    timeout_ms: u64,
    lock_timeout_ms: u32,
) -> Result<()> {
    let process = process::resolve(target)?;
    let directory = PathBuf::from(snapshot_directory);
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()?;
    let timeout = Duration::from_millis(timeout_ms);
    let poll = Duration::from_millis(25);
    journal(&directory, "freeze.started")?;

    cuda.wait_for_state(
        process.pid as i32,
        cuda::ProcessState::Running,
        timeout,
        poll,
    )?;
    println!("freeze: CUDA RUNNING → LOCKED");
    journal(&directory, "cuda.lock.started")?;
    cuda.lock(process.pid as i32, lock_timeout_ms)?;
    if let Err(error) = cuda.wait_for_state(
        process.pid as i32,
        cuda::ProcessState::Locked,
        timeout,
        poll,
    ) {
        let _ = recover_cuda(&cuda, process.pid as i32, timeout, poll);
        return Err(error).context("CUDA lock did not reach LOCKED state");
    }

    println!("freeze: LOCKED → CHECKPOINTED");
    journal(&directory, "cuda.checkpoint.started")?;
    if let Err(error) = cuda.checkpoint(process.pid as i32) {
        let _ = recover_cuda(&cuda, process.pid as i32, timeout, poll);
        return Err(error).context("CUDA checkpoint failed; process recovery was attempted");
    }
    if let Err(error) = cuda.wait_for_state(
        process.pid as i32,
        cuda::ProcessState::Checkpointed,
        timeout,
        poll,
    ) {
        let _ = recover_cuda(&cuda, process.pid as i32, timeout, poll);
        return Err(error)
            .context("CUDA checkpoint state was not reached; process recovery was attempted");
    }

    println!("freeze: CHECKPOINTED → CRIU DUMPING");
    journal(&directory, "criu.dump.started")?;
    if let Err(error) = criu::dump(process.pid, &directory) {
        let recovery = recover_cuda(&cuda, process.pid as i32, timeout, poll);
        return Err(error).context(format!(
            "CRIU dump failed; CUDA recovery result: {recovery:?}"
        ));
    }
    if let Err(error) = snapshot::write_kind(&directory, &process, "cuda-criu") {
        let _ = recover_cuda(&cuda, process.pid as i32, timeout, poll);
        return Err(error)
            .context("could not write CUDA snapshot manifest; CUDA recovery was attempted");
    }
    println!("CUDA+CRIU snapshot ready: {}", directory.display());
    journal(&directory, "snapshot.ready")?;
    Ok(())
}

fn summon(snapshot_directory: &str, timeout_ms: u64) -> Result<()> {
    let directory = PathBuf::from(snapshot_directory);
    let manifest = snapshot::read(&directory)?;
    snapshot::verify(&directory, &manifest)?;
    snapshot::require_cuda_kind(&manifest)?;
    criu::restore(&directory)?;
    let restored_pid: i32 = fs::read_to_string(directory.join("restored.pid"))?
        .trim()
        .parse()?;
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()?;
    let timeout = Duration::from_millis(timeout_ms);
    let poll = Duration::from_millis(25);
    journal(&directory, "summon.started")?;
    println!("summon: CRIU restored PID {restored_pid}; CUDA RESTORE");
    cuda.restore(restored_pid)?;
    cuda.wait_for_state(restored_pid, cuda::ProcessState::Locked, timeout, poll)?;
    cuda.unlock(restored_pid)?;
    cuda.wait_for_state(restored_pid, cuda::ProcessState::Running, timeout, poll)?;
    println!("CUDA+CRIU snapshot resumed: PID {restored_pid} RUNNING");
    journal(&directory, "summon.ready")?;
    Ok(())
}

fn recover_cuda(
    cuda: &cuda::CudaCheckpoint,
    pid: i32,
    timeout: Duration,
    poll: Duration,
) -> Result<()> {
    match cuda.state(pid)? {
        cuda::ProcessState::Checkpointed => {
            cuda.restore(pid)?;
            cuda.wait_for_state(pid, cuda::ProcessState::Locked, timeout, poll)?;
            cuda.unlock(pid)?;
        }
        cuda::ProcessState::Locked => cuda.unlock(pid)?,
        cuda::ProcessState::Running => {}
        state => anyhow::bail!("cannot recover CUDA process {pid} from state {state:?}"),
    }
    cuda.wait_for_state(pid, cuda::ProcessState::Running, timeout, poll)
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
    snapshot::verify(&directory, &manifest)?;
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

fn snapshot_check(snapshot_directory: &str) -> Result<()> {
    let directory = PathBuf::from(snapshot_directory);
    let manifest = snapshot::read(&directory)?;
    snapshot::verify(&directory, &manifest)?;
    println!(
        "snapshot compatible: kind={} schema={} files={} host={}",
        manifest.kind,
        manifest.schema_version,
        manifest.files.len(),
        manifest.host.hostname
    );
    Ok(())
}

fn health_check(target: &str, url: Option<&str>) -> Result<()> {
    let process = process::resolve(target)?;
    if let Some(url) = url {
        let output = std::process::Command::new("curl")
            .args(["--silent", "--show-error", "--fail", "--max-time", "5", url])
            .output()
            .with_context(|| "could not execute curl for health check")?;
        if !output.status.success() {
            anyhow::bail!(
                "health URL failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    println!("healthy: PID {} target {}", process.pid, target);
    Ok(())
}

fn snapshot_clean(snapshot_directory: &str, yes: bool) -> Result<()> {
    if !yes {
        anyhow::bail!("refusing to remove snapshot without --yes");
    }
    let directory = PathBuf::from(snapshot_directory);
    if !directory.is_dir() || directory.parent().is_none() {
        anyhow::bail!("snapshot path is not a directory: {}", directory.display());
    }
    fs::remove_dir_all(&directory)
        .with_context(|| format!("could not remove snapshot {}", directory.display()))?;
    println!("removed snapshot: {}", directory.display());
    Ok(())
}

fn journal(snapshot_directory: &Path, event: &str) -> Result<()> {
    let name = snapshot_directory
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("snapshot");
    let path = PathBuf::from(".edo")
        .join("journal")
        .join(format!("{name}.jsonl"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let entry = serde_json::json!({
        "event": event,
        "timestamp_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    });
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{entry}")?;
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
