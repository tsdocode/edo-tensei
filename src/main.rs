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
    sync::Arc,
    thread,
    time::Duration,
};

fn main() -> Result<()> {
    init_logging();
    match Cli::parse().command {
        Command::Doctor { json } => doctor::run(json),
        Command::Demo { name } => demo(&name),
        Command::Status { target, url } => health_check(&target, url.as_deref()),
        Command::Run { name, command } => {
            process::start(&name, &command)?;
            Ok(())
        }
        Command::CpuDump { target, snapshot } => cpu_dump(&target, &snapshot),
        Command::CpuRestore { snapshot } => cpu_restore(&snapshot),
        Command::Checkpoint { target, snapshot } => cpu_dump(&target, &snapshot),
        Command::Restore { snapshot } => cpu_restore(&snapshot),
        Command::Inspect { snapshot } => snapshot_check(&snapshot),
        Command::Diff { first, second } => snapshot_diff(&first, &second),
        Command::SnapshotCheck { snapshot } => snapshot_check(&snapshot),
        Command::HealthCheck { target, url } => health_check(&target, url.as_deref()),
        Command::SnapshotClean { snapshot, yes } => snapshot_clean(&snapshot, yes),
        Command::Completions { shell } => {
            let mut cli = Cli::command();
            generate(shell, &mut cli, "edo", &mut std::io::stdout());
            Ok(())
        }
        Command::CudaState { pid } => cuda_state(pid),
        Command::CudaInit => cuda_init(),
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
        Command::FreezeGroup {
            root,
            cuda_pids,
            snapshot,
            timeout_ms,
            lock_timeout_ms,
        } => freeze_group(&root, &cuda_pids, &snapshot, timeout_ms, lock_timeout_ms),
        Command::Summon {
            snapshot,
            timeout_ms,
            skip_integrity,
        } => summon(&snapshot, timeout_ms, skip_integrity),
        Command::SummonGroup {
            snapshot,
            timeout_ms,
            skip_integrity,
        } => summon_group(&snapshot, timeout_ms, skip_integrity),
    }
}

fn demo(name: &str) -> Result<()> {
    if name != "resume" {
        anyhow::bail!("unknown demo '{name}'; available demos: resume");
    }
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/00_hello_checkpoint/run.sh");
    let status = std::process::Command::new("bash")
        .arg(script)
        .status()
        .context("could not start the resume demo")?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("resume demo exited with {status}");
    }
}

fn snapshot_diff(first: &str, second: &str) -> Result<()> {
    let left = snapshot::read(Path::new(first))?;
    let right = snapshot::read(Path::new(second))?;
    println!("snapshot A: {} ({})", first, left.kind);
    println!("snapshot B: {} ({})", second, right.kind);
    println!(
        "architecture: {} → {}",
        left.host.architecture, right.host.architecture
    );
    println!("GPU: {:?} → {:?}", left.host.gpu_name, right.host.gpu_name);
    println!("image files: {} → {}", left.files.len(), right.files.len());
    println!("process: {} → {}", left.process_name, right.process_name);
    Ok(())
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

fn cuda_init() -> Result<()> {
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()
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

fn summon(snapshot_directory: &str, timeout_ms: u64, skip_integrity: bool) -> Result<()> {
    let directory = PathBuf::from(snapshot_directory);
    let manifest = snapshot::read(&directory)?;
    snapshot::verify_with_options(&directory, &manifest, !skip_integrity)?;
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

fn freeze_group(
    root_target: &str,
    cuda_pid_list: &str,
    snapshot_directory: &str,
    timeout_ms: u64,
    lock_timeout_ms: u32,
) -> Result<()> {
    let root = process::resolve(root_target)?;
    let tree = process::tree(root.pid)?;
    let cuda_pids = parse_pids(cuda_pid_list)?;
    let cuda_records = cuda_pids
        .iter()
        .map(|pid| {
            tree.iter()
                .find(|record| record.pid == *pid)
                .ok_or_else(|| {
                    anyhow::anyhow!("CUDA PID {pid} is not a descendant of root {}", root.pid)
                })
                .map(|record| snapshot::CudaProcessRecord {
                    pid: record.pid,
                    proc_start_time_ticks: record.proc_start_time_ticks,
                    executable: record.executable.clone(),
                    cmdline: record.cmdline.clone(),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let cuda = cuda::CudaCheckpoint::load()?;
    cuda.initialize()?;
    let timeout = Duration::from_millis(timeout_ms);
    let poll = Duration::from_millis(25);
    for record in &cuda_records {
        eprintln!(
            "freeze-group: waiting for CUDA PID {} to become RUNNING",
            record.pid
        );
        cuda.wait_for_state(
            record.pid as i32,
            cuda::ProcessState::Running,
            timeout,
            poll,
        )?;
    }
    let mut locked = Vec::new();
    for record in &cuda_records {
        eprintln!(
            "freeze-group: locking CUDA PID {} (timeout={}ms)",
            record.pid, lock_timeout_ms
        );
        if let Err(error) = cuda.lock(record.pid as i32, lock_timeout_ms) {
            let _ = recover_many(&cuda, &locked, timeout, poll);
            return Err(error).context(format!("CUDA group lock failed for PID {}", record.pid));
        }
        eprintln!(
            "freeze-group: CUDA PID {} lock returned; waiting for LOCKED",
            record.pid
        );
        if let Err(error) =
            cuda.wait_for_state(record.pid as i32, cuda::ProcessState::Locked, timeout, poll)
        {
            let _ = recover_many(&cuda, &locked, timeout, poll);
            return Err(error).context(format!(
                "CUDA group lock state failed for PID {}",
                record.pid
            ));
        }
        locked.push(record.pid as i32);
    }
    for record in &cuda_records {
        eprintln!("freeze-group: checkpointing CUDA PID {}", record.pid);
        if let Err(error) = cuda.checkpoint(record.pid as i32) {
            let _ = recover_many(&cuda, &locked, timeout, poll);
            return Err(error).context(format!(
                "CUDA group checkpoint failed for PID {}",
                record.pid
            ));
        }
        eprintln!(
            "freeze-group: CUDA PID {} checkpoint returned; waiting for CHECKPOINTED",
            record.pid
        );
        if let Err(error) = cuda.wait_for_state(
            record.pid as i32,
            cuda::ProcessState::Checkpointed,
            timeout,
            poll,
        ) {
            let _ = recover_many(&cuda, &locked, timeout, poll);
            return Err(error).context(format!(
                "CUDA group checkpoint state failed for PID {}",
                record.pid
            ));
        }
    }
    let directory = PathBuf::from(snapshot_directory);
    if let Err(error) = criu::dump_group(root.pid, &directory) {
        let _ = recover_many(&cuda, &locked, timeout, poll);
        return Err(error).context("CRIU group dump failed; CUDA recovery was attempted");
    }
    // Group dumps use CRIU's live-checkpoint mode so a Kubernetes container
    // init process is not reaped by the node-local agent. Resume the original
    // CUDA processes after the checkpoint has been materialized.
    if let Err(error) = recover_many(&cuda, &locked, timeout, poll) {
        return Err(error).context("CUDA recovery failed after CRIU group dump");
    }
    snapshot::write_group(&directory, &root, "cuda-criu-group", cuda_records)?;
    println!(
        "CUDA+CRIU process-group snapshot ready: {}",
        directory.display()
    );
    Ok(())
}

fn summon_group(snapshot_directory: &str, _timeout_ms: u64, skip_integrity: bool) -> Result<()> {
    let summon_started = std::time::Instant::now();
    let directory = PathBuf::from(snapshot_directory);
    let manifest = snapshot::read(&directory)?;
    snapshot::verify_with_options(&directory, &manifest, !skip_integrity)?;
    let verification_time = summon_started.elapsed();
    eprintln!(
        "summon-group timing: snapshot verification {:.3}s (skipped={})",
        verification_time.as_secs_f64(),
        skip_integrity
    );
    snapshot::require_group_kind(&manifest)?;
    // Load/initialize the driver before CRIU starts. This work is independent
    // of the restored target processes and avoids adding it to the serving
    // critical path after CRIU resumes them.
    let restore_started = std::time::Instant::now();
    let load_started = std::time::Instant::now();
    let cuda = Arc::new(cuda::CudaCheckpoint::load()?);
    let cuda_loaded = load_started.elapsed();
    let init_started = std::time::Instant::now();
    cuda.initialize()?;
    let cuda_initialized = init_started.elapsed();
    eprintln!(
        "summon-group timing: CUDA library load {:.3}s, cuInit {:.3}s",
        cuda_loaded.as_secs_f64(),
        cuda_initialized.as_secs_f64(),
    );
    let criu_started = std::time::Instant::now();
    criu::restore(&directory)?;
    let criu_restore_time = criu_started.elapsed();
    let restored_root: u32 = fs::read_to_string(directory.join("restored.pid"))?
        .trim()
        .parse()?;
    let mut restored_tree = process::tree(restored_root)?;
    let mut restored_pids = Vec::new();
    let mut used = std::collections::HashSet::new();
    for expected in &manifest.cuda_processes {
        // CRIU may report a PID from the restored child PID namespace in its
        // tree. CUDA driver ioctls require the host-visible /proc PID, so
        // prefer the node's global process view whenever it is available.
        let host_records =
            process::find_by_identity(&expected.executable, &expected.cmdline).unwrap_or_default();
        if !host_records.is_empty() {
            restored_tree.extend(host_records);
        } else if !restored_tree.iter().any(|record| {
            !used.contains(&record.pid)
                && record.executable == expected.executable
                && record.cmdline == expected.cmdline
        }) {
            // Keep the CRIU tree fallback for runtimes that do not expose a
            // host-visible process entry yet.
        }
        let match_record = restored_tree
            .iter()
            .filter(|record| {
                !used.contains(&record.pid)
                    && record.executable == expected.executable
                    && record.cmdline == expected.cmdline
            })
            .max_by_key(|record| record.pid)
            .ok_or_else(|| {
                anyhow::anyhow!("could not map restored CUDA process {:?}", expected.cmdline)
            })?;
        used.insert(match_record.pid);
        restored_pids.push(match_record.pid as i32);
    }
    let cuda_restore_started = std::time::Instant::now();
    parallel_cuda_calls(&cuda, &restored_pids, "restore", |cuda, pid| {
        cuda.restore(pid)
    })?;
    let cuda_restore_time = cuda_restore_started.elapsed();
    let cuda_unlock_started = std::time::Instant::now();
    parallel_cuda_calls(&cuda, &restored_pids, "unlock", |cuda, pid| {
        cuda.unlock(pid)
    })?;
    let cuda_unlock_time = cuda_unlock_started.elapsed();
    eprintln!(
        "summon-group timing: verify {:.3}s, CUDA load {:.3}s, cuInit {:.3}s, CRIU restore {:.3}s, CUDA restore+unlock {:.3}s, total {:.3}s",
        verification_time.as_secs_f64(),
        cuda_loaded.as_secs_f64(),
        cuda_initialized.as_secs_f64(),
        criu_restore_time.as_secs_f64(),
        (cuda_restore_time + cuda_unlock_time).as_secs_f64(),
        restore_started.elapsed().as_secs_f64(),
    );
    println!(
        "CUDA+CRIU process group resumed: root PID {} workers {}",
        restored_root,
        restored_pids.len()
    );
    Ok(())
}

fn parallel_cuda_calls<F>(
    cuda: &Arc<cuda::CudaCheckpoint>,
    pids: &[i32],
    operation: &'static str,
    call: F,
) -> Result<()>
where
    F: Fn(&cuda::CudaCheckpoint, i32) -> Result<()> + Send + Sync + 'static,
{
    let call = Arc::new(call);
    let handles = pids
        .iter()
        .copied()
        .map(|pid| {
            let cuda = Arc::clone(cuda);
            let call = Arc::clone(&call);
            thread::spawn(move || {
                call(&cuda, pid).with_context(|| format!("CUDA {operation} failed for PID {pid}"))
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("CUDA {operation} worker panicked"))??;
    }
    Ok(())
}

fn parse_pids(value: &str) -> Result<Vec<u32>> {
    let mut pids = value
        .split(',')
        .map(|pid| {
            pid.trim()
                .parse::<u32>()
                .with_context(|| format!("invalid CUDA PID '{pid}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    pids.sort_unstable();
    pids.dedup();
    if pids.is_empty() {
        anyhow::bail!("at least one CUDA PID is required");
    }
    Ok(pids)
}

fn recover_many(
    cuda: &cuda::CudaCheckpoint,
    pids: &[i32],
    timeout: Duration,
    poll: Duration,
) -> Result<()> {
    for pid in pids {
        recover_cuda(cuda, *pid, timeout, poll)?;
    }
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
