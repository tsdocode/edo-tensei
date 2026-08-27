use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use crate::process::ProcessRecord;

#[derive(Debug, Deserialize, Serialize)]
pub struct SnapshotManifest {
    pub schema_version: u32,
    pub kind: String,
    pub edo_version: String,
    pub created_at_unix: u64,
    pub captured_pid: u32,
    pub process_name: String,
    pub command: Vec<String>,
    pub proc_start_time_ticks: u64,
    pub executable: String,
    pub cmdline: Vec<String>,
    pub working_directory: String,
    pub environment_policy: String,
    pub process_tree: Vec<u32>,
    #[serde(default)]
    pub cuda_processes: Vec<CudaProcessRecord>,
    pub host: HostManifest,
    pub files: Vec<SnapshotFile>,
    pub snapshot_directory: PathBuf,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CudaProcessRecord {
    pub pid: u32,
    pub proc_start_time_ticks: u64,
    pub executable: String,
    pub cmdline: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HostManifest {
    pub hostname: String,
    pub kernel: String,
    pub architecture: String,
    pub criu_version: String,
    pub cuda_driver_version: Option<String>,
    pub gpu_uuid: Option<String>,
    pub gpu_name: Option<String>,
    pub gpu_compute_capability: Option<String>,
    pub gpu_memory_bytes: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SnapshotFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

pub fn write(directory: &Path, process: &ProcessRecord) -> Result<()> {
    write_kind(directory, process, "cpu-criu")
}

pub fn write_kind(directory: &Path, process: &ProcessRecord, kind: &str) -> Result<()> {
    write_group(directory, process, kind, Vec::new())
}

pub fn write_group(
    directory: &Path,
    process: &ProcessRecord,
    kind: &str,
    cuda_processes: Vec<CudaProcessRecord>,
) -> Result<()> {
    let files = snapshot_files(directory)?;
    let manifest = SnapshotManifest {
        schema_version: 2,
        kind: kind.to_owned(),
        edo_version: env!("CARGO_PKG_VERSION").to_owned(),
        created_at_unix: unix_time(),
        captured_pid: process.pid,
        process_name: process.name.clone(),
        command: process.command.clone(),
        proc_start_time_ticks: process.proc_start_time_ticks,
        executable: process.executable.clone(),
        cmdline: process.cmdline.clone(),
        working_directory: fs::read_link(format!("/proc/{}/cwd", process.pid))
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        environment_policy: "not captured; inherited environment must be recreated by the launcher"
            .to_owned(),
        process_tree: process_tree(process.pid),
        cuda_processes,
        host: host_manifest(),
        files,
        snapshot_directory: directory.to_path_buf(),
    };
    secure_directory(directory)?;
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    secure_file(&directory.join("manifest.json"))?;
    Ok(())
}

pub fn read(directory: &Path) -> Result<SnapshotManifest> {
    let path = directory.join("manifest.json");
    let bytes = fs::read(&path).with_context(|| format!("could not read {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn require_cuda_kind(manifest: &SnapshotManifest) -> Result<()> {
    if manifest.kind != "cuda-criu" {
        bail!(
            "snapshot kind '{}' is not a CUDA checkpoint snapshot",
            manifest.kind
        );
    }
    Ok(())
}

pub fn require_group_kind(manifest: &SnapshotManifest) -> Result<()> {
    if manifest.kind != "cuda-criu-group" {
        bail!(
            "snapshot kind '{}' is not a CUDA process-group snapshot",
            manifest.kind
        );
    }
    if manifest.cuda_processes.is_empty() {
        bail!("CUDA process-group snapshot has no recorded CUDA processes");
    }
    Ok(())
}

pub fn verify(directory: &Path, manifest: &SnapshotManifest) -> Result<()> {
    verify_with_options(directory, manifest, true)
}

pub fn verify_with_options(
    directory: &Path,
    manifest: &SnapshotManifest,
    check_integrity: bool,
) -> Result<()> {
    if manifest.schema_version != 2 {
        bail!(
            "unsupported snapshot schema {}; expected 2",
            manifest.schema_version
        );
    }
    let current = host_manifest();
    if manifest.host.architecture != current.architecture {
        bail!(
            "snapshot architecture '{}' does not match current '{}'",
            manifest.host.architecture,
            current.architecture
        );
    }
    if manifest.host.kernel != current.kernel {
        bail!(
            "snapshot kernel '{}' does not match current '{}'",
            manifest.host.kernel,
            current.kernel
        );
    }
    if manifest.host.criu_version != current.criu_version {
        bail!(
            "snapshot CRIU '{}' does not match current '{}'",
            manifest.host.criu_version,
            current.criu_version
        );
    }
    if matches!(manifest.kind.as_str(), "cuda-criu" | "cuda-criu-group") {
        if let (Some(required), Some(available)) =
            (&manifest.host.cuda_driver_version, &current.cuda_driver_version)
        {
            if required != available {
                bail!("snapshot NVIDIA driver does not match current driver");
            }
        }
        if let (Some(required), Some(available)) = (&manifest.host.gpu_name, &current.gpu_name) {
            if required != available {
                bail!("snapshot GPU model does not match current GPU");
            }
        }
        if let (Some(required), Some(available)) =
            (manifest.host.gpu_memory_bytes, current.gpu_memory_bytes)
        {
            if available < required {
                bail!("current GPU memory is insufficient for this snapshot");
            }
        }
    }
    for expected in &manifest.files {
        let path = directory.join(&expected.path);
        let metadata = fs::metadata(&path)
            .with_context(|| format!("snapshot file is missing: {}", expected.path))?;
        if metadata.len() != expected.size {
            bail!("snapshot file size changed: {}", expected.path);
        }
        if check_integrity {
            let actual = sha256_file(&path)?;
            if actual != expected.sha256 {
                bail!("snapshot integrity check failed: {}", expected.path);
            }
        }
    }
    Ok(())
}

fn host_manifest() -> HostManifest {
    HostManifest {
        hostname: command_output("hostname", &[]).unwrap_or_else(|| "unknown".to_owned()),
        kernel: command_output("uname", &["-r"]).unwrap_or_else(|| "unknown".to_owned()),
        architecture: std::env::consts::ARCH.to_owned(),
        criu_version: command_output(
            &std::env::var("EDO_CRIU").unwrap_or_else(|_| "criu".to_owned()),
            &["--version"],
        )
            .unwrap_or_else(|| "unknown".to_owned()),
        cuda_driver_version: command_output(
            "nvidia-smi",
            &[
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
        ),
        gpu_uuid: command_output(
            "nvidia-smi",
            &["--query-gpu=uuid", "--format=csv,noheader,nounits"],
        ),
        gpu_name: command_output(
            "nvidia-smi",
            &["--query-gpu=name", "--format=csv,noheader,nounits"],
        ),
        gpu_compute_capability: command_output(
            "nvidia-smi",
            &["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
        ),
        gpu_memory_bytes: command_output(
            "nvidia-smi",
            &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        )
        .and_then(|value| value.parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024),
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn snapshot_files(directory: &Path) -> Result<Vec<SnapshotFile>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.file_name().and_then(|name| name.to_str()) == Some("manifest.json")
            || !path.is_file()
        {
            continue;
        }
        paths.push(path);
    }
    paths.sort();
    let started = Instant::now();
    let total = paths.len();
    let mut files = Vec::with_capacity(total);
    for (index, path) in paths.into_iter().enumerate() {
        let metadata = fs::metadata(&path)?;
        eprintln!(
            "snapshot manifest: hashing {}/{} {} ({} bytes)",
            index + 1,
            total,
            path.file_name().unwrap_or_default().to_string_lossy(),
            metadata.len()
        );
        files.push(SnapshotFile {
            path: path
                .strip_prefix(directory)
                .map_err(|error| anyhow!(error))?
                .to_string_lossy()
                .into_owned(),
            size: metadata.len(),
            sha256: sha256_file(&path)?,
        });
        eprintln!(
            "snapshot manifest: hashed {}/{} in {:.1}s",
            index + 1,
            total,
            started.elapsed().as_secs_f32()
        );
    }
    Ok(files)
}

fn sha256_file(path: &Path) -> Result<String> {
    // GNU coreutils uses the platform's optimized SHA implementation (SHA-NI
    // on the test host). This is substantially faster than the portable
    // Rust implementation for multi-hundred-MiB CRIU page images. Keep the
    // Rust path as a portable fallback for minimal images.
    if let Ok(output) = Command::new("sha256sum").arg(path).output() {
        if output.status.success() {
            if let Some(hash) = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
            {
                if hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Ok(hash.to_owned());
                }
            }
        }
    }
    let mut file =
        fs::File::open(path).with_context(|| format!("could not read {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn process_tree(pid: u32) -> Vec<u32> {
    let mut tree = Vec::new();
    let mut current = pid;
    for _ in 0..32 {
        tree.push(current);
        let Ok(stat) = fs::read_to_string(format!("/proc/{current}/stat")) else {
            break;
        };
        let Some(end) = stat.rfind(')') else { break };
        let Some(ppid) = stat[end + 2..]
            .split_whitespace()
            .nth(1)
            .and_then(|value| value.parse().ok())
        else {
            break;
        };
        if ppid == 0 || ppid == current {
            break;
        }
        current = ppid;
    }
    tree
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn secure_directory(directory: &Path) -> Result<()> {
    #[cfg(unix)]
    fs::set_permissions(directory, permissions(0o700))?;
    Ok(())
}

fn secure_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    fs::set_permissions(path, permissions(0o600))?;
    Ok(())
}

#[cfg(unix)]
fn permissions(mode: u32) -> fs::Permissions {
    use std::os::unix::fs::PermissionsExt;
    fs::Permissions::from_mode(mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn hashes_snapshot_files() {
        let path = std::env::temp_dir().join(format!(
            "edo-snapshot-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("directory");
        let mut file = fs::File::create(path.join("image")).expect("file");
        file.write_all(b"checkpoint").expect("write");
        let files = snapshot_files(&path).expect("files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].size, 10);
        assert_eq!(files[0].sha256.len(), 64);
        fs::remove_dir_all(path).expect("cleanup");
    }
}
