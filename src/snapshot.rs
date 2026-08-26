use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::process::ProcessRecord;

#[derive(Debug, Deserialize, Serialize)]
pub struct SnapshotManifest {
    pub schema_version: u32,
    pub kind: String,
    pub edo_version: String,
    pub captured_pid: u32,
    pub process_name: String,
    pub command: Vec<String>,
    pub proc_start_time_ticks: u64,
    pub executable: String,
    pub cmdline: Vec<String>,
    pub snapshot_directory: PathBuf,
}

pub fn write(directory: &Path, process: &ProcessRecord) -> Result<()> {
    write_kind(directory, process, "cpu-criu")
}

pub fn write_kind(directory: &Path, process: &ProcessRecord, kind: &str) -> Result<()> {
    let manifest = SnapshotManifest {
        schema_version: 1,
        kind: kind.to_owned(),
        edo_version: env!("CARGO_PKG_VERSION").to_owned(),
        captured_pid: process.pid,
        process_name: process.name.clone(),
        command: process.command.clone(),
        proc_start_time_ticks: process.proc_start_time_ticks,
        executable: process.executable.clone(),
        cmdline: process.cmdline.clone(),
        snapshot_directory: directory.to_path_buf(),
    };
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
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
