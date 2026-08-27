use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Deserialize, Serialize)]
pub struct ProcessRecord {
    pub name: String,
    pub pid: u32,
    pub command: Vec<String>,
    pub started_at_unix: u64,
    pub proc_start_time_ticks: u64,
    pub executable: String,
    pub cmdline: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct ProcessIdentity {
    start_time_ticks: u64,
    executable: String,
    cmdline: Vec<String>,
}

pub fn start(name: &str, command: &[String]) -> Result<ProcessRecord> {
    if name.is_empty() || command.is_empty() {
        return Err(anyhow!("a process name and command are required"));
    }

    let child = Command::new(&command[0])
        .args(&command[1..])
        .spawn()
        .with_context(|| format!("could not start {}", command[0]))?;
    // Wrapper commands such as setsid/env may exec into the target shortly after spawn.
    // Capture identity after that transition so later validation fingerprints the real app.
    thread::sleep(Duration::from_millis(100));
    let identity = read_identity(child.id())?;
    let record = ProcessRecord {
        name: name.to_owned(),
        pid: child.id(),
        command: command.to_vec(),
        started_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        proc_start_time_ticks: identity.start_time_ticks,
        executable: identity.executable,
        cmdline: identity.cmdline,
    };
    write_record(&record)?;
    println!("started {} with PID {}", record.name, record.pid);
    Ok(record)
}

pub fn resolve(target: &str) -> Result<ProcessRecord> {
    let record = target
        .parse::<u32>()
        .ok()
        .map(|pid| ProcessRecord {
            name: pid.to_string(),
            pid,
            command: Vec::new(),
            started_at_unix: 0,
            proc_start_time_ticks: 0,
            executable: String::new(),
            cmdline: Vec::new(),
        })
        .map_or_else(|| read_record(target), Ok)?;

    let identity = read_identity(record.pid)?;
    if record.proc_start_time_ticks != 0
        && (record.proc_start_time_ticks != identity.start_time_ticks
            || record.executable != identity.executable
            || record.cmdline != identity.cmdline)
    {
        return Err(anyhow!(
            "target {} no longer matches its recorded process identity",
            target
        ));
    }
    Ok(record)
}

pub fn record_path(name: &str) -> PathBuf {
    PathBuf::from(".edo")
        .join("runs")
        .join(format!("{name}.json"))
}

/// Return the root process and all descendants currently visible in its PID namespace.
pub fn tree(root_pid: u32) -> Result<Vec<ProcessRecord>> {
    let mut pending = vec![root_pid];
    let mut records = Vec::new();
    while let Some(pid) = pending.pop() {
        let identity = read_identity(pid)?;
        records.push(ProcessRecord {
            name: pid.to_string(),
            pid,
            command: identity.cmdline.clone(),
            started_at_unix: 0,
            proc_start_time_ticks: identity.start_time_ticks,
            executable: identity.executable,
            cmdline: identity.cmdline,
        });
        let children =
            fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")).unwrap_or_default();
        pending.extend(
            children
                .split_whitespace()
                .filter_map(|child| child.parse::<u32>().ok()),
        );
    }
    Ok(records)
}

/// Find a restored process when a runtime changes the parent/PID namespace
/// relationship during CRIU restore.
pub fn find_by_identity(executable: &str, cmdline: &[String]) -> Result<Vec<ProcessRecord>> {
    let mut records = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(pid) = name.parse::<u32>() else { continue };
        let Ok(identity) = read_identity(pid) else { continue };
        if identity.executable == executable && identity.cmdline == cmdline {
            records.push(ProcessRecord {
                name: pid.to_string(),
                pid,
                command: identity.cmdline.clone(),
                started_at_unix: 0,
                proc_start_time_ticks: identity.start_time_ticks,
                executable: identity.executable,
                cmdline: identity.cmdline,
            });
        }
    }
    Ok(records)
}

fn write_record(record: &ProcessRecord) -> Result<()> {
    let path = record_path(&record.name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(record)?)?;
    Ok(())
}

fn read_record(name: &str) -> Result<ProcessRecord> {
    let path = record_path(name);
    let bytes = fs::read(&path)
        .with_context(|| format!("could not read process record {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn read_identity(pid: u32) -> Result<ProcessIdentity> {
    let proc_dir = PathBuf::from(format!("/proc/{pid}"));
    if !proc_dir.exists() {
        return Err(anyhow!("process PID {pid} is not running"));
    }

    let stat = fs::read_to_string(proc_dir.join("stat"))
        .with_context(|| format!("could not read /proc/{pid}/stat"))?;
    let closing_paren = stat
        .rfind(')')
        .ok_or_else(|| anyhow!("invalid /proc/{pid}/stat contents"))?;
    let fields: Vec<_> = stat[closing_paren + 2..].split_whitespace().collect();
    let start_time_ticks = fields
        .get(19)
        .ok_or_else(|| anyhow!("missing process start time for PID {pid}"))?
        .parse()?;

    let executable = fs::read_link(proc_dir.join("exe"))
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cmdline = fs::read(proc_dir.join("cmdline"))
        .unwrap_or_default()
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).into_owned())
        .collect();

    Ok(ProcessIdentity {
        start_time_ticks,
        executable,
        cmdline,
    })
}

#[cfg(test)]
mod tests {
    use super::read_identity;

    #[test]
    fn current_process_has_a_stable_proc_identity() {
        let identity = read_identity(std::process::id()).expect("current process identity");
        assert!(identity.start_time_ticks > 0);
        assert!(!identity.executable.is_empty());
        assert!(!identity.cmdline.is_empty());
    }
}
