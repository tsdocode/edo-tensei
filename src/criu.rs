use anyhow::{anyhow, Context, Result};
use std::{fs, path::Path, process::Command};

pub fn dump(pid: u32, directory: &Path) -> Result<()> {
    prepare_directory(directory)?;
    let result = run(
        Command::new("criu")
            .arg("dump")
            .arg("-t")
            .arg(pid.to_string())
            .arg("--images-dir")
            .arg(directory)
            .arg("--shell-job")
            // Recreate unlinked POSIX shared-memory/semaphore mappings.
            .arg("--link-remap")
            // Preserve accepted TCP connections owned by the server.
            .arg("--tcp-established")
            .arg("--log-file")
            .arg(directory.join("dump.log"))
            .arg("--verbosity=4"),
        "CRIU dump",
    );
    if result.is_err() {
        let result = if let Ok(log) = fs::read_to_string(directory.join("dump.log")) {
            let tail = log
                .lines()
                .rev()
                .take(40)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            if !tail.is_empty() {
                result.context(format!("CRIU dump log:\n{tail}"))
            } else {
                result
            }
        } else {
            result
        };
        cleanup_partial(directory);
        return result;
    }
    result
}

pub fn restore(directory: &Path) -> Result<()> {
    let pidfile = directory.join("restored.pid");
    run(
        Command::new("criu")
            .arg("restore")
            .arg("--images-dir")
            .arg(directory)
            .arg("--restore-detached")
            .arg("--shell-job")
            .arg("--tcp-established")
            .arg("--pidfile")
            .arg(&pidfile)
            .arg("--log-file")
            .arg(directory.join("restore.log"))
            .arg("--verbosity=4"),
        "CRIU restore",
    )
}

fn prepare_directory(directory: &Path) -> Result<()> {
    if directory.exists() {
        return Err(anyhow!(
            "snapshot directory already exists: {}",
            directory.display()
        ));
    }
    fs::create_dir_all(directory)?;
    Ok(())
}

fn cleanup_partial(directory: &Path) {
    // Partial CRIU images are never usable. Keep only the diagnostic log so a
    // failed integration test can explain the rejection without retaining
    // process-memory fragments.
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            if entry.file_name() != "dump.log" {
                let path = entry.path();
                if path.is_dir() {
                    let _ = fs::remove_dir_all(path);
                } else {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
}

fn run(command: &mut Command, operation: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("could not execute {operation}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow!(
            "{operation} failed with {}: {}",
            output.status,
            format!("{stdout}{stderr}").trim()
        ))
    }
}
