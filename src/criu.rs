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
        cleanup_partial(directory);
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
    let _ = fs::remove_dir_all(directory);
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
