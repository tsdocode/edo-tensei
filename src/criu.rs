use anyhow::{anyhow, Context, Result};
use std::{fs, path::Path, process::Command};

fn criu_program() -> std::ffi::OsString {
    std::env::var_os("EDO_CRIU").unwrap_or_else(|| "criu".into())
}

fn clear_link_remap_artifacts() {
    // CRIU creates these host-side names for deleted shared-memory objects.
    // A failed dump can leave one behind and make the next dump fail with
    // EEXIST. They contain no process state and are safe to remove before a
    // new dump; restrict cleanup to CRIU's exact generated prefix.
    if let Ok(entries) = fs::read_dir("/dev/shm") {
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("link_remap.")
            {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

pub fn dump(pid: u32, directory: &Path) -> Result<()> {
    dump_inner(pid, directory, &[])
}

pub fn dump_group(pid: u32, directory: &Path) -> Result<()> {
    let skipped_mounts = vllm_mounts(pid);
    dump_inner(pid, directory, &skipped_mounts)
}

fn dump_inner(pid: u32, directory: &Path, skipped_mounts: &[String]) -> Result<()> {
    prepare_directory(directory)?;
    clear_link_remap_artifacts();
    let mut command = Command::new(criu_program());
    command
        .arg("dump")
        .arg("-t")
        .arg(pid.to_string())
        .arg("--images-dir")
        .arg(directory)
        .arg("--shell-job")
        // A namespace-entered Kubernetes agent must not wait for PID 1 to
        // exit after dumping. The runtime owns that init process; keep the
        // checkpoint source alive and let Edo restore/unlock CUDA below.
        .arg("--leave-running")
        // Recreate unlinked POSIX shared-memory/semaphore mappings.
        .arg("--link-remap")
        // Preserve accepted TCP connections owned by the server.
        .arg("--tcp-established")
        // Kubernetes owns the target cgroup hierarchy. Dumping its cgroup
        // properties from a namespace-entered agent can block after CRIU has
        // already written the process images, so leave cgroup recreation to
        // the runtime and only checkpoint the process state.
        .arg("--manage-cgroups=ignore")
        // vLLM mounts rebuildable compilation/model caches into /root/.cache.
        // They are host-side artifacts, not process state, and CRIU cannot
        // checkpoint these bind mounts when their source is outside the
        // process root.
        .arg("--log-file")
        .arg(directory.join("dump.log"))
        .arg("--verbosity=4");
    for mount in skipped_mounts {
        command.args(["--skip-mnt", mount]);
    }
    let result = run(&mut command, "CRIU dump");
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

fn vllm_mounts(pid: u32) -> Vec<String> {
    let contents = fs::read_to_string(format!("/proc/{pid}/mountinfo")).unwrap_or_default();
    let mut mounts = contents
        .lines()
        .filter_map(|line| {
            let mount_info = line.split(" - ").next()?;
            let fields = mount_info.split_whitespace().collect::<Vec<_>>();
            let root = *fields.get(3)?;
            let mountpoint = *fields.get(4)?;
            // CRIU can recreate mounts rooted at `/`; vLLM's container and
            // cache injections are bind mounts with a different root.
            let runtime_owned = matches!(
                mountpoint,
                "/proc" | "/sys" | "/run" | "/etc/hosts" | "/etc/hostname" | "/etc/resolv.conf"
            ) || mountpoint.starts_with("/proc/")
                || mountpoint.starts_with("/sys/")
                || mountpoint.starts_with("/run/");
            (root != "/" || runtime_owned).then_some(mountpoint)
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    mounts.sort();
    mounts.dedup();
    mounts
}

pub fn restore(directory: &Path) -> Result<()> {
    let pidfile = directory.join("restored.pid");
    let mut command = if let Some(pid) = std::env::var_os("EDO_RESTORE_MOUNT_PID") {
        let mut command = Command::new("nsenter");
        command.args([
            "--target",
            &pid.to_string_lossy(),
            "--mount",
            "--uts",
            "--pid",
            "--",
        ]);
        command.arg(criu_program());
        command
    } else {
        Command::new(criu_program())
    };
    command
        .arg("restore")
        .arg("--images-dir")
        .arg(directory)
        .arg("--restore-detached")
        .arg("--shell-job")
        // The destination Pod has a different runtime-generated cgroup path;
        // Kubernetes remains responsible for placing restored processes.
        .arg("--manage-cgroups=ignore")
        .arg("--root")
        .arg("/")
        // The destination is a runtime-created container mount tree.  The
        // compatibility mount engine is less aggressive about replaying
        // runtime-owned overlay/bind mounts than mount-v2 here.
        .arg("--mntns-compat-mode")
        .arg("--tcp-established")
        // Give the CRIU fork a bounded worker budget for decompression and
        // direct-page paths; buffered restore remains the host default.
        // Upstream CRIU accepts this option too; the fork applies it to
        // both compressed work and independent raw page batches.
        .arg("--decompress-threads")
        .arg("16")
        .arg("--pidfile")
        .arg(&pidfile)
        .arg("--log-file")
        .arg(directory.join("restore.log"))
        .arg("--verbosity=4");
    if let Some(pid) = std::env::var_os("EDO_RESTORE_NET_PID") {
        let net_path = if std::env::var_os("EDO_RESTORE_MOUNT_PID").is_some() {
            // CRIU is launched inside the placeholder mount namespace, whose
            // private /proc exposes the placeholder as PID 1.
            "/proc/1/ns/net".to_owned()
        } else {
            format!("/proc/{}/ns/net", pid.to_string_lossy())
        };
        command.args(["--join-ns", &format!("net:{net_path}")]);
        if std::env::var_os("EDO_RESTORE_MOUNT_PID").is_some() {
            command.args(["--join-ns", "uts:/proc/1/ns/uts"]);
        }
    }
    run(&mut command, "CRIU restore")
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
