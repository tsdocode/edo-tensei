//! The 60-second onboarding demo. The shell script owns the privileged CRIU flow.
use std::process::Command;

fn main() {
    let status = Command::new("bash")
        .arg(format!(
            "{}/examples/00_hello_checkpoint/run.sh",
            env!("CARGO_MANIFEST_DIR")
        ))
        .status()
        .expect("start resume demo");
    std::process::exit(status.code().unwrap_or(1));
}
