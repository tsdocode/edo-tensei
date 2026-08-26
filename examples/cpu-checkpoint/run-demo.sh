#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/target/debug/edo"
process_name="edo-cpu-demo-$$"
snapshot_dir=$(mktemp -d /tmp/edo-cpu-checkpoint-demo.XXXXXX)
launch_log=$(mktemp /tmp/edo-cpu-checkpoint-launch.XXXXXX)

cleanup() {
    if [ -f "$snapshot_dir/restored.pid" ]; then
        restored_pid=$(sudo cat "$snapshot_dir/restored.pid" 2>/dev/null || true)
        if [ -n "${restored_pid:-}" ]; then
            sudo kill "$restored_pid" 2>/dev/null || true
        fi
    fi
    sudo rm -rf "$snapshot_dir"
    rm -f "$launch_log"
}
trap cleanup EXIT INT TERM

cd "$repo_root"
source ./env.sh
cargo build --quiet

echo "Launching counter fixture..."
"$binary" run --name "$process_name" -- setsid python3 examples/cpu-counter.py >"$launch_log" 2>&1
cat "$launch_log"
sleep 2

echo "Checkpointing CPU process..."
sudo "$binary" cpu-dump "$process_name" "$snapshot_dir"

echo "Restoring CPU process..."
sudo "$binary" cpu-restore "$snapshot_dir"

restored_pid=$(sudo cat "$snapshot_dir/restored.pid")
echo "Restored PID: $restored_pid"
sudo ps -o pid,ppid,stat,cmd -p "$restored_pid"
echo "CPU checkpoint demo passed."
