#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/target/debug/edo"
demo_name="${EDO_DEMO_NAME:-resume}"
name="edo-${demo_name}-$$"
parent=$(mktemp -d /tmp/edo-resume-demo.XXXXXX)
snapshot="$parent/checkpoint"
events="$parent/events.jsonl"
ready="$parent/ready"
launch_log="$parent/launch.log"
report_dir="$repo_root/.edo/runs"
report="$report_dir/$(date -u +%Y-%m-%dT%H-%M-%SZ)-${demo_name}.json"
start_ns=$(date +%s%N)

cleanup() {
    if [ -f "$snapshot/restored.pid" ]; then
        restored_pid=$(sudo cat "$snapshot/restored.pid" 2>/dev/null || true)
        [ -z "${restored_pid:-}" ] || sudo kill "$restored_pid" 2>/dev/null || true
    fi
    sudo rm -rf "$parent"
}
trap cleanup EXIT INT TERM

cd "$repo_root"
mkdir -p "$report_dir"
cargo build --quiet

echo "Edo Tensei — Resume a warm workload (${demo_name})"
"$binary" run --name "$name" -- setsid env \
    EDO_DEMO_REPORT="$events" EDO_DEMO_READY="$ready" \
    EDO_DEMO_STARTUP_SECONDS="${EDO_DEMO_STARTUP_SECONDS:-2}" \
    python3 examples/00_hello_checkpoint/workload.py >"$launch_log" 2>&1
for _ in $(seq 1 100); do [ -f "$ready" ] && break; sleep 0.1; done
test -f "$ready"
cold_end_ns=$(date +%s%N)
echo "✓ Started workload"
echo "✓ Warm-up completed"

request_pid=$(cat "$ready")
sudo kill -USR1 "$request_pid"
for _ in $(seq 1 50); do grep -q '"request"' "$events" && break; sleep 0.1; done

checkpoint_start_ns=$(date +%s%N)
sudo "$binary" cpu-dump "$name" "$snapshot" >/dev/null
checkpoint_end_ns=$(date +%s%N)
echo "✓ Checkpoint created: $snapshot"
# Edo keeps the source process alive so CUDA-backed callers can recover from
# a failed dump. For this CPU demo, explicitly simulate the failure boundary.
sudo kill "$request_pid" 2>/dev/null || true
for _ in $(seq 1 50); do
    state=$(ps -o stat= -p "$request_pid" 2>/dev/null | tr -d ' ' || true)
    [ -z "$state" ] || [[ "$state" == Z* ]] && break
    sleep 0.1
done
state=$(ps -o stat= -p "$request_pid" 2>/dev/null | tr -d ' ' || true)
if [ -n "$state" ] && [[ "$state" != Z* ]]; then
    echo "original process still exists after dump (state=$state)" >&2
    exit 1
fi
echo "✓ Original process disappeared"

restore_start_ns=$(date +%s%N)
sudo "$binary" cpu-restore "$snapshot" >/dev/null
restored_pid=$(sudo cat "$snapshot/restored.pid")
for _ in $(seq 1 50); do sudo kill -0 "$restored_pid" 2>/dev/null && break; sleep 0.1; done
sudo kill -USR1 "$restored_pid"
for _ in $(seq 1 50); do [ "$(grep -c '"request"' "$events" 2>/dev/null || true)" -ge 2 ] && break; sleep 0.1; done
restore_end_ns=$(date +%s%N)
test "$(grep -c '"request"' "$events")" -ge 2
echo "✓ Restored successfully"
echo "✓ Request completed after restore"

python3 - "$report" "$start_ns" "$cold_end_ns" "$checkpoint_start_ns" "$checkpoint_end_ns" "$restore_start_ns" "$restore_end_ns" "$snapshot" "$demo_name" <<'PY'
import json
import sys
from pathlib import Path

output, start, cold_end, checkpoint_start, checkpoint_end, restore_start, restore_end, snapshot, demo_name = sys.argv[1:]
report = {
    "schema_version": 1,
    "demo": demo_name,
    "platform": "cpu",
    "measured": True,
    "cold_start_ms": (int(cold_end) - int(start)) / 1_000_000,
    "checkpoint_ms": (int(checkpoint_end) - int(checkpoint_start)) / 1_000_000,
    "restore_to_request_ms": (int(restore_end) - int(restore_start)) / 1_000_000,
    "snapshot": snapshot,
    "request_after_restore": True,
}
Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"Cold start:    {report['cold_start_ms']:.1f} ms")
print(f"Restore time:  {report['restore_to_request_ms']:.1f} ms")
print(f"Run report:    {output}")
PY
