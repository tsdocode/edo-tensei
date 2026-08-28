#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/target/debug/edo"
process_name="edo-fastapi-demo-$$"
port="${EDO_PORT:-8000}"
snapshot_parent=$(mktemp -d /tmp/edo-fastapi-checkpoint-demo.XXXXXX)
snapshot_dir="$snapshot_parent/snapshot"
launch_log=$(mktemp /tmp/edo-fastapi-launch.XXXXXX)

cleanup() {
    if [ -f "$snapshot_dir/restored.pid" ]; then
        restored_pid=$(sudo cat "$snapshot_dir/restored.pid" 2>/dev/null || true)
        if [ -n "${restored_pid:-}" ]; then sudo kill "$restored_pid" 2>/dev/null || true; fi
    fi
    if [ ! -f "$snapshot_dir/restored.pid" ] && [ -f ".edo/runs/$process_name.json" ]; then
        original_pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$process_name.json")
        sudo kill "$original_pid" 2>/dev/null || true
    fi
    sudo rm -rf "$snapshot_parent"
    rm -f "$launch_log"
}
trap cleanup EXIT INT TERM

cd "$repo_root"
source ./env.sh
cargo build --quiet

echo "Launching FastAPI server with heavy startup..."
"$binary" run --name "$process_name" -- setsid env \
    EDO_PORT="$port" \
    EDO_STARTUP_SECONDS="${EDO_STARTUP_SECONDS:-8}" \
    EDO_MODEL_MB="${EDO_MODEL_MB:-64}" \
    EDO_USE_HF_MODEL="${EDO_USE_HF_MODEL:-1}" \
    EDO_HF_MODEL="${EDO_HF_MODEL:-sshleifer/tiny-distilbert-base-cased}" \
    python3 examples/02_fastapi_resume/server.py >"$launch_log" 2>&1

for _attempt in $(seq 1 120); do
    if curl --silent --fail "http://127.0.0.1:$port/ready" >/dev/null; then break; fi
    sleep 1
done

curl --silent --fail "http://127.0.0.1:$port/ready"
before_state=$(curl --silent --fail "http://127.0.0.1:$port/state")
curl --silent --fail "http://127.0.0.1:$port/infer?value=2"
echo
echo "Before checkpoint: $before_state"

curl --silent --fail -X POST "http://127.0.0.1:$port/quiesce"
if curl --silent --fail "http://127.0.0.1:$port/ready" >/dev/null; then
    echo "server remained ready during quiesce" >&2
    exit 1
fi

echo "Checkpointing FastAPI process..."
sudo "$binary" cpu-dump "$process_name" "$snapshot_dir"

echo "Restoring FastAPI process..."
sudo "$binary" cpu-restore "$snapshot_dir"

curl --silent --fail "http://127.0.0.1:$port/health"
curl --silent --fail -X POST "http://127.0.0.1:$port/resume"
for _attempt in $(seq 1 20); do
    if curl --silent --fail "http://127.0.0.1:$port/ready" >/dev/null; then break; fi
    sleep 1
done

curl --silent --fail "http://127.0.0.1:$port/ready"
after_state=$(curl --silent --fail "http://127.0.0.1:$port/state")
curl --silent --fail "http://127.0.0.1:$port/infer?value=2"
echo
echo "After restore:  $after_state"

before_checksum=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["model_checksum"])' "$before_state")
after_checksum=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["model_checksum"])' "$after_state")
test "$before_checksum" = "$after_checksum"
echo "FastAPI checkpoint demo passed: model checksum preserved."
