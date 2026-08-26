#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
edo="$repo_root/target/debug/edo"
python_bin="${EDO_MODEL_PYTHON:-/home/ubuntu/miniconda3/envs/edo-models/bin/python}"
model="${EDO_HF_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
name="edo-modal-qwen-$$"
parent=$(mktemp -d /tmp/edo-modal-qwen-demo.XXXXXX)
snapshot="$parent/snapshot"
log="$parent/model.log"
startup_marker="$parent/startup.marker"

cleanup() {
    if [ -f "$snapshot/restored.pid" ]; then sudo kill "$(sudo cat "$snapshot/restored.pid")" 2>/dev/null || true; fi
    if [ -f ".edo/runs/$name.json" ]; then sudo kill "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")" 2>/dev/null || true; fi
    sudo rm -rf "$parent"
}
trap cleanup EXIT INT TERM

cd "$repo_root"
"$edo" run --name "$name" -- setsid env EDO_HF_MODEL="$model" EDO_STARTUP_MARKER="$startup_marker" "$python_bin" examples/modal-gpu-snapshot/qwen_gpu_fixture.py >"$log" 2>&1
for _attempt in $(seq 1 180); do
    if grep -q 'model-ready' "$log"; then break; fi
    sleep 1
done
grep -q 'model-ready' "$log"

pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")
sudo kill -USR1 "$pid"
sleep 1
before=$(grep 'gpu-model-checksum' "$log" | tail -1)
before_checksum=$(printf '%s\n' "$before" | sed -n 's/.*checksum=\([^ ]*\).*/\1/p')
before_time=$(date +%s%N)
sudo "$edo" freeze "$name" "$snapshot"
freeze_time=$(date +%s%N)
sudo "$edo" summon "$snapshot"
restore_time=$(date +%s%N)
restored_pid=$(sudo cat "$snapshot/restored.pid")
sudo kill -USR1 "$restored_pid"
sleep 1
after=$(grep 'gpu-model-checksum' "$log" | tail -1)
after_checksum=$(printf '%s\n' "$after" | sed -n 's/.*checksum=\([^ ]*\).*/\1/p')
test -n "$before_checksum"
test "$before_checksum" = "$after_checksum"
test "$(grep -c '^model-initialized$' "$startup_marker")" -eq 1
freeze_ms=$(( (freeze_time - before_time) / 1000000 ))
restore_ms=$(( (restore_time - freeze_time) / 1000000 ))
echo "=== Modal-style GPU snapshot report ==="
grep 'model-ready' "$log" | tail -1
grep 'warmup-output' "$log" | tail -1
echo "Before restore: $before"
echo "After restore:  $after"
echo "Freeze time: ${freeze_ms} ms"
echo "Restore time: ${restore_ms} ms"
echo "GPU model checksum preserved."
echo "Model initialization count after restore: $(grep -c '^model-initialized$' "$startup_marker") (expected 1)."
