#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/target/debug/edo"
name="edo-gpu-model-$$"
parent=$(mktemp -d /tmp/edo-gpu-model-demo.XXXXXX)
snapshot="$parent/snapshot"
log="$parent/model.log"

cleanup() {
    if [ -f "$snapshot/restored.pid" ]; then sudo kill "$(sudo cat "$snapshot/restored.pid")" 2>/dev/null || true; fi
    if [ -f ".edo/runs/$name.json" ]; then sudo kill "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")" 2>/dev/null || true; fi
    sudo rm -rf "$parent"
    rm -f examples/gpu-model-snapshot/gpu-model-fixture
}
trap cleanup EXIT INT TERM

cd "$repo_root"
gcc -I/usr/local/cuda/include examples/gpu-model-snapshot/gpu-model-fixture.c \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcuda \
    -o examples/gpu-model-snapshot/gpu-model-fixture
"$binary" run --name "$name" -- setsid env EDO_MODEL_MB="${EDO_MODEL_MB:-64}" \
    examples/gpu-model-snapshot/gpu-model-fixture >"$log" 2>&1
pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")
for _attempt in $(seq 1 20); do
    if grep -q 'model-ready' "$log"; then break; fi
    sleep 1
done
grep -q 'model-ready' "$log"

sudo kill -USR1 "$pid"
sleep 1
before=$(grep 'gpu-model-checksum' "$log" | tail -1)
before_checksum=$(printf '%s\n' "$before" | sed -n 's/.*checksum=\([0-9]*\).*/\1/p')
before_time=$(date +%s%N)
sudo "$binary" freeze "$name" "$snapshot"
freeze_time=$(date +%s%N)
sudo "$binary" summon "$snapshot"
restore_time=$(date +%s%N)
restored_pid=$(sudo cat "$snapshot/restored.pid")
sudo kill -USR1 "$restored_pid"
sleep 1
after=$(grep 'gpu-model-checksum' "$log" | tail -1)
after_checksum=$(printf '%s\n' "$after" | sed -n 's/.*checksum=\([0-9]*\).*/\1/p')
test -n "$before_checksum"
test "$before_checksum" = "$after_checksum"
freeze_ms=$(( (freeze_time - before_time) / 1000000 ))
restore_ms=$(( (restore_time - freeze_time) / 1000000 ))
echo "GPU model before: $before"
echo "GPU model after:  $after"
echo "Freeze time: ${freeze_ms} ms"
echo "Restore time: ${restore_ms} ms"
echo "GPU model snapshot demo passed: checksum preserved."
