#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/target/debug/edo"
name="edo-cuda-combined-$$"
parent=$(mktemp -d /tmp/edo-cuda-combined-demo.XXXXXX)
snapshot="$parent/snapshot"
launch_log="$parent/fixture.log"

cleanup() {
    if [ -f "$snapshot/restored.pid" ]; then
        sudo kill "$(sudo cat "$snapshot/restored.pid")" 2>/dev/null || true
    fi
    if [ -f ".edo/runs/$name.json" ]; then
        sudo kill "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")" 2>/dev/null || true
    fi
    sudo rm -rf "$parent"
    rm -f examples/cuda-checkpoint/cuda-fixture
}
trap cleanup EXIT INT TERM

cd "$repo_root"
gcc -I/usr/local/cuda/include examples/cuda-checkpoint/cuda-fixture.c \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcuda \
    -o examples/cuda-checkpoint/cuda-fixture

./target/debug/edo run --name "$name" -- setsid \
    examples/cuda-checkpoint/cuda-fixture >"$launch_log" 2>&1
pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' ".edo/runs/$name.json")
sleep 1

sudo kill -USR1 "$pid"
sleep 1
grep -q 'gpu-pattern-ok' "$launch_log"
echo "Before combined checkpoint: GPU pattern verified."

sudo "$binary" freeze "$name" "$snapshot"
sudo "$binary" summon "$snapshot"

restored_pid=$(sudo cat "$snapshot/restored.pid")
sudo kill -USR1 "$restored_pid"
sleep 1
grep -q 'gpu-pattern-ok' "$launch_log"
echo "After combined restore: GPU pattern verified."
echo "Combined CUDA+CRIU demo passed."
