#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

input="${1:-artifacts/gemma-cold-vs-restore.cast}"
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/edo-real-panels.XXXXXX")
trap 'rm -rf "$out_dir"' EXIT

# These are the boundaries from the successful Gemma session currently
# checked in. The snapshot/hash interval is intentionally excluded from both
# panels so the two panels focus on serving readiness.
jq -c 'select(type == "object" or (type == "array" and .[0] <= 104.6))' "$input" \
  | jq -c 'select(type == "object" or .[0] <= 104.6)' > "$out_dir/cold.cast"
jq -c 'select(type == "object" or (type == "array" and .[0] >= 172.7))' "$input" \
  | jq -c 'if type == "object" then . else [.[0] - 172.7, .[1], .[2]] end' > "$out_dir/restore.cast"

agg "$out_dir/cold.cast" "$out_dir/cold.gif" --cols 120 --rows 36 --speed 4 --idle-time-limit 120 >/dev/null
agg "$out_dir/restore.cast" "$out_dir/restore.gif" --cols 120 --rows 36 --speed 4 --idle-time-limit 120 >/dev/null
ffmpeg -y -loglevel error -i "$out_dir/cold.gif" -i "$out_dir/restore.gif" \
  -filter_complex "[0:v]fps=10,scale=640:-1:flags=lanczos,setsar=1[left];[1:v]fps=10,scale=640:-1:flags=lanczos,setsar=1[right];[left][right]hstack=inputs=2:shortest=0,format=yuv420p" \
  -c:v libx264 -crf 18 -movflags +faststart artifacts/gemma-real-cold-vs-restore.mp4
ffmpeg -y -loglevel error -i artifacts/gemma-real-cold-vs-restore.mp4 \
  -vf "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=sierra2_4a" \
  artifacts/gemma-real-cold-vs-restore.gif
echo "wrote real panel video and GIF from $input"
