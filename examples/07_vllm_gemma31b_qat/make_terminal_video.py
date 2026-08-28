#!/usr/bin/env python3
"""Render a compact split-screen terminal video from the Gemma benchmark.

The animation is intentionally time-compressed for demos. The benchmark
values shown in the terminal are the measured values, not simulated timings.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 900
BG = (10, 14, 22)
PANEL = (18, 24, 35)
TERMINAL = (5, 9, 14)
GRID = (39, 51, 68)
WHITE = (226, 232, 240)
MUTED = (145, 158, 177)
GREEN = (67, 218, 145)
CYAN = (74, 190, 255)
YELLOW = (245, 193, 66)
RED = (255, 103, 103)


def font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(23)
BOLD = font(25, bold=True)
TITLE = font(30, bold=True)
BIG = font(46, bold=True)


def lines_for(side: str, t: float) -> tuple[str, list[str], tuple[int, int, int]]:
    if side == "cold":
        if t < 3:
            return "BOOTING", ["$ vllm serve gemma-3-27b-it-qat", "loading model shards..."], YELLOW
        if t < 7:
            return "LOADING WEIGHTS", ["[1/4] [2/4] [3/4] [4/4]", "weights: 17.42 GiB"], CYAN
        if t < 11:
            return "TORCH COMPILE", ["torch.compile: cache miss", "building optimized graph..."], YELLOW
        if t < 16:
            return "CUDA WARMUP", ["TRITON_ATTN selected", "capturing CUDA graphs..."], CYAN
        return "READY", ["/health 200 OK", "output: Ready."], GREEN
    if t < 2:
        return "SNAPSHOT READY", ["CRIU image: ~32 GiB", "KV pages released"], CYAN
    if t < 5:
        return "CRIU RESTORE", ["restoring process memory...", "io_uring state restored"], YELLOW
    if t < 7:
        return "CUDA RESTORE", ["CUDA context restored", "waking weights + fresh KV"], CYAN
    return "READY", ["/health 200 OK", "output: Ready."], GREEN


def terminal_lines(side: str, t: float) -> list[tuple[str, tuple[int, int, int]]]:
    """Return terminal-style output lines visible at the compressed time."""
    if side == "cold":
        events = [
            (0.0, "$ ./run.sh --full-snapshot", GREEN),
            (1.5, "[edo] starting vLLM Gemma QAT server", WHITE),
            (3.0, "[vllm] loading checkpoint shards 4/4", CYAN),
            (5.5, "[vllm] model memory: 17.42 GiB", WHITE),
            (7.0, "[torch] torch.compile cache miss", YELLOW),
            (10.5, "[torch] optimized graph ready", CYAN),
            (11.5, "[cuda] capturing CUDA graphs...", YELLOW),
            (15.5, "[triton] attention backend: TRITON_ATTN", CYAN),
            (17.0, "[http] GET /health 200 OK", GREEN),
            (17.5, "[test] output: Ready.", GREEN),
        ]
    else:
        events = [
            (0.0, "$ edo summon-group snapshot", GREEN),
            (1.0, "[edo] CRIU restore: process memory", YELLOW),
            (2.5, "[criu] io_uring state restored", CYAN),
            (4.0, "[cuda] context + graph state restored", CYAN),
            (5.5, "[vllm] waking weights; fresh KV cache", YELLOW),
            (7.0, "[http] GET /health 200 OK", GREEN),
            (7.5, "[test] output: Ready.", GREEN),
        ]
    return [(message, color) for start, message, color in events if start <= t]


def render_frame(index: int, fps: int, out: Path) -> None:
    t = index / fps
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((45, 28), "EDO TENSEI  ·  Gemma 3 QAT snapshot", font=TITLE, fill=WHITE)
    draw.text((45, 73), "time-compressed terminal replay · H100 · Triton attention · async vLLM", font=FONT, fill=MUTED)

    panel_y, panel_h, gap = 125, 700, 28
    panel_w = (WIDTH - 90 - gap) // 2
    for pos, side in enumerate(("cold", "restore")):
        x = 45 + pos * (panel_w + gap)
        draw.rounded_rectangle((x, panel_y, x + panel_w, panel_y + panel_h), 14, fill=TERMINAL, outline=GRID, width=2)
        draw.rounded_rectangle((x, panel_y, x + panel_w, panel_y + 42), 14, fill=(28, 36, 49))
        draw.rectangle((x, panel_y + 28, x + panel_w, panel_y + 42), fill=(28, 36, 49))
        for dot_x, dot_color in ((x + 18, RED), (x + 36, YELLOW), (x + 54, GREEN)):
            draw.ellipse((dot_x - 5, panel_y + 16, dot_x + 5, panel_y + 26), fill=dot_color)
        title = "LEFT  ·  COLD BOOT" if side == "cold" else "RIGHT  ·  RESTORE"
        draw.text((x + 80, panel_y + 12), title, font=FONT, fill=CYAN if side == "cold" else GREEN)
        measured = "103.052s to /health" if side == "cold" else "11.055s to /health"
        draw.text((x + 25, panel_y + 65), measured, font=FONT, fill=WHITE)

        local_t = min(t, 18.0) if side == "cold" else max(0.0, t - 18.0)
        shown_t = local_t if side == "cold" else min(local_t, 8.0)
        if side == "restore" and t < 18:
            state, messages, color = "WAITING", ["waiting for cold snapshot...", "snapshot boundary: pending"], MUTED
        else:
            state, messages, color = lines_for(side, shown_t)

        draw.text((x + 25, panel_y + 112), state, font=BIG, fill=color)
        draw.text((x + 25, panel_y + 185), "ubuntu@edo:~$", font=FONT, fill=GREEN)
        for row, (message, message_color) in enumerate(terminal_lines(side, shown_t)[-7:]):
            draw.text((x + 25, panel_y + 225 + row * 35), message, font=FONT, fill=message_color)

        progress = min(1.0, shown_t / (18.0 if side == "cold" else 8.0)) if not (side == "restore" and t < 18) else 0.0
        bar_x, bar_y, bar_w = x + 25, panel_y + 500, panel_w - 50
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 18), 8, fill=GRID)
        draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + 18), 8, fill=color)
        draw.text((x + 25, panel_y + 570), "no model reload", font=FONT, fill=GREEN if (side == "restore" and shown_t >= 7) else MUTED)
        draw.text((x + 25, panel_y + 605), "no torch.compile recapture", font=FONT, fill=GREEN if (side == "restore" and shown_t >= 7) else MUTED)
        draw.text((x + 25, panel_y + 640), "no CUDA graph recapture", font=FONT, fill=GREEN if (side == "restore" and shown_t >= 7) else MUTED)
        if side == "cold":
            draw.text((x + 25, panel_y + 675), "first-ever cold boot: ~180s", font=FONT, fill=YELLOW)
        else:
            draw.text((x + 25, panel_y + 675), "snapshot: ~32 GiB  ·  -55%", font=FONT, fill=GREEN)

    draw.text((45, 850), "Cold boot 103.052s  →  Restore 11.055s  ·  valid output: Ready.", font=BOLD, fill=WHITE)
    image.save(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma-cold-vs-restore.mp4"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edo-terminal-video-") as temp:
        frames = Path(temp)
        total = int(26 * args.fps)
        for index in range(total):
            render_frame(index, args.fps, frames / f"frame-{index:05d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                "-i", str(frames / "frame-%05d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "20", str(args.output),
            ],
            check=True,
        )
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
