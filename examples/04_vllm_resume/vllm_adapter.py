#!/usr/bin/env python3
"""vLLM warm-process adapter probe for Edo's future worker-group snapshots.

The default workflow warms vLLM normally, including its CUDA graph path. Full
CRIU freezing is deliberately not implicit because vLLM's engine workers, IPC,
and distributed state must be restored as a group.
"""

import argparse
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base: str, path: str, method: str = "GET", body=None, timeout: int = 10):
    data = None if body is None else json.dumps(body).encode()
    req = Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def wait_ready(base: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = request(base, "/health")
            if status == 200:
                return
        except (HTTPError, URLError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s")


def post_empty(base: str, path: str, timeout: int = 10) -> None:
    status, result = request(base, path, method="POST", timeout=timeout)
    if status != 200:
        raise RuntimeError(f"vLLM endpoint {path} failed with HTTP {status}: {result}")


def wait_sleeping(base: str, expected: bool, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, result = request(base, "/is_sleeping")
        if status == 200 and isinstance(result, dict) and result.get("is_sleeping") is expected:
            return
        time.sleep(0.25)
    raise TimeoutError(f"vLLM sleep state did not become {expected} within {timeout}s")


def warmup(base: str, model: str) -> dict:
    status, result = request(
        base,
        "/v1/chat/completions",
        method="POST",
        body={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: ready"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    if status != 200:
        raise RuntimeError(f"vLLM warmup request failed with HTTP {status}: {result}")
    return result


def timed_warmup(base: str, model: str) -> tuple[dict, float]:
    started = time.monotonic()
    result = warmup(base, model)
    return result, time.monotonic() - started


def timed_ttft(base: str, model: str) -> tuple[dict, float]:
    """Measure time to the first streamed token for the fixed warmup prompt."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: ready"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = Request(
        f"{base}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    started = time.monotonic()
    first_token = None
    chunks = []
    with urlopen(req, timeout=30) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            choices = event.get("choices", [])
            if choices and first_token is None:
                first_token = time.monotonic() - started
            if choices:
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    chunks.append(delta)
    if first_token is None:
        raise RuntimeError("stream ended without a token")
    return {"text": "".join(chunks)}, first_token


def process_tree(pid: int) -> list[dict[str, int | str]]:
    result = []
    pending = [pid]
    while pending:
        current = pending.pop(0)
        cmdline_path = Path(f"/proc/{current}/cmdline")
        if not cmdline_path.exists():
            continue
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        result.append({"pid": current, "ppid": int(Path(f"/proc/{current}/stat").read_text().split()[3]), "cmdline": cmdline})
        children = Path(f"/proc/{current}/task/{current}/children")
        if children.exists():
            pending.extend(int(child) for child in children.read_text().split())
    return result


def run(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}"
    if args.inspect_only:
        if args.pid is None:
            raise SystemExit("--inspect-only requires --pid and never launches a server")
        print(json.dumps(process_tree(args.pid), indent=2))
        return 0
    command = shlex.split(args.vllm_command) + [
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.kv_cache_memory_bytes is not None:
        command.extend(["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.no_async_scheduling:
        command.append("--no-async-scheduling")
    if args.cudagraph_capture_sizes:
        command.extend(
            ["--cudagraph-capture-sizes"]
            + [str(size) for size in args.cudagraph_capture_sizes]
        )
    if args.compilation_config:
        command.extend(["--compilation-config", args.compilation_config])
    if args.attention_config:
        command.extend(["--attention-config", args.attention_config])
    # The sleep endpoint is only useful for snapshot minimization when vLLM
    # allocates weights/KV through CuMemAllocator. Without this flag the
    # endpoint can still respond, but the allocator tracks little or none of
    # the model state and CRIU captures the large CUDA allocation unchanged.
    if args.release_kv_cache:
        command.append("--enable-sleep-mode")
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    # When the launcher is an absolute path inside a uv/venv environment,
    # preserve that environment for subprocesses such as Triton and
    # FlashInfer, which discover tools (notably ninja) through PATH.
    launcher = Path(command[0]).expanduser()
    if not launcher.is_absolute():
        resolved_launcher = shutil.which(command[0], path=env.get("PATH"))
        if resolved_launcher:
            launcher = Path(resolved_launcher).resolve()
    else:
        launcher = launcher.resolve()
    if launcher.is_absolute() and launcher.parent.name == "bin":
        env["PATH"] = f"{launcher.parent}:{env.get('PATH', '')}"
    if args.no_async_scheduling:
        # CRIU 4.2.1 cannot dump uvloop's anon_inode:[io_uring] mappings.
        # Keep vLLM scheduling synchronous and force libuv's epoll path.
        env["UVLOOP_NO_URING"] = "1"
    print("Starting dedicated vLLM server:", " ".join(command), flush=True)
    cold_started = time.monotonic()
    server = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        wait_ready(base, args.startup_timeout)
        cold_ready_seconds = time.monotonic() - cold_started
        print("vLLM health: ready")
        print(f"cold startup to /health: {cold_ready_seconds:.3f}s")
        _, models = request(base, "/v1/models")
        print("model endpoint:", json.dumps(models, sort_keys=True))
        warmup_result, cold_warmup_seconds = timed_warmup(base, args.model)
        print("warmup inference:", json.dumps(warmup_result, sort_keys=True))
        print(f"cold startup to warm inference: {time.monotonic() - cold_started:.3f}s")
        print(f"cold warmup request latency: {cold_warmup_seconds:.3f}s")
        _, cold_ttft_seconds = timed_ttft(base, args.model)
        print(f"cold TTFT: {cold_ttft_seconds:.3f}s")
        print("vLLM process is warm and ready for a snapshot")
        tree = process_tree(server.pid)
        print("worker process group:")
        print(json.dumps(tree, indent=2))
        if args.release_kv_cache:
            print("Warm snapshot boundary passed; vLLM KV backing will be released before checkpoint.")
        else:
            print("Warm snapshot boundary passed; Sleep Mode was not used.")
        if args.full_snapshot:
            if args.release_kv_cache:
                release_started = time.monotonic()
                # Copying a large model's tagged weights to pinned host memory
                # can take longer than the normal health/request timeout.
                post_empty(
                    base,
                    f"/sleep?level={args.sleep_level}&mode=wait",
                    timeout=max(120, args.startup_timeout),
                )
                wait_sleeping(base, True, args.startup_timeout)
                print(
                    f"Dynamo-style KV-cache release: {time.monotonic() - release_started:.3f}s",
                    flush=True,
                )
            api = next(
                record for record in tree
                if "/vllm serve " in str(record["cmdline"])
                and "nsenter" not in str(record["cmdline"])
                and "sudo" not in str(record["cmdline"])
            )
            workers = [
                record for record in tree
                if str(record["cmdline"]).strip() == "VLLM::EngineCore"
            ]
            if not workers:
                raise RuntimeError("could not identify a VLLM::EngineCore CUDA worker")
            cuda_pids = ",".join([str(api["pid"])] + [str(worker["pid"]) for worker in workers])
            snapshot = Path(tempfile.mkdtemp(prefix="edo-vllm-group-")) / "snapshot"
            print(f"Freezing vLLM group root={api['pid']} CUDA_PIDs={cuda_pids}", flush=True)
            subprocess.run(
                ["sudo", "--preserve-env=EDO_CRIU", args.edo, "freeze-group",
                 str(api["pid"]), cuda_pids, str(snapshot)],
                check=True,
            )
            # CRIU restores the recorded numeric PIDs.  The dumped instance
            # must therefore be fully reaped before summon-group starts.
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            restore_started = time.monotonic()
            preserved_env = "EDO_CRIU"
            if args.io_uring_restore:
                preserved_env += ",EDO_IO_URING_RESTORE"
            summon_command = [
                "sudo", f"--preserve-env={preserved_env}", args.edo,
                "summon-group", str(snapshot)
            ]
            if args.fast_restore:
                summon_command.append("--skip-integrity")
            restore_env = os.environ.copy()
            if args.io_uring_restore:
                restore_env["EDO_IO_URING_RESTORE"] = "1"
            subprocess.run(
                summon_command,
                check=True,
                env=restore_env,
            )
            if args.release_kv_cache:
                wake_started = time.monotonic()
                # Rehydrate weights first, then recreate the discarded KV
                # backing as empty GPU memory.  Keeping these as two calls is
                # important: a single wake_up() would hide whether the
                # checkpoint actually excluded KV pages.
                post_empty(base, "/wake_up?tags=weights")
                post_empty(base, "/wake_up?tags=kv_cache")
                wait_sleeping(base, False, args.startup_timeout)
                print(
                    "Dynamo-style staged wake (weights + fresh KV cache): "
                    f"{time.monotonic() - wake_started:.3f}s"
                )
            wait_ready(base, args.startup_timeout)
            restore_ready_seconds = time.monotonic() - restore_started
            after, restore_warmup_seconds = timed_warmup(base, args.model)
            print(f"restore to /health: {restore_ready_seconds:.3f}s")
            print(f"post-restore warmup request latency: {restore_warmup_seconds:.3f}s")
            print(f"restore to warm inference: {time.monotonic() - restore_started:.3f}s")
            _, restore_ttft_seconds = timed_ttft(base, args.model)
            print(f"post-restore TTFT: {restore_ttft_seconds:.3f}s")
            print(f"TTFT delta: {restore_ttft_seconds - cold_ttft_seconds:+.3f}s")
            print("post-restore warmup inference:", json.dumps(after, sort_keys=True))
            print("vLLM group restore passed")
        return 0
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        for record in process_tree(server.pid):
            if record["pid"] != os.getpid():
                try:
                    os.kill(int(record["pid"]), signal.SIGTERM)
                except ProcessLookupError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("EDO_VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--vllm-command", default=os.environ.get("EDO_VLLM_COMMAND", "vllm"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--pid", type=int, help="vLLM API process PID for --inspect-only")
    parser.add_argument("--full-snapshot", action="store_true", help="run the destructive Edo group freeze/restore test")
    parser.add_argument(
        "--fast-restore",
        action="store_true",
        help="skip large image SHA-256 reads during restore (trusted local snapshots only)",
    )
    parser.add_argument(
        "--release-kv-cache",
        action="store_true",
        help="release vLLM KV-cache backing before snapshot and wake it after restore",
    )
    parser.add_argument(
        "--sleep-level",
        type=int,
        choices=(1, 2),
        default=1,
        help="vLLM sleep level used before checkpoint (level 2 discards weights too)",
    )
    parser.add_argument("--edo", default=os.environ.get("EDO_BIN", "target/debug/edo"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        help="explicit per-GPU KV-cache budget; useful for bounded snapshot artifacts",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graphs for compatibility diagnosis; graphs remain the default",
    )
    parser.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="disable vLLM async scheduling to avoid CRIU-unsupported io_uring state",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        nargs="+",
        type=int,
        help="CUDA graph capture sizes, e.g. '1' to minimize graph state",
    )
    parser.add_argument(
        "--compilation-config",
        help="vLLM compilation JSON, e.g. '{\"cudagraph_mode\":\"FULL\"}'",
    )
    parser.add_argument(
        "--attention-config",
        help='vLLM attention JSON, e.g. \'{"backend":"TRITON_ATTN"}\'',
    )
    parser.add_argument(
        "--io-uring-restore",
        action="store_true",
        help="opt into buffered io_uring page restore (experimental)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
