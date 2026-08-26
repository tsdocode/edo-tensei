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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base: str, path: str, method: str = "GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(req, timeout=10) as response:
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
    if args.enforce_eager:
        command.append("--enforce-eager")
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    print("Starting dedicated vLLM server:", " ".join(command), flush=True)
    server = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        wait_ready(base, args.startup_timeout)
        print("vLLM health: ready")
        _, models = request(base, "/v1/models")
        print("model endpoint:", json.dumps(models, sort_keys=True))
        warmup_result = warmup(base, args.model)
        print("warmup inference:", json.dumps(warmup_result, sort_keys=True))
        print("vLLM process is warm and ready for a snapshot")
        tree = process_tree(server.pid)
        print("worker process group:")
        print(json.dumps(tree, indent=2))
        print("Warm snapshot boundary passed; Sleep Mode was not used.")
        if args.full_snapshot:
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
                ["sudo", args.edo, "freeze-group", str(api["pid"]), cuda_pids, str(snapshot)],
                check=True,
            )
            subprocess.run(["sudo", args.edo, "summon-group", str(snapshot)], check=True)
            wait_ready(base, args.startup_timeout)
            after = warmup(base, args.model)
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
    parser.add_argument("--edo", default=os.environ.get("EDO_BIN", "target/debug/edo"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graphs for compatibility diagnosis; graphs remain the default",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
