#!/usr/bin/env python3
"""Small node-local Edo snapshot agent.

The agent intentionally accepts process IDs rather than discovering pods. A
controller or an operator can resolve a container ID through CRI and submit
the request after it has quiesced the workload.
"""
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


EDO = os.environ.get("EDO_BIN", "/opt/edo/bin/edo")
CRIU = os.environ.get("EDO_CRIU", "/opt/edo/bin/criu")
SNAPSHOTS = Path(os.environ.get("SNAPSHOT_ROOT", "/var/lib/edo-snapshots"))


def prewarm_cuda() -> None:
    """Move one-time driver initialization out of restore's critical path."""
    if os.environ.get("EDO_CUDA_PREWARM", "false").lower() not in ("1", "true", "yes"):
        return
    result = subprocess.run([EDO, "cuda-init"], text=True, capture_output=True, env=os.environ.copy())
    if result.returncode:
        print(f"CUDA prewarm failed (restore will retry): {result.stderr.strip()}", flush=True)
    else:
        print("CUDA checkpoint driver prewarmed", flush=True)


def inner_pid(host_pid: int) -> int:
    values = Path(f"/proc/{host_pid}/status").read_text().splitlines()
    nspid = next((x for x in values if x.startswith("NSpid:")), "")
    return int(nspid.split()[-1]) if nspid else host_pid


def run_in_namespace(host_pid: int, args: list[str]) -> dict:
    command = [
        "nsenter", "--target", str(host_pid), "--mount", "--uts", "--ipc",
        "--net", "--pid", "--", *args,
    ]
    env = os.environ.copy()
    env["EDO_CRIU"] = CRIU
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/v1/snapshot", "/v1/restore"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            host_pid = int(body["host_pid"])
            snapshot = str(SNAPSHOTS / body["name"])
            if not snapshot.startswith(str(SNAPSHOTS) + os.sep):
                raise ValueError("snapshot name must be relative")
            if self.path.endswith("snapshot"):
                cuda = ",".join(str(inner_pid(int(pid))) for pid in body["cuda_pids"])
                args = [EDO, "freeze-group", str(inner_pid(host_pid)), cuda, snapshot]
            else:
                args = [EDO, "summon-group", snapshot]
                if body.get("skip_integrity", False):
                    args.append("--skip-integrity")
            result = run_in_namespace(host_pid, args)
            self.send_json(200 if result["returncode"] == 0 else 500, result)
        except Exception as exc:  # request errors are returned as JSON
            self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    prewarm_cuda()
    HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8787"))), Handler).serve_forever()
