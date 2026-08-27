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


def namespace_ipv4(host_pid: int) -> str | None:
    result = subprocess.run(
        ["nsenter", "--target", str(host_pid), "--net", "--", "ip", "-4", "-o", "addr", "show", "scope", "global"],
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        for field in line.split():
            if "/" in field and field[0].isdigit():
                return field.split("/", 1)[0]
    return None


def run_in_namespace(host_pid: int, args: list[str]) -> dict:
    command = [
        "nsenter", "--target", str(host_pid), "--mount", "--uts", "--ipc",
        "--net", "--pid", "--", *args,
    ]
    env = os.environ.copy()
    env["EDO_CRIU"] = CRIU
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_restore_in_namespace(host_pid: int, args: list[str]) -> dict:
    # CRIU must remain in the node's PID namespace so it can create the fresh
    # PID namespace required by restore. The destination pod's mount, IPC,
    # UTS, and network namespaces are still used.
    command = [
        "nsenter", "--target", str(host_pid), "--mount", "--uts", "--ipc",
        "--net", "--", *args,
    ]
    env = os.environ.copy()
    env["EDO_CRIU"] = CRIU
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_on_node(args: list[str]) -> dict:
    # For dump, keep the node's PID and mount namespaces. CRIU must see host
    # PIDs and the container mount namespace through /proc; entering the pod's
    # mount namespace first would hide host PIDs behind its private /proc.
    env = os.environ.copy()
    env["EDO_CRIU"] = CRIU
    result = subprocess.run(args, text=True, capture_output=True, env=env)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_restore_on_node(host_pid: int, args: list[str], source_ip: str | None = None) -> dict:
    env = os.environ.copy()
    env["EDO_CRIU"] = CRIU
    # Keep the placeholder Pod's K8s network namespace; its veth cannot be
    # recreated from the source Pod image because the peer is runtime-owned.
    env["EDO_RESTORE_NET_PID"] = str(host_pid)
    # Edo performs CUDA initialization in the node namespace.  The Rust
    # restore path enters the placeholder mount namespace only for CRIU.
    env["EDO_RESTORE_MOUNT_PID"] = str(host_pid)
    destination_ip = namespace_ipv4(host_pid)
    if source_ip and destination_ip and source_ip != destination_ip:
        env["EDO_RESTORE_REMAP_IP_FROM"] = source_ip
        env["EDO_RESTORE_REMAP_IP_TO"] = destination_ip
    result = subprocess.run(args, text=True, capture_output=True, env=env)
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
                # Keep CRIU in the node PID namespace. Passing host PIDs lets
                # it capture the pod's PID namespace as part of the image;
                # entering the pod PID namespace here loses that metadata.
                cuda = ",".join(str(int(pid)) for pid in body["cuda_pids"])
                args = [EDO, "freeze-group", str(host_pid), cuda, snapshot]
                result = run_on_node(args)
                if result["returncode"] == 0:
                    source_ip = namespace_ipv4(host_pid)
                    if source_ip:
                        (Path(snapshot) / "network.json").write_text(
                            json.dumps({"source_ipv4": source_ip}) + "\n"
                        )
            else:
                args = [EDO, "summon-group", snapshot]
                if body.get("skip_integrity", False):
                    args.append("--skip-integrity")
                # The snapshot contains the source container's mount and PID
                # namespaces. Restore from the node namespace so host PID,
                # driver visibility, and CRIU namespace creation are intact.
                source_ip = None
                network_file = Path(snapshot) / "network.json"
                if network_file.exists():
                    source_ip = json.loads(network_file.read_text()).get("source_ipv4")
                source_ip = body.get("source_ipv4", source_ip)
                result = run_restore_on_node(host_pid, args, source_ip)
            self.send_json(200 if result["returncode"] == 0 else 500, result)
        except Exception as exc:  # request errors are returned as JSON
            self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    prewarm_cuda()
    HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8787"))), Handler).serve_forever()
