#!/usr/bin/env python3
import json
import os
import signal
import socket
import time

running = True
counter = 0
state_path = os.environ.get("EDO_FIXTURE_STATE", "/tmp/edo-resource-fixture.json")
held_file = open(os.environ.get("EDO_FIXTURE_HELD_FILE", "/tmp/edo-held-file.txt"), "a+", encoding="utf-8")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 0))
server.listen(1)
server.settimeout(0.2)


def stop(_signal, _frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

while running:
    counter += 1
    try:
        client, _address = server.accept()
        client.close()
    except socket.timeout:
        pass
    with open(state_path, "w", encoding="utf-8") as state:
        json.dump(
            {
                "pid": os.getpid(),
                "counter": counter,
                "cwd": os.getcwd(),
                "marker": os.environ.get("EDO_FIXTURE_MARKER"),
                "socket_port": server.getsockname()[1],
                "monotonic": time.monotonic(),
            },
            state,
        )
    time.sleep(0.5)

server.close()
held_file.close()
