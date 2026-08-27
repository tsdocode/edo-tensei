#!/usr/bin/env python3
import json
import os
import signal
import time

report_path = os.environ["EDO_DEMO_REPORT"]
ready_path = os.environ["EDO_DEMO_READY"]
counter = 0
running = True


def write_event(event):
    with open(report_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "counter": counter}) + "\n")


def request(_signum, _frame):
    global counter
    counter += 1
    write_event("request")


def stop(_signum, _frame):
    global running
    running = False


signal.signal(signal.SIGUSR1, request)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

time.sleep(float(os.environ.get("EDO_DEMO_STARTUP_SECONDS", "2")))
write_event("warmup")
open(ready_path, "w", encoding="utf-8").write(str(os.getpid()))

while running:
    time.sleep(0.1)
