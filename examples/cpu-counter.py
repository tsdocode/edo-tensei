#!/usr/bin/env python3
import signal
import time

counter = 0
running = True


def stop(_signal, _frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

while running:
    counter += 1
    print(f"counter={counter}", flush=True)
    time.sleep(1)
