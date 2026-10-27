"""Resource limits for a manual GPU experiment; failure stops this process only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def sample():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0])
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=5)
    thermal = []
    for path in Path("/host/thermal").glob("thermal_zone*/temp"):
        thermal.append(float(path.read_text()) / 1000)
    if not thermal:
        raise RuntimeError("Host thermal sensors are unavailable")
    return {"available_gib": memory["MemAvailable"] / 1024**2,
            "gpu_c": max(float(value) for value in gpu.splitlines()),
            "thermal_c": max(thermal), "time": time.time()}


def violation(reading, config, startup=False):
    minimum = config["startup_available_gib"] if startup else config["min_available_gib"]
    if reading["available_gib"] < minimum:
        return "available_memory"
    if reading["gpu_c"] >= config["max_gpu_c"]:
        return "gpu_temperature"
    if reading["thermal_c"] >= config["max_thermal_c"]:
        return "thermal_temperature"
    return None


def check(config, startup=False):
    reading = sample()
    print(json.dumps({"event": "resources", **reading}), flush=True)
    reason = violation(reading, config, startup)
    if reason:
        raise RuntimeError("Safety stop: " + reason)
    return reading


def watch(config, stopped):
    while not stopped.wait(2):
        try:
            check(config)
        except Exception as exc:
            print(json.dumps({"event": "safety_stop", "reason": str(exc)}),
                  file=sys.stderr, flush=True)
            os._exit(70)


def start_watchdog(config):
    stopped = threading.Event()
    threading.Thread(target=watch, args=(config, stopped), daemon=True).start()
    return stopped
