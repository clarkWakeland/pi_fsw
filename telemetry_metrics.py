import json
import os
import shutil
import subprocess
from pathlib import Path

from stream_settings import load_bitrate_bps


FIRMWARE_VERSION_PATH = Path(
    os.environ.get("QWATERCAM_FIRMWARE_VERSION_PATH", "/opt/fsw/device_version.json")
)
THERMAL_ZONE_PATH = Path(
    os.environ.get("QWATERCAM_THERMAL_ZONE_PATH", "/sys/class/thermal/thermal_zone0/temp")
)
HAILO_DEVICE_PATH = Path(os.environ.get("QWATERCAM_HAILO_DEVICE_PATH", "/dev/hailo0"))
SYSTEMD_UNITS = ("streamer.service", "updater.service", "hailort.service")


def read_firmware_version(path=FIRMWARE_VERSION_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "unknown"

    for key in ("version_number", "version"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, str) and value:
            return value
    return "unknown"


def read_uptime_seconds(path="/proc/uptime"):
    try:
        return float(Path(path).read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_cpu_temperature_celsius(path=THERMAL_ZONE_PATH):
    try:
        return float(Path(path).read_text(encoding="utf-8").strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_memory_used_ratio(path="/proc/meminfo"):
    try:
        values = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
        total = values["MemTotal"]
        available = values["MemAvailable"]
        return max(0.0, min(1.0, (total - available) / total))
    except (OSError, ValueError, KeyError, ZeroDivisionError, IndexError):
        return None


def read_disk_used_ratio(path="/"):
    try:
        usage = shutil.disk_usage(path)
        return usage.used / usage.total
    except (OSError, ZeroDivisionError):
        return None


def read_cpu_throttled(run=subprocess.run):
    try:
        result = run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0 or "=" not in result.stdout:
            return None
        flags = int(result.stdout.strip().split("=", 1)[1], 16)
        return 1 if flags else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def read_systemd_unit(unit, run=subprocess.run):
    try:
        result = run(
            ["systemctl", "show", unit, "-p", "ActiveState", "-p", "NRestarts"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"unit": unit, "up": 0, "restart_count": 0}

    properties = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value

    try:
        restart_count = int(properties.get("NRestarts", 0))
    except ValueError:
        restart_count = 0

    return {
        "unit": unit,
        "up": 1 if properties.get("ActiveState") == "active" else 0,
        "restart_count": restart_count,
    }


def collect_status_snapshot():
    return {
        "device_up": 1,
        "device_uptime_seconds": read_uptime_seconds(),
        "cpu_temperature_celsius": read_cpu_temperature_celsius(),
        "cpu_throttled": read_cpu_throttled(),
        "memory_used_ratio": read_memory_used_ratio(),
        "disk_used_ratio": read_disk_used_ratio(),
        "hailo_present": 1 if HAILO_DEVICE_PATH.exists() else 0,
        "stream_bitrate_bits_per_second": load_bitrate_bps(),
        "services": [read_systemd_unit(unit) for unit in SYSTEMD_UNITS],
    }
