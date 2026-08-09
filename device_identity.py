import json
import os
import re
from pathlib import Path


DEVICE_METADATA_PATH = Path(
    os.environ.get("QWATERCAM_DEVICE_METADATA_PATH", "/opt/device_metadata.json")
)
SERIAL_NUMBER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class DeviceIdentityError(ValueError):
    pass


def load_device_identity(path=None):
    metadata_path = Path(path) if path is not None else DEVICE_METADATA_PATH

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeviceIdentityError(f"device metadata not found: {metadata_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DeviceIdentityError(f"unable to read device metadata: {metadata_path}") from exc

    if not isinstance(data, dict):
        raise DeviceIdentityError("device metadata must be a JSON object")

    serial_number = data.get("serial_number")
    if not isinstance(serial_number, str) or not SERIAL_NUMBER_PATTERN.fullmatch(serial_number):
        raise DeviceIdentityError(
            "serial_number must be a 1-64 character string containing letters, numbers, '.', '_' or '-'"
        )

    identity = {"serial_number": serial_number}
    hardware_serial = data.get("hardware_serial")
    if isinstance(hardware_serial, str) and hardware_serial:
        identity["hardware_serial"] = hardware_serial

    return identity
