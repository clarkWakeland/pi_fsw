import json
import os
from pathlib import Path


DEFAULT_BITRATE_BPS = 15_000_000
ALLOWED_BITRATE_PRESETS_MBPS = (15, 30, 60)
MIN_CUSTOM_BITRATE_BPS = 1_000_000
MAX_CUSTOM_BITRATE_BPS = 100_000_000
STREAM_SETTINGS_PATH = Path(
    os.environ.get("QCAM_STREAM_SETTINGS_PATH", Path.home() / ".qwatercam" / "stream_settings.json")
)


def parse_bitrate_bps(value):
    try:
        bitrate_bps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bitrate_bps must be an integer") from exc

    if bitrate_bps < MIN_CUSTOM_BITRATE_BPS or bitrate_bps > MAX_CUSTOM_BITRATE_BPS:
        raise ValueError(
            f"bitrate_bps must be between {MIN_CUSTOM_BITRATE_BPS} and {MAX_CUSTOM_BITRATE_BPS}"
        )

    return bitrate_bps


def load_bitrate_bps(path=STREAM_SETTINGS_PATH):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return parse_bitrate_bps(data.get("bitrate_bps"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return DEFAULT_BITRATE_BPS


def save_bitrate_bps(bitrate_bps, path=STREAM_SETTINGS_PATH):
    bitrate_bps = parse_bitrate_bps(bitrate_bps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps({"bitrate_bps": bitrate_bps}, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return bitrate_bps


def bitrate_settings_payload(bitrate_bps):
    bitrate_bps = parse_bitrate_bps(bitrate_bps)
    return {
        "bitrate_bps": bitrate_bps,
        "bitrate_mbps": bitrate_bps / 1_000_000,
        "allowed_presets_mbps": list(ALLOWED_BITRATE_PRESETS_MBPS),
        "custom_min_mbps": MIN_CUSTOM_BITRATE_BPS / 1_000_000,
        "custom_max_mbps": MAX_CUSTOM_BITRATE_BPS / 1_000_000,
    }
