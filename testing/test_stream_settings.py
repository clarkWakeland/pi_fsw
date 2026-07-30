#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stream_settings import (
    ALLOWED_BITRATE_PRESETS_MBPS,
    DEFAULT_BITRATE_BPS,
    bitrate_settings_payload,
    load_bitrate_bps,
    parse_bitrate_bps,
    save_bitrate_bps,
)


def test_parse_bitrate_accepts_supported_presets_and_custom_values():
    assert ALLOWED_BITRATE_PRESETS_MBPS == (15, 30, 60)
    assert parse_bitrate_bps(15_000_000) == 15_000_000
    assert parse_bitrate_bps("30000000") == 30_000_000
    assert parse_bitrate_bps(42_500_000) == 42_500_000


def test_parse_bitrate_rejects_invalid_values():
    for value in (0, -1, "abc", 999_999, 100_000_001):
        try:
            parse_bitrate_bps(value)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {value!r}")


def test_load_bitrate_uses_default_for_missing_or_invalid_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "stream_settings.json"
        assert load_bitrate_bps(path) == DEFAULT_BITRATE_BPS

        path.write_text(json.dumps({"bitrate_bps": 0}), encoding="utf-8")
        assert load_bitrate_bps(path) == DEFAULT_BITRATE_BPS


def test_save_and_load_bitrate_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "nested" / "stream_settings.json"

        save_bitrate_bps(60_000_000, path)

        assert load_bitrate_bps(path) == 60_000_000


def test_bitrate_payload_includes_presets_and_mbps_value():
    payload = bitrate_settings_payload(30_000_000)

    assert payload == {
        "bitrate_bps": 30_000_000,
        "bitrate_mbps": 30.0,
        "allowed_presets_mbps": [15, 30, 60],
        "custom_min_mbps": 1.0,
        "custom_max_mbps": 100.0,
    }


if __name__ == "__main__":
    test_parse_bitrate_accepts_supported_presets_and_custom_values()
    test_parse_bitrate_rejects_invalid_values()
    test_load_bitrate_uses_default_for_missing_or_invalid_file()
    test_save_and_load_bitrate_round_trip()
    test_bitrate_payload_includes_presets_and_mbps_value()
