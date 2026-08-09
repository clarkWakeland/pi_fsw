import json

import pytest

from device_identity import DeviceIdentityError, load_device_identity


def test_load_device_identity_preserves_leading_zeroes(tmp_path):
    path = tmp_path / "device_metadata.json"
    path.write_text(
        json.dumps({"serial_number": "0002", "hardware_serial": "abc123"}),
        encoding="utf-8",
    )

    assert load_device_identity(path) == {
        "serial_number": "0002",
        "hardware_serial": "abc123",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"serial_number": 2},
        {"serial_number": ""},
        {"serial_number": "serial number with spaces"},
    ],
)
def test_load_device_identity_rejects_invalid_serials(tmp_path, metadata):
    path = tmp_path / "device_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DeviceIdentityError):
        load_device_identity(path)


def test_load_device_identity_reports_missing_file(tmp_path):
    with pytest.raises(DeviceIdentityError, match="device metadata not found"):
        load_device_identity(tmp_path / "missing.json")
