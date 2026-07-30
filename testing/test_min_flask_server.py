import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import min_flask_server


def test_version_merges_device_metadata(tmp_path, monkeypatch):
    version_path = tmp_path / "device_version.json"
    metadata_path = tmp_path / "device_metadata.json"
    version_path.write_text(
        json.dumps({"name": "Test Camera", "version_number": "1.2.3"}),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"serial_number": "0001"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(min_flask_server, "VERSION_PATH", version_path)
    monkeypatch.setattr(min_flask_server, "DEVICE_METADATA_PATH", metadata_path)

    response = min_flask_server.app.test_client().get("/version")

    assert response.status_code == 200
    assert response.get_json() == {
        "name": "Test Camera",
        "serial_number": "0001",
        "version_number": "1.2.3",
    }
