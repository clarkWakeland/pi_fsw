#!/usr/bin/env python3
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apply_update


def make_payload(path, release_name):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {
            f"{release_name}/streamer.py": "print('streamer')\n",
            f"{release_name}/device_version.json": '{"version_number": "test"}\n',
        }.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_duplicate_release_failure_does_not_delete_existing_release():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        update_root = root / "update"
        releases = root / "releases"
        release_name = "clarkWakeland-pi_fsw-existing"
        existing_release = releases / release_name
        existing_release.mkdir(parents=True)
        sentinel = existing_release / "sentinel.txt"
        sentinel.write_text("do not delete", encoding="utf-8")
        update_root.mkdir()
        payload = update_root / "firmware_update.tar.gz"
        make_payload(payload, release_name)

        old_values = {
            "UPDATE_ROOT": apply_update.UPDATE_ROOT,
            "PAYLOAD": apply_update.PAYLOAD,
            "STAGING": apply_update.STAGING,
            "RELEASES": apply_update.RELEASES,
            "CURRENT_LINK": apply_update.CURRENT_LINK,
        }

        apply_update.UPDATE_ROOT = str(update_root)
        apply_update.PAYLOAD = str(payload)
        apply_update.STAGING = str(update_root / "staging")
        apply_update.RELEASES = str(releases)
        apply_update.CURRENT_LINK = str(root / "fsw")

        try:
            try:
                apply_update.main()
            except Exception as exc:
                assert "release already exists" in str(exc)
            else:
                raise AssertionError("expected duplicate release failure")

            assert sentinel.exists()
            assert sentinel.read_text(encoding="utf-8") == "do not delete"
        finally:
            for key, value in old_values.items():
                setattr(apply_update, key, value)


if __name__ == "__main__":
    test_duplicate_release_failure_does_not_delete_existing_release()
