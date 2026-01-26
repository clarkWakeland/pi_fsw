#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tarfile

UPDATE_ROOT = "/opt/update"
PAYLOAD = f"{UPDATE_ROOT}/firmware_update.tar.gz"
STAGING = f"{UPDATE_ROOT}/staging"

CAM_ROOT = "/opt/fsw"
RELEASES = f"{CAM_ROOT}/releases"
CURRENT_LINK = f"{CAM_ROOT}/current"

SERVICE = "streamer.service"


def run(cmd):
    subprocess.check_call(cmd)


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise Exception(msg)


def main():
    if not os.path.exists(PAYLOAD):
        fail("firmware_update.tar.gz not found")

    # Ensure dirs exist
    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(RELEASES, exist_ok=True)

    target = None
    service_stopped = False
    
    try:
        # Clean staging (SAFE — not live code)
        for item in os.listdir(STAGING):
            shutil.rmtree(os.path.join(STAGING, item))

        # Extract payload
        with tarfile.open(PAYLOAD, "r:gz") as tar:
            tar.extractall(STAGING)

        entries = os.listdir(STAGING)
        if len(entries) != 1:
            fail("payload must contain exactly one top-level directory")

        extracted = os.path.join(STAGING, entries[0])
        if not os.path.isdir(extracted):
            fail("extracted payload is not a directory")

        # Basic sanity check
        required = ["streamer.py", "device_version.json"]
        for f in required:
            if not os.path.exists(os.path.join(extracted, f)):
                fail(f"missing required file: {f}")

        target = os.path.join(RELEASES, entries[0])
        if os.path.exists(target):
            fail(f"release already exists: {target}")

        # Move into durable releases directory
        shutil.move(extracted, target)

        # Stop service before switching
        run(["systemctl", "stop", SERVICE])
        service_stopped = True

        # Atomically flip symlink
        run(["ln", "-sfn", target, CURRENT_LINK])

        # Restart service
        run(["systemctl", "start", SERVICE])
        service_stopped = False

        # Cleanup
        os.remove(PAYLOAD)
        shutil.rmtree(STAGING)

        print(f"Update applied successfully: {target}")
        return True
        
    except Exception as e:
        print(f"Update failed: {e}", file=sys.stderr)
        
        # Attempt to restart service if it was stopped
        if service_stopped:
            try:
                print("Attempting to restart service...", file=sys.stderr)
                run(["systemctl", "start", SERVICE])
            except Exception as restart_err:
                print(f"Failed to restart service: {restart_err}", file=sys.stderr)
        
        # Cleanup failed release if it was created
        if target and os.path.exists(target):
            try:
                shutil.rmtree(target)
                print(f"Cleaned up failed release: {target}", file=sys.stderr)
            except Exception as cleanup_err:
                print(f"Failed to cleanup: {cleanup_err}", file=sys.stderr)
        
        raise


if __name__ == "__main__":
    main()
