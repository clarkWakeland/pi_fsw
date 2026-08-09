from datetime import datetime, timezone
from flask import Flask, request, jsonify
from pathlib import Path
import json
import apply_update
from device_identity import load_device_identity
from telemetry_metrics import collect_status_snapshot, read_firmware_version

app = Flask(__name__)
VERSION_PATH = Path("/opt/fsw/device_version.json")
DEVICE_METADATA_PATH = Path("/opt/device_metadata.json")
TELEMETRY_SCHEMA_VERSION = 1

@app.route('/')
def index():
    return "Ping!\n"

@app.route('/version', methods=['GET'])
def get_version():
    data = json.loads(VERSION_PATH.read_text(encoding='utf-8'))
    if DEVICE_METADATA_PATH.exists():
        data.update(load_device_identity(DEVICE_METADATA_PATH))
    return jsonify(data)


def build_telemetry_payload(observed_at=None):
    identity = load_device_identity(DEVICE_METADATA_PATH)
    snapshot = collect_status_snapshot()
    services = snapshot.pop("services", [])

    payload = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        **identity,
        "firmware_version": read_firmware_version(VERSION_PATH),
        "observed_at": observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": snapshot,
        "services": services,
    }
    return payload


@app.route('/telemetry', methods=['GET'])
def get_telemetry():
    return jsonify(build_telemetry_payload())

@app.route('/update', methods=['POST'])
def update_firmware():
    # Placeholder for firmware update logic
    if "firmware" not in request.files:
        return "No firmware file provided.\n", 400

    firmware_file = request.files['firmware']
    firmware_file.save('/opt/update/firmware_update.tar.gz')

    try:
        apply_update.main()
    except Exception as e:
        return f"Update failed: {str(e)}\n", 500

    return "Firmware update initiated.\n", 202

    

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5001)
