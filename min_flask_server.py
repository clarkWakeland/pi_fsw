from flask import Flask, request, jsonify
from pathlib import Path
import json
import apply_update

app = Flask(__name__)
VERSION_PATH = Path("/opt/fsw/device_version.json")
DEVICE_METADATA_PATH = Path("/opt/device_metadata.json")

@app.route('/')
def index():
    return "Ping!\n"

@app.route('/version', methods=['GET'])
def get_version():
    data = json.loads(VERSION_PATH.read_text(encoding='utf-8'))
    if DEVICE_METADATA_PATH.exists():
        device_metadata = json.loads(DEVICE_METADATA_PATH.read_text(encoding='utf-8'))
        data.update(device_metadata)
    return jsonify(data)

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
