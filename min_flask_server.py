from flask import Flask, request, jsonify
from pathlib import Path
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "Ping!\n"

@app.route('/version', methods=['GET'])
def get_version():
    data = json.loads(Path('device_version.json').read_text(encoding='utf-8'))
    return jsonify(data)

@app.route('/update', methods=['POST'])
def update_firmware():
    # Placeholder for firmware update logic
    if "firmware" not in request.files:
        return "No firmware file provided.\n", 400

    firmware_file = request.files['firmware']
    firmware_file.save('/opt/update/firmware_update.tar.gz')

    # unzip tarball    

    return "Firmware update initiated.\n", 202
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)