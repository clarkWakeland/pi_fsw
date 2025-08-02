from flask import Flask, Response
from picamera2 import Picamera2
import io
import time
from PIL import Image
from detection_test import PersonTracking
import threading
import numpy as np
from flask_socketio import SocketIO, emit

app = Flask(__name__)

class CameraStreamer:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (1040, 640)}))
        self.picam2.start()
        self.frame = None
        self.frame_copy = None
        self.condition = threading.Condition()
        self.update_frame()
        self.tracker = PersonTracking()
        threading.Thread(target = self.tracking, daemon=True).start()

    def update_frame(self):
        # This function runs continuously to update the current frame
        def _update():
            while True:
                buffer = io.BytesIO()
                self.frame = self.picam2.capture_array()
                self.frame_copy = np.copy(self.frame)
                # self.tracker.basic_video(frame)
                img = Image.fromarray(self.frame)
                img.save(buffer, format='JPEG')
                with self.condition:
                    self.frame = buffer.getvalue()
                    self.condition.notify_all()
                # time.sleep(1 / 40)  # ~30 fps
        
        threading.Thread(target=_update, daemon=True).start()
        
    def tracking(self):
        while True:
            self.tracker.basic_video(self.frame_copy)
            
            
    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame

camera = CameraStreamer()
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("click")
def handle_click(data):
    print('Click event received', data)

@app.route('/video')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Raspberry Pi Camera Stream</h1><img src="/video" />'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
    #socketio.run(app, host = '0.0.0.0', port=5000, threaded=True)
