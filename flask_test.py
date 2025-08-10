from flask import Flask, Response
from picamera2 import Picamera2
import io
import time
from PIL import Image
from detection_test import PersonTracking
import threading
import numpy as np
from flask_socketio import SocketIO, emit
import websockets
import asyncio

app = Flask(__name__)

class Websocket_handler():
    async def main(self):
        async with websockets.serve(self.handle, "0.0.0.0", port=5000):
            print("websocket server started on port 5000")
            await asyncio.Future() # run forever

    async def handle(self, websocket):
        async for message in websocket:
            print(message)

class CameraStreamer:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (1920, 1080)}))
        self.picam2.start()
        self.frame = None
        self.frame_copy = None
        self.condition = threading.Condition()
        self.update_frame()
        print("TESTTESTSETSETST")
        # self.tracker = PersonTracking()
        # threading.Thread(target = self.tracking, daemon=True).start()
    
            
    def update_frame(self):
        # This function runs continuously to update the current frame
        def _update():
            while True:
                print("test1")
                buffer = io.BytesIO()
                self.frame = self.picam2.capture_array()
                self.frame_copy = np.copy(self.frame)
                # self.tracker.basic_video(frame)
                img = Image.fromarray(self.frame)
                print(img)
                img.save(buffer, format='JPEG')
                with self.condition:
                    self.frame = buffer.getvalue()
                    self.condition.notify_all()
                # time.sleep(1 / 40)  # ~30 fps
        
        threading.Thread(target=_update(), daemon=True).start()
        
    def tracking(self):
        while True:
            # self.tracker.basic_video(self.frame_copy)
            pass
            
    
    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame

camera = CameraStreamer()

wsHandler = Websocket_handler()


@app.route('/video')
def video_feed():
    def generate():
        while True:
            print("test2")
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Raspberry Pi Camera Stream</h1><img src="/video" />'

def start_ws():
    asyncio.run(wsHandler.main())
    
    
if __name__ == '__main__':
    threading.Thread(target=start_ws, daemon=True).start()
    print("test")
    app.run(host='0.0.0.0', port=8000, threaded=True)
    
