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
import json

app = Flask(__name__)

class Websocket_handler():
    def __init__(self, person_tracking_instance):
        self.tracker = person_tracking_instance
        self.trackingPrimed = False
        self.websocket = None

    async def main(self):
        async with websockets.serve(self.handle, "0.0.0.0", port=5000):
            print("websocket server started on port 5000")
            await asyncio.Future() # run forever

    async def handle(self, websocket):
        self.websocket = websocket
        async for message in websocket:
            message_dict = json.loads(message)
            match message_dict["type"]:
                case "box-drawn":
                    print(message_dict)
                
                case "canvas-click":
                    print(message_dict)
                    if self.trackingPrimed:
                        self.tracker.start_tracking(message_dict["position"]["x"], message_dict["position"]["y"])
                    else:
                        await websocket.send(json.dumps({"error": "Click doesn't matter, not tracking"}))

                case "toggle-tracking":
                    # self.tracker.toggle_tracking()
                    self.trackingPrimed = not self.trackingPrimed
                    if not self.trackingPrimed:
                        self.tracker.stop_tracking()
                    await websocket.send(json.dumps({"tracking": self.trackingPrimed}))

                case "manual-control":
                    print(f"manual control: {message_dict['direction']}")
                    self.tracker.manual_control(message_dict["direction"])


class CameraStreamer:
    def __init__(self, person_tracking_instance):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (640, 640)}))
        self.picam2.start()
        self.frame = None
        self.frame_copy = None
        self.condition = threading.Condition()
        self.update_frame()
        self.tracker = person_tracking_instance
        threading.Thread(target = self.tracking, daemon=True).start()

    def update_frame(self):
        # This function runs continuously to update the current frame
        def _update():
            while True:
                buffer = io.BytesIO()
                self.frame = self.picam2.capture_array()
                self.frame_copy = np.copy(self.frame)
                img = Image.fromarray(self.frame)
                img.save(buffer, format='JPEG')
                with self.condition:
                    self.frame = buffer.getvalue()
                    self.condition.notify_all()
        
        threading.Thread(target=_update, daemon=True).start()
        
    def tracking(self):
        while True:
            self.tracker.basic_video(self.frame_copy)
    
    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame
def send_ws_message(message):
    if wsHandler.websocket:
        asyncio.run_coroutine_threadsafe(
            wsHandler.websocket.send(json.dumps(message)),
            wsHandler.websocket.loop
        )
        print(message)
pTrack = PersonTracking(send_ws_message)
camera = CameraStreamer(pTrack)

wsHandler = Websocket_handler(pTrack)


@app.route('/video')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()

            # only yield if frame isn't a np array
            if not isinstance(frame, np.ndarray):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Raspberry Pi Camera Stream</h1><img src="/video" />'

@app.route('/tracking')
def return_tracking_value():
    return {"tracking": wsHandler.trackingPrimed}

def start_ws():
    asyncio.run(wsHandler.main())
    
    
if __name__ == '__main__':
    threading.Thread(target=start_ws, daemon=True).start()
    print("test")
    app.run(host='0.0.0.0', port=8000, threaded=True)
    
