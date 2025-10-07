from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import Transform, controls
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
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
    def __init__(self, person_tracking_instance, camera_instance):
        self.tracker = person_tracking_instance
        self.camera = camera_instance
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
                
                case "get-tracking-status":
                    await websocket.send(json.dumps({"tracking": self.trackingPrimed}))

                case "autofocus":
                    print("autofocusing")
                    result = self.camera.picam2.autofocus_cycle(wait = False)

class CameraStreamer:
    def __init__(self, person_tracking_instance):
        self.picam2 = Picamera2()
        encoder = H264Encoder(bitrate=50000000)
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (1920, 1080)},
                                                                     transform=Transform(hflip=1, vflip=1)))
        import subprocess
        ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-i', 'pipe:0',  # Input comes from stdin
            '-c:v', 'copy',  # Copy the video codec
            '-f', 'rtsp',  # Output format
            '-g', '60',
            '-pix_fmt', 'yuv420p',
            '-rtsp_transport', 'tcp',
            'rtsp://0.0.0.0:8554/live.stream'  # Output file
        ], stdin=subprocess.PIPE)        
        self.picam2.start_recording(encoder, FileOutput(ffmpeg_process.stdin))
        self.picam2.start()
        
        self.picam2.set_controls({"AfRange": controls.AfRangeEnum.Macro})
        success = self.picam2.autofocus_cycle(wait = False)
        self.condition = threading.Condition()
        self.tracker = person_tracking_instance
        threading.Thread(target = self.tracking, daemon=True).start()
        
        # start mediamtx server
        subprocess.Popen(["mediamtx"], cwd="../../Downloads")
        print("started mediamtx server")
           
    def tracking(self):
        while True:
            self.tracker.basic_video(self.picam2.capture_array())
    
def send_ws_message(message):
    if wsHandler.websocket:
        asyncio.run_coroutine_threadsafe(
            wsHandler.websocket.send(json.dumps(message)),
            wsHandler.websocket.loop
        )
        print(message)

pTrack = PersonTracking(send_ws_message)
camera = CameraStreamer(pTrack)

wsHandler = Websocket_handler(pTrack, camera)

@app.route('/video')
def video_feed():
    pass

def start_ws():
    asyncio.run(wsHandler.main())
    
    
if __name__ == '__main__':
    threading.Thread(target=start_ws, daemon=True).start()
    print("test")
    app.run(host='0.0.0.0', port=8000, threaded=True)
    
