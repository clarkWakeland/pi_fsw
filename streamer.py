from picamera2 import Picamera2
from libcamera import Transform, controls
from picamera2.encoders import H264Encoder, JpegEncoder, Quality
from picamera2.outputs import FileOutput
import io
import time
from PIL import Image
from control import PersonTracking
import threading
import numpy as np
import websockets
import asyncio
import json
import subprocess


class Websocket_handler():
    def __init__(self, person_tracking_instance, camera_instance):
        self.tracker = person_tracking_instance
        self.camera = camera_instance
        self.runML_pi = False
        self.websocket = None

        self.message_handlers = {
            "box-drawn": self.handle_box_draw,
            "canvas-click": self.handle_canvas_click,
            "toggle-tracking": self.handle_toggle_tracking,
            "get-tracking-status": self.handle_get_tracking_status,
            "manual-control": self.handle_manual_control,
            "autofocus": self.handle_autofocus,
        }

    async def handle_box_draw(self, data, websocket):
        print(data) # TODO: implement

    async def handle_canvas_click(self, data, websocket):
        print("handling_canvas_click")
        if self.runML_pi:
            coords = data["position"]
            x = float(coords['x'])
            y = float(coords['y'])
            self.tracker.start_tracking(x, y)

    async def handle_toggle_tracking(self, data, websocket):
        self.runML_pi = not self.runML_pi
        if not self.runML_pi:
            self.tracker.stop_tracking()
        await websocket.send(json.dumps({"tracking": self.runML_pi}))

    async def handle_get_tracking_status(self, data, websocket):
        await websocket.send(json.dumps({"tracking": self.runML_pi}))

    async def handle_manual_control(self, data, websocket):
        self.tracker.manual_control(data['direction'])

    async def handle_autofocus(self, data, websocket):
        print("autofocus command received")
        self.camera.picam2.autofocus_cycle(wait = False)

    async def main(self):
        async with websockets.serve(self.handle, "0.0.0.0", port=5000):
            print("websocket server started on port 5000")
            await asyncio.Future() # run forever

    async def handle(self, websocket):
        self.websocket = websocket # bit of spaghetti to allow the callback
        async for message in websocket:
            message_dict = json.loads(message)
            msg_type = message_dict.get("type")
            func  = self.message_handlers.get(msg_type)
            if func:
                await func(message_dict, websocket)
            else:
                print(f"Unknown message type: {msg_type}")

class CameraStreamer:
    def __init__(self):
        subprocess.Popen(["./mediamtx"], cwd="../../Downloads", )
        print("started mediamtx server")
        self.picam2 = Picamera2()
        encoder = H264Encoder(qp = 10, iperiod=10)
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1)))
        ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-i', 'pipe:0',  # Input comes from stdin
            '-c:v', 'copy',  # Copy the video codec
            '-f', 'rtsp',  # Output format
            '-rtsp_transport', 'tcp',
            'rtsp://0.0.0.0:8554/live.stream'  # Output file
        ], stdin=subprocess.PIPE)        

        print("started ffmpeg process")
        self.picam2.start_recording(encoder, FileOutput(ffmpeg_process.stdin), quality=Quality.VERY_HIGH)

        # wait for camera
        time.sleep(2)

        self.picam2.autofocus_cycle(wait = False)
        # Wait 5 seconds, then save the image
    
    def capture_array(self):
        return self.picam2.capture_array()

def send_ws_message(message):
    if wsHandler.websocket: # if websocket is connected
        asyncio.run_coroutine_threadsafe(
            wsHandler.websocket.send(json.dumps(message)),
            wsHandler.websocket.loop
        )
        print(message)

camera = CameraStreamer()
pTrack = PersonTracking(send_ws_message, camera)
wsHandler = Websocket_handler(pTrack, camera)

if __name__ == '__main__':
    try:
        asyncio.run(wsHandler.main())
    except KeyboardInterrupt:
        print("Shutting down...")

    
