from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FileOutput
import time
from control import PersonTracking
import websockets
import asyncio
import json
import subprocess
import smbus2
import logging

# Battery poller task
I2C_BUS = 1
ADDR = 0x36
VOLTAGE_REG = 0x02
POLL_INTERVAL = 120 # in seconds

bus = smbus2.SMBus(I2C_BUS)


class Websocket_handler():
    def __init__(self, person_tracking_instance, camera_instance):
        self.tracker = person_tracking_instance
        self.camera = camera_instance
        self.runML_pi = False
        self.websockets = set()
        self.loop = None

        self.message_handlers = {
            "box-drawn": self.handle_box_draw,
            "canvas-click": self.handle_canvas_click,
            "toggle-tracking": self.handle_toggle_tracking,
            "get-tracking-status": self.handle_get_tracking_status,
            "manual-control": self.handle_manual_control,
            "autofocus": self.handle_autofocus,
            "toggle-auto-acquire": self.handle_auto_acquire,
            "show-boxes": self.handle_show_boxes
        }

    async def handle_box_draw(self, data, websocket):
        print(data) # TODO: implement
        box = data['box']
        box_data = [box['x'], box['y'], box['width'], box['height']]
        self.tracker.user_intent.set_ROI(box_data)  # box is [x, y, w, h]

    async def handle_canvas_click(self, data, websocket):
        print("handling_canvas_click")

        coords = data["position"]
        x = float(coords['x'])
        y = float(coords['y'])
        self.tracker.user_intent.set_click_coordinates(x, y)
        self.tracker.user_intent.clear_ROI()

    async def handle_toggle_tracking(self, data, websocket):
        run_ml = self.tracker.toggle_tracking()
        await websocket.send(json.dumps({"tracking_primed": run_ml}))
        await websocket.send(json.dumps({
            "type": "tracking-state",
            "payload": self.tracker.get_tracking_status()
        }))

    async def handle_get_tracking_status(self, data, websocket):
        status = self.tracker.get_tracking_status()
        await websocket.send(json.dumps({"tracking_primed": status["run_ml"]}))
        await websocket.send(json.dumps({"type": "tracking-state", "payload": status}))

    async def handle_manual_control(self, data, websocket):
        self.tracker.manual_control(data['direction'])

    async def handle_autofocus(self, data, websocket):
        self.camera.picam2.autofocus_cycle(wait = False)
        print("autofocus command received")

    async def handle_auto_acquire(self, data, websocket):
        self.tracker.user_intent.auto_acquire = not self.tracker.user_intent.auto_acquire
        print(f"auto acquire set to {self.tracker.user_intent.auto_acquire}")
        self.tracker.emit_tracking_state(force=True)

    async def handle_show_boxes(self, data, websocket):
        self.tracker.show_boxes = not self.tracker.show_boxes
        print(f"show boxes set to {self.tracker.show_boxes}")

    def read_voltage(self) -> float:
        val = bus.read_word_data(ADDR, VOLTAGE_REG)
        swapped = ((val << 8) & 0xFF00) + (val >> 8)
        return (swapped >> 3) * 1.25 / 1000.0

    async def poll_battery(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                voltage = await loop.run_in_executor(None, self.read_voltage)
                logging.info(f"Battery voltage: {voltage:.3f} V")
                send_ws_message({"battery_voltage": voltage})

            except Exception as e:
                logging.error(f"Error polling battery voltage: {e}")

            await asyncio.sleep(POLL_INTERVAL)

    async def main(self):
        try:
            self.loop = asyncio.get_running_loop()
            async with websockets.serve(self.handle, "0.0.0.0", port=5000):
                print("websocket server started on port 5000")
                battery_task = asyncio.create_task(self.poll_battery())                    
                await asyncio.Future() # run forever
        finally:
            battery_task.cancel()

    async def broadcast(self, message):
        if not self.websockets:
            return

        data = json.dumps(message)
        disconnected = []
        for ws in list(self.websockets):
            try:
                await ws.send(data)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.websockets.discard(ws)

    def send_from_thread(self, message):
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def handle(self, websocket):
        self.websockets.add(websocket)
        try:
            await websocket.send(json.dumps({
                "type": "tracking-state",
                "payload": self.tracker.get_tracking_status()
            }))
            async for message in websocket:
                message_dict = json.loads(message)
                msg_type = message_dict.get("type")
                func  = self.message_handlers.get(msg_type)
                if func:
                    await func(message_dict, websocket)
                else:
                    print(f"Unknown message type: {msg_type}")
        finally:
            self.websockets.discard(websocket)

class CameraStreamer:
    def __init__(self):
        subprocess.Popen(["./mediamtx"], cwd="/home/clark64/Downloads", )
        self.picam2 = Picamera2()
        encoder = H264Encoder(bitrate=15_000_000, iperiod=30)
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'BGR888', "size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1)))
        ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-i', 'pipe:0',
            '-c:v', 'copy', 
            '-f', 'rtsp',  
            '-rtsp_transport', 'tcp',
            'rtsp://0.0.0.0:8554/live.stream'  # Output to mediamtx server
        ], stdin=subprocess.PIPE)        

        print("started ffmpeg process")
        self.picam2.start_recording(encoder, FileOutput(ffmpeg_process.stdin), quality=Quality.VERY_HIGH)

        # wait for camera
        time.sleep(2)
        self.picam2.autofocus_cycle(wait = False)
    
    def capture_array(self):
        return self.picam2.capture_array()

def send_ws_message(message):
    if "wsHandler" not in globals():
        return
    wsHandler.send_from_thread(message)
    print(message)

camera = CameraStreamer()
pTrack = PersonTracking(send_ws_message, camera)
wsHandler = Websocket_handler(pTrack, camera)

if __name__ == '__main__':
    try:
        asyncio.run(wsHandler.main())
    except KeyboardInterrupt:
        print("Shutting down...")

    
