# ESP32 brushless gimbal handoff

## Current state

- Repository: `/home/clark/repos/pi_fsw`
- Branch: `esp32_gimbal_control`
- Latest implementation commit: `e4a8667` (`Reduce pan tracking gain`)
- Base branch: `smooth_movement_branch` at `dabe6f8`
- Deployed target: `qcam1:/opt/releases/pi_fsw-dev`
- Last verification: `streamer.service` active with zero restarts; both ESP32s connected
- Test result: 68 passed, 1 skipped
- Leave the unrelated untracked `frame_test_59_94.mp4` alone.

Commits on this branch, oldest first:

1. `63fe69d` Add ESP32 gimbal serial control
2. `c8f29b3` Rotate camera output for inverted mount
3. `a601864` Map connected ESP32 to tilt motor
4. `6c8ce8e` Reverse manual pan joystick direction
5. `951ac12` Use calibrated default as tilt lower limit
6. `536a1bc` Reverse pan direction for automatic tracking
7. `00ab661` Reduce tilt tracking gain
8. `e4a8667` Reduce pan tracking gain

## Hardware and USB mapping

The two SimpleFOC ESP32 controllers are distinguished by physical USB path
because both CP2102 bridges report the same serial number.

| Axis | USB path | Current tty |
| --- | --- | --- |
| TILT | `/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0-port0` | `/dev/ttyUSB0` |
| PAN | `/dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0` | `/dev/ttyUSB1` |

The defaults are in `gimbal_serial.py`. They can be overridden with
`QCAM_TILT_SERIAL_PORT` and `QCAM_PAN_SERIAL_PORT`.

Both ESP32s currently run the v1 interface firmware. The firmware reports the
generic name `AXIS1`; the Pi assigns PAN/TILT from the physical USB path.

## Pi-side architecture

`servo_control.MotorControl` retains the API used by `control.PersonTracking`,
but direct `pantilthat` calls have been replaced by one persistent
`gimbal_serial.GimbalAxis` worker per motor.

The tracking path is:

```text
Hailo inference -> BYTETrack -> smoothed image error -> P/D correction
-> software angle limit -> absolute ESP32 TARGET command -> SimpleFOC motor loop
```

Important serial behavior:

- Standard-library `termios` is used, so the Pi FSW needs no new Python package.
- Only the newest pending absolute target is retained.
- A pending target expires after 0.5 seconds rather than moving late after a reconnect.
- Workers reconnect automatically and keep the controller alive with `PING`.
- A controller must return a valid v1 `STATE` before the Pi sends `PING` or `TARGET`.
  This prevents legacy character-command firmware from interpreting protocol text
  as motor commands.
- A missing or incompatible controller does not crash RTSP streaming.
- ESP32 feedback initializes the Pi's virtual angle after startup/reconnect.

The current rates are:

- ML inference target: 20 Hz
- Pi tracking loop: about 30 Hz
- ESP32 motor-control loop: 200 Hz
- ESP32 state telemetry: 10 Hz

## Motion configuration

Software angle limits in `servo_control.py`:

- PAN: `-90` to `+90` degrees
- TILT: `0` to `+90` degrees

Tracking gains:

| Axis | P | D |
| --- | ---: | ---: |
| PAN | 0.006 | 0.00025 |
| TILT | 0.006 | 0.00025 |

PAN automatic tracking is intentionally sign-inverted relative to the legacy
servo convention. Manual PAN is also reversed from the old PanTilt HAT code,
but this is implemented separately in `set_manual_input`. TILT manual and
tracking directions were visually confirmed as correct.

Manual joystick target increments use:

- Deadzone: 0.08
- Minimum nonzero step: 0.02 degrees
- Precision-band maximum step: 0.08 degrees
- Full-stick maximum step: 0.28 degrees per 30 Hz update

The ESP32 remains responsible for actual motor acceleration and velocity
limiting. Current firmware values are 0.30 rad/s^2 acceleration and 0.15 rad/s
maximum velocity, with a 3 V motor limit.

## Stored calibration

Both encoders have persistent zeros in ESP32 NVS (`zero_persisted=1`).

- TILT zero raw encoder angle: approximately `129.551` degrees
- PAN zero raw encoder angle: approximately `137.373` degrees
- The PAN zero is about 89.7 degrees left of its original calibrated default.

`ZERO` is accepted only while the controller is disabled. PAN relaxes by about
12 degrees when motor torque is removed. Its final calibration therefore used
back-to-back `STOP` and `ZERO` commands on one serial connection so NVS captured
the energized target position before mechanical relaxation.

Known nuance: a persisted zero defines the logical HOME position, but the
current Pi startup code does **not** automatically issue `HOME`. After a cold
power cycle, the motor remains at its physical position until it receives a
target. If automatic boot homing is desired, add it deliberately with suitable
safety/interlock behavior rather than assuming calibration itself causes motion.

Both AS5600s have reported valid encoder data and zero I2C errors. The AS5600
magnet-weak flag has also been reported, so magnet spacing should be revisited
if encoder stability becomes a problem.

## ESP32 firmware

Firmware source on the development machine:

```text
/home/clark/repos/qcam_gimbal_firmware
```

Copy on `qcam1`:

```text
/home/clark64/dev/qcam_gimbal_firmware
```

The firmware directory is not currently a Git repository. Key files are:

- `src/gimbal_interface.cpp`
- `PROTOCOL.md`
- `platformio.ini`
- `tools/gimbal_serial.py`

Protocol commands are `TARGET`, `HOME`, `STOP`, `ZERO`, `STATUS`, and `PING`.
The first target after boot/STOP performs 750 ms electrical alignment. After 5
seconds without commands the controller holds its measured position; after 30
seconds it disables the driver. Pi keepalives normally prevent those timeouts.

To flash a controller, stop `streamer.service` first so it releases the port,
then select the corresponding USB path explicitly:

```bash
ssh qcam1
sudo systemctl stop streamer.service
cd /home/clark64/dev/qcam_gimbal_firmware
pio run -e gimbal_interface -t upload \
  --upload-port /dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0
sudo systemctl start streamer.service
```

The example above flashes PAN; substitute the `.0` path for TILT.

## Camera orientation change

The camera on this build is physically opposite the original hardware. The old
`Transform(hflip=1, vflip=1)` made its image upside down, so `streamer.py` now
uses `Transform()`. Because this is applied in the Picamera2 configuration, the
RTSP main stream and 640x640 ML stream share the same orientation without an
extra decode/re-encode step.

## Deployment and verification

Deploy the current branch:

```bash
cd /home/clark/repos/pi_fsw
scripts/qcam-dev --no-restart-updater deploy
```

Follow logs:

```bash
scripts/qcam-dev logs --follow
```

Useful direct checks:

```bash
ssh qcam1 'systemctl status streamer.service --no-pager -l'
ssh qcam1 'systemctl show streamer.service -p MainPID -p NRestarts -p Result'
```

Healthy startup logs should show both physical paths connected and the
WebSocket server listening on port 5000. The RTSP publisher remains
`rtsp://qcam1:8554/live.stream`.

Run the local tests before deployment:

```bash
cd /home/clark/repos/pi_fsw
python3 -m pytest -q
git diff --check
```

## Suggested next checks

1. Repeat closed-loop swimmer tracking with the reduced gains and watch for
   oscillation, lag, and limit behavior on both axes.
2. Tune one axis/parameter at a time. The axis-specific gain fields are in
   `servo_control.py`.
3. Decide whether automatic boot homing is required and, if so, define when it
   is safe for the enclosure to move before implementing it.
4. Check AS5600 magnet spacing because both controllers have reported the weak
   magnet status bit.
