#!/usr/bin/env python3
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gimbal_serial import DEFAULT_PAN_PORT, DEFAULT_TILT_PORT, GimbalAxis


def test_default_usb_paths_match_physical_motor_wiring():
    assert "xhci-hcd.0" in DEFAULT_TILT_PORT
    assert "xhci-hcd.1" in DEFAULT_PAN_PORT


def test_state_line_is_parsed_and_emitted():
    received = []
    axis = GimbalAxis("PAN", "/dev/null", state_callback=lambda name, state: received.append((name, state)), autostart=False)

    axis._handle_line(
        "STATE,42,1234,AXIS1,HOLDING,10.500,11.000,0.500,-0.010,1,1,1,0,3"
    )

    state = axis.state
    assert state.connected
    assert state.sequence == 42
    assert state.reported_axis == "AXIS1"
    assert state.controller_state == "HOLDING"
    assert state.angle_degrees == 10.5
    assert state.target_degrees == 11.0
    assert state.encoder_ok
    assert state.driver_fault_ok
    assert state.magnet_weak
    assert not state.zero_persisted
    assert state.i2c_errors == 3
    assert received[-1] == ("PAN", state)
    assert axis._protocol_ready


def test_only_newest_pending_target_is_retained():
    axis = GimbalAxis("PAN", "/dev/null", autostart=False)
    axis.set_target(1.0)
    axis.set_target(2.0)

    assert axis._take_pending_target(time.monotonic()) == 2.0
    assert axis._take_pending_target(time.monotonic()) is None


def test_stale_pending_target_is_discarded():
    axis = GimbalAxis("PAN", "/dev/null", autostart=False)
    axis.set_target(5.0)

    assert axis._take_pending_target(time.monotonic() + 1.0) is None
