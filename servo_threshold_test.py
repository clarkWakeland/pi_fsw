import argparse
import time

import pantilthat

from servo_control import MotorControl


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def prompt_yes_no_quit(message):
    while True:
        reply = input(message).strip().lower()
        if reply in {"y", "yes"}:
            return "yes"
        if reply in {"n", "no", ""}:
            return "no"
        if reply in {"q", "quit"}:
            return "quit"
        print("Please answer y, n, or q.")


def build_magnitude_sequence(start, end, step):
    values = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def apply_test_pulses(mc, axis, magnitude, pulses, pulse_interval_s, direction_sign):
    x_input = 0.0
    y_input = 0.0
    value = direction_sign * magnitude

    for _ in range(pulses):
        if axis == "x":
            x_input = value
            y_input = 0.0
        else:
            x_input = 0.0
            y_input = value
        mc.set_manual_input(x_input, y_input)
        time.sleep(pulse_interval_s)


def center_servos(pan_center, tilt_center):
    pantilthat.pan(pan_center)
    pantilthat.tilt(tilt_center)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially test manual-control input magnitudes to find the lowest "
            "input where the servo physically responds."
        )
    )
    parser.add_argument("--axis", choices=["x", "y"], default="x", help="Servo axis to test: x=pan, y=tilt")
    parser.add_argument("--start", type=float, default=0.08, help="Starting input magnitude")
    parser.add_argument("--end", type=float, default=0.5, help="Maximum input magnitude to test")
    parser.add_argument("--step", type=float, default=0.01, help="Input increment between tests")
    parser.add_argument("--pulses", type=int, default=20, help="Number of repeated set_manual_input calls per magnitude")
    parser.add_argument("--pulse-interval", type=float, default=0.033, help="Delay between pulses in seconds")
    parser.add_argument("--settle", type=float, default=0.4, help="Settling delay after each magnitude test")
    parser.add_argument("--direction", choices=["pos", "neg"], default="neg", help="Direction to test")
    parser.add_argument("--pan-center", type=float, default=0.0, help="Pan angle to return to between tests")
    parser.add_argument("--tilt-center", type=float, default=20.0, help="Tilt angle to return to between tests")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be > 0")

    mc = MotorControl()
    direction_sign = -1.0 if args.direction == "neg" else 1.0

    start = clamp(args.start, 0.0, 1.0)
    end = clamp(args.end, 0.0, 1.0)
    if end < start:
        raise ValueError("--end must be >= --start")

    magnitudes = build_magnitude_sequence(start, end, args.step)
    print("=== Servo Manual-Input Threshold Test ===")
    print(f"Axis: {args.axis} | Direction: {args.direction} | Magnitude range: {start:.3f}..{end:.3f} step {args.step:.3f}")
    print(f"Pulses/magnitude: {args.pulses} at {args.pulse_interval:.3f}s interval")
    print("Respond after each test: y=yes movement, n=no movement, q=quit")

    detected = None

    try:
        center_servos(args.pan_center, args.tilt_center)
        time.sleep(0.5)

        for magnitude in magnitudes:
            mapped_step = mc._manual_axis_to_step(direction_sign * magnitude)
            print(f"\nTesting magnitude={magnitude:.3f}, mapped_step={mapped_step:.4f}")

            apply_test_pulses(
                mc,
                axis=args.axis,
                magnitude=magnitude,
                pulses=args.pulses,
                pulse_interval_s=args.pulse_interval,
                direction_sign=direction_sign,
            )
            time.sleep(args.settle)

            reply = prompt_yes_no_quit("Did the servo physically move? [y/N/q]: ")
            if reply == "yes":
                detected = magnitude
                break
            if reply == "quit":
                break

            center_servos(args.pan_center, args.tilt_center)
            time.sleep(0.35)

    finally:
        center_servos(args.pan_center, args.tilt_center)

    if detected is None:
        print("\nNo movement threshold confirmed in tested range.")
        return

    mapped = mc._manual_axis_to_step(direction_sign * detected)
    print("\nDetected lowest responsive magnitude:")
    print(f"  input magnitude: {detected:.3f}")
    print(f"  mapped step:     {mapped:.4f}")


if __name__ == "__main__":
    main()
