#!/usr/bin/env python3
"""Run a Hailo HEF on a video and write an annotated video.

This utility is intentionally headless so it can run over SSH on a camera unit.
By default it swaps OpenCV's BGR input to RGB before Hailo inference, matching
the previously successful Pi test path. Drawing remains on the untouched BGR
frame so the saved video retains its true colors. Hailo boxes are treated as
y1,x1,y2,x2 by default and converted to x1,y1,x2,y2 before NMS and drawing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = Path("/opt/fsw/hailo_models/dashing_dolphin_masters.hef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hailo inference on an MP4 and save visible bounding boxes."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"HEF model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.30,
        help="Minimum detection confidence (default: 0.30)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.70,
        help="Class-agnostic NMS IoU threshold (default: 0.70)",
    )
    parser.add_argument(
        "--label",
        default="swimmer",
        help="Label drawn above each box (default: swimmer)",
    )
    parser.add_argument(
        "--input-color",
        choices=("rgb", "bgr"),
        default="rgb",
        help=(
            "Channel order fed to Hailo; rgb swaps OpenCV's BGR channels "
            "while keeping output colors unchanged (default: rgb)"
        ),
    )
    parser.add_argument(
        "--box-order",
        choices=("yxyx", "xyxy"),
        default="yxyx",
        help=(
            "Coordinate order returned by the HEF; yxyx swaps x/y before "
            "drawing (default: yxyx)"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 processes the entire video",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N frames; 0 disables progress output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    return args


def non_max_suppression_xyxy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Return indices kept by class-agnostic NMS for pixel-space xyxy boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        intersection_x1 = np.maximum(x1[current], x1[remaining])
        intersection_y1 = np.maximum(y1[current], y1[remaining])
        intersection_x2 = np.minimum(x2[current], x2[remaining])
        intersection_y2 = np.minimum(y2[current], y2[remaining])
        intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
        intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def prepare_detections(
    raw_results: object,
    input_width: int,
    input_height: int,
    confidence: float,
    iou_threshold: float,
    box_order: str = "yxyx",
) -> np.ndarray:
    """Convert normalized HEF results to filtered input-space xyxy pixels."""
    detections = np.asarray(raw_results, dtype=np.float32)
    if detections.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if detections.ndim == 1:
        detections = detections.reshape(1, -1)
    if detections.ndim != 2 or detections.shape[1] < 5:
        raise ValueError(f"Unexpected Hailo output shape: {detections.shape}")

    detections = detections[:, :5].copy()
    detections = detections[detections[:, 4] >= confidence]
    if not len(detections):
        return np.empty((0, 5), dtype=np.float32)

    if box_order == "yxyx":
        detections[:, :4] = detections[:, [1, 0, 3, 2]]
    elif box_order != "xyxy":
        raise ValueError(f"Unexpected box order: {box_order}")

    detections[:, [0, 2]] *= input_width
    detections[:, [1, 3]] *= input_height
    keep = non_max_suppression_xyxy(
        detections[:, :4], detections[:, 4], iou_threshold
    )
    return detections[keep]


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    input_width: int,
    input_height: int,
    label: str,
) -> None:
    """Draw input-space detections on a source-resolution BGR frame."""
    frame_height, frame_width = frame.shape[:2]
    scale_x = frame_width / input_width
    scale_y = frame_height / input_height

    for x1, y1, x2, y2, score in detections:
        left = int(np.clip(round(x1 * scale_x), 0, frame_width - 1))
        top = int(np.clip(round(y1 * scale_y), 0, frame_height - 1))
        right = int(np.clip(round(x2 * scale_x), 0, frame_width - 1))
        bottom = int(np.clip(round(y2 * scale_y), 0, frame_height - 1))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        text = f"{label} {score:.2f}"
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_bottom = max(top, text_size[1] + baseline + 2)
        cv2.rectangle(
            frame,
            (left, text_bottom - text_size[1] - baseline - 4),
            (min(frame_width - 1, left + text_size[0] + 6), text_bottom),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (left + 3, text_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def prepare_inference_frame(
    frame: np.ndarray, input_width: int, input_height: int, input_color: str
) -> np.ndarray:
    """Resize a BGR video frame and optionally swap it to RGB for Hailo."""
    inference_frame = cv2.resize(
        frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    if input_color == "rgb":
        return cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
    return inference_frame


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"HEF model not found: {model_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path} (pass --overwrite to replace it)"
        )
    if output_path == input_path:
        raise ValueError("Input and output paths must be different")

    try:
        from picamera2.devices import Hailo
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2's Hailo API is unavailable; run this on the camera unit"
        ) from exc

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError("Input video reports an invalid frame size")
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (source_width, source_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    hailo = None
    frame_count = 0
    detection_count = 0
    inference_seconds = 0.0
    started_at = time.monotonic()

    try:
        hailo = Hailo(str(model_path))
        input_shape = tuple(hailo.get_input_shape())
        if len(input_shape) < 2:
            raise RuntimeError(f"Unexpected HEF input shape: {input_shape}")
        input_height, input_width = map(int, input_shape[:2])
        print(
            f"Model input: {input_width}x{input_height}; "
            f"video: {source_width}x{source_height} at {source_fps:.2f} FPS; "
            f"frames: {total_frames or 'unknown'}; "
            f"Hailo input color: {args.input_color.upper()}; "
            f"box order: {args.box_order}"
        )

        while args.max_frames == 0 or frame_count < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            inference_frame = prepare_inference_frame(
                frame, input_width, input_height, args.input_color
            )
            inference_started = time.monotonic()
            outputs = hailo.run(inference_frame)
            inference_seconds += time.monotonic() - inference_started
            if not outputs:
                raise RuntimeError("Hailo returned no output tensors")

            detections = prepare_detections(
                outputs[0],
                input_width,
                input_height,
                args.confidence,
                args.iou_threshold,
                args.box_order,
            )
            draw_detections(
                frame, detections, input_width, input_height, args.label
            )
            writer.write(frame)
            frame_count += 1
            detection_count += len(detections)

            if args.progress_every and frame_count % args.progress_every == 0:
                average_fps = frame_count / max(time.monotonic() - started_at, 1e-9)
                print(
                    f"Processed {frame_count}/{total_frames or '?'} frames "
                    f"({average_fps:.2f} end-to-end FPS)"
                )

    finally:
        capture.release()
        writer.release()
        if hailo is not None:
            hailo.close()

    elapsed = time.monotonic() - started_at
    inference_fps = frame_count / inference_seconds if inference_seconds else 0.0
    end_to_end_fps = frame_count / elapsed if elapsed else 0.0
    print(f"Wrote: {output_path}")
    print(
        f"Frames: {frame_count}; detections: {detection_count}; "
        f"Hailo inference: {inference_fps:.2f} FPS; "
        f"end-to-end: {end_to_end_fps:.2f} FPS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
