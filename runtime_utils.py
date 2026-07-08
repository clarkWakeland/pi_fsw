import cv2
import numpy as np


ML_INPUT_SIZE = 640
ML_TARGET_HZ = 20.0
ML_INTERVAL_SECONDS = 1.0 / ML_TARGET_HZ
SERVO_RAMP_DURATION_SECONDS = 0.5
SERVO_RAMP_START_STEP = 0.75
SERVO_RAMP_FULL_STEP = 3.0
REACQUIRE_MAX_CENTER_DISTANCE_PX = 120.0
REACQUIRE_MIN_IOU = 0.05
REACQUIRE_MIN_SCORE = 0.3
TARGET_CONFIRMATION_FRAMES = 3
TARGET_CENTER_SMOOTHING_ALPHA = 0.35
SOFT_DEADBAND_PX = 15.0
SOFT_DEADBAND_FULL_RESPONSE_PX = 120.0
SERVO_STEP_ACCEL_LIMIT = 0.4
MANUAL_CONTROL_HZ = 30.0
MANUAL_CONTROL_INTERVAL_SECONDS = 1.0 / MANUAL_CONTROL_HZ
MANUAL_INPUT_TIMEOUT_SECONDS = 0.15
MANUAL_STEP_ACCEL_LIMIT = 0.3


def deadline_sleep_seconds(now, deadline):
    return max(0.0, deadline - now)


def ensure_inference_size(image, image_size=ML_INPUT_SIZE):
    if image is None:
        return None

    if image.shape[0] == image_size and image.shape[1] == image_size:
        return image

    return cv2.resize(image, (image_size, image_size))


def non_max_suppression_xyxy(boxes, scores, iou_threshold):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        widths = np.maximum(0.0, xx2 - xx1)
        heights = np.maximum(0.0, yy2 - yy1)
        intersection = widths * heights
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

        order = remaining[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def prepare_hailo_detections(results, image_size=ML_INPUT_SIZE, iou_threshold=0.7):
    detections = np.asarray(results, dtype=np.float32)

    if detections.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    if detections.ndim == 1:
        detections = detections.reshape(1, -1)

    if detections.shape[1] < 5:
        return np.empty((0, 5), dtype=np.float32)

    detections = detections[:, :5].copy()
    detections[:, :4] *= image_size

    keep = non_max_suppression_xyxy(detections[:, :4], detections[:, 4], iou_threshold)
    return detections[keep]


def box_center(box):
    box = np.asarray(box, dtype=np.float32)
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def smooth_point(previous, current, alpha=TARGET_CENTER_SMOOTHING_ALPHA):
    current = np.asarray(current, dtype=np.float32)
    if previous is None:
        return current.copy()

    previous = np.asarray(previous, dtype=np.float32)
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha * current + (1.0 - alpha) * previous


def apply_soft_deadband(error, deadband=SOFT_DEADBAND_PX, full_response=SOFT_DEADBAND_FULL_RESPONSE_PX):
    error = float(error)
    magnitude = abs(error)
    deadband = abs(float(deadband))
    full_response = max(deadband, abs(float(full_response)))

    if magnitude <= deadband:
        return 0.0

    if magnitude >= full_response:
        return error

    response = (magnitude - deadband) * full_response / max(1e-6, full_response - deadband)
    return float(np.copysign(response, error))


def limit_step_acceleration(requested_step, previous_step=0.0, max_step_change=SERVO_STEP_ACCEL_LIMIT):
    requested_step = float(requested_step)
    previous_step = float(previous_step)
    max_step_change = abs(float(max_step_change))
    delta = requested_step - previous_step
    delta = max(-max_step_change, min(max_step_change, delta))
    return previous_step + delta


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def find_reacquire_track(
    last_box,
    tracks,
    max_center_distance=REACQUIRE_MAX_CENTER_DISTANCE_PX,
    min_iou=REACQUIRE_MIN_IOU,
    min_score=REACQUIRE_MIN_SCORE,
):
    if last_box is None:
        return None

    last_center = box_center(last_box)
    best_track = None
    best_score = None

    for track in tracks:
        if getattr(track, "score", 0.0) < min_score:
            continue

        candidate_box = np.asarray(track.tlbr, dtype=np.float32)
        iou = box_iou(last_box, candidate_box)
        distance = float(np.linalg.norm(box_center(candidate_box) - last_center))

        if iou < min_iou and distance > max_center_distance:
            continue

        # Higher IoU is better; for non-overlapping nearby tracks, shorter distance wins.
        candidate_score = (iou, -distance, getattr(track, "score", 0.0))
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_track = track

    return best_track


def servo_ramp_step_limit(
    now,
    started_at,
    ramp_duration=SERVO_RAMP_DURATION_SECONDS,
    start_step=SERVO_RAMP_START_STEP,
    full_step=SERVO_RAMP_FULL_STEP,
):
    if started_at is None:
        return full_step

    if ramp_duration <= 0:
        return full_step

    progress = max(0.0, min(1.0, (now - started_at) / ramp_duration))
    return start_step + (full_step - start_step) * progress


def tracking_motion_confirmed(confirm_frames, required_frames=TARGET_CONFIRMATION_FRAMES):
    return confirm_frames >= required_frames
