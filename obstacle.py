from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
from base_ctrl import BaseController

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 580, 440

BASE_SPEED = 0.12
STEER_SENSITIVITY = 0.0025

EXPECTED_LANE_WIDTH = 340
TAPE_WIDTH_PIXELS = 14
TAPE_MARGIN = 6
CONFIDENCE_THRESHOLD = 500

OBJ_CONF_THRESH = 0.4
ALLOWED_CLASSES = {7: "car", 15: "person"}

# -----------------------------
# Perspective Transform
# -----------------------------
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# -----------------------------
# Hardware Init
# -----------------------------
def is_raspberry_pi5():
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "Raspberry Pi 5" in f.read()
    except:
        return False

serial_device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(serial_device, 115200)

# -----------------------------
# Camera Init
# -----------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
time.sleep(1)

# -----------------------------
# Object Detection Model
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
is_running = False

# -----------------------------
# Helpers
# -----------------------------
def find_tape_center(histogram, start_x, end_x):
    best_center = None
    max_score = 0
    for x in range(start_x, end_x - TAPE_WIDTH_PIXELS - TAPE_MARGIN):
        score = histogram[x] + histogram[x + TAPE_WIDTH_PIXELS]
        if score > max_score and score > CONFIDENCE_THRESHOLD:
            max_score = score
            best_center = x + (TAPE_WIDTH_PIXELS // 2)
    return best_center

def detect_obstacle(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < OBJ_CONF_THRESH:
            continue

        idx = int(detections[0, 0, i, 1])
        if idx not in ALLOWED_CLASSES:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        sx, sy, ex, ey = box.astype(int)

        cx = (sx + ex) // 2
        bottom = ey

        if (w * 0.35 < cx < w * 0.65) and (bottom > h * 0.6):
            return True, (sx, sy, ex, ey), ALLOWED_CLASSES[idx]

    return False, None, None

# -----------------------------
# Core Logic
# -----------------------------
def process_and_drive(frame):
    h, w = frame.shape[:2]
    img_center = w // 2

    obstacle, box, label = detect_obstacle(frame)

    if obstacle:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        viz = frame.copy()
        sx, sy, ex, ey = box
        cv2.rectangle(viz, (sx, sy), (ex, ey), (0, 0, 255), 3)
        cv2.putText(
            viz,
            f"STOP: {label}",
            (sx, sy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return viz

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    warped = cv2.warpPerspective(edges, M, (w, h))

    hist = np.sum(warped[int(h * 0.6):, :], axis=0)

    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w - 1)

    if left_x is None and right_x is None:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2
    elif left_x is not None:
        lane_center = left_x + EXPECTED_LANE_WIDTH // 2
    else:
        lane_center = right_x - EXPECTED_LANE_WIDTH // 2

    error = lane_center - img_center
    steering = error * STEER_SENSITIVITY

    left_wheel = np.clip(BASE_SPEED + steering, -0.1, 0.5)
    right_wheel = np.clip(BASE_SPEED - steering, -0.1, 0.5)

    base.send_command({
        "T": 1,
        "L": round(left_wheel, 3),
        "R": round(right_wheel, 3)
    })

    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    cv2.circle(viz, (lane_center, h - 60), 12, (255, 120, 0), -1)

    return viz

# -----------------------------
# Streaming
# -----------------------------
def generate_frames():
    while True:
        frame = picam2.capture_array()

        if is_running:
            output = process_and_drive(frame)
        else:
            output = frame

        ret, jpeg = cv2.imencode(".jpg", output)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

# -----------------------------
# Routes
# -----------------------------
@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/start_robot")
def start_robot():
    global is_running
    is_running = True
    return "Robot started"

@app.route("/stop_robot")
def stop_robot():
    global is_running
    is_running = False
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    return "Robot stopped"

# -----------------------------
# Cleanup
# -----------------------------
def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)

