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

# Object detection
DNN_CONFIDENCE = 0.3
LEFT_BIAS_PIXELS = 80   # how aggressively to go left

ALLOWED_CLASSES = {7: "car", 15: "person"}

# -----------------------------
# Perspective Transform
# -----------------------------
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# -----------------------------
# Load DNN
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# -----------------------------
# Hardware Init
# -----------------------------
def is_raspberry_pi5():
    try:
        with open("/proc/cpuinfo") as f:
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
# Flask
# -----------------------------
app = Flask(__name__)
is_running = False

# -----------------------------
# Helpers
# -----------------------------
def find_tape_center(hist, start_x, end_x):
    best, score_max = None, 0
    for x in range(start_x, end_x - TAPE_WIDTH_PIXELS - TAPE_MARGIN):
        score = hist[x] + hist[x + TAPE_WIDTH_PIXELS]
        if score > score_max and score > CONFIDENCE_THRESHOLD:
            score_max = score
            best = x + TAPE_WIDTH_PIXELS // 2
    return best

def detect_object(frame):
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
        if conf < DNN_CONFIDENCE:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id not in ALLOWED_CLASSES:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        return True, box.astype(int), conf

    return False, None, None

# -----------------------------
# Core Logic
# -----------------------------
def process_and_drive(frame):
    h, w = frame.shape[:2]
    img_center = w // 2
    viz = frame.copy()

    # Object detection
    obj_detected, obj_box, obj_conf = detect_object(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    warped = cv2.warpPerspective(edges, M, (w, h))

    hist = np.sum(warped[int(h * 0.6):, :], axis=0)
    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w)

    if left_x is None and right_x is None:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        return viz

    if left_x and right_x:
        lane_center = (left_x + right_x) // 2
    elif left_x:
        lane_center = left_x + EXPECTED_LANE_WIDTH // 2
    else:
        lane_center = right_x - EXPECTED_LANE_WIDTH // 2

    # Apply left bias if object detected
    if obj_detected:
        lane_center -= LEFT_BIAS_PIXELS
        x1, y1, x2, y2 = obj_box
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            viz,
            "OBJECT AHEAD -> LEFT",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    error = lane_center - img_center
    steering = error * STEER_SENSITIVITY

    left_wheel = np.clip(BASE_SPEED + steering, -0.1, 0.5)
    right_wheel = np.clip(BASE_SPEED - steering, -0.1, 0.5)

    base.send_command({
        "T": 1,
        "L": round(left_wheel, 3),
        "R": round(right_wheel, 3)
    })

    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    cv2.circle(viz, (lane_center, h - 60), 10, (0, 255, 255), -1)

    return viz

# -----------------------------
# Streaming
# -----------------------------
def generate_frames():
    while True:
        frame = picam2.capture_array()
        output = process_and_drive(frame) if is_running else frame
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
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_robot")
def start_robot():
    global is_running
    is_running = True
    return "Started"

@app.route("/stop_robot")
def stop_robot():
    global is_running
    is_running = False
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    return "Stopped"

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
