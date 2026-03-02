from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
from base_ctrl import BaseController

# -----------------------------
# Configuration & Calibration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 580, 440
BASE_SPEED = 0.12
STEER_SENSITIVITY = 0.0025

SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

EXPECTED_LANE_WIDTH = 340
TAPE_WIDTH_PIXELS = 14
TAPE_MARGIN = 6
CONFIDENCE_THRESHOLD = 500

last_error = 0
integral = 0

# -----------------------------
# Object Detection Config
# -----------------------------
DNN_CONFIDENCE = 0.3
ALLOWED_CLASSES = {7: "car"}

STOP_TIME = 0.5
TURN_TIME = 0.8
TURN_LEFT_L = -0.1
TURN_LEFT_R = 0.25

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

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
time.sleep(1)

app = Flask(__name__)
is_running = False

# -----------------------------
# ORIGINAL LANE HELPERS (UNCHANGED)
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

# -----------------------------
# Object Detection
# -----------------------------
def detect_car(frame):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < DNN_CONFIDENCE:
            continue
        class_id = int(detections[0, 0, i, 1])
        if class_id in ALLOWED_CLASSES:
            return True
    return False

# -----------------------------
# CORE LOGIC
# -----------------------------
def process_and_drive(frame):
    global last_error, integral
    h, w = frame.shape[:2]
    img_center = w // 2

    # -------- Added behavior ONLY --------
    if detect_car(frame):
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        time.sleep(STOP_TIME)

        base.send_command({"T": 1, "L": TURN_LEFT_L, "R": TURN_LEFT_R})
        time.sleep(TURN_TIME)

        return frame
    # ------------------------------------

    # -------- ORIGINAL LANE CODE BELOW (UNCHANGED) --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    warped = cv2.warpPerspective(edges, M, (w, h))
    hist = np.sum(warped[int(h * 0.6):, :], axis=0)

    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w - 1)

    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2
    elif left_x is not None:
        lane_center = left_x + (EXPECTED_LANE_WIDTH // 2)
    elif right_x is not None:
        lane_center = right_x - (EXPECTED_LANE_WIDTH // 2)
    else:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    error = lane_center - img_center
    integral += error
    steering_adj = error * STEER_SENSITIVITY

    left_wheel = np.clip(BASE_SPEED + steering_adj, -0.1, 0.5)
    right_wheel = np.clip(BASE_SPEED - steering_adj, -0.1, 0.5)

    base.send_command({
        "T": 1,
        "L": round(left_wheel, 3),
        "R": round(right_wheel, 3)
    })

    last_error = error

    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    cv2.circle(viz, (lane_center, h - 80), 15, (255, 120, 0), -1)

    return viz
    # ----------------------------------------------------

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

def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
