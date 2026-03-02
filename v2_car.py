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

# Obstacle detection (contour based)
OBSTACLE_MIN_AREA = 200
OBSTACLE_MAX_CONTOURS = 1
OBSTACLE_SOLIDITY_MIN = 0.3

# Object detection
DNN_CONFIDENCE = 0.3
ALLOWED_CLASSES = {7: "car"}

# -----------------------------
# Perspective Transform
# -----------------------------
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# -----------------------------
# Load DNN model
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)
print("DNN model loaded")

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
# Flask App
# -----------------------------
app = Flask(__name__)
is_running = False

# -----------------------------
# Lane Helpers
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
# Car Detection
# -----------------------------
def detect_car(frame):
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
        confidence = detections[0, 0, i, 2]
        if confidence < DNN_CONFIDENCE:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id not in ALLOWED_CLASSES:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        return True, box.astype(int), confidence

    return False, None, None

# -----------------------------
# Contour Obstacle Detection
# -----------------------------
def detect_solid_obstacle(edges_frame):
    h, w = edges_frame.shape

    y1 = int(h * 0.50)
    y2 = int(h * 0.85)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)

    roi = edges_frame[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < OBSTACLE_MIN_AREA:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity >= OBSTACLE_SOLIDITY_MIN:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            valid.append((bx + x1, by + y1, bw, bh, area, solidity))

    detected = len(valid) > 0 and len(valid) <= OBSTACLE_MAX_CONTOURS
    return detected, (x1, y1, x2, y2), valid

# -----------------------------
# Core Logic
# -----------------------------
def process_and_drive(frame):
    h, w = frame.shape[:2]
    img_center = w // 2
    viz = frame.copy()

    # 1. Car detection (highest priority)
    car_detected, car_box, car_conf = detect_car(frame)
    if car_detected:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        x1, y1, x2, y2 = car_box
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(viz, "CAR DETECTED", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return viz

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. Solid obstacle detection
    obstacle, roi_box, obstacles = detect_solid_obstacle(edges)
    if obstacle:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        for (ox, oy, ow, oh, _, _) in obstacles:
            cv2.rectangle(viz, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 3)
        return viz

    # 3. Lane following
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

    error = lane_center - img_center
    steering = error * STEER_SENSITIVITY

    left = np.clip(BASE_SPEED + steering, -0.1, 0.5)
    right = np.clip(BASE_SPEED - steering, -0.1, 0.5)

    base.send_command({"T": 1, "L": round(left, 3), "R": round(right, 3)})
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)

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
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

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
