from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
import os
from base_ctrl import BaseController

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 580, 440
BASE_SPEED = 0.12
STEER_SENSITIVITY = 0.0025

# Perspective Transform
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Lane parameters
EXPECTED_LANE_WIDTH = 340
TAPE_WIDTH_PIXELS = 14
TAPE_MARGIN = 6
CONFIDENCE_THRESHOLD = 500

last_error = 0

# -----------------------------
# Hardware Initialization
# -----------------------------
def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except:
        return False

serial_device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(serial_device, 115200)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

# -----------------------------
# Control Flags
# -----------------------------
is_running = False
turn_phase = None
turn_phase_end_time = 0

# -----------------------------
# Helper
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
# Main Processing
# -----------------------------
def process_and_drive(frame):
    global last_error
    global turn_phase, turn_phase_end_time

    h, w = frame.shape[:2]
    img_center = w // 2

    # =============================
    # LEFT-RIGHT MANEUVER MODE
    # =============================
    if turn_phase is not None:

        if turn_phase == "LEFT":
            if time.time() < turn_phase_end_time:
                base.send_command({"T": 1, "L": -0.15, "R": 0.45})
                return frame
            else:
                turn_phase = "RIGHT"
                turn_phase_end_time = time.time() + 0.6
                return frame

        elif turn_phase == "RIGHT":
            if time.time() < turn_phase_end_time:
                base.send_command({"T": 1, "L": 0.45, "R": -0.15})
                return frame
            else:
                turn_phase = None

    # =============================
    # Normal Lane Processing
    # =============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # Noise reduction
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    warped = cv2.warpPerspective(edges, M, (w, h))

    hist = np.sum(warped[int(h * 0.6):, :], axis=0)

    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w - 1)

    l_valid, r_valid = left_x is not None, right_x is not None

    if l_valid and r_valid:
        lane_center = (left_x + right_x) // 2
    elif l_valid:
        lane_center = left_x + (EXPECTED_LANE_WIDTH // 2)
    elif r_valid:
        lane_center = right_x - (EXPECTED_LANE_WIDTH // 2)
    else:
        steering_adj = last_error * STEER_SENSITIVITY
        left_wheel = np.clip(BASE_SPEED + steering_adj, -0.1, 0.5)
        right_wheel = np.clip(BASE_SPEED - steering_adj, -0.1, 0.5)
        base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    error = lane_center - img_center
    steering_adj = error * STEER_SENSITIVITY

    left_wheel = np.clip(BASE_SPEED + steering_adj, -0.1, 0.5)
    right_wheel = np.clip(BASE_SPEED - steering_adj, -0.1, 0.5)

    base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})

    last_error = error

    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    cv2.circle(viz, (lane_center, h - 80), 15, (255, 120, 0), -1)

    return viz

# -----------------------------
# Flask Endpoints
# -----------------------------
@app.route("/turn_left")
def turn_left():
    global turn_phase, turn_phase_end_time
    turn_phase = "LEFT"
    turn_phase_end_time = time.time() + 0.4
    return "Left-Right maneuver started"

@app.route("/start_robot")
def start_robot():
    global is_running
    is_running = True
    return "Robot movement started!"

@app.route("/stop_robot")
def stop_robot():
    global is_running
    is_running = False
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    return "Robot stopped!"

@app.route("/start")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# Frame Generator
# -----------------------------
def generate_frames():
    while True:
        frame = picam2.capture_array()

        if is_running:
            processed = process_and_drive(frame)
        else:
            processed = frame

        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

# -----------------------------
# Cleanup
# -----------------------------
def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    try:
        picam2.stop()
    except:
        pass
    os._exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
