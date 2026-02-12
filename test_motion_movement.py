from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import signal
import sys
from base_ctrl import BaseController

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.3
BASE_SPEED = 0.15  # Default speed for both wheels
KP = 0.0015        # Steering sensitivity (adjust this if turning is too weak/strong)

# -----------------------------
# Initialization
# -----------------------------
def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except: return False

device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(device, 115200)

# Load Object Detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Lane Perspective Setup
SRC_POINTS = np.float32([[100, 480], [540, 480], [220, 320], [420, 320]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

# -----------------------------
# Logic
# -----------------------------
def process_frame(frame):
    h, w = frame.shape[:2]

    # 1. Lane Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    warped = cv2.warpPerspective(thresh, M, (w, h))
    
    hist = np.sum(warped[h//2:, :], axis=0)
    midpoint = int(hist.shape[0]/2)
    left_x = np.argmax(hist[:midpoint])
    right_x = np.argmax(hist[midpoint:]) + midpoint
    lane_center = (left_x + right_x) // 2
    
    # 2. Steering Calculation (Differential Drive)
    deviation = lane_center - (w // 2)
    steering_adj = deviation * KP

    # Adjust L and R wheel speeds
    # If lane_center is to the right (positive deviation), slow down R or speed up L to turn right
    left_wheel = BASE_SPEED + steering_adj
    right_wheel = BASE_SPEED - steering_adj

    # Constrain speeds to safe limits
    left_wheel = max(0.0, min(0.4, left_wheel))
    right_wheel = max(0.0, min(0.4, right_wheel))

    # Send command to chassis
    base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})

    # UI Overlays
    cv2.circle(frame, (lane_center, h-30), 10, (0, 0, 255), -1)
    cv2.putText(frame, f"L:{left_wheel:.2f} R:{right_wheel:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_frame(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(sig, frame):
    print("\nStopping...")
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
