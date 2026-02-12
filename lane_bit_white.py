from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
from base_ctrl import BaseController

# -----------------------------
# Configuration & Calibration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
BASE_SPEED = 0.12          
STEER_SENSITIVITY = 0.0028 

# Perspective Transform
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# --- NEW ROBUST PARAMETERS ---
EXPECTED_LANE_WIDTH = 340 # The fixed pixel distance between your lanes
MIN_AREA = 500            # Minimum size of a white object to be considered a "line"
MAX_AREA = 5000           # Ignore objects that are too "fat" (big white patches)

# -----------------------------
# Hardware Initialization
# -----------------------------
def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except: return False

serial_device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(serial_device, 115200)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

# -----------------------------
# Robust Detection Logic
# -----------------------------

def process_and_drive(frame):
    h, w = frame.shape[:2]
    img_center = w // 2

    # 1. CLEAN THE IMAGE (Ignore big patches)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    
    # Use Connected Components to filter by size (Removes non-line patches)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    clean_mask = np.zeros_like(thresh)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Only keep objects that look like thin lanes (Not too small, not too big)
        if MIN_AREA < area < MAX_AREA:
            clean_mask[labels == i] = 255

    # 2. WARP & HISTOGRAM
    warped = cv2.warpPerspective(clean_mask, M, (w, h))
    # Look at the top 60% of the image to see curves early
    hist = np.sum(warped[int(h*0.4):, :], axis=0)

    # 3. FIND PEAKS
    mid = w // 2
    left_x = np.argmax(hist[:mid]) if np.max(hist[:mid]) > 0 else None
    right_x = (np.argmax(hist[mid:]) + mid) if np.max(hist[mid:]) > 0 else None

    # 4. GEOMETRIC CONSTANCY (Maintain distance on curves)
    l_col, r_col = (0, 255, 0), (0, 0, 255) # Green, Red

    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2
    elif left_x is not None:
        # Sharp Right Curve: Predicted Right based on Fixed Distance
        right_x = left_x + EXPECTED_LANE_WIDTH
        lane_center = (left_x + right_x) // 2
        r_col = (0, 255, 255) # Yellow (Ghost Lane)
    elif right_x is not None:
        # Sharp Left Curve: Predicted Left based on Fixed Distance
        left_x = right_x - EXPECTED_LANE_WIDTH
        lane_center = (left_x + right_x) // 2
        l_col = (0, 255, 255) # Yellow (Ghost Lane)
    else:
        # Totally lost: stop
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # 5. CONTROL MATH (Differential Drive)
    error = lane_center - img_center
    steering_adj = error * STEER_SENSITIVITY
    
    # Pivot Turning Logic
    # If Error is Negative (Left), slow Left wheel, speed up Right wheel
    left_wheel = np.clip(BASE_SPEED + steering_adj, -0.1, 0.4)
    right_wheel = np.clip(BASE_SPEED - steering_adj, -0.1, 0.4)
    
    base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})

    # 6. VISUALIZATION
    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.circle(viz, (int(left_x), h - 40), 20, l_col, -1)
    cv2.circle(viz, (int(right_x), h - 40), 20, r_col, -1)
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    
    return viz

# --- Flask Streaming Logic ---
def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_and_drive(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)