from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import signal
import sys
from base_ctrl import BaseController

# -----------------------------
# 1. Configuration & Tuning
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# GAINS: Tune these if the robot is too slow or too shaky
BASE_SPEED = 0.16   # Forward speed on straights
KP = 0.0022         # Steering aggressiveness (Increased for sharper turns)
MAX_STEER = 0.25    # Cap on steering effort to prevent spin-outs

# -----------------------------
# 2. Initialization
# -----------------------------
def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except: return False

# Setup Chassis
device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(device, 115200)

# Setup Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

# Perspective Matrix (Adjust SRC if your "Bird's Eye" looks skewed)
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

app = Flask(__name__)

# -----------------------------
# 3. Processing Pipeline
# -----------------------------
def process_frame(frame):
    h, w = frame.shape[:2]

    # --- Pre-processing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    
    # --- Warp to Bird's Eye View ---
    warped = cv2.warpPerspective(thresh, M, (w, h))

    # --- Detection ROI (Look Ahead) ---
    # We look from 40% height to 80% height to catch turns early
    roi = warped[int(h*0.4):int(h*0.8), :]
    hist = np.sum(roi, axis=0)
    
    if np.max(hist) < 50:
        # SAFETY: If no lanes are detected, stop moving
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        cv2.putText(frame, "STATUS: LANE LOST", (50, 50), 2, 0.8, (0,0,255), 2)
        return frame

    # --- Find Lane Center ---
    midpoint = int(hist.shape[0]/2)
    left_x = np.argmax(hist[:midpoint])
    right_x = np.argmax(hist[midpoint:]) + midpoint
    lane_center = (left_x + right_x) // 2

    # --- Calculate Control Signals ---
    deviation = lane_center - (w // 2)
    steering_adj = deviation * KP
    
    # Clip steering
    steering_adj = max(-MAX_STEER, min(MAX_STEER, steering_adj))

    # DYNAMIC SPEED: Automatically slow down as the turn gets sharper
    # This prevents the robot from "drifting" off the track
    speed_penalty = abs(steering_adj) * 0.4 
    current_speed = max(0.05, BASE_SPEED - speed_penalty)

    # Convert to Differential Drive (Left/Right)
    left_wheel = current_speed + steering_adj
    right_wheel = current_speed - steering_adj

    # Send command
    base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})

    # --- Visual Overlays for Debugging ---
    # Draw Picture-in-Picture Warped View
    warped_viz = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), (160, 120))
    frame[10:130, w-170:w-10] = warped_viz
    
    # Draw steering indicators
    cv2.circle(frame, (lane_center, h-30), 10, (0, 255, 255), -1)
    cv2.putText(frame, f"Speed: {current_speed:.2f} | Steer: {steering_adj:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

# -----------------------------
# 4. Flask & Execution
# -----------------------------
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
    print("\nShutting down safely...")
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    print("UGV Ready. Video stream at http://<pi_ip>:5001/video")
    app.run(host="0.0.0.0", port=5001, threaded=True)
