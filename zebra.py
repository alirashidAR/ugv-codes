from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
from base_ctrl import BaseController

# --- Config ---
FRAME_WIDTH, FRAME_HEIGHT = 580, 440
BASE_SPEED = 0.12          
STEER_SENSITIVITY = 0.0025 

# Perspective Transform
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Zebra Detection - Look at the very bottom of the image
ZEBRA_THRESHOLD = 3000   # Total white pixel density to trigger
HARD_TURN_POWER = 0.45   # Sharp turn speed
PIVOT_REVERSE = -0.15    # Reverse inner wheel for pivot

# Global Variables
is_running = False
last_error = 0

# --- Hardware Init ---
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

# --- Logic Functions ---

def find_tape_center(histogram, start_x, end_x):
    best_center = None
    max_score = 0
    width = 14
    for x in range(start_x, end_x - width - 6):
        score = histogram[x] + histogram[x + width]
        if score > 500:
            if score > max_score:
                max_score = score
                best_center = x + (width // 2)
    return best_center

def process_and_drive(frame):
    global last_error, is_running
    h, w = frame.shape[:2]
    img_center = w // 2

    # 1. Image Processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    warped = cv2.warpPerspective(edges, M, (w, h))

    # 2. Zebra Detection - ONLY look at the bottom 15% of the warped image
    # This ensures the robot is "On" the crossing
    trigger_zone = warped[int(h*0.85):, :] 
    trigger_hist = np.sum(trigger_zone, axis=0)
    
    # Calculate mass in the trigger zone
    middle_mass = np.sum(trigger_hist[int(w*0.3):int(w*0.7)])
    left_mass = np.sum(trigger_hist[:int(w*0.3)])
    right_mass = np.sum(trigger_hist[int(w*0.7):])
    
    is_on_zebra = middle_mass > ZEBRA_THRESHOLD

    # 3. Decision Making
    status = "SCANNING"
    l_cmd, r_cmd = 0.0, 0.0

    if is_on_zebra:
        status = "ON ZEBRA - TURNING"
        # Determine which way to turn based on where the zebra is densest
        if left_mass > right_mass:
            l_cmd, r_cmd = PIVOT_REVERSE, HARD_TURN_POWER # Hard Left
        else:
            l_cmd, r_cmd = HARD_TURN_POWER, PIVOT_REVERSE # Hard Right
    else:
        # Standard Lane Following (looks slightly higher up for anticipation)
        follow_hist = np.sum(warped[int(h*0.5):int(h*0.8)], axis=0)
        left_x = find_tape_center(follow_hist, 0, int(w * 0.35))
        right_x = find_tape_center(follow_hist, int(w * 0.65), w - 1)
        
        if left_x and right_x:
            lane_center = (left_x + right_x) // 2
            status = "LANE"
        elif left_x:
            lane_center = left_x + 170
            status = "LEFT ONLY"
        elif right_x:
            lane_center = right_x - 170
            status = "RIGHT ONLY"
        else:
            lane_center = img_center
            status = "SEARCHING"

        error = lane_center - img_center
        steering_adj = error * STEER_SENSITIVITY
        l_cmd = np.clip(BASE_SPEED + steering_adj, -0.1, 0.5)
        r_cmd = np.clip(BASE_SPEED - steering_adj, -0.1, 0.5)

    # 4. Motor Execution
    if is_running:
        base.send_command({"T": 1, "L": round(l_cmd, 3), "R": round(r_cmd, 3)})
    else:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})

    # 5. Viz with Trigger Box
    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    # Draw the trigger zone box so you can see where it "feels" the zebra
    cv2.rectangle(viz, (0, int(h*0.85)), (w, h), (0, 0, 255), 2)
    cv2.putText(viz, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return viz

# --- Flask & Server ---

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_and_drive(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/start")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_robot")
def start_robot():
    global is_running
    is_running = True
    return "Robot Started"

@app.route("/stop_robot")
def stop_robot():
    global is_running
    is_running = False
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    return "Robot Stopped"

def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
