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

# Obstacle detection - look for solid objects, not stripe patterns
OBSTACLE_MIN_AREA = 200           # Minimum contour area for a single obstacle
OBSTACLE_MAX_CONTOURS = 1         # Max number of contours (zebra = many, obstacle = few)
OBSTACLE_SOLIDITY_MIN = 0.3       # Minimum solidity (filled ratio)
FORWARD_ROI_Y_START = 0.45
FORWARD_ROI_Y_END = 0.70

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
# 3D Obstacle Detection (ignores zebra crossings)
# -----------------------------
def detect_solid_obstacle(edges_frame):
    """
    Detects solid 3D obstacles in the FRONTAL view (not bird's eye).
    Uses the original frame edges before perspective transform.
    Zebra crossings create MANY small stripe contours.
    Real obstacles create FEW large solid blobs.
    """
    h, w = edges_frame.shape

    # Define the forward-looking ROI in FRONTAL view
    # Focus on the center-forward area where obstacles would appear
    y1 = int(h * 0.50)  # Start from middle of frame
    y2 = int(h * 0.85)  # Go down to near bottom
    
    # Center region (where obstacle would be between lanes)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    
    roi = edges_frame[y1:y2, x1:x2]

    # Find contours in the center region
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter significant contours
    significant_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < OBSTACLE_MIN_AREA:
            continue
        
        # Calculate solidity (how "filled" the contour is)
        # Zebra stripes are thin lines (low solidity)
        # Solid objects are filled blobs (high solidity)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
            
        solidity = area / hull_area
        
        # Only keep solid objects
        if solidity >= OBSTACLE_SOLIDITY_MIN:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            significant_contours.append((x + x1, y + y1, w_box, h_box, area, solidity))

    # Real obstacles = few large solid blobs
    # Zebra crossings = many thin stripe contours
    obstacle_detected = (len(significant_contours) > 0 and 
                        len(significant_contours) <= OBSTACLE_MAX_CONTOURS)

    return obstacle_detected, (x1, y1, x2, y2), significant_contours

# -----------------------------
# Core Logic
# -----------------------------
def process_and_drive(frame):
    h, w = frame.shape[:2]
    img_center = w // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Check for solid obstacles in FRONTAL view (before warping)
    obstacle_detected, roi_box, obstacles = detect_solid_obstacle(edges)

    # Create frontal view visualization
    viz_frontal = frame.copy()
    x1_roi, y1_roi, x2_roi, y2_roi = roi_box

    if obstacle_detected:
        # STOP when solid obstacle detected
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})

        # Visualize obstacles on FRONTAL view
        cv2.rectangle(viz_frontal, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)
        
        for (ox, oy, ow, oh, area, solidity) in obstacles:
            cv2.rectangle(viz_frontal, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 3)
            cv2.putText(
                viz_frontal,
                f"A:{int(area)} S:{solidity:.2f}",
                (ox, oy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        cv2.putText(
            viz_frontal,
            "OBSTACLE DETECTED",
            (x1_roi + 10, y1_roi - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        return viz_frontal

    # Warp for lane detection
    warped = cv2.warpPerspective(edges, M, (w, h))
    
    # Normal lane following (continues through zebra crossings)
    hist = np.sum(warped[int(h * 0.6):, :], axis=0)

    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w - 1)

    if left_x is None and right_x is None:
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        # Draw ROI on frontal view
        cv2.rectangle(viz_frontal, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 1)
        return viz_frontal

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

    # Visualization on FRONTAL view
    # Draw ROI (green when clear)
    cv2.rectangle(viz_frontal, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
    cv2.putText(
        viz_frontal,
        "CLEAR",
        (x1_roi + 5, y1_roi - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
    
    # Show lane center indicator
    cv2.line(viz_frontal, (img_center, 0), (img_center, h), (255, 255, 0), 2)

    return viz_frontal

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

