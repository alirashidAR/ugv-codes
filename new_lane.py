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
Kp_steer = 0.005           # Proportional steering gain

# Perspective Transform (Keep your calibration)
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Sliding Window Config
N_WINDOWS = 9              # Number of stacking windows
MARGIN = 60                # Width of the windows +/- margin
MINPIX = 40                # Minimum pixels found to recenter window

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
is_running = False  # Global flag for start/stop

# -----------------------------
# Robust Logic: Sliding Windows
# -----------------------------

def find_lane_windows(binary_warped):
    """
    Scans the image using sliding windows to track curved lines.
    Returns: left_lane_inds, right_lane_inds, visualization_image
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set height of windows
    window_height = int(binary_warped.shape[0]//N_WINDOWS)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(N_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > MINPIX pixels, recenter next window on their mean position
        if len(good_left_inds) > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return leftx, lefty, rightx, righty, out_img

def process_and_drive(frame):
    h, w = frame.shape[:2]
    
    # 1. Edge Detection (Combined Canny + Threshold usually best)
    # Using simple thresholding for speed on RPi, tweak strictly for your tape color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Using Canny as per your request, but adding dilation to make lines thicker
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Warping
    warped = cv2.warpPerspective(edges, M, (w, h))

    # 2. Sliding Window Search
    leftx, lefty, rightx, righty, viz_img = find_lane_windows(warped)

    # 3. Calculate Lane Center
    # We prioritize the "Look Ahead" point (middle of the screen) for steering
    # rather than the bottom of the screen. This helps with cornering.
    
    center_img = w // 2
    lane_center_points = []
    
    if len(leftx) > 100 and len(rightx) > 100:
        # Both lanes found: Average the means
        l_mean = np.mean(leftx)
        r_mean = np.mean(rightx)
        lane_target = (l_mean + r_mean) // 2
        color = (0, 255, 0) # Green: Good lock
        
    elif len(leftx) > 100:
        # Only Left found
        l_mean = np.mean(leftx)
        lane_target = l_mean + (340 // 2) # Use your expected width
        color = (0, 255, 255) # Yellow: Left only
        
    elif len(rightx) > 100:
        # Only Right found
        r_mean = np.mean(rightx)
        lane_target = r_mean - (340 // 2)
        color = (0, 0, 255) # Red: Right only
        
    else:
        # LOST
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        cv2.putText(viz_img, "LOST LANE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return viz_img

    # 4. Steering Control
    error = lane_target - center_img
    
    # Visualization
    cv2.circle(viz_img, (int(lane_target), h//2), 15, color, -1)
    cv2.line(viz_img, (center_img, 0), (center_img, h), (255, 255, 255), 2)

    # Simple Proportional Control is often smoother than PID for "Look Ahead" tracking
    # because the look-ahead naturally acts as a damping (D) term.
    steering = error * Kp_steer
    
    left_wheel = np.clip(BASE_SPEED + steering, -0.2, 0.4)
    right_wheel = np.clip(BASE_SPEED - steering, -0.2, 0.4)
    
    if is_running:
        base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})
    
    cv2.putText(viz_img, f"Err: {int(error)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return viz_img

# -----------------------------
# Flask Routes
# -----------------------------
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

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_and_drive(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/start")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
