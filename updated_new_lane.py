from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
from base_ctrl import BaseController
from collections import deque

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 580, 440
BASE_SPEED = 0.12          
Kp_steer = 0.005           # Proportional steering gain
Ki_steer = 0.0001          # Integral gain (small)
Kd_steer = 0.002           # Derivative gain

# Perspective Transform (Keep your calibration)
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Sliding Window Config
N_WINDOWS = 9              # Number of stacking windows
MARGIN = 60                # Width of the windows +/- margin
MINPIX = 40                # Minimum pixels found to recenter window

# Color Detection Thresholds for WHITE LINES ONLY
WHITE_THRESHOLD_LOW = 180   # Detect white tape (adjust 170-200 based on lighting)
WHITE_THRESHOLD_HIGH = 255

# Noise Filtering
MIN_LINE_PIXELS = 100       # Minimum pixels to consider a valid lane
SMOOTHING_FRAMES = 5        # Number of frames to average for stability

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
is_running = False

# Temporal smoothing buffers
lane_center_buffer = deque(maxlen=SMOOTHING_FRAMES)
error_buffer = deque(maxlen=3)
last_error = 0
integral_error = 0

# -----------------------------
# IMPROVED: White Line Detection
# -----------------------------

def isolate_white_lines(frame):
    """
    Isolates ONLY white lane lines, rejecting:
    - Zebra crossings (horizontal lines)
    - Red borders
    - Green grass
    - Shadows
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This helps in varying lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold to get ONLY white areas
    _, white_mask = cv2.threshold(blur, WHITE_THRESHOLD_LOW, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up noise
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # CRITICAL: Filter out horizontal lines (zebra crossings)
    # Use Sobel to detect vertical gradients (lane lines are mostly vertical in bird's eye)
    sobelx = cv2.Sobel(white_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(white_mask, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of gradient
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
    
    # Direction of gradient (we want vertical lines, so gradient should be horizontal)
    gradient_dir = np.arctan2(sobely, sobelx)
    
    # Keep only edges that are roughly horizontal in gradient direction
    # (which means vertical lines in the image)
    # Allow angles between -30° and +30° from horizontal
    angle_mask = np.zeros_like(gradient_dir, dtype=np.uint8)
    angle_mask[(gradient_dir > -np.pi/6) & (gradient_dir < np.pi/6)] = 255
    angle_mask[(gradient_dir > 5*np.pi/6) | (gradient_dir < -5*np.pi/6)] = 255
    
    # Combine white detection with vertical line filter
    vertical_lines = cv2.bitwise_and(white_mask, white_mask, mask=angle_mask)
    
    # Additional cleanup
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, kernel)
    
    return vertical_lines

def create_roi_mask(shape):
    """
    Create ROI mask to ignore outer red borders and center grass
    Focus only on the lane area
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define a trapezoid focusing on the actual lane area
    # Adjust these based on your warped perspective
    roi_vertices = np.array([[
        (100, h),           # Bottom left
        (w - 100, h),       # Bottom right
        (w - 200, 0),       # Top right
        (200, 0)            # Top left
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    
    return mask

# -----------------------------
# Sliding Windows with Filtering
# -----------------------------

def find_lane_windows(binary_warped):
    """
    Scans the image using sliding windows to track curved lines.
    IMPROVED: Better initialization and noise rejection
    """
    # Take a histogram of the bottom third of the image (more stable than half)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] * 0.7):,:], axis=0)
    
    # Smooth histogram to reduce noise spikes
    from scipy.ndimage import gaussian_filter1d
    histogram = gaussian_filter1d(histogram, sigma=5)
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Find peaks with minimum distance to avoid false detections
    midpoint = int(histogram.shape[0] // 2)
    
    # Left lane: search in left half with minimum height threshold
    left_hist = histogram[:midpoint]
    if np.max(left_hist) > 50:  # Minimum threshold
        leftx_base = np.argmax(left_hist)
    else:
        leftx_base = midpoint // 2  # Default fallback
    
    # Right lane: search in right half
    right_hist = histogram[midpoint:]
    if np.max(right_hist) > 50:
        rightx_base = np.argmax(right_hist) + midpoint
    else:
        rightx_base = midpoint + midpoint // 2  # Default fallback

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set height of windows
    window_height = int(binary_warped.shape[0] // N_WINDOWS)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(N_WINDOWS):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                     (win_xleft_high, win_y_high), (0, 255, 0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                     (win_xright_high, win_y_high), (0, 255, 0), 2) 
        
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
    global last_error, integral_error
    
    h, w = frame.shape[:2]
    
    # 1. Isolate WHITE LINES ONLY (reject zebra crossings, grass, red borders)
    white_lines = isolate_white_lines(frame)
    
    # 2. Apply ROI mask
    roi_mask = create_roi_mask(white_lines.shape)
    masked = cv2.bitwise_and(white_lines, white_lines, mask=roi_mask)
    
    # 3. Warp perspective
    warped = cv2.warpPerspective(masked, M, (w, h))

    # 4. Sliding Window Search
    leftx, lefty, rightx, righty, viz_img = find_lane_windows(warped)

    # 5. Calculate Lane Center with Temporal Smoothing
    center_img = w // 2
    
    if len(leftx) > MIN_LINE_PIXELS and len(rightx) > MIN_LINE_PIXELS:
        # Both lanes found: Use median for robustness (less affected by outliers)
        l_median = np.median(leftx)
        r_median = np.median(rightx)
        lane_target = (l_median + r_median) / 2
        color = (0, 255, 0)  # Green: Good lock
        confidence = "BOTH LANES"
        
    elif len(leftx) > MIN_LINE_PIXELS:
        # Only Left found - estimate right lane
        l_median = np.median(leftx)
        lane_width = 340  # Expected lane width in pixels (adjust if needed)
        lane_target = l_median + (lane_width / 2)
        color = (0, 255, 255)  # Yellow: Left only
        confidence = "LEFT ONLY"
        
    elif len(rightx) > MIN_LINE_PIXELS:
        # Only Right found - estimate left lane
        r_median = np.median(rightx)
        lane_width = 340
        lane_target = r_median - (lane_width / 2)
        color = (255, 165, 0)  # Orange: Right only
        confidence = "RIGHT ONLY"
        
    else:
        # LOST - Use last known good position or stop
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        cv2.putText(viz_img, "LOST LANE - STOPPING", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        lane_center_buffer.clear()  # Reset smoothing
        return viz_img
    
    # 6. Temporal Smoothing - Average last N lane centers
    lane_center_buffer.append(lane_target)
    smoothed_target = np.mean(lane_center_buffer)
    
    # 7. PID Control for Smooth Steering
    error = smoothed_target - center_img
    error_buffer.append(error)
    
    # Proportional
    P = error * Kp_steer
    
    # Integral (with anti-windup)
    integral_error += error
    integral_error = np.clip(integral_error, -500, 500)  # Prevent windup
    I = integral_error * Ki_steer
    
    # Derivative (use smoothed error)
    if len(error_buffer) >= 2:
        error_change = error_buffer[-1] - error_buffer[-2]
        D = error_change * Kd_steer
    else:
        D = 0
    
    steering = P + I + D
    last_error = error
    
    # 8. Wheel Speed Calculation with Speed Reduction on Turns
    # Reduce speed on sharp turns for stability
    turn_severity = abs(steering)
    if turn_severity > 0.05:
        speed_factor = 0.7  # Slow down 30% on turns
    else:
        speed_factor = 1.0
    
    adjusted_speed = BASE_SPEED * speed_factor
    
    left_wheel = np.clip(adjusted_speed + steering, -0.3, 0.4)
    right_wheel = np.clip(adjusted_speed - steering, -0.3, 0.4)
    
    # 9. Send command
    if is_running:
        base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})
    
    # 10. Visualization
    cv2.circle(viz_img, (int(smoothed_target), h // 2), 15, color, -1)
    cv2.line(viz_img, (center_img, 0), (center_img, h), (255, 255, 255), 2)
    
    # Display info
    cv2.putText(viz_img, f"{confidence}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(viz_img, f"Error: {int(error)}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz_img, f"L:{left_wheel:.2f} R:{right_wheel:.2f}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
    global is_running, integral_error
    is_running = False
    integral_error = 0  # Reset integral term
    lane_center_buffer.clear()
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
