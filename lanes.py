from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Perspective Transform Points (Based on your green trapezoid overlays)
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# -----------------------------
# Initialization
# -----------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

def process_lane_view(frame):
    h, w = frame.shape[:2]

    # 1. Grayscale and Threshold (Isolating white lines)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    
    # 2. Warp to Top-Down View
    warped = cv2.warpPerspective(thresh, M, (w, h))

    # 3. Find Lane Peaks (Histogram)
    hist = np.sum(warped[int(h*0.5):, :], axis=0)
    midpoint = int(hist.shape[0]/2)
    left_x = np.argmax(hist[:midpoint])
    right_x = np.argmax(hist[midpoint:]) + midpoint
    lane_center = (left_x + right_x) // 2

    # 4. Create Binary Visualization (Matching your uploaded screenshots)
    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    
    # Draw central cyan target line
    cv2.line(viz, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
    
    # Draw Green (Left) and Red (Right) detection dots
    cv2.circle(viz, (left_x, h - 40), 20, (0, 255, 0), -1) 
    cv2.circle(viz, (right_x, h - 40), 20, (0, 0, 255), -1)
    
    # Display Lane Center Text
    cv2.putText(viz, f"Lane Center: {lane_center}", (w - 300, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 5. Add Picture-in-Picture Raw View (Top Left)
    pip_h, pip_w = 120, 160
    raw_small = cv2.resize(frame, (pip_w, pip_h))
    
    # Draw the green trapezoid on the small raw view
    pts = (SRC_POINTS // (FRAME_WIDTH / pip_w)).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(raw_small, [pts], True, (0, 255, 0), 2)
    
    viz[10:10+pip_h, 10:10+pip_w] = raw_small

    return viz

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_lane_view(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(sig, frame):
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
