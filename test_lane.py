from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import signal
import sys

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.3
ALLOWED_CLASSES = {7: "car", 15: "person"}

# Lane Transformation Points (Bird's Eye View)
# Tune these based on your camera angle!
SRC_POINTS = np.float32([
    [100, 480], [540, 480], # Bottom corners
    [220, 320], [420, 320]  # Top corners (trapezoid)
])
DST_POINTS = np.float32([
    [150, 480], [490, 480], # Bottom mapped to rectangle
    [150, 0],   [490, 0]    # Top mapped to rectangle
])

# -----------------------------
# Initialization
# -----------------------------
# Load Object Detection Model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
# Compute Perspective Matrix for Lane Detection
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Camera Setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

# -----------------------------
# Detection Logic
# -----------------------------
def process_frame(frame):
    h, w = frame.shape[:2]

    # --- 1. Lane Detection (Bird's Eye View) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    warped = cv2.warpPerspective(thresh, M, (w, h))
    
    # Histogram logic to find center
    hist = np.sum(warped[h//2:, :], axis=0)
    midpoint = int(hist.shape[0]/2)
    left_x = np.argmax(hist[:midpoint])
    right_x = np.argmax(hist[midpoint:]) + midpoint
    lane_center = (left_x + right_x) // 2
    deviation = lane_center - (w // 2)

    # Visual overlays for Lane
    cv2.circle(frame, (lane_center, h-30), 10, (0, 0, 255), -1) # Lane center (Red)
    cv2.line(frame, (w//2, h), (w//2, h-60), (255, 0, 0), 2)    # Target (Blue)
    cv2.putText(frame, f"Steer: {deviation}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- 2. Object Detection ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if idx in ALLOWED_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{ALLOWED_CLASSES[idx]}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    return frame

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)
        
        ret, jpeg = cv2.imencode(".jpg", processed_frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# -----------------------------
# Routes & Cleanup
# -----------------------------
@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(sig, frame):
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
