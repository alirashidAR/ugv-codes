from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import signal
import sys

from base_ctrl import BaseController

# -----------------------------
# Rover base setup (UNCHANGED)
# -----------------------------
def is_raspberry_pi5():
    with open('/proc/cpuinfo', 'r') as file:
        for line in file:
            if 'Model' in line:
                return 'Raspberry Pi 5' in line
    return False

if is_raspberry_pi5():
    base = BaseController('/dev/ttyAMA0', 115200)
else:
    base = BaseController('/dev/serial0', 115200)

def stop():
    base.send_command({"T": 1, "L": 0, "R": 0})

def forward(speed=0.2):
    base.send_command({"T": 1, "L": speed, "R": speed})

def turn_left(speed=0.2):
    base.send_command({"T": 1, "L": -speed, "R": speed})

def turn_right(speed=0.2):
    base.send_command({"T": 1, "L": speed, "R": -speed})

# -----------------------------
# Vision configuration
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.4

# Bounding box width thresholds (tune these)
FAR_THRESHOLD = 120
CLOSE_THRESHOLD = 220

ALLOWED_CLASSES = {
    7: "car",
    15: "person"
}

# -----------------------------
# Load detection model
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# -----------------------------
# Camera setup
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
# Flask app
# -----------------------------
app = Flask(__name__)

last_motion_time = time.time()

def generate_frames():
    global last_motion_time

    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        action_taken = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            idx = int(detections[0, 0, i, 1])
            if idx not in ALLOWED_CLASSES:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            box_width = endX - startX
            center_x = (startX + endX) // 2

            label = f"{ALLOWED_CLASSES[idx]} {confidence*100:.1f}%"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------- Motion logic --------
            if box_width > CLOSE_THRESHOLD:
                stop()
                action_taken = True

            elif box_width > FAR_THRESHOLD:
                if center_x < w // 2:
                    turn_right()
                else:
                    turn_left()
                action_taken = True

            break  # Only react to the closest object

        if not action_taken:
            forward()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------
# Clean shutdown
# -----------------------------
def cleanup(sig, frame):
    stop()
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)

