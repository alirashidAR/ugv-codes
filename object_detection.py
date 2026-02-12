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

# Only allow these classes
ALLOWED_CLASSES = {
    7: "car",
    15: "person"
}

# -----------------------------
# Load model
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

print("Model loaded")

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

print("Pi Camera started")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            idx = int(detections[0, 0, i, 1])

            if idx not in ALLOWED_CLASSES:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            label = f"{ALLOWED_CLASSES[idx]}: {confidence*100:.1f}%"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY > 20 else startY + 20
            cv2.putText(
                frame,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

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
    print("Shutting down")
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
