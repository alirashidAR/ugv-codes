import serial
import json
import time


def is_raspberry_pi5():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "Model" in line:
                    return "Raspberry Pi 5" in line
    except Exception:
        pass
    return False


device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
baudrate = 115200

print(f"[INFO] Using serial device: {device}")

ser = serial.Serial(
    port=device,
    baudrate=baudrate,
    timeout=1
)

buffer = bytearray()

print("[INFO] Listening for chassis data...")
print("[INFO] Measuring receive frequency (Hz)")
print("--------------------------------------------------")

count = 0
t0 = time.time()

try:
    while True:
        data = ser.read(ser.in_waiting or 1)
        if not data:
            continue

        buffer.extend(data)

        while b"\n" in buffer:
            line, _, buffer = buffer.partition(b"\n")
            print(line)
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                continue

            clean = text[start:end + 1]
            clean = clean.replace(",}", "}")

            try:
                obj = json.loads(clean)
            except Exception:
                continue

            if obj.get("T") == 1001:
                count += 1
                now = time.time()

                if now - t0 >= 1.0:
                    print(f"[RATE] {count} Hz")
                    count = 0
                    t0 = now

except KeyboardInterrupt:
    print("\n[INFO] Exiting")
    ser.close()

