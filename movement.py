from base_ctrl import BaseController
import time


def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except:
        return False


device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(device, 115200)

# ---------------- TEST SEQUENCE ----------------

print("[TEST] Move forward")
base.send_command({"T": 1, "L": 0.2, "R": 0.2})
time.sleep(2)

print("[TEST] Stop")
base.send_command({"T": 1, "L": 0.0, "R": 0.0})
time.sleep(1)

print("[TEST] Turn right (left wheel forward)")
base.send_command({"T": 1, "L": 0.2, "R": 0.0})
time.sleep(2)

print("[TEST] Stop")
base.send_command({"T": 1, "L": 0.0, "R": 0.0})
time.sleep(1)

print("[TEST] Turn left (right wheel forward)")
base.send_command({"T": 1, "L": 0.0, "R": 0.2})
time.sleep(2)

print("[TEST] Stop")
base.send_command({"T": 1, "L": 0.0, "R": 0.0})

print("[DONE]")
