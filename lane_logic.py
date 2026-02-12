import cv2
import numpy as np

class LaneDetector:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Define the trapezoid points for the Bird's Eye View
        # These depend on your camera's mounting angle
        self.src = np.float32([
            [100, h],       # Bottom Left
            [540, h],       # Bottom Right
            [220, 320],     # Top Left
            [420, 320]      # Top Right
        ])
        self.dst = np.float32([
            [150, h], [490, h],
            [150, 0], [490, 0]
        ])
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def get_deviation(self, frame):
        # 1. Pre-process: Grayscale and Threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 2. Warp to Bird's Eye View
        warped = cv2.warpPerspective(thresh, self.matrix, (self.w, self.h))

        # 3. Calculate Histogram of the bottom half
        hist = np.sum(warped[self.h//2:, :], axis=0)
        midpoint = int(hist.shape[0]/2)
        
        # Find the peak of the left and right halves
        left_base = np.argmax(hist[:midpoint])
        right_base = np.argmax(hist[midpoint:]) + midpoint
        
        # Determine lane center and deviation from frame center
        lane_center = (left_base + right_base) // 2
        deviation = lane_center - (self.w // 2)

        return deviation, warped, lane_center
