import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import requests

STOP_DISTANCE = 0.5  # meters
TURN_ENDPOINT = "http://localhost:5001/turn_left"

class LidarFront180Turn(Node):

    def __init__(self):
        super().__init__('lidar_front_180_turn')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        # 180° front region (-90° to +90°)
        front_mask = np.logical_and(
            angles >= -math.pi/4,
            angles <= math.pi/4
        )

        front_ranges = ranges[front_mask]
        front_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(front_ranges) == 0:
            return

        min_distance = np.min(front_ranges)

        if min_distance < STOP_DISTANCE:
            self.get_logger().warn(
                f"Obstacle detected at {min_distance:.2f} meters"
            )
            try:
                requests.get(TURN_ENDPOINT, timeout=0.1)
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = LidarFront180Turn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
