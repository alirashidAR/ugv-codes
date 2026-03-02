import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import requests

STOP_DISTANCE = 0.5
TURN_ENDPOINT = "http://localhost:5001/turn_left"

class LidarFrontTurn(Node):

    def __init__(self):
        super().__init__('lidar_front_turn')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.obstacle_present = False  # State flag

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        # Narrow cone: -45° to +45°
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

            # Trigger only once when obstacle first appears
            if not self.obstacle_present:
                self.get_logger().warn(
                    f"Obstacle detected at {min_distance:.2f} meters"
                )
                try:
                    requests.get(TURN_ENDPOINT, timeout=0.1)
                except:
                    pass

                self.obstacle_present = True

        else:
            # Reset when path becomes clear
            if self.obstacle_present:
                self.get_logger().info("Path clear")
                self.obstacle_present = False


def main(args=None):
    rclpy.init(args=args)
    node = LidarFrontTurn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
