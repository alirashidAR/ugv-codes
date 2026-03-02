import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

STOP_DISTANCE = 0.5  # meters

class LidarFront180(Node):

    def __init__(self):
        super().__init__('lidar_front_180')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        # Select angles between -90° and +90°
        front_mask = np.logical_and(
            angles >= -math.pi/2,
            angles <= math.pi/2
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

def main(args=None):
    rclpy.init(args=args)
    node = LidarFront180()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
