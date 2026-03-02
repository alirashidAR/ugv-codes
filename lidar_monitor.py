import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

STOP_DISTANCE = 0.5  # meters

class LidarFrontMonitor(Node):

    def __init__(self):
        super().__init__('lidar_front_monitor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

    def scan_callback(self, msg):
        ranges = msg.ranges

        # Front beam = center of scan
        front_index = len(ranges) // 2
        front_distance = ranges[front_index]

        if math.isfinite(front_distance):
            if front_distance < STOP_DISTANCE:
                self.get_logger().warn(
                    f"Obstacle detected at {front_distance:.2f} meters"
                )

def main(args=None):
    rclpy.init(args=args)
    node = LidarFrontMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
