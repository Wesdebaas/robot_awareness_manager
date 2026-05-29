"""
scan_relay_node.py — Re-stamps /scan_raw with ROS2 sim time and republishes as /scan.

The Gazebo lidar plugin (libgazebo_ros_ray_sensor.so) stamps scans with the
plugin node's wall-clock time. Nav2's costmap does TF lookups at the scan's
header timestamp; when that timestamp is wall-clock (~1.78e9 s) but TF is in
Gazebo sim time (~5500 s), every lookup fails and the MPPI controller either
jitters or falsely declares "goal reached".

This node runs with use_sim_time:=true so get_clock().now() returns sim time.
It replaces the header stamp before republishing, making the scan transparent
to Nav2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanRelayNode(Node):

    def __init__(self) -> None:
        super().__init__('scan_relay')
        self._pub = self.create_publisher(LaserScan, 'scan', 10)
        self.create_subscription(LaserScan, 'scan_raw', self._relay, 10)
        self.get_logger().info("scan_relay: /scan_raw → /scan with sim-time stamp")

    def _relay(self, msg: LaserScan) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
