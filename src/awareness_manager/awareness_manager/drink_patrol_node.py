"""
drink_patrol_node.py — Minimal two-room patrol for the Drink Serving scenario.

Drives MIRTE between the kitchen and living_room in a continuous loop using
the Nav2 NavigateToPose action. No awareness management logic — pure navigation.

This is Step 1: verify that Nav2 navigation works reliably before adding the AM.

Parameters:
    dwell_time   float  8.0    Seconds to stay in each room before moving
    start_delay  float  38.0   Seconds to wait before the first goal (Nav2 init)
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


_WAYPOINTS: dict[str, tuple[float, float]] = {
    "kitchen":     ( 0.0,  0.0),
    "living_room": (-8.0,  0.35),
}

_PATROL_SEQUENCE = ["kitchen", "living_room"]

_SPAWN_POS: tuple[float, float] = (1.0527, 0.5096)

_RETRY_DELAY = 5.0
_MAX_RETRIES = 30


class DrinkPatrolNode(Node):

    def __init__(self) -> None:
        super().__init__('drink_patrol')

        self.declare_parameter('dwell_time',  8.0)
        self.declare_parameter('start_delay', 38.0)

        self._dwell_time  = self.get_parameter('dwell_time').value
        self._start_delay = self.get_parameter('start_delay').value
        self._current_pos = _SPAWN_POS

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info(
            f"DrinkPatrolNode ready — patrol: {' ↔ '.join(_PATROL_SEQUENCE)}  "
            f"dwell={self._dwell_time}s  start_delay={self._start_delay}s"
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        time.sleep(self._start_delay)

        self.get_logger().info("Waiting for navigate_to_pose action server…")
        if not self._nav_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error(
                "navigate_to_pose not available after 120 s — patrol aborting."
            )
            return
        self.get_logger().info("Nav2 action server up. Starting patrol.")

        idx = 0
        while rclpy.ok():
            room = _PATROL_SEQUENCE[idx % len(_PATROL_SEQUENCE)]
            idx += 1

            x, y = _WAYPOINTS[room]
            self.get_logger().info(f"[PATROL] → '{room}'  ({x:.2f}, {y:.2f})")

            succeeded = False
            for attempt in range(_MAX_RETRIES):
                pose = self._make_pose(x, y)
                status = self._navigate_blocking(pose)

                if status == GoalStatus.STATUS_SUCCEEDED:
                    succeeded = True
                    break

                self.get_logger().warn(
                    f"[RETRY {attempt + 1}/{_MAX_RETRIES}] '{room}' "
                    f"status={status} — waiting {_RETRY_DELAY:.0f}s…"
                )
                time.sleep(_RETRY_DELAY)

            if succeeded:
                self._current_pos = (x, y)
                self.get_logger().info(
                    f"[ARRIVED] '{room}' — dwelling {self._dwell_time:.0f}s"
                )
                time.sleep(self._dwell_time)
            else:
                self.get_logger().error(
                    f"[GIVE UP] '{room}' failed after {_MAX_RETRIES} attempts — skipping."
                )

    def _navigate_blocking(self, pose: PoseStamped) -> int:
        done = threading.Event()
        status_holder: list[int] = [GoalStatus.STATUS_ABORTED]

        def on_accepted(future) -> None:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Goal rejected by Nav2.")
                status_holder[0] = GoalStatus.STATUS_ABORTED
                done.set()
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(on_result)

        def on_result(future) -> None:
            status_holder[0] = future.result().status
            done.set()

        send_future = self._nav_client.send_goal_async(
            NavigateToPose.Goal(pose=pose)
        )
        send_future.add_done_callback(on_accepted)

        done.wait()
        return status_holder[0]

    def _make_pose(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        dx = x - self._current_pos[0]
        dy = y - self._current_pos[1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            yaw = math.atan2(dy, dx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            pose.pose.orientation.w = 1.0
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DrinkPatrolNode()
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
