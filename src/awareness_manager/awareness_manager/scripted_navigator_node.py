"""
scripted_navigator_node.py — Awareness-driven room-tour navigator for the Find My Book demo.

Drives the MIRTE robot through rooms, publishing Nav2 goals. The AwarenessManager
publishes `awareness/suggested_room` with the top-priority room at each tick.
When a suggestion is available, the navigator follows it (closing the AM→navigation
loop). When no suggestion has arrived since the last step, it falls back to the
fixed room sequence.

This creates a reactive navigation strategy where the AM's epistemic priorities
directly guide exploration order: rooms with high-E book instances get visited
first, rather than cycling in a predetermined order.

Fallback sequence (when no suggestion): kitchen → bathroom → bedroom → study → living_room
The robot starts in living_room; the AM observes it there on startup.

Parameters:
    dwell_time   float  8.0    Seconds to stay in each room before advancing
    loop         bool   true   Whether to keep cycling after all rooms visited
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
from std_msgs.msg import String


# Explicit nav waypoints verified against the nav2 map (free or allow_unknown reachable).
# Kept here rather than deriving from ROOM_BOUNDS centers because several room centers
# fall on furniture obstacles (bedroom/study have couches at y≈2.0) and the old kitchen
# bounds pointed outside the map.
_WAYPOINTS: dict[str, tuple[float, float]] = {
    "kitchen":     ( 0.0,  1.75),   # between y=1.5 wall and kitchen appliances
    "bathroom":    ( 0.5, -1.25),   # south corridor, clearly free
    "bedroom":     (-3.5,  1.0),    # bedroom floor, avoids furniture walls at y≈2.0
    "study":       (-6.5,  1.0),    # study floor, avoids walls at y=2.0
    "living_room": ( 0.5,  0.5),    # central living room, near spawn
}

# Robot spawn position in living_room — used as initial orientation reference.
_SPAWN_POS: tuple[float, float] = (1.0527, 0.5096)


class ScriptedNavigatorNode(Node):

    # Visit order — robot starts in living_room so we skip it first;
    # it returns at the end of each cycle, closing the loop naturally.
    ROOM_SEQUENCE = ["kitchen", "bathroom", "bedroom", "study", "living_room"]

    def __init__(self) -> None:
        super().__init__('scripted_navigator')

        self.declare_parameter('dwell_time',  8.0)
        self.declare_parameter('loop',        True)
        self.declare_parameter('start_delay', 38.0)

        self._dwell_time  = self.get_parameter('dwell_time').value
        self._loop        = self.get_parameter('loop').value
        self._start_delay = self.get_parameter('start_delay').value

        self._waypoints = {room: _WAYPOINTS[room] for room in self.ROOM_SEQUENCE}
        self._current_pos: tuple[float, float] = _SPAWN_POS

        # AM-suggested room (latest received); None = no suggestion yet.
        # Protected by _suggested_lock; consumed once per navigation step.
        self._suggested_room: str | None = None
        self._suggested_lock = threading.Lock()

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._goal_pub   = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscribe to AM room suggestions
        self.create_subscription(
            String, 'awareness/suggested_room', self._on_suggested_room, 10
        )

        self.get_logger().info(
            f"ScriptedNavigator: {' → '.join(self.ROOM_SEQUENCE)} (fallback), "
            f"dwell={self._dwell_time}s  loop={self._loop}  "
            f"start_delay={self._start_delay}s  AM-driven=True"
        )

        # Run the navigation loop in a background thread so spin() stays free
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── Suggestion callback ───────────────────────────────────────────────────

    def _on_suggested_room(self, msg: String) -> None:
        """Store the latest AM room suggestion; consumed at the next navigation step."""
        room = msg.data
        if room in self._waypoints:
            with self._suggested_lock:
                self._suggested_room = room

    # ── Navigation loop ───────────────────────────────────────────────────────

    _RETRY_DELAY = 5.0   # seconds between retries when Nav2 rejects a goal
    _MAX_RETRIES = 30    # give Nav2 up to ~2.5 min to become fully ready

    def _run(self) -> None:
        time.sleep(self._start_delay)

        self.get_logger().info("Waiting for navigate_to_pose action server…")
        if not self._nav_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error(
                "navigate_to_pose action server not available after 120 s — "
                "scripted navigator aborting."
            )
            return
        self.get_logger().info("Nav2 action server up. Starting room tour.")

        idx = 0
        while rclpy.ok():
            # Consume the latest AM suggestion (if any); fall back to fixed sequence.
            with self._suggested_lock:
                suggested = self._suggested_room
                self._suggested_room = None  # consume

            if suggested and suggested in self._waypoints:
                room = suggested
                self.get_logger().info(f"[AM]  Awareness-driven destination → '{room}'")
            else:
                room = self.ROOM_SEQUENCE[idx % len(self.ROOM_SEQUENCE)]
                self.get_logger().info(f"[SEQ] Fallback sequence → '{room}'")
                idx += 1

            x, y = self._waypoints[room]

            # Publish to /goal_pose first so BookFindingNode queues the
            # anticipatory goal (F2) before the robot starts moving.
            pose = self._make_pose(x, y)
            self._goal_pub.publish(pose)
            self.get_logger().info(f"[NAV] → '{room}'  ({x:.2f}, {y:.2f})")

            succeeded = False
            for attempt in range(self._MAX_RETRIES):
                pose = self._make_pose(x, y)  # fresh timestamp each attempt
                status = self._navigate_blocking(pose)

                if status == GoalStatus.STATUS_SUCCEEDED:
                    succeeded = True
                    break

                # Nav2 may still be initialising (TF not ready) — retry
                self.get_logger().warn(
                    f"[RETRY {attempt + 1}/{self._MAX_RETRIES}] '{room}' "
                    f"status={status} — waiting {self._RETRY_DELAY:.0f}s…"
                )
                time.sleep(self._RETRY_DELAY)

            if succeeded:
                self._current_pos = (x, y)
                self.get_logger().info(
                    f"[ARRIVED] '{room}' — dwelling {self._dwell_time:.0f}s"
                )
                time.sleep(self._dwell_time)
            else:
                self.get_logger().error(
                    f"[GIVE UP] '{room}' failed after {self._MAX_RETRIES} "
                    "attempts — skipping room."
                )

            if not self._loop and idx >= len(self.ROOM_SEQUENCE):
                self.get_logger().info("Room tour complete.")
                break

    def _navigate_blocking(self, pose: PoseStamped) -> int:
        """Send a NavigateToPose goal and block until it completes.

        Runs from the background thread; callbacks fire on the spin thread.
        Returns the GoalStatus int code.
        """
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

    # ── Helpers ───────────────────────────────────────────────────────────────

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
    node = ScriptedNavigatorNode()
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
