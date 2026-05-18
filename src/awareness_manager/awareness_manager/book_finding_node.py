"""
book_finding_node.py — ROS2 awareness node for the Find My Book scenario.

Integrates the AwarenessManager with a navigating MIRTE robot in the
KRR_Course_Small_house Gazebo world.  The node:

  1. Ticks the AM at tick_rate Hz, publishing awareness state.
  2. Subscribes to /amcl_pose → maps (x,y) → current room zone → updates
     the AM's robot_zone so F6 spatial costs reflect the robot's location.
  3. Subscribes to /goal_pose (simple Nav2 goal) → when the robot is sent
     to a room, queue_goal is called with an ETA so F2 anticipatory
     pre-tuning fires before the robot arrives.
  4. Publishes /awareness/schedule (String[], one concept per slot) so
     downstream nodes know which book locations have highest priority.

Topics published (same as awareness_node):
    awareness/state            (std_msgs/String  JSON)
    awareness/schedule         (std_msgs/String  JSON list)
    awareness/goal             (std_msgs/String)
    awareness/controller_state (std_msgs/String  JSON)  — book-finding status

Topics subscribed:
    /amcl_pose                 (geometry_msgs/PoseWithCovarianceStamped)
    /goal_pose                 (geometry_msgs/PoseStamped)  — Nav2 simple goal

Parameters:
    budget           int     3       Max concepts scheduled per tick
    tick_rate        float   10.0    Hz
    observation_interval float 5.0  Seconds between auto-observations
    alpha            float   0.5     F1 spreading activation decay
    nav_eta          float   15.0    Assumed travel time (s) when a nav goal fires
    f6               bool    true    Enable spatial opportunity cost
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.feature_config import FeatureConfig
from awareness_manager.scenarios.book_finding import (
    ZONE_TRAVEL_TIMES,
    build_book_finding_kb,
    build_book_finding_instance_kb,
    load_zone_assignment,
    get_room_for_pose,
)


class BookFindingNode(Node):

    def __init__(self) -> None:
        super().__init__('book_finding_awareness')

        self.declare_parameter('budget',               3)
        self.declare_parameter('tick_rate',            10.0)
        self.declare_parameter('observation_interval', 5.0)
        self.declare_parameter('alpha',                0.5)
        self.declare_parameter('nav_eta',              15.0)
        self.declare_parameter('f6',                   True)

        budget    = self.get_parameter('budget').value
        tick_rate = self.get_parameter('tick_rate').value
        obs_int   = self.get_parameter('observation_interval').value
        alpha     = self.get_parameter('alpha').value
        f6        = self.get_parameter('f6').value
        self._nav_eta   = self.get_parameter('nav_eta').value
        self._obs_int   = obs_int
        self._dt        = 1.0 / tick_rate

        fc = FeatureConfig() if f6 else FeatureConfig.with_disabled('f6')

        kb              = build_book_finding_kb()
        ikb             = build_book_finding_instance_kb()
        zone_assignment = load_zone_assignment()

        self._am = AwarenessManager(
            kb=kb,
            instance_kb=ikb,
            goal_id='find_book',
            budget=budget,
            alpha=alpha,
            feature_config=fc,
            zone_travel_times=ZONE_TRAVEL_TIMES,
            zone_assignment=zone_assignment,
        )

        # State
        self._lock               = threading.Lock()
        self._current_zone: str | None = 'living_room'
        self._current_robot_pos: str | None = 'living_room'  # concept_id for robot_pos
        self._time_since_obs     = 0.0
        self._last_schedule: list[str] = []
        self._found_book: str | None   = None  # concept_id once found
        self._queued_nav_goal: str | None = None

        # Publishers
        self._pub_state    = self.create_publisher(String, 'awareness/state',            10)
        self._pub_schedule = self.create_publisher(String, 'awareness/schedule',         10)
        self._pub_goal     = self.create_publisher(String, 'awareness/goal',             10)
        self._pub_ctrl     = self.create_publisher(String, 'awareness/controller_state', 10)

        # Subscriptions
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl_pose, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self._on_goal_pose, 10)

        # Tick timer
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f"BookFindingNode started — budget={budget}  "
            f"tick={tick_rate}Hz  obs_interval={obs_int}s  f6={f6}"
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        zone = get_room_for_pose(x, y)
        if zone is None:
            return
        with self._lock:
            if zone != self._current_zone:
                self._current_zone = zone
                # Use the room concept as robot_pos so F6 zone lookup works
                self._current_robot_pos = zone
                self.get_logger().info(f"[ZONE] Robot entered '{zone}'")
                # Observe the book in this room immediately on entry
                book_cid = f"book_{zone}"
                if (self._am.instance_kb and
                        book_cid in self._am.instance_kb.instance_ids()):
                    self._am.observe(book_cid)
                    self.get_logger().info(f"[OBS]  Observed '{book_cid}' on room entry")

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        """When a Nav2 goal is sent, infer the target room and queue it."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        target_zone = get_room_for_pose(x, y)
        if target_zone is None:
            return
        book_cid = f"book_{target_zone}"
        eta = self._nav_eta
        with self._lock:
            if self._queued_nav_goal != book_cid:
                self._queued_nav_goal = book_cid
                # Use a surrogate goal: 'find_book' (the class KB goal) with
                # the matching book instance as the focus.  F2 will boost
                # attention for concepts near the target room.
                self._am.queue_goal(book_cid, eta=eta, level='task')
                self.get_logger().info(
                    f"[NAV]  Queued goal '{book_cid}' ETA={eta:.0f}s "
                    f"(navigating to '{target_zone}')"
                )

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        with self._lock:
            schedule = self._am.tick(dt=self._dt, robot_pos=self._current_robot_pos)
            self._time_since_obs += self._dt

            if schedule and self._time_since_obs >= self._obs_int:
                for cid in schedule:
                    self._am.observe(cid)
                self._time_since_obs = 0.0
                self._last_schedule = list(schedule)

            self._publish_state(schedule)

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_state(self, schedule: list[str]) -> None:
        am = self._am
        attn   = am.attention()
        epist  = {c: am.kb.get_concept(c).epistemic_error
                  for c in am.kb.concept_ids()}
        prio   = am.priorities()

        # Build per-concept state dict (mirrors awareness_node format)
        ch = am.attention_channels()
        state: dict[str, dict] = {}
        for cid in am.kb.concept_ids():
            concept = am.kb.get_concept(cid)
            a = attn.get(cid, 0.0)
            state[cid] = {
                'A':    round(a, 4),
                'E':    round(concept.epistemic_error, 4),
                'P':    round(prio.get(cid, 0.0), 6),
                'ch_m': round(ch['mission'].get(cid, 0.0), 4),
                'ch_an':round(ch['anticipatory'].get(cid, 0.0), 4),
                'ch_r': round(ch['relational'].get(cid, 0.0), 4),
                'ch_s': round(ch['surprise'].get(cid, 0.0), 4),
            }

        if am.instance_kb:
            for iid in am.instance_kb.instance_ids():
                inst = am.instance_kb.get_instance(iid)
                ia   = attn.get(iid, 0.0)
                state[iid] = {
                    'A':           round(ia, 4),
                    'E':           round(inst.epistemic_error, 4),
                    'P':           round(prio.get(iid, 0.0), 6),
                    'urgency':     round(inst.urgency, 4),
                    'urgency_rate': inst.urgency_rate,
                    'ch_m':        round(ch['mission'].get(iid, 0.0), 4),
                    'ch_an':       round(ch['anticipatory'].get(iid, 0.0), 4),
                    'ch_r':        round(ch['relational'].get(iid, 0.0), 4),
                    'ch_s':        round(ch['surprise'].get(iid, 0.0), 4),
                }

        self._pub_state.publish(String(data=json.dumps(state)))
        self._pub_schedule.publish(String(data=json.dumps(schedule)))
        self._pub_goal.publish(String(data=am.goal_id))

        ctrl = {
            'zone':       self._current_zone or 'unknown',
            'found_book': self._found_book,
            'schedule':   schedule[:3],
        }
        self._pub_ctrl.publish(String(data=json.dumps(ctrl)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BookFindingNode()
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
