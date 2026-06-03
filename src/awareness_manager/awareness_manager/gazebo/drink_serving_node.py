"""
drink_serving_node.py — Awareness-managed drink-serving Gazebo scenario.

Two-room patrol (kitchen ↔ living_room) with Awareness Manager integration.

During kitchen dwell (default 30 s):
  - AM schedules kitchen items by E×A priority every observation_interval seconds
  - Drinks (A=0.50 via F1) consistently win the budget=2 slots over distractors (A=0.125)
  - ReactiveBaseline (hop_distance=1) round-robins all 5 items without epistemic priority

On first living_room arrival after visiting the kitchen:
  - Trigger fires once: robot reports each drink's epistemic state
  - E < 0.5 → "AVAILABLE"  (AM observed it reliably during patrol)
  - E ≥ 0.5 → "UNCERTAIN"  (reactive may have missed it while cycling distractors)

Parameters:
    budget               int    2                  Max items scheduled per observation interval
    observation_interval float  15.0               Seconds between auto-observations
    dwell_time           float  30.0               Seconds to stay in each room
    start_delay          float  38.0               Seconds to wait for Nav2 init
    alpha                float  0.5                Spreading activation decay (F1)
    strategy             str    awareness_manager  'awareness_manager' or 'reactive'
"""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.baselines.reactive import ReactiveBaseline
from awareness_manager.concept import InstanceConcept
from awareness_manager.feature_config import FeatureConfig
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.gazebo.drink_serving_gazebo import (
    DRINK_INSTANCE_IDS,
    DISTRACTOR_INSTANCE_IDS,
    ITEM_DECAY_RATE,
    build_drink_serving_kb,
    get_room_for_pose,
)


_WAYPOINTS: dict[str, tuple[float, float]] = {
    'kitchen':     ( 0.0,  0.0),
    'living_room': (-8.0,  0.35),
}
_PATROL_SEQUENCE = ['kitchen', 'living_room']
_SPAWN_POS: tuple[float, float] = (1.0527, 0.5096)

_RETRY_DELAY  = 5.0
_MAX_RETRIES  = 30
_E_THRESHOLD  = 0.5   # below this the robot considers a drink "observed recently enough"
_ZONE_CONFIRM = 2     # consecutive AMCL callbacks in the same zone to confirm transition


class DrinkServingNode(Node):

    def __init__(self) -> None:
        super().__init__('drink_serving')

        self.declare_parameter('budget',               2)
        self.declare_parameter('observation_interval', 15.0)
        self.declare_parameter('dwell_time',           5.0)
        self.declare_parameter('start_delay',          38.0)
        self.declare_parameter('alpha',                0.5)
        self.declare_parameter('strategy',             'awareness_manager')

        budget               = self.get_parameter('budget').value
        obs_interval         = self.get_parameter('observation_interval').value
        self._dwell_time     = self.get_parameter('dwell_time').value
        self._start_delay    = self.get_parameter('start_delay').value
        alpha                = self.get_parameter('alpha').value
        strategy             = self.get_parameter('strategy').value

        # ── Knowledge bases ──────────────────────────────────────────────
        kb        = build_drink_serving_kb()
        self._ikb = InstanceKnowledgeBase()

        # ── Attention strategy ───────────────────────────────────────────
        if strategy == 'reactive':
            self._am: AwarenessManager | ReactiveBaseline = ReactiveBaseline(
                kb=kb,
                goal_id='serve_drinks',
                budget=budget,
                observation_interval=obs_interval,
                ikb=self._ikb,
                hop_distance=1,
            )
        else:
            self._am = AwarenessManager(
                kb=kb,
                goal_id='serve_drinks',
                alpha=alpha,
                budget=budget,
                observation_interval=obs_interval,
                instance_kb=self._ikb,
                feature_config=FeatureConfig(),
            )

        self._strategy    = strategy
        self._budget      = budget
        self._obs_interval = obs_interval

        # ── State ────────────────────────────────────────────────────────
        self._current_zone:    str | None = None
        self._zone_candidate:  str | None = None
        self._zone_hold_count: int        = 0
        self._instances_added: bool       = False
        self._kitchen_visited: bool       = False
        self._trigger_fired:   bool       = False
        self._last_obs_time:   float      = 0.0
        self._current_pos:     tuple[float, float] = _SPAWN_POS
        # Track which instances were observed at least once during kitchen patrol.
        # Used as the binary pass/fail criterion at trigger time (not E, which
        # degrades during travel because F3 drift > refresh for small δ).
        self._observed_in_kitchen: set[str] = set()

        # ── ROS publishers / subscribers ─────────────────────────────────
        self._pub_state    = self.create_publisher(String, 'awareness/state',    10)
        self._pub_schedule = self.create_publisher(String, 'awareness/schedule', 10)
        self._pub_goal     = self.create_publisher(String, 'awareness/goal',     10)

        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl_pose, 10,
        )

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── 10 Hz AM tick timer ──────────────────────────────────────────
        self.create_timer(0.1, self._tick_cb)

        self.get_logger().info(
            f'DrinkServingNode ready — strategy={strategy}  budget={budget}  '
            f'obs_interval={obs_interval}s  dwell={self._dwell_time}s'
        )

        # ── Navigation patrol runs in a background thread ─────────────────
        threading.Thread(target=self._run_patrol, daemon=True).start()

    # ------------------------------------------------------------------
    # AM tick — runs at 10 Hz on the ROS spin thread
    # ------------------------------------------------------------------

    def _tick_cb(self) -> None:
        schedule = self._am.tick(dt=0.1)
        self._auto_observe(schedule)
        self._publish_state(schedule)

    def _auto_observe(self, schedule: list[str]) -> None:
        if self._current_zone != 'kitchen':
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_obs_time < self._obs_interval or not schedule:
            return
        observed = schedule[:self._budget]
        for cid in observed:
            self._am.observe(cid)
            self._observed_in_kitchen.add(cid)
        self._last_obs_time = now
        self.get_logger().info(
            f'[OBS/{self._strategy[:3].upper()}] {observed}'
            + (f'  beer_seen={("kitchen_beer" in self._observed_in_kitchen)}'
               f'  wine_seen={("kitchen_wine" in self._observed_in_kitchen)}'
               if self._instances_added else '')
        )

    def _inst_e(self, cid: str) -> float:
        inst = self._ikb._instances.get(cid)
        return inst.epistemic_error if inst is not None else -1.0

    # ------------------------------------------------------------------
    # Zone tracking — AMCL subscription, spin thread
    # ------------------------------------------------------------------

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        zone = get_room_for_pose(x, y)
        if zone is None:
            self._zone_hold_count = 0
            self._current_zone = None  # robot left all known zones (transit)
            return
        if zone == self._zone_candidate:
            self._zone_hold_count += 1
        else:
            self._zone_candidate  = zone
            self._zone_hold_count = 1
        if self._zone_hold_count >= _ZONE_CONFIRM and zone != self._current_zone:
            self._current_zone = zone
            self._on_zone_entry(zone)

    def _on_zone_entry(self, zone: str) -> None:
        self.get_logger().info(f'[ZONE] Entered {zone!r}')
        if zone == 'kitchen' and not self._instances_added:
            self._discover_kitchen_instances()
        elif zone == 'living_room' and self._kitchen_visited and not self._trigger_fired:
            self._fire_trigger()

    def _discover_kitchen_instances(self) -> None:
        for cid in DRINK_INSTANCE_IDS:
            self._ikb.add_instance(InstanceConcept(
                concept_id=cid,
                concept_type='object',
                decay_rate=ITEM_DECAY_RATE,
                class_id='drink_class',
                epistemic_error=1.0,
                observation_cost=1.0,
            ))
        for cid in DISTRACTOR_INSTANCE_IDS:
            self._ikb.add_instance(InstanceConcept(
                concept_id=cid,
                concept_type='object',
                decay_rate=ITEM_DECAY_RATE,
                class_id='distractor',
                epistemic_error=1.0,
                observation_cost=1.0,
            ))

        # Reactive must rebuild its active list to include the new instances.
        if isinstance(self._am, ReactiveBaseline):
            self._am._rebuild_active_set()
        elif isinstance(self._am, AwarenessManager):
            self._am.set_goal(self._am.goal_id)

        self._instances_added = True
        self._kitchen_visited = True
        self.get_logger().info(
            f'[DISCOVER] {len(DRINK_INSTANCE_IDS)} drinks + '
            f'{len(DISTRACTOR_INSTANCE_IDS)} distractors added to instance KB'
        )

    # ------------------------------------------------------------------
    # Trigger — fires once on first living_room arrival after kitchen
    # ------------------------------------------------------------------

    def _fire_trigger(self) -> None:
        self._trigger_fired = True
        sep = '=' * 60
        self.get_logger().info(sep)
        self.get_logger().info(f'DRINK AVAILABILITY QUERY  [{self._strategy.upper()}]')
        for drink_id in DRINK_INSTANCE_IDS:
            inst = self._ikb._instances.get(drink_id)
            if inst is None:
                self.get_logger().warn(f'  {drink_id}: never discovered → UNCERTAIN')
                continue
            e = inst.epistemic_error
            # Answer based on observation history: was this drink ever scheduled
            # and observed during the kitchen patrol?  E at trigger time is not
            # reliable because F3 drift accumulates during travel regardless of
            # how well the AM tracked the item.
            observed = drink_id in self._observed_in_kitchen
            answer = 'AVAILABLE' if observed else 'UNCERTAIN'
            self.get_logger().info(
                f'  {drink_id}: observed_during_patrol={observed}  E_now={e:.3f} → {answer}'
            )
        self.get_logger().info(sep)

    # ------------------------------------------------------------------
    # State publishing
    # ------------------------------------------------------------------

    def _publish_state(self, schedule: list[str]) -> None:
        self._pub_schedule.publish(String(data=json.dumps(schedule)))
        self._pub_goal.publish(String(data=self._am.goal_id))

        attn = self._am.attention()
        state: dict[str, dict] = {}

        for cid in self._am.kb.concept_ids():
            c = self._am.kb.get_concept(cid)
            state[cid] = {
                'E': round(c.epistemic_error, 4),
                'A': round(attn.get(cid, 0.0), 4),
            }

        for iid in self._ikb.instance_ids():
            inst = self._ikb.get_instance(iid)
            state[iid] = {
                'E':           round(inst.epistemic_error, 4),
                'A':           round(attn.get(iid, 0.0), 4),
                'in_schedule': iid in schedule,
            }

        self._pub_state.publish(String(data=json.dumps(state)))

    # ------------------------------------------------------------------
    # Navigation patrol loop — background thread
    # ------------------------------------------------------------------

    def _run_patrol(self) -> None:
        time.sleep(self._start_delay)

        self.get_logger().info('Waiting for navigate_to_pose action server…')
        if not self._nav_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error('navigate_to_pose not available — patrol aborting.')
            return
        self.get_logger().info('Nav2 up. Starting patrol.')

        idx = 0
        while rclpy.ok():
            room = _PATROL_SEQUENCE[idx % len(_PATROL_SEQUENCE)]
            idx += 1

            x, y = _WAYPOINTS[room]
            self.get_logger().info(f'[PATROL] → {room!r}  ({x:.2f}, {y:.2f})')

            succeeded = False
            for attempt in range(_MAX_RETRIES):
                status = self._navigate_blocking(self._make_pose(x, y))
                if status == GoalStatus.STATUS_SUCCEEDED:
                    succeeded = True
                    break
                self.get_logger().warn(
                    f'[RETRY {attempt + 1}/{_MAX_RETRIES}] {room!r} '
                    f'status={status} — waiting {_RETRY_DELAY:.0f}s…'
                )
                time.sleep(_RETRY_DELAY)

            if succeeded:
                self._current_pos = (x, y)
                self.get_logger().info(
                    f'[ARRIVED] {room!r} — dwelling {self._dwell_time:.0f}s'
                )
                time.sleep(self._dwell_time)
            else:
                self.get_logger().error(f'[GIVE UP] {room!r} — skipping.')

    def _navigate_blocking(self, pose: PoseStamped) -> int:
        done = threading.Event()
        status_holder: list[int] = [GoalStatus.STATUS_ABORTED]

        def on_accepted(future) -> None:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn('Goal rejected by Nav2.')
                done.set()
                return
            goal_handle.get_result_async().add_done_callback(on_result)

        def on_result(future) -> None:
            status_holder[0] = future.result().status
            done.set()

        self._nav_client.send_goal_async(
            NavigateToPose.Goal(pose=pose)
        ).add_done_callback(on_accepted)

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
    node = DrinkServingNode()
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
