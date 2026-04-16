"""
ROS2 node wrapper for the AwarenessManager.

Run:
    ros2 run awareness_manager awareness_node

Parameters (override with --ros-args -p name:=value):
    scenario             string   birdhouse       KB scenario to load
    goal_id              string   build_birdhouse Initial mission goal
    alpha                double   0.5             Spreading activation decay
    max_distance         double   4.0             Max weighted graph distance
    budget               int      3               Top-N schedule size
    tick_rate            double   10.0            Hz — am.tick() frequency
    observation_interval double   2.0             Seconds between auto-observations

Topics:
    awareness/set_goal            (sub)  std_msgs/String  — switch mission goal
    awareness/schedule            (pub)  std_msgs/String  — JSON schedule each tick
    awareness/state               (pub)  std_msgs/String  — JSON {cid:{E,A,P}} each tick
    awareness/observation_feedback(sub)  std_msgs/String  — report an observation was made
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb


def _load_scenario(name: str):
    """Return a KnowledgeBase for the named scenario."""
    if name == 'birdhouse':
        return build_birdhouse_kb()
    if name == 'pv_inspection':
        return build_pv_inspection_kb()
    raise ValueError(
        f"Unknown scenario '{name}'. Available: ['birdhouse', 'pv_inspection']"
    )


class AwarenessNode(Node):
    """
    ROS2 node that runs the AwarenessManager and exposes its interface via topics.

    Two independent timers drive the node:
        tick_timer  — advances simulation and publishes schedule + state
        obs_timer   — executes one scheduled observation per interval
    """

    def __init__(self) -> None:
        super().__init__('awareness_manager')

        # ---- Parameters ----
        self.declare_parameter('scenario',             'birdhouse')
        self.declare_parameter('goal_id',              'build_birdhouse')
        self.declare_parameter('alpha',                0.5)
        self.declare_parameter('max_distance',         4.0)
        self.declare_parameter('budget',               3)
        self.declare_parameter('tick_rate',            10.0)
        self.declare_parameter('observation_interval', 2.0)

        scenario    = self.get_parameter('scenario').value
        goal_id     = self.get_parameter('goal_id').value
        alpha       = self.get_parameter('alpha').value
        max_dist    = self.get_parameter('max_distance').value
        budget      = self.get_parameter('budget').value
        tick_rate   = self.get_parameter('tick_rate').value
        obs_interval = self.get_parameter('observation_interval').value

        # ---- Knowledge base + awareness manager ----
        kb = _load_scenario(scenario)
        self._am = AwarenessManager(
            kb,
            goal_id=goal_id,
            alpha=alpha,
            max_distance=max_dist,
            budget=budget,
            observation_interval=obs_interval,
        )
        self._schedule: list[str] = []
        self._tick_dt = 1.0 / tick_rate

        # ---- Publishers ----
        self._pub_schedule = self.create_publisher(String, 'awareness/schedule', 10)
        self._pub_state    = self.create_publisher(String, 'awareness/state',    10)

        # ---- Subscribers ----
        self.create_subscription(
            String, 'awareness/set_goal',
            self._on_set_goal, 10,
        )
        self.create_subscription(
            String, 'awareness/observation_feedback',
            self._on_observation_feedback, 10,
        )

        # ---- Timers ----
        self.create_timer(self._tick_dt,  self._tick_cb)
        self.create_timer(obs_interval,   self._obs_cb)

        self.get_logger().info(
            f"AwarenessNode started — scenario={scenario}  goal={goal_id}  "
            f"budget={budget}  tick_rate={tick_rate}Hz  obs_interval={obs_interval}s"
        )

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _tick_cb(self) -> None:
        """Advance simulation, publish schedule and state."""
        self._schedule = self._am.tick(dt=self._tick_dt)

        self._pub_schedule.publish(String(data=json.dumps(self._schedule)))

        state = {
            cid: {
                'E': round(self._am._kb.get_concept(cid).epistemic_error, 4),
                'A': round(self._am.attention().get(cid, 0.0), 4),
                'P': round(self._am.priorities().get(cid, 0.0), 6),
            }
            for cid in self._am._kb.concept_ids()
        }
        self._pub_state.publish(String(data=json.dumps(state)))

    def _obs_cb(self) -> None:
        """Execute one scheduled observation (Formula 3 refresh)."""
        if not self._schedule:
            return
        target = self._schedule[0]
        before = self._am._kb.get_concept(target).epistemic_error
        refresh = self._am.observe(target)
        after  = self._am._kb.get_concept(target).epistemic_error
        self.get_logger().info(
            f"[OBS] '{target}'  E: {before:.3f} → {after:.3f}  "
            f"(refresh={refresh:.3f})"
        )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_set_goal(self, msg: String) -> None:
        goal_id = msg.data.strip()
        try:
            self._am.set_goal(goal_id)
            self.get_logger().info(f"[GOAL] Mission changed → '{goal_id}'")
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_observation_feedback(self, msg: String) -> None:
        concept_id = msg.data.strip()
        try:
            before = self._am._kb.get_concept(concept_id).epistemic_error
            refresh = self._am.observe(concept_id)
            after  = self._am._kb.get_concept(concept_id).epistemic_error
            self.get_logger().info(
                f"[FEEDBACK] '{concept_id}'  E: {before:.3f} → {after:.3f}  "
                f"(refresh={refresh:.3f})"
            )
        except KeyError:
            self.get_logger().warn(f"Unknown concept in observation feedback: '{concept_id}'")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AwarenessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
