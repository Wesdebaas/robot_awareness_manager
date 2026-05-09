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
    awareness/observation_feedback(sub)  std_msgs/String  — JSON {"concept_id": "x"}
    awareness/violations          (pub)  std_msgs/String  — JSON list of violated concept IDs
    awareness/violations_feedback (sub)  std_msgs/String  — JSON {"concept_id":"x","observed_value":0.7}

Services:
    awareness/query_concept   awareness_manager_msgs/srv/QueryConcept
        Request:  concept_id (string)
        Response: concept_id, epistemic_error, attention, priority, found

    awareness/set_attention   awareness_manager_msgs/srv/SetAttention
        Request:  concept_id (string), attention_override (float64)
        Response: success (bool), message (string)
        Effect:   replaces the concept's computed attention for one tick

Actions:
    awareness/queue_goal      awareness_manager_msgs/action/QueueGoal
        Goal:     goal_id (string), eta (float64), level ('global'|'phase'|'task')
        Feedback: goal_id, current_eta, attention_boost (published every tick)
        Result:   promoted (bool), promoted_goal_id (string)
        Effect:   calls am.queue_goal() and streams progress until promotion
"""

import json

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb
from awareness_manager_msgs.srv import QueryConcept, SetAttention
from awareness_manager_msgs.action import QueueGoal


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
    ROS2 node that runs the AwarenessManager and exposes its full interface.

    Two independent timers drive the node:
        tick_timer  - advances simulation, publishes schedule + state,
                      and sends feedback to all active QueueGoal action handles
        obs_timer   - executes one scheduled observation per interval

    Services (synchronous, respond immediately):
        query_concept  - returns current E / A / P for any concept
        set_attention  - injects a one-tick attention override

    Action server (asynchronous, long-running):
        queue_goal     - queues a future goal and streams tick-by-tick feedback
                         until the goal is promoted (ETA → 0)
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

        scenario     = self.get_parameter('scenario').value
        goal_id      = self.get_parameter('goal_id').value
        alpha        = self.get_parameter('alpha').value
        max_dist     = self.get_parameter('max_distance').value
        budget       = self.get_parameter('budget').value
        tick_rate    = self.get_parameter('tick_rate').value
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

        # Active QueueGoal action handles: goal_id → (handle, initial_eta)
        # Supports at most one pending handle per goal_id (last one wins).
        self._queued_handles: dict[str, tuple] = {}

        # Reentrant callback group so the action server callbacks and timers
        # can execute concurrently in the multi-threaded executor.
        self._cb_group = ReentrantCallbackGroup()

        # ---- Publishers ----
        self._pub_schedule   = self.create_publisher(String, 'awareness/schedule',   10)
        self._pub_state      = self.create_publisher(String, 'awareness/state',      10)
        self._pub_violations = self.create_publisher(String, 'awareness/violations', 10)

        # ---- Subscribers ----
        self.create_subscription(
            String, 'awareness/set_goal',
            self._on_set_goal, 10,
        )
        self.create_subscription(
            String, 'awareness/observation_feedback',
            self._on_observation_feedback, 10,
        )
        self.create_subscription(
            String, 'awareness/violations_feedback',
            self._on_violations_feedback, 10,
        )

        # ---- Services ----
        self.create_service(
            QueryConcept, 'awareness/query_concept',
            self._handle_query_concept,
            callback_group=self._cb_group,
        )
        self.create_service(
            SetAttention, 'awareness/set_attention',
            self._handle_set_attention,
            callback_group=self._cb_group,
        )

        # ---- Action server ----
        self._action_server = ActionServer(
            self,
            QueueGoal,
            'awareness/queue_goal',
            goal_callback=self._action_goal_callback,
            cancel_callback=self._action_cancel_callback,
            execute_callback=self._action_execute_callback,
            callback_group=self._cb_group,
        )

        # ---- Timers ----
        self.create_timer(self._tick_dt, self._tick_cb,
                          callback_group=self._cb_group)
        self.create_timer(obs_interval,  self._obs_cb,
                          callback_group=self._cb_group)

        self.get_logger().info(
            f"AwarenessNode started - scenario={scenario}  goal={goal_id}  "
            f"budget={budget}  tick_rate={tick_rate}Hz  obs_interval={obs_interval}s"
        )

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _tick_cb(self) -> None:
        """Advance simulation, publish schedule + state, push action feedback."""
        # Track which goals are in the queue before and after the tick so we
        # can detect promotions without modifying AwarenessManager.
        queue_before = {entry[0]: entry[1] for entry in self._am.mission_queue}

        self._schedule = self._am.tick(dt=self._tick_dt)

        queue_after = {entry[0]: entry[1] for entry in self._am.mission_queue}

        # Publish schedule
        self._pub_schedule.publish(String(data=json.dumps(self._schedule)))

        # Publish state (class-level concepts)
        state = {
            cid: {
                'E': round(self._am._kb.get_concept(cid).epistemic_error, 4),
                'A': round(self._am.attention().get(cid, 0.0), 4),
                'P': round(self._am.priorities().get(cid, 0.0), 6),
            }
            for cid in self._am._kb.concept_ids()
        }
        self._pub_state.publish(String(data=json.dumps(state)))

        # Detect and publish prediction-error violations
        violations = [
            cid for cid in self._am._kb.concept_ids()
            if self._am._kb.get_concept(cid).prediction_error > 0.0
        ]
        self._pub_violations.publish(String(data=json.dumps(violations)))

        # Send action feedback for all pending QueueGoal handles
        promoted_ids = set(queue_before) - set(queue_after)
        for goal_id, (handle, initial_eta) in list(self._queued_handles.items()):
            if not handle.is_active:
                del self._queued_handles[goal_id]
                continue

            if goal_id in promoted_ids:
                # Goal promoted - send result and finish
                result = QueueGoal.Result()
                result.promoted = True
                result.promoted_goal_id = goal_id
                handle.succeed(result)
                del self._queued_handles[goal_id]
                self.get_logger().info(
                    f"[ACTION] QueueGoal '{goal_id}' promoted → result sent"
                )
            elif goal_id in queue_after:
                # Still pending - send feedback
                current_eta = queue_after[goal_id]
                # attention_boost = how much the queued goal is currently contributing
                # to the concept with the highest attention under the queued goal
                am_attn = self._am.attention()
                boost = max(
                    (v for k, v in am_attn.items() if k != self._am.goal_id),
                    default=0.0,
                )
                feedback = QueueGoal.Feedback()
                feedback.goal_id       = goal_id
                feedback.current_eta   = current_eta
                feedback.attention_boost = round(boost, 4)
                handle.publish_feedback(feedback)

    def _obs_cb(self) -> None:
        """Execute one scheduled observation (Formula 3 refresh)."""
        if not self._schedule:
            return
        target = self._schedule[0]
        try:
            before  = self._am._kb.get_concept(target).epistemic_error
        except Exception:
            before = 0.0
        refresh = self._am.observe(target)
        try:
            after = self._am._kb.get_concept(target).epistemic_error
        except Exception:
            after = 0.0
        self.get_logger().info(
            f"[OBS] '{target}'  E: {before:.3f} → {after:.3f}  "
            f"(refresh={refresh:.3f})"
        )

    # ------------------------------------------------------------------
    # Topic subscription callbacks
    # ------------------------------------------------------------------

    def _on_set_goal(self, msg: String) -> None:
        goal_id = msg.data.strip()
        try:
            self._am.set_goal(goal_id)
            self.get_logger().info(f"[GOAL] Mission changed → '{goal_id}'")
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_observation_feedback(self, msg: String) -> None:
        """
        Handle a plain observation report: JSON {"concept_id": "x"}.
        Applies Formula 3 refresh without prediction-error checking.
        """
        try:
            data = json.loads(msg.data)
            concept_id = data.get('concept_id', msg.data.strip())
        except (json.JSONDecodeError, AttributeError):
            concept_id = msg.data.strip()

        try:
            refresh = self._am.observe(concept_id)
            self.get_logger().info(
                f"[FEEDBACK] '{concept_id}'  refresh={refresh:.3f}"
            )
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_violations_feedback(self, msg: String) -> None:
        """
        Handle a valued observation (prediction error feedback):
        JSON {"concept_id": "x", "observed_value": 0.7}.
        Calls observe_with_feedback() and logs any violation.
        """
        try:
            data = json.loads(msg.data)
            concept_id     = data['concept_id']
            observed_value = float(data['observed_value'])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(
                f"[VIOLATIONS_FEEDBACK] Malformed message: {exc}  raw='{msg.data}'"
            )
            return

        try:
            refresh, violated = self._am.observe_with_feedback(
                concept_id, observed_value
            )
            if violated:
                self.get_logger().warn(
                    f"[VIOLATION] '{concept_id}' violated - "
                    f"observed={observed_value:.3f}  "
                    f"predicted={self._am._kb.get_concept(concept_id).predicted_value:.3f}  "
                    f"error={self._am._kb.get_concept(concept_id).prediction_error:.3f}"
                )
            else:
                self.get_logger().info(
                    f"[FEEDBACK+] '{concept_id}'  refresh={refresh:.3f}  no_violation"
                )
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def _handle_query_concept(
        self,
        request: QueryConcept.Request,
        response: QueryConcept.Response,
    ) -> QueryConcept.Response:
        """
        awareness/query_concept - synchronous on-demand concept lookup.

        Checks the class KB first, then the instance KB. Returns current
        epistemic error, attention, and priority for the requested concept.
        """
        cid = request.concept_id.strip()
        attn  = self._am.attention()
        prio  = self._am.priorities()

        if cid in self._am._kb.concept_ids():
            concept = self._am._kb.get_concept(cid)
            response.concept_id      = cid
            response.epistemic_error = concept.epistemic_error
            response.attention       = attn.get(cid, 0.0)
            response.priority        = prio.get(cid, 0.0)
            response.found           = True
        elif (self._am.instance_kb is not None
              and cid in self._am.instance_kb.instance_ids()):
            inst = self._am.instance_kb.get_instance(cid)
            response.concept_id      = cid
            response.epistemic_error = inst.epistemic_error
            response.attention       = attn.get(cid, 0.0)
            response.priority        = prio.get(cid, 0.0)
            response.found           = True
        else:
            response.concept_id      = cid
            response.epistemic_error = 0.0
            response.attention       = 0.0
            response.priority        = 0.0
            response.found           = False
            self.get_logger().warn(
                f"[QUERY] Unknown concept '{cid}'"
            )

        self.get_logger().info(
            f"[QUERY] '{cid}'  found={response.found}  "
            f"E={response.epistemic_error:.3f}  "
            f"A={response.attention:.3f}  "
            f"P={response.priority:.4f}"
        )
        return response

    def _handle_set_attention(
        self,
        request: SetAttention.Request,
        response: SetAttention.Response,
    ) -> SetAttention.Response:
        """
        awareness/set_attention - inject a one-tick attention override.

        The override replaces the normally-computed attention for the given
        concept on the next tick, then is cleared. Useful for debugging and
        external forcing of attention allocation.
        """
        cid   = request.concept_id.strip()
        value = request.attention_override

        try:
            self._am.override_attention(cid, value)
            response.success = True
            response.message = (
                f"Attention override set for '{cid}' → {value:.3f} (one tick)"
            )
            self.get_logger().info(
                f"[SET_ATTN] '{cid}' override={value:.3f}"
            )
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().warn(f"[SET_ATTN] {exc}")

        return response

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _action_goal_callback(self, goal_request) -> GoalResponse:
        """Validate and accept a QueueGoal action goal."""
        goal_id = goal_request.goal_id.strip()
        eta     = goal_request.eta
        level   = goal_request.level.strip() or 'task'

        if goal_id not in self._am._kb.concept_ids():
            self.get_logger().warn(
                f"[ACTION] Rejected QueueGoal: unknown goal '{goal_id}'"
            )
            return GoalResponse.REJECT

        if eta <= 0.0:
            self.get_logger().warn(
                f"[ACTION] Rejected QueueGoal: eta must be > 0, got {eta}"
            )
            return GoalResponse.REJECT

        from awareness_manager.awareness_manager import _LAMBDA_BY_LEVEL
        if level not in _LAMBDA_BY_LEVEL:
            self.get_logger().warn(
                f"[ACTION] Rejected QueueGoal: unknown level '{level}'"
            )
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, goal_handle) -> CancelResponse:
        """Always accept cancellation of a pending QueueGoal."""
        return CancelResponse.ACCEPT

    async def _action_execute_callback(self, goal_handle) -> QueueGoal.Result:
        """
        Queue the goal in the AwarenessManager and return when promoted.

        Feedback is NOT published here - it is published each tick from
        _tick_cb() to keep feedback cadence aligned with the simulation clock.
        The execute callback simply blocks until _tick_cb() marks the handle
        as succeeded (promotion) or the client cancels.
        """
        import asyncio

        goal_id = goal_handle.request.goal_id.strip()
        eta     = goal_handle.request.eta
        level   = goal_handle.request.level.strip() or 'task'

        try:
            self._am.queue_goal(goal_id, eta, level)
        except ValueError as exc:
            self.get_logger().error(f"[ACTION] queue_goal failed: {exc}")
            goal_handle.abort()
            result = QueueGoal.Result()
            result.promoted         = False
            result.promoted_goal_id = ''
            return result

        # Register handle so _tick_cb can push feedback and detect promotion
        self._queued_handles[goal_id] = (goal_handle, eta)

        self.get_logger().info(
            f"[ACTION] QueueGoal accepted: '{goal_id}'  eta={eta}s  level='{level}'"
        )

        # Await until the handle transitions out of EXECUTING state
        while goal_handle.is_active:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                # Remove from tracking and remove from AM queue
                self._queued_handles.pop(goal_id, None)
                self.get_logger().info(
                    f"[ACTION] QueueGoal '{goal_id}' cancelled by client"
                )
                result = QueueGoal.Result()
                result.promoted         = False
                result.promoted_goal_id = ''
                return result
            await asyncio.sleep(self._tick_dt)

        # _tick_cb already called handle.succeed() - just return the result
        result = QueueGoal.Result()
        result.promoted         = True
        result.promoted_goal_id = goal_id
        return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AwarenessNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
