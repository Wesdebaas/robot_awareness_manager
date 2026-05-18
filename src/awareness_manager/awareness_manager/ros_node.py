"""
ROS2 node wrapper for the AwarenessManager.

Run:
    ros2 run awareness_manager awareness_node

Parameters (override with --ros-args -p name:=value):
    scenario             string   social_serving  KB scenario to load
    goal_id              string   serve_people_drinks  Initial mission goal
    alpha                double   0.5             Spreading activation decay
    max_distance         double   4.0             Max weighted graph distance
    budget               int      3               Top-N schedule size
    tick_rate            double   10.0            Hz — am.tick() frequency
    observation_interval double   2.0             Seconds between auto-observations
    memory_budget        int      -1              F4 budget B; -1 = disabled (uses max_distance)
    instance_relational_weight double 0.3         Instance relational boost scale
    suspected_absent_priority_scale double 0.5   Priority multiplier for SUSPECTED_ABSENT instances
    certainty_threshold  double   0.0             Probabilistic forgetting gate (0 = disabled)
    f1..f6               bool     True            Enable/disable individual grounding formulas

Topics:
    awareness/set_goal             (sub)  std_msgs/String  — switch mission goal (concept_id)
    awareness/robot_pos            (sub)  std_msgs/String  — update robot position (concept_id)
    awareness/schedule             (pub)  std_msgs/String  — JSON schedule each tick
    awareness/state                (pub)  std_msgs/String  — JSON {cid:{E,A,P}} each tick (class + instances)
    awareness/observation_feedback (sub)  std_msgs/String  — JSON {"concept_id": "x"}
    awareness/violations           (pub)  std_msgs/String  — JSON list of violated concept IDs
    awareness/violations_feedback  (sub)  std_msgs/String  — JSON {"concept_id":"x","observed_value":0.7}
    awareness/add_instance         (sub)  std_msgs/String  — JSON {concept_id, concept_type, decay_rate,
                                                              class_id, extra_class_ids?, observation_cost?, zone?}
    awareness/mark_suspected_absent(sub)  std_msgs/String  — JSON {"instance_id": "x"}
    awareness/confirm_present      (sub)  std_msgs/String  — JSON {"instance_id": "x"}
    awareness/confirm_absent       (sub)  std_msgs/String  — JSON {"instance_id": "x"}
    awareness/create_serve_goal    (sub)  std_msgs/String  — JSON {"person_id":"x","drink_class":"y"}
                                                             Creates goal node + switches to it immediately.

Services:
    awareness/query_concept   awareness_manager_msgs/srv/QueryConcept
        Request:  concept_id (string)
        Response: concept_id, epistemic_error, attention, priority, found

    awareness/set_attention   awareness_manager_msgs/srv/SetAttention
        Request:  concept_id (string), attention_override (float64)
        Response: success (bool), message (string)

Actions:
    awareness/queue_goal      awareness_manager_msgs/action/QueueGoal
        Goal:     goal_id (string), eta (float64), level ('global'|'phase'|'task')
        Feedback: goal_id, current_eta, attention_boost (published every tick)
        Result:   promoted (bool), promoted_goal_id (string)
"""

import json

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from awareness_manager.awareness_manager import AwarenessManager, _LAMBDA_BY_LEVEL
from awareness_manager.concept import InstanceConcept, PresenceState
from awareness_manager.feature_config import FeatureConfig
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.pv_inspection import (
    build_pv_inspection_kb,
    build_pv_inspection_instance_kb,
)
from awareness_manager.scenarios.social_serving import (
    ZONE_TRAVEL_TIMES,
    build_social_serving_instance_kb,
    build_social_serving_kb,
    create_serve_goal,
    load_zone_assignment,
    preferred_drinks_for,
)
from awareness_manager_msgs.action import QueueGoal
from awareness_manager_msgs.srv import QueryConcept, SetAttention


# Social serving controller constants (mirror run_dashboard.py)
_SS_DRINK_CLASSES  = {'juice', 'cola', 'beer', 'wine', 'champagne'}
_SS_DISAPPEAR_RATE = 0.003       # per drink per second
_SS_FINISH_RATE    = 1 / 40.0   # person finishes drink ~every 40 s
_SS_SERVE_DELAY    = 4.0        # simulated seconds to fetch + deliver
_SS_SCAN_PERIOD    = 3.0        # seconds between person scans in MONITORING


def _load_scenario(
    name: str,
) -> tuple[KnowledgeBase, InstanceKnowledgeBase | None, dict[str, str], dict[tuple[str, str], float]]:
    """Return (kb, instance_kb, zone_assignment, zone_travel_times) for the named scenario."""
    if name == 'pv_inspection':
        return (
            build_pv_inspection_kb(),
            build_pv_inspection_instance_kb(),
            {},
            {},
        )
    if name == 'social_serving':
        return (
            build_social_serving_kb(),
            build_social_serving_instance_kb(),
            load_zone_assignment(),
            ZONE_TRAVEL_TIMES,
        )
    raise ValueError(
        f"Unknown scenario '{name}'. Available: ['pv_inspection', 'social_serving']"
    )


class AwarenessNode(Node):
    """
    ROS2 node that runs the AwarenessManager and exposes its full interface.

    Two independent timers drive the node:
        tick_timer - advances simulation, publishes schedule + state,
                     and sends feedback to all active QueueGoal action handles.
        obs_timer  - executes one scheduled observation per interval.
    """

    def __init__(self) -> None:
        super().__init__('awareness_manager')

        # ---- Parameters ----
        self.declare_parameter('scenario',                       'social_serving')
        self.declare_parameter('goal_id',                        'serve_people_drinks')
        self.declare_parameter('alpha',                          0.5)
        self.declare_parameter('max_distance',                   4.0)
        self.declare_parameter('budget',                         3)
        self.declare_parameter('tick_rate',                      10.0)
        self.declare_parameter('observation_interval',           2.0)
        self.declare_parameter('memory_budget',                  -1)
        self.declare_parameter('instance_relational_weight',     0.3)
        self.declare_parameter('suspected_absent_priority_scale', 0.5)
        self.declare_parameter('certainty_threshold',            0.0)
        self.declare_parameter('f1', True)
        self.declare_parameter('f2', True)
        self.declare_parameter('f3', True)
        self.declare_parameter('f4', True)
        self.declare_parameter('f5', True)
        self.declare_parameter('f6', True)

        scenario     = self.get_parameter('scenario').value
        goal_id      = self.get_parameter('goal_id').value
        alpha        = self.get_parameter('alpha').value
        max_dist     = self.get_parameter('max_distance').value
        budget       = self.get_parameter('budget').value
        tick_rate    = self.get_parameter('tick_rate').value
        obs_interval = self.get_parameter('observation_interval').value
        mem_budget   = self.get_parameter('memory_budget').value
        irw          = self.get_parameter('instance_relational_weight').value
        sa_scale     = self.get_parameter('suspected_absent_priority_scale').value
        cert_thresh  = self.get_parameter('certainty_threshold').value

        feature_config = FeatureConfig(
            use_f1_spreading_activation=self.get_parameter('f1').value,
            use_f2_anticipatory_horizon=self.get_parameter('f2').value,
            use_f3_utility_saturation  =self.get_parameter('f3').value,
            use_f4_memory_budget       =self.get_parameter('f4').value,
            use_f5_epistemic_drift     =self.get_parameter('f5').value,
            use_f6_observation_cost    =self.get_parameter('f6').value,
        )

        # ---- Knowledge base + awareness manager ----
        kb, instance_kb, zone_assignment, zone_travel_times = _load_scenario(scenario)
        self._am = AwarenessManager(
            kb,
            goal_id=goal_id,
            alpha=alpha,
            max_distance=max_dist,
            budget=budget,
            observation_interval=obs_interval,
            memory_budget=mem_budget if mem_budget >= 0 else None,
            instance_kb=instance_kb,
            instance_relational_weight=irw,
            suspected_absent_priority_scale=sa_scale,
            certainty_threshold=cert_thresh,
            feature_config=feature_config,
            zone_assignment=zone_assignment,
            zone_travel_times=zone_travel_times,
        )
        self._schedule: list[str] = []
        self._tick_dt = 1.0 / tick_rate
        self._robot_pos: str | None = None
        self._scenario = scenario

        # Social serving controller state machine (only active for that scenario)
        import random as _random
        self._controller_ctx: dict | None = None
        if scenario == 'social_serving':
            self._controller_ctx = {
                'initialized': False,
                'persons':     [],
                'drinks':      [],
                'gt_has_drink': {},
                'gt_drinks':    {},
                'state':        'monitoring',
                'target':       None,
                'check_timer':  0.0,
                'serve_timer':  0.0,
                'rng':          _random.Random(42),
            }

        # Active QueueGoal action handles: goal_id → (handle, initial_eta)
        self._queued_handles: dict[str, tuple] = {}
        self._cb_group = ReentrantCallbackGroup()

        # ---- Publishers ----
        self._pub_schedule          = self.create_publisher(String, 'awareness/schedule',          10)
        self._pub_state             = self.create_publisher(String, 'awareness/state',             10)
        self._pub_violations        = self.create_publisher(String, 'awareness/violations',        10)
        self._pub_goal              = self.create_publisher(String, 'awareness/goal',              10)
        self._pub_controller_state  = self.create_publisher(String, 'awareness/controller_state', 10)

        # ---- Subscribers ----
        self.create_subscription(String, 'awareness/set_goal',              self._on_set_goal,              10)
        self.create_subscription(String, 'awareness/robot_pos',             self._on_robot_pos,             10)
        self.create_subscription(String, 'awareness/observation_feedback',  self._on_observation_feedback,  10)
        self.create_subscription(String, 'awareness/violations_feedback',   self._on_violations_feedback,   10)
        self.create_subscription(String, 'awareness/add_instance',          self._on_add_instance,          10)
        self.create_subscription(String, 'awareness/mark_suspected_absent', self._on_mark_suspected_absent, 10)
        self.create_subscription(String, 'awareness/confirm_present',       self._on_confirm_present,       10)
        self.create_subscription(String, 'awareness/confirm_absent',        self._on_confirm_absent,        10)
        self.create_subscription(String, 'awareness/create_serve_goal',     self._on_create_serve_goal,     10)

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
        self.create_timer(self._tick_dt, self._tick_cb, callback_group=self._cb_group)
        self.create_timer(obs_interval,  self._obs_cb,  callback_group=self._cb_group)

        self.get_logger().info(
            f"AwarenessNode started — scenario={scenario}  goal={goal_id}  "
            f"budget={budget}  tick_rate={tick_rate}Hz  obs_interval={obs_interval}s  "
            f"instance_kb={'yes' if instance_kb else 'no'}  f6={feature_config.use_f6_observation_cost}"
        )

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _tick_cb(self) -> None:
        """Advance simulation, publish schedule + state, push action feedback."""
        queue_before = {entry[0]: entry[1] for entry in self._am.mission_queue}
        self._schedule = self._am.tick(dt=self._tick_dt, robot_pos=self._robot_pos)
        queue_after = {entry[0]: entry[1] for entry in self._am.mission_queue}

        self._pub_schedule.publish(String(data=json.dumps(self._schedule)))
        self._pub_goal.publish(String(data=self._am.goal_id))

        # State: class concepts + active instance concepts, with channel breakdown
        attn = self._am.attention()
        prio = self._am.priorities()
        ch   = self._am.attention_channels()
        state: dict = {}
        for cid in self._am._kb.concept_ids():
            cm = ch['mission'].get(cid, 0.0)
            ca = ch['anticipatory'].get(cid, 0.0)
            state[cid] = {
                'E':    round(self._am._kb.get_concept(cid).epistemic_error, 4),
                'A':    round(attn.get(cid, 0.0), 4),
                'P':    round(prio.get(cid, 0.0), 6),
                'ch_m': round(cm, 4),
                'ch_an': round(ca, 4),
                'ch_r': 0.0,
                'ch_s': round(ch['surprise'].get(cid, 0.0), 4),
            }
        if self._am.instance_kb is not None:
            for iid in self._am.instance_kb.active_instance_ids():
                inst = self._am.instance_kb.get_instance(iid)
                class_id = inst.class_id
                cm = ch['mission'].get(class_id, 0.0)
                ca = ch['anticipatory'].get(class_id, 0.0)
                state[iid] = {
                    'E':        round(inst.epistemic_error, 4),
                    'A':        round(attn.get(iid, 0.0), 4),
                    'P':        round(prio.get(iid, 0.0), 6),
                    'ch_m':     round(cm, 4),
                    'ch_an':    round(ca, 4),
                    'ch_r':     round(ch['relational'].get(iid, 0.0), 4),
                    'ch_s':     round(ch['surprise'].get(iid, 0.0), 4),
                    'presence': inst.presence_state.value,
                }
        self._pub_state.publish(String(data=json.dumps(state)))

        # Violations: class concepts + active instance concepts
        violations = [
            cid for cid in self._am._kb.concept_ids()
            if self._am._kb.get_concept(cid).prediction_error > 0.0
        ]
        if self._am.instance_kb is not None:
            violations += [
                iid for iid in self._am.instance_kb.active_instance_ids()
                if self._am.instance_kb.get_instance(iid).prediction_error > 0.0
            ]
        self._pub_violations.publish(String(data=json.dumps(violations)))

        # Social serving controller (only active for that scenario)
        if self._controller_ctx is not None:
            cs = self._run_controller(self._tick_dt)
            self._pub_controller_state.publish(String(data=json.dumps(cs)))

        # Action feedback and promotion detection
        promoted_ids = set(queue_before) - set(queue_after)
        for goal_id, (handle, initial_eta) in list(self._queued_handles.items()):
            if not handle.is_active:
                del self._queued_handles[goal_id]
                continue
            if goal_id in promoted_ids:
                result = QueueGoal.Result()
                result.promoted = True
                result.promoted_goal_id = goal_id
                handle.succeed(result)
                del self._queued_handles[goal_id]
                self.get_logger().info(f"[ACTION] QueueGoal '{goal_id}' promoted")
            elif goal_id in queue_after:
                current_eta = queue_after[goal_id]
                boost = max(
                    (v for k, v in attn.items() if k != self._am.goal_id),
                    default=0.0,
                )
                feedback = QueueGoal.Feedback()
                feedback.goal_id        = goal_id
                feedback.current_eta    = current_eta
                feedback.attention_boost = round(boost, 4)
                handle.publish_feedback(feedback)

    def _run_controller(self, dt: float) -> dict:
        """
        Tick the MONITORING/SERVING social-serving state machine.

        Mirrors the _make_ss_controller_hook logic from run_dashboard.py so the
        ROS node drives goal transitions autonomously without needing an external
        controller node.  Publishes result to awareness/controller_state.
        """
        ctx = self._controller_ctx
        ikb = self._am.instance_kb
        if ikb is None:
            return {'state': 'monitoring', 'target': None}

        if not ctx['initialized']:
            for iid in ikb.instance_ids():
                inst = ikb.get_instance(iid)
                if any(c in _SS_DRINK_CLASSES for c in inst.all_class_ids):
                    ctx['drinks'].append(iid)
                else:
                    ctx['persons'].append(iid)
            ctx['gt_has_drink'] = {p: True for p in ctx['persons']}
            ctx['gt_drinks']    = {d: True for d in ctx['drinks']}
            for pid in ctx['persons']:
                ikb.get_instance(pid).properties['has_drink'] = True
            ctx['initialized'] = True

        rng = ctx['rng']

        for pid in ctx['persons']:
            if rng.random() < _SS_FINISH_RATE * dt:
                ctx['gt_has_drink'][pid] = False

        for did in ctx['drinks']:
            if rng.random() < _SS_DISAPPEAR_RATE * dt:
                if ctx['gt_drinks'].get(did, True):
                    ctx['gt_drinks'][did] = False
                    inst = ikb.get_instance(did)
                    if inst.presence_state != PresenceState.CONFIRMED_ABSENT:
                        ikb.mark_suspected_absent(did)
                        ikb.confirm_absent(did)

        if ctx['state'] == 'monitoring':
            ctx['check_timer'] -= dt
            if ctx['check_timer'] <= 0.0:
                ctx['check_timer'] = _SS_SCAN_PERIOD
                for pid in ctx['persons']:
                    if not ctx['gt_has_drink'].get(pid, True):
                        dc = self._pick_ss_drink_class(ikb, ctx, pid)
                        if dc:
                            goal_id = create_serve_goal(self._am._kb, pid, dc)
                            try:
                                self._am.set_goal(goal_id)
                                ctx['state']       = 'serving'
                                ctx['target']      = pid
                                ctx['serve_timer'] = _SS_SERVE_DELAY
                                ikb.get_instance(pid).properties['has_drink'] = False
                                self.get_logger().info(f"[CTRL] SERVING → {pid} ({dc})")
                            except ValueError as exc:
                                self.get_logger().warn(f"[CTRL] set_goal failed: {exc}")
                            break

        elif ctx['state'] == 'serving':
            ctx['serve_timer'] -= dt
            if ctx['serve_timer'] <= 0.0:
                pid = ctx['target']
                dc = self._pick_ss_drink_class(ikb, ctx, pid) if pid else None
                if pid and dc:
                    ctx['gt_has_drink'][pid] = True
                    ikb.get_instance(pid).properties['has_drink'] = True
                self._am.set_goal('serve_people_drinks')
                ctx['state']       = 'monitoring'
                ctx['target']      = None
                ctx['check_timer'] = 1.0
                self.get_logger().info(f"[CTRL] MONITORING (served {pid})")

        return {'state': ctx['state'], 'target': ctx['target']}

    def _pick_ss_drink_class(self, ikb, ctx: dict, person_id: str) -> str | None:
        """Return the first available preferred drink class for person_id, or None."""
        inst = ikb.get_instance(person_id)
        for dc in preferred_drinks_for(inst.all_class_ids):
            active = [
                did for did in ikb.instances_of_class(dc)
                if (ikb.get_instance(did).presence_state != PresenceState.CONFIRMED_ABSENT
                    and ctx['gt_drinks'].get(did, True))
            ]
            if active:
                return dc
        return None

    def _obs_cb(self) -> None:
        """Execute one scheduled observation (Formula 3 refresh)."""
        if not self._schedule:
            return
        target = self._schedule[0]
        try:
            if target in self._am._kb.concept_ids():
                before = self._am._kb.get_concept(target).epistemic_error
            elif self._am.instance_kb and target in self._am.instance_kb.instance_ids():
                before = self._am.instance_kb.get_instance(target).epistemic_error
            else:
                self.get_logger().warn(f"[OBS] '{target}' not found in KB or instance KB")
                return
            refresh = self._am.observe(target)
            if target in self._am._kb.concept_ids():
                after = self._am._kb.get_concept(target).epistemic_error
            else:
                after = self._am.instance_kb.get_instance(target).epistemic_error
            self.get_logger().info(
                f"[OBS] '{target}'  E: {before:.3f} → {after:.3f}  (refresh={refresh:.3f})"
            )
        except ValueError as exc:
            self.get_logger().warn(f"[OBS] {exc}")

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

    def _on_robot_pos(self, msg: String) -> None:
        """Update current robot position (concept_id) used by F6 travel cost."""
        pos = msg.data.strip()
        self._robot_pos = pos if pos else None

    def _on_observation_feedback(self, msg: String) -> None:
        """Handle plain observation: JSON {"concept_id": "x"} or bare concept_id string."""
        try:
            data = json.loads(msg.data)
            concept_id = data.get('concept_id', msg.data.strip())
        except (json.JSONDecodeError, AttributeError):
            concept_id = msg.data.strip()
        try:
            refresh = self._am.observe(concept_id)
            self.get_logger().info(f"[FEEDBACK] '{concept_id}'  refresh={refresh:.3f}")
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_violations_feedback(self, msg: String) -> None:
        """Handle valued observation: JSON {"concept_id": "x", "observed_value": 0.7}."""
        try:
            data = json.loads(msg.data)
            concept_id     = data['concept_id']
            observed_value = float(data['observed_value'])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"[VIOLATIONS_FEEDBACK] Malformed: {exc}  raw='{msg.data}'")
            return
        try:
            refresh, violated = self._am.observe_with_feedback(concept_id, observed_value)
            if violated:
                self.get_logger().warn(
                    f"[VIOLATION] '{concept_id}' violated  observed={observed_value:.3f}"
                )
            else:
                self.get_logger().info(f"[FEEDBACK+] '{concept_id}'  refresh={refresh:.3f}  no_violation")
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_add_instance(self, msg: String) -> None:
        """
        Add a new instance at runtime (e.g. discovered by a detector node).

        JSON payload:
            {
              "concept_id":       "person_11",
              "concept_type":     "object",
              "decay_rate":       0.02,
              "class_id":         "person",
              "extra_class_ids":  ["VIP"],        // optional
              "observation_cost": 2.0,             // optional, default 1.0
              "zone":             "table_area"     // optional, for F6
            }
        """
        try:
            data = json.loads(msg.data)
            instance = InstanceConcept(
                concept_id      = data['concept_id'],
                concept_type    = data['concept_type'],
                decay_rate      = float(data['decay_rate']),
                class_id        = data['class_id'],
                extra_class_ids = data.get('extra_class_ids', []),
                observation_cost = float(data.get('observation_cost', 1.0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"[ADD_INSTANCE] Malformed: {exc}")
            return
        self._am.add_instance(instance)
        if 'zone' in data:
            self._am._zone_assignment[data['concept_id']] = data['zone']
        self.get_logger().info(
            f"[ADD_INSTANCE] '{instance.concept_id}' (class={instance.class_id})"
        )

    def _on_mark_suspected_absent(self, msg: String) -> None:
        """JSON {"instance_id": "x"} — transition instance to SUSPECTED_ABSENT."""
        try:
            iid = json.loads(msg.data).get('instance_id', msg.data.strip())
            self._am.mark_suspected_absent(iid)
            self.get_logger().info(f"[ABSENT?] '{iid}' → SUSPECTED_ABSENT")
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            self.get_logger().warn(str(exc))

    def _on_confirm_present(self, msg: String) -> None:
        """JSON {"instance_id": "x"} — restore instance to PRESENT."""
        try:
            iid = json.loads(msg.data).get('instance_id', msg.data.strip())
            self._am.confirm_present(iid)
            self.get_logger().info(f"[PRESENT] '{iid}' → PRESENT")
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            self.get_logger().warn(str(exc))

    def _on_confirm_absent(self, msg: String) -> None:
        """JSON {"instance_id": "x"} — mark instance as CONFIRMED_ABSENT."""
        try:
            iid = json.loads(msg.data).get('instance_id', msg.data.strip())
            self._am.confirm_absent(iid)
            self.get_logger().info(f"[ABSENT!] '{iid}' → CONFIRMED_ABSENT")
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            self.get_logger().warn(str(exc))

    def _on_create_serve_goal(self, msg: String) -> None:
        """
        Create a parameterised serve goal and switch to it immediately.

        JSON payload: {"person_id": "person_01", "drink_class": "beer"}

        Adds serve_<person_id> task node to the class KB (if absent) and
        adds an edge to drink_class. Then calls set_goal() automatically.
        The caller does not need to separately publish to awareness/set_goal.
        """
        try:
            data = json.loads(msg.data)
            person_id   = data['person_id']
            drink_class = data['drink_class']
        except (json.JSONDecodeError, KeyError) as exc:
            self.get_logger().warn(f"[SERVE_GOAL] Malformed: {exc}")
            return
        goal_id = create_serve_goal(self._am._kb, person_id, drink_class)
        try:
            self._am.set_goal(goal_id)
            self.get_logger().info(
                f"[SERVE_GOAL] '{goal_id}' created + active  (drink={drink_class})"
            )
        except ValueError as exc:
            self.get_logger().warn(f"[SERVE_GOAL] set_goal failed: {exc}")

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def _handle_query_concept(
        self,
        request: QueryConcept.Request,
        response: QueryConcept.Response,
    ) -> QueryConcept.Response:
        """awareness/query_concept — synchronous concept lookup (class or instance)."""
        cid  = request.concept_id.strip()
        attn = self._am.attention()
        prio = self._am.priorities()

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
            self.get_logger().warn(f"[QUERY] Unknown concept '{cid}'")

        self.get_logger().info(
            f"[QUERY] '{cid}'  found={response.found}  "
            f"E={response.epistemic_error:.3f}  A={response.attention:.3f}  P={response.priority:.4f}"
        )
        return response

    def _handle_set_attention(
        self,
        request: SetAttention.Request,
        response: SetAttention.Response,
    ) -> SetAttention.Response:
        """awareness/set_attention — inject a one-tick attention override."""
        cid   = request.concept_id.strip()
        value = request.attention_override
        try:
            self._am.override_attention(cid, value)
            response.success = True
            response.message = f"Attention override set for '{cid}' → {value:.3f} (one tick)"
            self.get_logger().info(f"[SET_ATTN] '{cid}' override={value:.3f}")
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
            self.get_logger().warn(f"[ACTION] Rejected QueueGoal: unknown goal '{goal_id}'")
            return GoalResponse.REJECT
        if eta <= 0.0:
            self.get_logger().warn(f"[ACTION] Rejected QueueGoal: eta must be > 0, got {eta}")
            return GoalResponse.REJECT
        if level not in _LAMBDA_BY_LEVEL:
            self.get_logger().warn(f"[ACTION] Rejected QueueGoal: unknown level '{level}'")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    async def _action_execute_callback(self, goal_handle) -> QueueGoal.Result:
        """
        Queue the goal and block until promoted or cancelled.

        Feedback is published each tick from _tick_cb(), not here.
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

        self._queued_handles[goal_id] = (goal_handle, eta)
        self.get_logger().info(f"[ACTION] QueueGoal accepted: '{goal_id}'  eta={eta}s  level='{level}'")

        while goal_handle.is_active:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._queued_handles.pop(goal_id, None)
                self.get_logger().info(f"[ACTION] QueueGoal '{goal_id}' cancelled")
                result = QueueGoal.Result()
                result.promoted         = False
                result.promoted_goal_id = ''
                return result
            await asyncio.sleep(self._tick_dt)

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
