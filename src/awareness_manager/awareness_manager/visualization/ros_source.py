"""
ros_source.py - ROS2-backed live source for the Awareness Manager dashboard.

Subscribes to the topics published by AwarenessNode and provides the same
snapshot() / history() / start() / stop() interface as SimulationRunner.
The Dash app sees no difference; only the data origin changes.

Topics consumed:
    awareness/state    — {cid: {E, A, P, ch_m, ch_an, ch_r, ch_s, presence?}}
    awareness/schedule — JSON list of scheduled concept IDs
    awareness/goal     — current goal_id (plain string)

Usage:
    source = RosStateSource(scenario='social_serving')
    source.start()          # begins spinning rclpy in a daemon thread
    snap = source.snapshot()
    source.stop()
"""

import json
import threading
import time
from collections import deque
from collections import defaultdict

from awareness_manager.visualization.snapshot import (
    channel_color,
    compute_positions_from_structure,
)


class RosStateSource:
    """
    Live dashboard source backed by a running AwarenessNode.

    Loads the scenario KB once at startup to derive graph structure and
    fixed node positions (identical to what SimulationRunner would produce).
    All live state (E, A, channels, schedule, goal) comes from ROS2 topics.
    """

    def __init__(self, scenario: str, history_maxlen: int = 300) -> None:
        self._scenario = scenario
        self._history_maxlen = history_maxlen
        self._dt = 0.1  # expected publish interval from awareness_node (10 Hz)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Latest data from topics
        self._state: dict[str, dict] = {}
        self._schedule: list[str] = []
        self._goal: str = ''
        self._elapsed: float = 0.0
        self._first_msg_time: float | None = None
        self._controller_state: dict | None = None

        # Rolling attention history: concept_id → deque of HistoryEntry
        # Entry format mirrors SimulationRunner:
        #   (elapsed, total_attn, mission, relational, surprise, e, pe, ch_anticipatory)
        self._history: dict[str, deque] = {}

        # Load scenario structure once (concepts, edges, instances) for positions
        self._structure = _load_structure(scenario)
        self._positions = compute_positions_from_structure(self._structure)

        # Static metadata per node (concept_type, decay_rate, class_id)
        self._meta: dict[str, dict] = {}
        for c in self._structure['classes']:
            self._meta[c['id']] = {
                'concept_type': c.get('concept_type', 'object'),
                'decay_rate':   c.get('decay_rate', 0.0),
                'type':         'class',
                'class_id':     None,
            }
        for inst in self._structure.get('instances', []):
            self._meta[inst['id']] = {
                'concept_type': inst.get('concept_type', 'object'),
                'decay_rate':   inst.get('decay_rate', 0.0),
                'type':         'instance',
                'class_id':     inst.get('class_id', ''),
            }

        # Populated classes: class nodes that have at least one instance
        self._populated_classes: set[str] = {
            inst['class_id']
            for inst in self._structure.get('instances', [])
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background rclpy listener thread."""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener thread to shut down."""
        self._running = False

    @property
    def available_seconds(self) -> float:
        with self._lock:
            return min(self._elapsed, self._history_maxlen * 0.1)

    # ------------------------------------------------------------------
    # Public interface (mirrors SimulationRunner)
    # ------------------------------------------------------------------

    def snapshot(self, threshold: float = 0.0) -> dict:
        """Build a Cytoscape-ready snapshot from the latest ROS topic data."""
        with self._lock:
            state             = dict(self._state)
            schedule          = list(self._schedule)
            goal              = self._goal
            elapsed           = self._elapsed
            controller_state  = self._controller_state

        schedule_set = set(schedule)
        elements: list[dict] = []

        # Compute priority rank across all concepts
        gamma = 1.0  # urgency_weight not exposed over ROS; use default
        all_prios = [(state.get(n, {}).get('P', 0.0), n) for n in state]
        positive_prios = sorted([(p, n) for p, n in all_prios if p > 0], reverse=True)
        n_schedulable = len(positive_prios)
        rank_lookup: dict[str, int] = {n: i + 1 for i, (_, n) in enumerate(positive_prios)}

        # Class nodes
        for c in self._structure['classes']:
            cid  = c['id']
            meta = self._meta.get(cid, {})
            vals = state.get(cid, {})
            a    = vals.get('A', 0.0)
            e    = vals.get('E', 0.0)
            ch_m = vals.get('ch_m', 0.0)
            ch_an = vals.get('ch_an', 0.0)
            ch_r = vals.get('ch_r', 0.0)
            ch_s = vals.get('ch_s', 0.0)
            p    = vals.get('P', 0.0)
            pos  = self._positions.get(cid, {'x': 0.0, 'y': 0.0})

            elements.append({
                'data': {
                    'id':                   cid,
                    'label':                f"{cid}\nA={a:.2f}  E={e:.2f}",
                    'type':                 'class',
                    'concept_type':         meta.get('concept_type', 'object'),
                    'attention':            round(a, 4),
                    'epistemic_error':      round(e, 4),
                    'prediction_error':     0.0,
                    'decay_rate':           meta.get('decay_rate', 0.0),
                    'urgency':              0.0,
                    'urgency_contribution': 0.0,
                    'priority':             round(p, 4),
                    'priority_rank':        rank_lookup.get(cid, 0),
                    'n_schedulable':        n_schedulable,
                    'in_schedule':          1 if cid in schedule_set else 0,
                    'below_threshold':      1 if a < threshold else 0,
                    'is_active_goal':       1 if cid == goal else 0,
                    'has_instances':        1 if cid in self._populated_classes else 0,
                    'ch_mission':           round(ch_m, 4),
                    'ch_anticipatory':      round(ch_an, 4),
                    'ch_relational':        round(ch_r, 4),
                    'ch_surprise':          round(ch_s, 4),
                    'color':                channel_color(ch_m, ch_an, ch_r, ch_s, e),
                },
                'position': pos,
            })

        # Instance nodes
        for inst in self._structure.get('instances', []):
            iid  = inst['id']
            meta = self._meta.get(iid, {})
            vals = state.get(iid, {})
            a    = vals.get('A', 0.0)
            e    = vals.get('E', 0.0)
            ch_m  = vals.get('ch_m', 0.0)
            ch_an = vals.get('ch_an', 0.0)
            ch_r  = vals.get('ch_r', 0.0)
            ch_s  = vals.get('ch_s', 0.0)
            u     = vals.get('urgency', 0.0)
            ur    = vals.get('urgency_rate', 0.0)
            p     = vals.get('P', 0.0)
            uc    = round(gamma * u, 4)
            pos   = self._positions.get(iid, {'x': 0.0, 'y': 0.0})

            label = f"{iid}\nA={a:.2f}  E={e:.2f}"
            if u > 0.05:
                label += f"\nU={u:.2f}"

            elements.append({
                'data': {
                    'id':                   iid,
                    'label':                label,
                    'parent':               meta.get('class_id', ''),
                    'type':                 'instance',
                    'class_id':             meta.get('class_id', ''),
                    'attention':            round(a, 4),
                    'epistemic_error':      round(e, 4),
                    'prediction_error':     0.0,
                    'urgency':              round(u, 4),
                    'urgency_rate':         ur,
                    'urgency_contribution': uc,
                    'priority':             round(p, 4),
                    'priority_rank':        rank_lookup.get(iid, 0),
                    'n_schedulable':        n_schedulable,
                    'decay_rate':           meta.get('decay_rate', 0.0),
                    'in_schedule':          1 if iid in schedule_set else 0,
                    'below_threshold':      1 if a < threshold else 0,
                    'is_active_goal':       0,
                    'has_instances':        0,
                    'ch_mission':           round(ch_m, 4),
                    'ch_anticipatory':      round(ch_an, 4),
                    'ch_relational':        round(ch_r, 4),
                    'ch_surprise':          round(ch_s, 4),
                    'color':                channel_color(ch_m, ch_an, ch_r, ch_s, e),
                },
                'position': pos,
            })

        # Semantic edges (class–class)
        for id_a, id_b, weight in self._structure['semantic_edges']:
            elements.append({'data': {
                'id':        f"sem__{id_a}__{id_b}",
                'source':    id_a,
                'target':    id_b,
                'edge_type': 'semantic',
                'weight':    weight,
                'label':     f"{weight:.1f}",
            }})

        # Relational edges (instance–instance)
        for id_a, id_b, weight, rtype in self._structure.get('relational_edges', []):
            elements.append({'data': {
                'id':           f"rel__{id_a}__{id_b}",
                'source':       id_a,
                'target':       id_b,
                'edge_type':    'relational',
                'weight':       weight,
                'relation_type': rtype or 'related',
                'label':        rtype or 'related',
            }})

        snap = {
            'elements':         elements,
            'goal_id':          goal,
            'queue_text':       'no queued goals',
            'n_scheduled':      len(schedule),
            'budget':           3,
            'budget_unlimited': False,
            'strategy':         'awareness_manager',
            'elapsed':          round(elapsed, 1),
            'schedule':         schedule,
        }
        if controller_state is not None:
            snap['controller_state'] = controller_state
        return snap

    def history(self, concept_id: str) -> list:
        """Return the rolling attention history for one concept (thread-safe)."""
        with self._lock:
            return list(self._history.get(concept_id, []))

    # ------------------------------------------------------------------
    # Background rclpy thread
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String as RosString

        rclpy.init()

        node = Node('awareness_dashboard_listener')
        node.create_subscription(RosString, 'awareness/state',            self._on_state,            10)
        node.create_subscription(RosString, 'awareness/schedule',         self._on_schedule,         10)
        node.create_subscription(RosString, 'awareness/goal',             self._on_goal,             10)
        node.create_subscription(RosString, 'awareness/controller_state', self._on_controller_state, 10)

        while self._running:
            rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Topic callbacks (called from rclpy spin in background thread)
    # ------------------------------------------------------------------

    def _on_state(self, msg) -> None:
        try:
            data: dict = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        now = time.monotonic()
        with self._lock:
            if self._first_msg_time is None:
                self._first_msg_time = now
            self._elapsed = now - self._first_msg_time
            self._state = data
            self._update_history(data, self._elapsed)

    def _on_schedule(self, msg) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._schedule = data

    def _on_goal(self, msg) -> None:
        with self._lock:
            self._goal = msg.data.strip()

    def _on_controller_state(self, msg) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._controller_state = data

    def _update_history(self, state: dict, elapsed: float) -> None:
        """Append one history entry per concept. Called under lock."""
        max_priority = max((v.get('P', 0.0) for v in state.values()), default=0.0)
        for cid, vals in state.items():
            if cid not in self._history:
                self._history[cid] = deque(maxlen=self._history_maxlen)
            a     = vals.get('A', 0.0)
            e     = vals.get('E', 0.0)
            ch_m  = vals.get('ch_m', 0.0)
            ch_an = vals.get('ch_an', 0.0)
            ch_r  = vals.get('ch_r', 0.0)
            ch_s  = vals.get('ch_s', 0.0)
            prio  = vals.get('P', 0.0)
            # History entry: (elapsed, total_attn, mission, relational, surprise,
            #                 epistemic_error, prediction_error, ch_anticipatory,
            #                 priority, max_priority)
            self._history[cid].append(
                (elapsed, a, ch_m, ch_r, ch_s, e, 0.0, ch_an, prio, max_priority)
            )


# ------------------------------------------------------------------
# Structure loader
# ------------------------------------------------------------------

def _load_structure(scenario: str) -> dict:
    """Load KB + IKB once to extract graph structure for position computation."""
    if scenario == 'social_serving':
        from awareness_manager.scenarios.social_serving import (
            build_social_serving_instance_kb,
            build_social_serving_kb,
        )
        kb  = build_social_serving_kb()
        ikb = build_social_serving_instance_kb()
    elif scenario == 'pv_inspection':
        from awareness_manager.scenarios.pv_inspection import (
            build_pv_inspection_instance_kb,
            build_pv_inspection_kb,
        )
        kb  = build_pv_inspection_kb()
        ikb = build_pv_inspection_instance_kb()
    else:
        raise ValueError(f"Unknown scenario '{scenario}' for RosStateSource.")

    return {
        'classes': [
            {
                'id':           cid,
                'concept_type': kb.get_concept(cid).concept_type,
                'decay_rate':   kb.get_concept(cid).decay_rate,
            }
            for cid in kb.concept_ids()
        ],
        'instances': [
            {
                'id':           iid,
                'concept_type': ikb.get_instance(iid).concept_type,
                'decay_rate':   ikb.get_instance(iid).decay_rate,
                'class_id':     ikb.get_instance(iid).class_id,
            }
            for iid in ikb.instance_ids()
        ],
        'semantic_edges':   kb.semantic_edges(),
        'relational_edges': [
            (id_a, id_b, w, rtype)
            for id_a, id_b, w, rtype in ikb.relational_edges()
        ],
    }
