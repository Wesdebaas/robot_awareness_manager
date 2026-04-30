import math
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons, CheckButtons
import networkx as nx

from awareness_manager.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase


# --- Visual constants --------------------------------------------------------

_NODE_SIZE_MIN = 200        # size for concepts outside the attention window
_NODE_SIZE_MAX = 2000       # size for the goal node (attention = 1.0)
_INST_NODE_SIZE_MIN = 60    # instance equivalent
_INST_NODE_SIZE_MAX = 700
_EDGE_WIDTH_SCALE = 4.0     # base numerator for width = scale / weight²
_HIT_RADIUS = 0.15          # graph-coordinate radius for click detection
_CMAP = plt.cm.RdYlGn_r     # green=fresh (E=0), red=stale (E=1)
_HIGHLIGHT_COLOR = '#ffdd00' # yellow ring: just observed
_HIGHLIGHT_SIZE_BONUS = 600  # extra size added to the highlight ring
_INST_HIGHLIGHT_SIZE_BONUS = 200

# Instance-specific colours
_ISINSTANCE_EDGE_COLOR  = '#5555aa'   # dashed: class → instance (typeof link)
_INST_REL_EDGE_COLOR    = '#446688'   # dotted: instance ↔ instance (relational)
_INSTANCE_NODE_ALPHA    = 0.82
_INSTANCE_EDGE_ALPHA    = 0.45

# Radial layout: instances orbit their parent class node
_INST_ORBIT_RADIUS_BASE = 0.12   # minimum orbit radius
_INST_ORBIT_RADIUS_STEP = 0.025  # extra radius per additional instance


class KBVisualizer:
    """
    Live visualizer for a KnowledgeBase, with optional instance layer.

    Renders the class-level semantic graph in a matplotlib window that updates
    on a timer. When an InstanceKnowledgeBase is provided, a second layer of
    instance nodes can be toggled on/off via a "Show instances" checkbox.

    Node encoding (class nodes — circles):
        Color  — epistemic error (green = fresh, red = stale)
        Size   — attention value from the current goal (larger = more attention)
        Label  — concept_id + live E and A values

    Node encoding (instance nodes — diamonds):
        Color  — epistemic error (same colour map)
        Size   — attention value (scaled to 35% of class range)
        Label  — instance_id + E and A values (smaller font)

    Edge encoding:
        Class edges    — solid grey,  width ∝ 1/weight²
        typeof edges   — dashed blue-grey (class → instance)
        Relational edges — dotted teal (instance ↔ instance)

    Interaction:
        Click a task class node    → set as new goal
        Click any other class node → manual refresh (E drops)
        Click an instance node     → manual refresh of that instance
        RadioButtons panel         → switch active goal
        "Show instances" checkbox  → toggle instance layer visibility

    Performance: static elements are drawn once at startup. Per-frame updates
    only mutate node colours, sizes, and label strings in-place.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        sim_interval: float = 1.0,
        frame_interval: float = 0.1,
        awareness_manager: "AwarenessManager | None" = None,
        refresh_interval: float | None = None,
        highlight_duration: float = 1.5,
        instance_kb: "InstanceKnowledgeBase | None" = None,
    ) -> None:
        """
        Args:
            kb:                 The class-level knowledge base to visualise.
            goal_id:            Initial mission goal concept ID.
            sim_interval:       Simulated seconds that pass per real tick.
            frame_interval:     Real seconds between visual redraws.
            awareness_manager:  Optional AM; when set, tick and observe are
                                delegated to the AM (Formulas 1-5).
            refresh_interval:   Real seconds between automatic observations.
                                None disables auto-refresh.
            highlight_duration: Seconds the yellow ring is visible after a refresh.
            instance_kb:        Optional instance-level knowledge base. When provided,
                                a "Show instances" checkbox appears in the right panel.
        """
        self._kb = kb
        self._instance_kb = instance_kb
        self._goal_id = goal_id
        self._sim_interval = sim_interval
        self._frame_interval = frame_interval
        self._am = awareness_manager
        self._refresh_interval = refresh_interval
        self._highlight_duration = highlight_duration

        self._elapsed = 0.0
        self._attention: dict[str, float] = {}
        self._last_tick_time = time.time()
        self._last_refresh_time = time.time()
        self._last_schedule: list[str] = []
        self._dirty = True
        self._show_instances = False   # toggled by checkbox

        # Perf counters
        self._perf_frames = 0
        self._perf_artist_ms = 0.0
        self._perf_render_ms = 0.0

        # Highlight state: concept_id → expiry timestamp
        self._highlighted: dict[str, float] = {}

        # Fix class layout once so the graph never jumps
        self._pos = nx.spring_layout(kb._graph, seed=42)

        # Instance positions: orbit parent class node
        self._instance_pos: dict[str, tuple[float, float]] = {}
        if self._instance_kb is not None:
            self._instance_pos = self._compute_instance_pos()

        self._setup_figure()
        self._recompute_attention()
        self._setup_artists()

    # ------------------------------------------------------------------
    # Instance layout
    # ------------------------------------------------------------------

    def _compute_instance_pos(self) -> dict[str, tuple[float, float]]:
        """
        Place each instance radially around its parent class node.

        Instances of the same class are evenly distributed on a circle whose
        radius scales slightly with the number of instances to prevent overlap.
        """
        class_instances: dict[str, list[str]] = defaultdict(list)
        for iid in self._instance_kb.instance_ids():
            inst = self._instance_kb.get_instance(iid)
            if inst.class_id in self._pos:
                class_instances[inst.class_id].append(iid)

        pos: dict[str, tuple[float, float]] = {}
        for class_id, instances in class_instances.items():
            cx, cy = self._pos[class_id]
            n = len(instances)
            radius = _INST_ORBIT_RADIUS_BASE + _INST_ORBIT_RADIUS_STEP * max(0, n - 1)
            for i, iid in enumerate(sorted(instances)):   # sorted for determinism
                angle = 2 * math.pi * i / n + math.pi / 6
                pos[iid] = (
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                )
        return pos

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        self._fig = plt.figure(figsize=(14, 7))
        self._fig.patch.set_facecolor('#1e1e2e')

        # Left panel: graph
        self._ax_graph = self._fig.add_axes([0.01, 0.05, 0.72, 0.90])
        self._ax_graph.set_facecolor('#1e1e2e')
        self._ax_graph.axis('off')

        # Right panel: RadioButtons for goal selection
        radio_bottom = 0.25 if self._instance_kb is not None else 0.20
        self._ax_radio = self._fig.add_axes([0.76, radio_bottom, 0.22, 0.60])
        self._ax_radio.set_facecolor('#2a2a3e')

        concept_ids = self._kb.concept_ids()
        self._radio = RadioButtons(
            self._ax_radio,
            labels=concept_ids,
            active=concept_ids.index(self._goal_id) if self._goal_id in concept_ids else 0,
        )
        self._radio.on_clicked(self._on_radio_select)
        for label in self._radio.labels:
            label.set_color('white')
            label.set_fontsize(9)
        self._ax_radio.set_title('Active Goal', color='white', fontsize=10, pad=6)

        # "Show instances" checkbox — only when instance_kb is provided
        self._ax_check = None
        self._check = None
        if self._instance_kb is not None:
            self._ax_check = self._fig.add_axes([0.76, 0.08, 0.22, 0.13])
            self._ax_check.set_facecolor('#2a2a3e')
            self._check = CheckButtons(
                self._ax_check,
                labels=['Show instances'],
                actives=[False],
            )
            self._check.on_clicked(self._toggle_instances)
            for label in self._check.labels:
                label.set_color('white')
                label.set_fontsize(9)

        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_artists(self) -> None:
        """
        Draw static elements once and create persistent artist handles.

        Static (drawn once, never updated):
            - Class edges (width ∝ 1/weight²)
            - Class edge weight labels
            - Instance typeof edges (class → instance, dashed)
            - Instance relational edges (instance ↔ instance, dotted)

        Dynamic (handles kept, data mutated in-place each frame):
            - Class node collection  (colours + sizes)
            - Class label Text objects
            - Class highlight PathCollections
            - Instance node scatter  (colours + sizes)
            - Instance label Text objects
            - Instance highlight scatter collections
        """
        graph = self._kb._graph
        concept_ids = self._kb.concept_ids()

        # --- Title ---
        self._title = self._ax_graph.set_title(
            f'Robot Awareness — Knowledge Graph    '
            f'[t={self._elapsed:5.1f}s  goal: {self._goal_id}]',
            color='white', fontsize=11, pad=8,
        )

        # --- Static: class edges ---
        edge_widths = [
            _EDGE_WIDTH_SCALE / (graph[u][v]['weight'] ** 2)
            for u, v in graph.edges()
        ]
        nx.draw_networkx_edges(
            graph, self._pos, ax=self._ax_graph,
            width=edge_widths, edge_color='#888888', alpha=0.6,
        )

        # --- Static: class edge weight labels ---
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.1f}"
            for u, v in graph.edges()
        }
        nx.draw_networkx_edge_labels(
            graph, self._pos, edge_labels=edge_labels, ax=self._ax_graph,
            font_size=7, font_color='#cccccc', bbox=dict(alpha=0),
        )

        # --- Static: instance edges (drawn before instance nodes so nodes sit on top) ---
        self._instance_edge_artists: list = []
        if self._instance_kb is not None:
            # typeof edges: class pos → instance pos (dashed)
            for iid in self._instance_kb.instance_ids():
                inst = self._instance_kb.get_instance(iid)
                if inst.class_id in self._pos and iid in self._instance_pos:
                    x0, y0 = self._pos[inst.class_id]
                    x1, y1 = self._instance_pos[iid]
                    line, = self._ax_graph.plot(
                        [x0, x1], [y0, y1],
                        color=_ISINSTANCE_EDGE_COLOR, linewidth=0.8,
                        linestyle='--', alpha=_INSTANCE_EDGE_ALPHA,
                        zorder=1, visible=False,
                    )
                    self._instance_edge_artists.append(line)

            # Relational edges: instance ↔ instance (dotted)
            for u, v in self._instance_kb._graph.edges():
                if u in self._instance_pos and v in self._instance_pos:
                    x0, y0 = self._instance_pos[u]
                    x1, y1 = self._instance_pos[v]
                    line, = self._ax_graph.plot(
                        [x0, x1], [y0, y1],
                        color=_INST_REL_EDGE_COLOR, linewidth=0.8,
                        linestyle=':', alpha=_INSTANCE_EDGE_ALPHA + 0.1,
                        zorder=1, visible=False,
                    )
                    self._instance_edge_artists.append(line)

        # --- Dynamic: class node collection ---
        node_colors = [
            _CMAP(self._kb.get_concept(cid).epistemic_error)
            for cid in concept_ids
        ]
        node_sizes = [
            _NODE_SIZE_MIN + (_NODE_SIZE_MAX - _NODE_SIZE_MIN) * self._attention.get(cid, 0.0)
            for cid in concept_ids
        ]
        self._node_collection = nx.draw_networkx_nodes(
            graph, self._pos, ax=self._ax_graph,
            nodelist=concept_ids,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.92,
        )
        self._node_collection.set_zorder(3)
        self._concept_ids = concept_ids

        # --- Dynamic: class highlight rings ---
        self._highlight_collections: dict[str, object] = {}
        for cid in concept_ids:
            coll = nx.draw_networkx_nodes(
                graph, self._pos, ax=self._ax_graph,
                nodelist=[cid],
                node_color=_HIGHLIGHT_COLOR,
                node_size=_NODE_SIZE_MIN,
                alpha=0.0,
            )
            coll.set_zorder(2)
            self._highlight_collections[cid] = coll

        # --- Dynamic: class label Text objects ---
        self._label_texts: dict[str, plt.Text] = {}
        for cid in concept_ids:
            x, y = self._pos[cid]
            txt = self._ax_graph.text(
                x, y,
                f"{cid}\nE={self._kb.get_concept(cid).epistemic_error:.2f}  "
                f"A={self._attention.get(cid, 0.0):.2f}",
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e',
                          alpha=0.6, edgecolor='none'),
                zorder=5,
            )
            self._label_texts[cid] = txt

        # --- Dynamic: instance nodes, labels, highlights ---
        self._instance_node_collection = None
        self._instance_label_texts: dict[str, plt.Text] = {}
        self._instance_highlight_collections: dict[str, object] = {}
        self._instance_ids: list[str] = []

        if self._instance_kb is not None:
            self._instance_ids = list(self._instance_kb.instance_ids())
            if self._instance_ids:
                xs = [self._instance_pos[iid][0] for iid in self._instance_ids]
                ys = [self._instance_pos[iid][1] for iid in self._instance_ids]
                inst_colors = [
                    _CMAP(self._instance_kb.get_instance(iid).epistemic_error)
                    for iid in self._instance_ids
                ]
                inst_sizes = [
                    _INST_NODE_SIZE_MIN
                    + (_INST_NODE_SIZE_MAX - _INST_NODE_SIZE_MIN)
                    * self._attention.get(iid, 0.0)
                    for iid in self._instance_ids
                ]
                self._instance_node_collection = self._ax_graph.scatter(
                    xs, ys, s=inst_sizes, c=inst_colors,
                    marker='D', alpha=_INSTANCE_NODE_ALPHA,
                    edgecolors='white', linewidths=0.4,
                    zorder=4, visible=False,
                )

                # Highlight rings for instances
                for iid in self._instance_ids:
                    x, y = self._instance_pos[iid]
                    h = self._ax_graph.scatter(
                        [x], [y], s=[_INST_NODE_SIZE_MIN],
                        c=_HIGHLIGHT_COLOR, marker='D',
                        alpha=0.0, zorder=3, visible=True,
                    )
                    self._instance_highlight_collections[iid] = h

                # Labels for instances (smaller, no heavy bbox)
                for iid in self._instance_ids:
                    x, y = self._instance_pos[iid]
                    inst = self._instance_kb.get_instance(iid)
                    txt = self._ax_graph.text(
                        x, y,
                        f"{iid}\nE={inst.epistemic_error:.2f} A={self._attention.get(iid, 0.0):.2f}",
                        ha='center', va='center',
                        fontsize=6, color='#ddddff',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e2e',
                                  alpha=0.5, edgecolor='none'),
                        zorder=6, visible=False,
                    )
                    self._instance_label_texts[iid] = txt

    # ------------------------------------------------------------------
    # Instance layer visibility
    # ------------------------------------------------------------------

    def _toggle_instances(self, _label: str) -> None:
        self._show_instances = not self._show_instances
        self._set_instance_visibility(self._show_instances)
        self._dirty = True

    def _set_instance_visibility(self, visible: bool) -> None:
        """Show or hide all instance-layer artists at once."""
        for line in self._instance_edge_artists:
            line.set_visible(visible)
        if self._instance_node_collection is not None:
            self._instance_node_collection.set_visible(visible)
        for txt in self._instance_label_texts.values():
            txt.set_visible(visible)
        # Highlight rings: control via alpha, not visibility (so fade logic still works)
        if not visible:
            for coll in self._instance_highlight_collections.values():
                coll.set_alpha(0.0)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the timer-driven animation. Blocks until the window is closed."""
        interval_ms = int(self._frame_interval * 1000)
        self._anim = animation.FuncAnimation(
            self._fig,
            self._frame,
            interval=interval_ms,
            cache_frame_data=False,
        )
        plt.show()

    def _frame(self, _frame_number) -> None:
        now = time.time()

        # Drain all accumulated simulation ticks
        while now - self._last_tick_time >= self._sim_interval:
            if self._am is not None:
                self._last_schedule = self._am.tick(dt=self._sim_interval)
                if self._am.goal_id != self._goal_id:
                    self._goal_id = self._am.goal_id
                    labels = [lbl.get_text() for lbl in self._radio.labels]
                    if self._goal_id in labels:
                        self._radio.set_active(labels.index(self._goal_id))
            else:
                self._kb.tick(dt=self._sim_interval)
            self._elapsed += self._sim_interval
            self._last_tick_time += self._sim_interval
            self._dirty = True
            self._print_terminal()

        # Auto-refresh
        if (self._am is not None
                and self._refresh_interval is not None
                and self._last_schedule
                and now - self._last_refresh_time >= self._refresh_interval):
            self._do_scheduled_refresh(now)
            self._dirty = True

        if self._highlighted:
            self._dirty = True

        if self._dirty:
            self._recompute_attention()
            t0 = time.perf_counter()
            self._draw_graph()
            t1 = time.perf_counter()
            self._fig.canvas.draw()
            t2 = time.perf_counter()
            self._dirty = False

            self._perf_artist_ms += (t1 - t0) * 1000
            self._perf_render_ms += (t2 - t1) * 1000
            self._perf_frames += 1
            if self._perf_frames >= 30:
                print(
                    f"[PERF  ]  artist={self._perf_artist_ms/self._perf_frames:.1f}ms  "
                    f"render={self._perf_render_ms/self._perf_frames:.1f}ms  "
                    f"(avg over {self._perf_frames} redraws)"
                )
                self._perf_frames = 0
                self._perf_artist_ms = 0.0
                self._perf_render_ms = 0.0

    # ------------------------------------------------------------------
    # Scheduled observation
    # ------------------------------------------------------------------

    def _do_scheduled_refresh(self, now: float) -> None:
        self._last_refresh_time = now
        for target in self._last_schedule:
            before = self._get_epistemic_error(target)
            if self._am is not None:
                refresh_used = self._am.observe(target)
            else:
                self._kb.refresh_concept(target, refresh=0.4)
                refresh_used = 0.4
            after = self._get_epistemic_error(target)
            self._highlighted[target] = now + self._highlight_duration
            print(f"[OBS   ]  '{target}'  E: {before:.3f} → {after:.3f}  (refresh={refresh_used:.3f})")

    # ------------------------------------------------------------------
    # Drawing — in-place artist updates, no ax.clear()
    # ------------------------------------------------------------------

    def _recompute_attention(self) -> None:
        if self._am is not None:
            self._attention = self._am.attention()
        else:
            try:
                self._attention = self._kb.compute_attention(self._goal_id)
            except ValueError:
                self._attention = {}

    def _draw_graph(self) -> None:
        # --- Title ---
        self._title.set_text(
            f'Robot Awareness — Knowledge Graph    '
            f'[t={self._elapsed:5.1f}s  goal: {self._goal_id}]'
        )

        # --- Class node colours and sizes ---
        node_colors = [
            _CMAP(self._kb.get_concept(cid).epistemic_error)
            for cid in self._concept_ids
        ]
        node_sizes = [
            _NODE_SIZE_MIN + (_NODE_SIZE_MAX - _NODE_SIZE_MIN) * self._attention.get(cid, 0.0)
            for cid in self._concept_ids
        ]
        self._node_collection.set_facecolor(node_colors)
        self._node_collection.set_sizes(node_sizes)

        # --- Class label text ---
        for cid in self._concept_ids:
            self._label_texts[cid].set_text(
                f"{cid}\n"
                f"E={self._kb.get_concept(cid).epistemic_error:.2f}  "
                f"A={self._attention.get(cid, 0.0):.2f}"
            )

        # --- Class highlight rings ---
        now = time.time()
        for cid, coll in self._highlight_collections.items():
            if cid in self._highlighted:
                expiry = self._highlighted[cid]
                if now >= expiry:
                    del self._highlighted[cid]
                    coll.set_alpha(0.0)
                else:
                    remaining = expiry - now
                    hi_size = (
                        _NODE_SIZE_MIN
                        + (_NODE_SIZE_MAX - _NODE_SIZE_MIN) * self._attention.get(cid, 0.0)
                        + _HIGHLIGHT_SIZE_BONUS
                    )
                    coll.set_sizes([hi_size])
                    coll.set_alpha(remaining / self._highlight_duration)
            else:
                coll.set_alpha(0.0)

        # --- Instance layer (only update data when visible) ---
        if self._show_instances and self._instance_kb is not None:
            self._draw_instance_layer(now)

        self._fig.canvas.draw_idle()

    def _draw_instance_layer(self, now: float) -> None:
        """Update instance node colours, sizes, labels, and highlight rings."""
        if not self._instance_ids or self._instance_node_collection is None:
            return

        inst_colors = [
            _CMAP(self._instance_kb.get_instance(iid).epistemic_error)
            for iid in self._instance_ids
        ]
        inst_sizes = [
            _INST_NODE_SIZE_MIN
            + (_INST_NODE_SIZE_MAX - _INST_NODE_SIZE_MIN)
            * self._attention.get(iid, 0.0)
            for iid in self._instance_ids
        ]
        self._instance_node_collection.set_facecolor(inst_colors)
        self._instance_node_collection.set_sizes(inst_sizes)

        for iid in self._instance_ids:
            inst = self._instance_kb.get_instance(iid)
            self._instance_label_texts[iid].set_text(
                f"{iid}\n"
                f"E={inst.epistemic_error:.2f} A={self._attention.get(iid, 0.0):.2f}"
            )

        # Instance highlight rings
        for iid, coll in self._instance_highlight_collections.items():
            if iid in self._highlighted:
                expiry = self._highlighted[iid]
                if now >= expiry:
                    del self._highlighted[iid]
                    coll.set_alpha(0.0)
                else:
                    remaining = expiry - now
                    hi_size = (
                        _INST_NODE_SIZE_MIN
                        + (_INST_NODE_SIZE_MAX - _INST_NODE_SIZE_MIN)
                        * self._attention.get(iid, 0.0)
                        + _INST_HIGHLIGHT_SIZE_BONUS
                    )
                    coll.set_sizes([hi_size])
                    coll.set_alpha(remaining / self._highlight_duration)
            else:
                coll.set_alpha(0.0)

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def _print_terminal(self) -> None:
        top = sorted(
            self._attention.items(), key=lambda x: x[1], reverse=True
        )[:5]
        parts = '  '.join(
            f"{cid}(E={self._get_epistemic_error(cid):.2f} A={a:.2f})"
            for cid, a in top
        )
        schedule_str = ''
        queue_str = ''
        if self._am is not None:
            schedule_str = f"  → schedule: {self._last_schedule}"
            queue = self._am.mission_queue
            if queue:
                queue_parts = ', '.join(f"{gid}(Δt={eta:.1f}s)" for gid, eta in queue)
                queue_str = f"  | queue: [{queue_parts}]"
        print(f"[t={self._elapsed:6.1f}s]  goal={self._goal_id:<20}  {parts}{schedule_str}{queue_str}")

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_radio_select(self, label: str) -> None:
        self._goal_id = label
        if self._am is not None:
            self._am.set_goal(label)
        self._recompute_attention()
        self._draw_graph()
        self._fig.canvas.draw_idle()
        print(f"[GOAL  ]  Mission changed → '{label}'")

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_graph:
            return
        if event.xdata is None or event.ydata is None:
            return

        clicked = self._nearest_node(event.xdata, event.ydata)
        if clicked is None:
            return

        # --- Instance node clicked ---
        if self._instance_kb is not None and clicked in self._instance_kb.instance_ids():
            if not self._show_instances:
                return  # ignore clicks on hidden instances
            before = self._instance_kb.get_instance(clicked).epistemic_error
            if self._am is not None:
                self._am.observe(clicked)
            else:
                self._instance_kb.refresh_instance(clicked, refresh=0.5)
            after = self._instance_kb.get_instance(clicked).epistemic_error
            self._highlighted[clicked] = time.time() + self._highlight_duration
            print(f"[MANUAL]  '{clicked}' (instance)  E: {before:.3f} → {after:.3f}")
            self._draw_graph()
            self._fig.canvas.draw_idle()
            return

        # --- Class node clicked ---
        concept = self._kb.get_concept(clicked)
        if concept.concept_type == 'task':
            self._goal_id = clicked
            if self._am is not None:
                self._am.set_goal(clicked)
            self._recompute_attention()
            labels = [lbl.get_text() for lbl in self._radio.labels]
            if clicked in labels:
                self._radio.set_active(labels.index(clicked))
            print(f"[GOAL  ]  Mission changed → '{clicked}'")
        else:
            before = concept.epistemic_error
            self._kb.refresh_concept(clicked, refresh=0.5)
            after = concept.epistemic_error
            self._highlighted[clicked] = time.time() + self._highlight_duration
            print(f"[MANUAL]  '{clicked}'  E: {before:.3f} → {after:.3f}")

        self._draw_graph()
        self._fig.canvas.draw_idle()

    def _nearest_node(self, x: float, y: float) -> str | None:
        """
        Return the concept_id or instance_id of the closest node within
        _HIT_RADIUS. Searches class positions first, then instance positions
        (only when the instance layer is visible).
        """
        best_id = None
        best_dist = _HIT_RADIUS

        for cid, (nx_, ny_) in self._pos.items():
            dist = math.hypot(x - nx_, y - ny_)
            if dist < best_dist:
                best_dist = dist
                best_id = cid

        if self._show_instances:
            for iid, (ix, iy) in self._instance_pos.items():
                dist = math.hypot(x - ix, y - iy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = iid

        return best_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_epistemic_error(self, concept_id: str) -> float:
        """Return epistemic error for either a class concept or an instance."""
        if concept_id in self._kb._concepts:
            return self._kb.get_concept(concept_id).epistemic_error
        if (self._instance_kb is not None
                and concept_id in self._instance_kb.instance_ids()):
            return self._instance_kb.get_instance(concept_id).epistemic_error
        return 0.0
