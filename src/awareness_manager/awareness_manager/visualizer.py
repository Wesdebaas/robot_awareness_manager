import math
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons
import networkx as nx

from awareness_manager.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from awareness_manager.awareness_manager import AwarenessManager


# --- Visual constants --------------------------------------------------------

_NODE_SIZE_MIN = 200        # size for concepts outside the attention window
_NODE_SIZE_MAX = 2000       # size for the goal node (attention = 1.0)
_EDGE_WIDTH_SCALE = 4.0     # base numerator for width = scale / weight²
_HIT_RADIUS = 0.15          # graph-coordinate radius for click detection
_CMAP = plt.cm.RdYlGn_r     # green=fresh (E=0), red=stale (E=1)
_HIGHLIGHT_COLOR = '#ffdd00' # yellow ring drawn over a just-refreshed node
_HIGHLIGHT_SIZE_BONUS = 600  # extra node size for the highlight ring


class KBVisualizer:
    """
    Live visualizer for a KnowledgeBase.

    Renders the semantic graph in a matplotlib window that updates on a timer.
    Simultaneously prints a one-line summary to the terminal each tick.

    Node encoding:
        Color  — epistemic error (green = fresh, red = stale)
        Size   — attention value from the current goal (larger = more attention)
        Label  — concept_id + live E and A values

    Edge encoding:
        Width  — 1 / semantic_weight (thicker = tighter coupling)

    Interaction:
        Click a non-task node  → refresh it (simulate an observation, E drops)
        Click a task node      → set it as the new goal
        RadioButtons panel     → switch active goal from a list

    When awareness_manager and refresh_interval are both provided the visualizer
    automatically refreshes the top-priority concept on that cadence and draws a
    fading yellow ring around it to make the observation visible.

    Performance: static elements (edges, edge labels) are drawn once at startup.
    Per-frame updates only touch node colors, node sizes, label text strings, and
    highlight ring alphas — all via in-place artist mutations, no ax.clear().
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
    ) -> None:
        """
        Args:
            sim_interval:       Simulated time (seconds) that passes per real tick.
            frame_interval:     Real time (seconds) between visual redraws.
            awareness_manager:  Optional AM; when set, tick delegates to am.tick()
                                and observations are executed via am.observe()
                                (Formula 3 refresh values).
            refresh_interval:   Real seconds between automatic observations.
                                None disables auto-refresh (manual clicks only).
            highlight_duration: Seconds the yellow ring stays visible after a refresh.
        """
        self._kb = kb
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
        self._dirty = True   # redraw only when state actually changed

        # Perf counters — printed every 30 dirty redraws
        self._perf_frames = 0
        self._perf_artist_ms = 0.0
        self._perf_render_ms = 0.0

        # Highlight state: maps concept_id → expiry timestamp for the yellow ring.
        self._highlighted: dict[str, float] = {}

        # Fix layout positions once so the graph does not jump between frames
        self._pos = nx.spring_layout(kb._graph, seed=42)

        self._setup_figure()
        self._recompute_attention()
        self._setup_artists()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        self._fig = plt.figure(figsize=(14, 7))
        self._fig.patch.set_facecolor('#1e1e2e')

        # Left panel: graph (wide)
        self._ax_graph = self._fig.add_axes([0.01, 0.05, 0.72, 0.90])
        self._ax_graph.set_facecolor('#1e1e2e')
        self._ax_graph.axis('off')

        # Right panel: RadioButtons for goal selection
        self._ax_radio = self._fig.add_axes([0.76, 0.25, 0.22, 0.55])
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

        # Click handler for node interaction
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_artists(self) -> None:
        """
        Draw static elements once and create persistent artist handles.

        Static (drawn once, never updated):
            - Edges with widths proportional to 1/weight²
            - Edge weight labels

        Dynamic (handles kept, data updated in-place each frame):
            - Node collection  : colors (E) and sizes (A) change every tick
            - Label Text objects: E and A values in the text change every tick
            - Highlight PathCollections: one per node, alpha toggled on observe
        """
        graph = self._kb._graph
        concept_ids = self._kb.concept_ids()

        # --- Static: title (updated in-place via set_text, never cleared) ---
        self._title = self._ax_graph.set_title(
            f'Robot Awareness — Knowledge Graph    '
            f'[t={self._elapsed:5.1f}s  goal: {self._goal_id}]',
            color='white', fontsize=11, pad=8,
        )

        # --- Static: edges (weights never change) ---
        edge_widths = [
            _EDGE_WIDTH_SCALE / (graph[u][v]['weight'] ** 2)
            for u, v in graph.edges()
        ]
        nx.draw_networkx_edges(
            graph, self._pos, ax=self._ax_graph,
            width=edge_widths, edge_color='#888888', alpha=0.6,
        )

        # --- Static: edge weight labels ---
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.1f}"
            for u, v in graph.edges()
        }
        nx.draw_networkx_edge_labels(
            graph, self._pos, edge_labels=edge_labels, ax=self._ax_graph,
            font_size=7, font_color='#cccccc', bbox=dict(alpha=0),
        )

        # --- Dynamic: node collection (colors + sizes updated each frame) ---
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
        # Store the node order so _draw_graph updates the right indices
        self._concept_ids = concept_ids

        # --- Dynamic: highlight rings (one PathCollection per node, alpha=0 by default) ---
        self._highlight_collections: dict[str, object] = {}
        for cid in concept_ids:
            coll = nx.draw_networkx_nodes(
                graph, self._pos, ax=self._ax_graph,
                nodelist=[cid],
                node_color=_HIGHLIGHT_COLOR,
                node_size=_NODE_SIZE_MIN,
                alpha=0.0,
            )
            self._highlight_collections[cid] = coll

        # --- Dynamic: node label Text objects (text updated each frame) ---
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

        # --- Simulation ticks: drain all accumulated ticks in one frame pass.
        #     Using += sim_interval (not = now) keeps timing precise if a frame
        #     fires late — no ticks are silently skipped.
        while now - self._last_tick_time >= self._sim_interval:
            if self._am is not None:
                self._last_schedule = self._am.tick(dt=self._sim_interval)
                # Sync goal if the mission queue auto-promoted a new goal
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

        # --- Auto-refresh: one observation every refresh_interval real seconds ---
        if (self._am is not None
                and self._refresh_interval is not None
                and self._last_schedule
                and now - self._last_refresh_time >= self._refresh_interval):
            self._do_scheduled_refresh(now)
            self._dirty = True

        # --- Expiring highlights need one final redraw after they clear ---
        if self._highlighted:
            self._dirty = True

        if self._dirty:
            self._recompute_attention()
            t0 = time.perf_counter()
            self._draw_graph()          # artist data updates (Python side)
            t1 = time.perf_counter()
            self._fig.canvas.draw()     # synchronous render (backend side)
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
            before = self._kb.get_concept(target).epistemic_error
            if self._am is not None:
                refresh_used = self._am.observe(target)
            else:
                self._kb.refresh_concept(target, refresh=0.4)
                refresh_used = 0.4
            after = self._kb.get_concept(target).epistemic_error
            self._highlighted[target] = now + self._highlight_duration
            print(f"[OBS   ]  '{target}'  E: {before:.3f} → {after:.3f}  (refresh={refresh_used:.3f})")

    # ------------------------------------------------------------------
    # Drawing — in-place artist updates, no ax.clear()
    # ------------------------------------------------------------------

    def _recompute_attention(self) -> None:
        if self._am is not None:
            # Use the AM's already-computed blended attention (Formulas 1+2).
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

        # --- Node colors and sizes ---
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

        # --- Node label text ---
        for cid in self._concept_ids:
            self._label_texts[cid].set_text(
                f"{cid}\n"
                f"E={self._kb.get_concept(cid).epistemic_error:.2f}  "
                f"A={self._attention.get(cid, 0.0):.2f}"
            )

        # --- Highlight rings ---
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

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def _print_terminal(self) -> None:
        top = sorted(
            self._attention.items(), key=lambda x: x[1], reverse=True
        )[:5]
        parts = '  '.join(
            f"{cid}(E={self._kb.get_concept(cid).epistemic_error:.2f} A={a:.2f})"
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
        # Only respond to clicks inside the graph axes
        if event.inaxes is not self._ax_graph:
            return
        if event.xdata is None or event.ydata is None:
            return

        clicked = self._nearest_node(event.xdata, event.ydata)
        if clicked is None:
            return

        concept = self._kb.get_concept(clicked)

        if concept.concept_type == 'task':
            # Clicking a task node switches the goal
            self._goal_id = clicked
            if self._am is not None:
                self._am.set_goal(clicked)
            self._recompute_attention()
            labels = [lbl.get_text() for lbl in self._radio.labels]
            if clicked in labels:
                self._radio.set_active(labels.index(clicked))
            print(f"[GOAL  ]  Mission changed → '{clicked}'")
        else:
            # Clicking any other node simulates a manual observation (refresh)
            before = concept.epistemic_error
            self._kb.refresh_concept(clicked, refresh=0.5)
            after = concept.epistemic_error
            self._highlighted[clicked] = time.time() + self._highlight_duration
            print(f"[MANUAL]  '{clicked}'  E: {before:.3f} → {after:.3f}")

        self._draw_graph()
        self._fig.canvas.draw_idle()

    def _nearest_node(self, x: float, y: float) -> str | None:
        """Return the concept_id of the closest node within _HIT_RADIUS, or None."""
        best_id = None
        best_dist = _HIT_RADIUS
        for cid, (nx_, ny_) in self._pos.items():
            dist = math.hypot(x - nx_, y - ny_)
            if dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id
