import math
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RadioButtons
import networkx as nx

from awareness_manager.knowledge_base import KnowledgeBase


# --- Visual constants --------------------------------------------------------

_NODE_SIZE_MIN = 200        # size for concepts outside the attention window
_NODE_SIZE_MAX = 2000       # size for the goal node (attention = 1.0)
_EDGE_WIDTH_SCALE = 4.0     # base numerator for width = scale / weight²
_HIT_RADIUS = 0.15          # graph-coordinate radius for click detection
_CMAP = plt.cm.RdYlGn_r     # green=fresh (E=0), red=stale (E=1)


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
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        sim_interval: float = 1.0,
        frame_interval: float = 0.1,
    ) -> None:
        """
        Args:
            sim_interval:   Simulated time (seconds) that passes per tick.
                            Controls how fast epistemic error grows.
            frame_interval: Real time (seconds) between visual redraws.
                            Decoupled from sim_interval so the graph can
                            update smoothly without speeding up the simulation.
        """
        self._kb = kb
        self._goal_id = goal_id
        self._sim_interval = sim_interval
        self._frame_interval = frame_interval
        self._elapsed = 0.0
        self._attention: dict[str, float] = {}
        self._last_tick_time = time.time()

        # Fix layout positions once so the graph does not jump between frames
        self._pos = nx.spring_layout(kb._graph, seed=42)

        self._setup_figure()
        self._recompute_attention()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        self._fig = plt.figure(figsize=(14, 7))
        self._fig.patch.set_facecolor('#1e1e2e')

        # Left panel: graph (wide)
        self._ax_graph = self._fig.add_axes([0.01, 0.05, 0.72, 0.90])
        self._ax_graph.set_facecolor('#1e1e2e')
        self._ax_graph.set_title(
            'Robot Awareness — Knowledge Graph',
            color='white', fontsize=13, pad=10
        )

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
        # Style radio button labels
        for label in self._radio.labels:
            label.set_color('white')
            label.set_fontsize(9)
        self._ax_radio.set_title('Active Goal', color='white', fontsize=10, pad=6)

        # Click handler for node interaction
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

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
        """Called every frame_interval ms. Ticks the simulation only when
        sim_interval real seconds have elapsed since the last tick."""
        now = time.time()
        if now - self._last_tick_time >= self._sim_interval:
            self._kb.tick(dt=self._sim_interval)
            self._elapsed += self._sim_interval
            self._last_tick_time = now
            self._print_terminal()
        self._recompute_attention()
        self._draw_graph()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _recompute_attention(self) -> None:
        try:
            self._attention = self._kb.compute_attention(self._goal_id)
        except ValueError:
            self._attention = {}

    def _draw_graph(self) -> None:
        self._ax_graph.clear()
        self._ax_graph.set_facecolor('#1e1e2e')
        self._ax_graph.set_title(
            f'Robot Awareness — Knowledge Graph    '
            f'[t={self._elapsed:5.1f}s  goal: {self._goal_id}]',
            color='white', fontsize=11, pad=8
        )
        self._ax_graph.axis('off')

        graph = self._kb._graph
        concept_ids = self._kb.concept_ids()

        # Node colors from epistemic error
        node_colors = [
            _CMAP(self._kb.get_concept(cid).epistemic_error)
            for cid in concept_ids
        ]

        # Node sizes from attention
        node_sizes = [
            _NODE_SIZE_MIN + (_NODE_SIZE_MAX - _NODE_SIZE_MIN) * self._attention.get(cid, 0.0)
            for cid in concept_ids
        ]

        # Edge widths: scale / weight² gives much stronger contrast than
        # scale / weight. weight=1.0 → 4.0px, weight=1.5 → 1.78px, weight=2.0 → 1.0px
        edge_widths = [
            _EDGE_WIDTH_SCALE / (graph[u][v]['weight'] ** 2)
            for u, v in graph.edges()
        ]

        # Node labels with live values
        labels = {
            cid: (
                f"{cid}\n"
                f"E={self._kb.get_concept(cid).epistemic_error:.2f}  "
                f"A={self._attention.get(cid, 0.0):.2f}"
            )
            for cid in concept_ids
        }

        # Edge weight labels (semantic distance value on each edge)
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.1f}"
            for u, v in graph.edges()
        }

        nx.draw_networkx_edges(
            graph, self._pos, ax=self._ax_graph,
            width=edge_widths, edge_color='#888888', alpha=0.6,
        )
        nx.draw_networkx_nodes(
            graph, self._pos, ax=self._ax_graph,
            nodelist=concept_ids,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.92,
        )
        nx.draw_networkx_labels(
            graph, self._pos, labels=labels, ax=self._ax_graph,
            font_size=8, font_color='white', font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e', alpha=0.6, edgecolor='none'),
        )
        nx.draw_networkx_edge_labels(
            graph, self._pos, edge_labels=edge_labels, ax=self._ax_graph,
            font_size=7, font_color='#cccccc', bbox=dict(alpha=0),
        )

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
        print(f"[t={self._elapsed:6.1f}s]  goal={self._goal_id:<20}  {parts}")

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_radio_select(self, label: str) -> None:
        self._goal_id = label
        self._recompute_attention()
        self._draw_graph()
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
            self._recompute_attention()
            # Sync RadioButtons selection
            labels = [lbl.get_text() for lbl in self._radio.labels]
            if clicked in labels:
                self._radio.set_active(labels.index(clicked))
            print(f"[GOAL  ]  Mission changed → '{clicked}'")
        else:
            # Clicking any other node simulates an observation (refresh)
            before = concept.epistemic_error
            self._kb.refresh_concept(clicked, refresh=0.5)
            after = concept.epistemic_error
            print(f"[REFRESH]  '{clicked}'  E: {before:.3f} → {after:.3f}")

        self._draw_graph()

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
