import math
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import networkx as nx
import matplotlib.pyplot as _mpl  # used only for RdYlGn_r colormap conversion

from awareness_manager.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from awareness_manager.awareness_manager import AwarenessManager


# --- Visual constants --------------------------------------------------------

_NODE_SIZE_MIN = 12      # pixels — nodes outside attention window
_NODE_SIZE_MAX = 55      # pixels — goal node (attention = 1.0)
_EDGE_WIDTH_SCALE = 4.0  # same formula as matplotlib version: scale / weight²
_HIT_RADIUS = 0.15       # graph-coordinate radius for click detection
_CMAP = _mpl.cm.RdYlGn_r
_HIGHLIGHT_SIZE_BONUS = 14
_BG_COLOR = '#1e1e2e'
_PANEL_COLOR = '#2a2a3e'


def _cmap_to_qcolor(value: float) -> pg.QtGui.QColor:
    r, g, b, _ = _CMAP(float(np.clip(value, 0.0, 1.0)))
    return pg.mkColor(int(r * 255), int(g * 255), int(b * 255), 230)


class KBVisualizerQt:
    """
    PyQtGraph-based live visualizer for a KnowledgeBase.

    Drop-in replacement for KBVisualizer with the same public interface.
    Uses Qt + OpenGL-backed rendering for ~10x lower frame cost vs matplotlib.

    Node encoding, edge encoding, and interaction are identical to
    KBVisualizer — see that class for the full docstring.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        sim_interval: float = 1.0,
        frame_interval: float = 0.05,
        awareness_manager: "AwarenessManager | None" = None,
        refresh_interval: float | None = None,
        highlight_duration: float = 1.5,
    ) -> None:
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
        self._highlighted: dict[str, float] = {}

        self._pos = nx.spring_layout(kb._graph, seed=42)

        pg.setConfigOption('background', _BG_COLOR)
        pg.setConfigOption('foreground', 'w')

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self._setup_window()
        self._recompute_attention()
        self._setup_artists()

    # ------------------------------------------------------------------
    # Window and panel setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle('Robot Awareness — Knowledge Graph')
        self._win.resize(1200, 680)

        central = QtWidgets.QWidget()
        central.setStyleSheet(f'background-color: {_BG_COLOR};')
        self._win.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Left: graph plot ---
        self._plot = pg.PlotWidget()
        self._plot.setBackground(_BG_COLOR)
        self._plot.hideAxis('left')
        self._plot.hideAxis('bottom')
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setAspectLocked(True)
        # Fix view so nodes don't move as we update
        self._plot.setXRange(-1.4, 1.4, padding=0)
        self._plot.setYRange(-1.4, 1.4, padding=0)
        layout.addWidget(self._plot, stretch=5)

        # --- Right: goal selector ---
        right = QtWidgets.QWidget()
        right.setFixedWidth(220)
        right.setStyleSheet(
            f'background-color: {_PANEL_COLOR}; color: white;'
            'border-radius: 6px;'
        )
        rlayout = QtWidgets.QVBoxLayout(right)
        rlayout.setContentsMargins(10, 12, 10, 12)
        rlayout.setSpacing(6)

        title_label = QtWidgets.QLabel('Active Goal')
        title_label.setStyleSheet('font-size: 11pt; font-weight: bold; color: white;')
        rlayout.addWidget(title_label)

        self._radio_buttons: dict[str, QtWidgets.QRadioButton] = {}
        self._radio_group = QtWidgets.QButtonGroup(right)
        for i, cid in enumerate(self._kb.concept_ids()):
            rb = QtWidgets.QRadioButton(cid)
            rb.setStyleSheet('color: white; font-size: 9pt;')
            if cid == self._goal_id:
                rb.setChecked(True)
            rlayout.addWidget(rb)
            self._radio_group.addButton(rb, i)
            self._radio_buttons[cid] = rb

        rlayout.addStretch()
        self._radio_group.buttonClicked.connect(self._on_radio_clicked)
        layout.addWidget(right, stretch=0)

        # Scene click handler for node interaction
        self._plot.scene().sigMouseClicked.connect(self._on_scene_click)

    # ------------------------------------------------------------------
    # Static and dynamic artist setup
    # ------------------------------------------------------------------

    def _setup_artists(self) -> None:
        graph = self._kb._graph
        concept_ids = self._kb.concept_ids()
        self._concept_ids = concept_ids

        # Node position arrays (fixed for the life of the window)
        self._xs = np.array([self._pos[cid][0] for cid in concept_ids])
        self._ys = np.array([self._pos[cid][1] for cid in concept_ids])

        # --- Static: title text ---
        self._title_item = pg.TextItem(anchor=(0.5, 0.0), color='white')
        self._title_item.setFont(QtGui.QFont('monospace', 9))
        self._title_item.setPos(0.0, 1.35)
        self._plot.addItem(self._title_item)

        # --- Static: edges (drawn once — weights never change) ---
        for u, v in graph.edges():
            weight = graph[u][v]['weight']
            width = max(0.5, _EDGE_WIDTH_SCALE / (weight ** 2))
            xu, yu = self._pos[u]
            xv, yv = self._pos[v]
            self._plot.addItem(pg.PlotCurveItem(
                [xu, xv], [yu, yv],
                pen=pg.mkPen(color=(136, 136, 136, 150), width=width),
            ))

        # --- Static: edge weight labels ---
        for u, v in graph.edges():
            weight = graph[u][v]['weight']
            xu, yu = self._pos[u]
            xv, yv = self._pos[v]
            lbl = pg.TextItem(
                f'{weight:.1f}', color=(180, 180, 180), anchor=(0.5, 0.5),
            )
            lbl.setFont(QtGui.QFont('monospace', 7))
            lbl.setPos((xu + xv) / 2, (yu + yv) / 2)
            self._plot.addItem(lbl)

        # --- Dynamic: highlight rings (drawn below nodes) ---
        self._highlight_scatter = pg.ScatterPlotItem(
            x=self._xs, y=self._ys,
            brush=[pg.mkBrush(255, 221, 0, 0)] * len(concept_ids),
            size=[_NODE_SIZE_MIN] * len(concept_ids),
            pen=pg.mkPen(None),
            pxMode=True,
        )
        self._plot.addItem(self._highlight_scatter)

        # --- Dynamic: main nodes ---
        init_brushes = [pg.mkBrush(_cmap_to_qcolor(0.0))] * len(concept_ids)
        init_sizes = [_NODE_SIZE_MIN] * len(concept_ids)
        self._scatter = pg.ScatterPlotItem(
            x=self._xs, y=self._ys,
            brush=init_brushes,
            size=init_sizes,
            pen=pg.mkPen(None),
            pxMode=True,
        )
        self._plot.addItem(self._scatter)

        # --- Dynamic: node labels ---
        self._label_items: dict[str, pg.TextItem] = {}
        for cid in concept_ids:
            x, y = self._pos[cid]
            item = pg.TextItem(
                text=f'{cid}\nE=0.00  A=0.00',
                color='white',
                anchor=(0.5, 0.5),
            )
            item.setFont(QtGui.QFont('monospace', 7))
            item.setPos(x, y)
            self._plot.addItem(item)
            self._label_items[cid] = item

        self._update_artists()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Show the window and enter the Qt event loop. Blocks until closed."""
        self._win.show()
        self._timer = QtCore.QTimer()
        self._timer.setInterval(int(self._frame_interval * 1000))
        self._timer.timeout.connect(self._frame)
        self._timer.start()
        try:
            self._app.exec()
        except AttributeError:
            self._app.exec_()  # PyQt5 fallback

    def _frame(self) -> None:
        now = time.time()
        dirty = False

        # Drain all accumulated simulation ticks
        while now - self._last_tick_time >= self._sim_interval:
            if self._am is not None:
                self._last_schedule = self._am.tick(dt=self._sim_interval)
                if self._am.goal_id != self._goal_id:
                    self._goal_id = self._am.goal_id
                    if self._goal_id in self._radio_buttons:
                        self._radio_buttons[self._goal_id].setChecked(True)
            else:
                self._kb.tick(dt=self._sim_interval)
            self._elapsed += self._sim_interval
            self._last_tick_time += self._sim_interval
            dirty = True
            self._print_terminal()

        # Auto-refresh
        if (self._am is not None
                and self._refresh_interval is not None
                and self._last_schedule
                and now - self._last_refresh_time >= self._refresh_interval):
            self._do_scheduled_refresh(now)
            dirty = True

        if self._highlighted:
            dirty = True

        if dirty:
            self._recompute_attention()
            self._update_artists()

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
    # Artist updates — in-place, no scene rebuild
    # ------------------------------------------------------------------

    def _recompute_attention(self) -> None:
        if self._am is not None:
            self._attention = self._am.attention()
        else:
            try:
                self._attention = self._kb.compute_attention(self._goal_id)
            except ValueError:
                self._attention = {}

    def _update_artists(self) -> None:
        concept_ids = self._concept_ids

        # Title
        self._title_item.setText(
            f'Robot Awareness — Knowledge Graph'
            f'    [t={self._elapsed:5.1f}s  goal: {self._goal_id}]'
        )

        # Node colors and sizes
        brushes = [pg.mkBrush(_cmap_to_qcolor(
            self._kb.get_concept(cid).epistemic_error)) for cid in concept_ids]
        sizes = [self._node_size(cid) for cid in concept_ids]
        self._scatter.setBrush(brushes)
        self._scatter.setSize(sizes)

        # Highlight rings
        now = time.time()
        hi_brushes = []
        hi_sizes = []
        for cid in concept_ids:
            if cid in self._highlighted:
                expiry = self._highlighted[cid]
                if now >= expiry:
                    del self._highlighted[cid]
                    hi_brushes.append(pg.mkBrush(255, 221, 0, 0))
                else:
                    remaining = expiry - now
                    alpha = int(200 * remaining / self._highlight_duration)
                    hi_brushes.append(pg.mkBrush(255, 221, 0, alpha))
            else:
                hi_brushes.append(pg.mkBrush(255, 221, 0, 0))
            hi_sizes.append(self._node_size(cid) + _HIGHLIGHT_SIZE_BONUS)
        self._highlight_scatter.setBrush(hi_brushes)
        self._highlight_scatter.setSize(hi_sizes)

        # Node label text
        for cid in concept_ids:
            e = self._kb.get_concept(cid).epistemic_error
            a = self._attention.get(cid, 0.0)
            self._label_items[cid].setText(f'{cid}\nE={e:.2f}  A={a:.2f}')

    def _node_size(self, cid: str) -> float:
        a = self._attention.get(cid, 0.0)
        return _NODE_SIZE_MIN + (_NODE_SIZE_MAX - _NODE_SIZE_MIN) * a

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def _print_terminal(self) -> None:
        top = sorted(self._attention.items(), key=lambda x: x[1], reverse=True)[:5]
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

    def _on_radio_clicked(self, button: QtWidgets.QRadioButton) -> None:
        label = button.text()
        self._goal_id = label
        if self._am is not None:
            self._am.set_goal(label)
        self._recompute_attention()
        self._update_artists()
        print(f"[GOAL  ]  Mission changed → '{label}'")

    def _on_scene_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self._plot.plotItem.vb
        pos = vb.mapSceneToView(event.scenePos())
        x, y = pos.x(), pos.y()

        clicked = self._nearest_node(x, y)
        if clicked is None:
            return

        concept = self._kb.get_concept(clicked)
        if concept.concept_type == 'task':
            self._goal_id = clicked
            if self._am is not None:
                self._am.set_goal(clicked)
            if clicked in self._radio_buttons:
                self._radio_buttons[clicked].setChecked(True)
            self._recompute_attention()
            self._update_artists()
            print(f"[GOAL  ]  Mission changed → '{clicked}'")
        else:
            before = concept.epistemic_error
            self._kb.refresh_concept(clicked, refresh=0.5)
            after = concept.epistemic_error
            self._highlighted[clicked] = time.time() + self._highlight_duration
            print(f"[MANUAL]  '{clicked}'  E: {before:.3f} → {after:.3f}")
            self._update_artists()

    def _nearest_node(self, x: float, y: float) -> str | None:
        best_id = None
        best_dist = _HIT_RADIUS
        for cid, (nx_, ny_) in self._pos.items():
            dist = math.hypot(x - nx_, y - ny_)
            if dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id
