"""
Side-by-side live comparison: Full System vs. one formula disabled.

Both panels run the same birdhouse scenario in lockstep. The left panel always
uses all five formulas. The right panel disables the chosen formula and shows
its degraded baseline behaviour. A shared metric strip at the bottom tracks
attention-weighted epistemic error Σ(E·A)/Σ(A) in real time.

Run:
    python3 src/awareness_manager/demos/run_birdhouse_compare.py
    python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_drift
    python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_spreading
    python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_anticipatory
    python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_saturation
    python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_budget

Both AMs use the extended birdhouse KB (which includes a disconnected store_tools
subgraph). At t=25 simulated seconds the store_tools goal auto-promotes, causing
a visible attention reshuffle. F2 OFF is most striking here: store_tools concepts
were invisible (A=0) until the switch, so their E spiked unchecked while the full
system was pre-tuning.
"""

import argparse
import math
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import networkx as nx

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.feature_config import FeatureConfig
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb_extended

# ---------------------------------------------------------------------------
# Named configs
# ---------------------------------------------------------------------------

_NAMED_CONFIGS: dict[str, FeatureConfig] = {
    'no_spreading':    FeatureConfig.with_disabled('f1'),
    'no_anticipatory': FeatureConfig.with_disabled('f2'),
    'no_saturation':   FeatureConfig.with_disabled('f3'),
    'no_budget':       FeatureConfig.with_disabled('f4'),
    'no_drift':        FeatureConfig.with_disabled('f5'),
}

_LABELS: dict[str, str] = {
    'no_spreading':    'F1 OFF — Uniform attention',
    'no_anticipatory': 'F2 OFF — No lookahead',
    'no_saturation':   'F3 OFF — Over-refresh',
    'no_budget':       'F4 OFF — No memory budget',
    'no_drift':        'F5 OFF — Attention-only priority',
}

_FORMULA_NAMES: dict[str, str] = {
    'no_spreading':    'F1: Spreading Activation',
    'no_anticipatory': 'F2: Anticipatory Horizon',
    'no_saturation':   'F3: Utility Saturation',
    'no_budget':       'F4: Memory Budget Constraint',
    'no_drift':        'F5: Epistemic Drift',
}

# ---------------------------------------------------------------------------
# Visual constants  (dark theme matching the existing KBVisualizer)
# ---------------------------------------------------------------------------

_BG         = '#1e1e2e'
_NODE_MIN   = 200
_NODE_MAX   = 1600
_CMAP       = plt.cm.RdYlGn_r     # green=fresh (E=0), red=stale (E=1)
_HI_COLOR   = '#ffdd00'            # yellow ring: just observed
_HI_BONUS   = 450                  # extra scatter size for highlight ring
_HI_DUR     = 1.5                  # real seconds the ring stays visible

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

_SIM_STEP     = 0.2    # simulated seconds per animation frame  (2× real-time)
_FRAME_MS     = 100    # real milliseconds per frame (10 fps)
_OBS_SIM_S    = 2.0    # simulated seconds between observation events
_F3_INTERVAL  = 10.0   # observation_interval fed to AM for Formula 3 calibration
_BUDGET       = 2
_ALPHA        = 0.5
_MAX_DIST     = 4.0
_MEMORY_BUDGET = 9     # used only for no_budget config  (depth = √9-1 = 2)

_INITIAL_GOAL = 'build_birdhouse'
_QUEUED_GOAL  = 'store_tools'
_SWITCH_ETA   = 25.0   # sim-seconds until goal promotes

_METRIC_MAXLEN = 600   # rolling window length (frames) for the metric chart


# ---------------------------------------------------------------------------
# _Panel — one side of the comparison
# ---------------------------------------------------------------------------

class _Panel:
    """
    Manages one knowledge base + awareness manager + set of matplotlib artists
    for one half of the comparison window.
    """

    def __init__(
        self,
        ax: plt.Axes,
        kb,
        am: AwarenessManager,
        pos: dict,
    ) -> None:
        self._ax = ax
        self._kb = kb
        self._am = am
        self._pos = pos
        self._concept_ids = kb.concept_ids()
        self._highlighted: dict[str, float] = {}   # cid → real-time expiry

        ax.set_facecolor(_BG)
        ax.axis('off')

        graph = kb._graph

        # Static: edges (drawn once, never touched again)
        widths = [3.5 / graph[u][v]['weight'] ** 2 for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, ax=ax, width=widths,
                               edge_color='#777777', alpha=0.5)

        # Dynamic: node scatter
        attn = am.attention()
        self._nodes = nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            nodelist=self._concept_ids,
            node_color=[_CMAP(kb.get_concept(c).epistemic_error) for c in self._concept_ids],
            node_size=[_NODE_MIN + (_NODE_MAX - _NODE_MIN) * attn.get(c, 0)
                       for c in self._concept_ids],
            alpha=0.92,
        )
        self._nodes.set_zorder(3)

        # Highlight rings (one per concept, initially invisible)
        self._hi: dict[str, object] = {}
        for cid in self._concept_ids:
            h = nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[cid],
                                       node_color=_HI_COLOR, node_size=_NODE_MIN, alpha=0.0)
            h.set_zorder(2)
            self._hi[cid] = h

        # Dynamic: text labels
        self._texts: dict[str, plt.Text] = {}
        for cid in self._concept_ids:
            x, y = pos[cid]
            t = ax.text(
                x, y, self._lbl(cid, attn),
                ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=_BG, alpha=0.6, edgecolor='none'),
                zorder=5,
            )
            self._texts[cid] = t

        # Mutable title (updated with goal + sim time each frame)
        self._title = ax.set_title('', color='white', fontsize=10, pad=6)

    # --- helpers ---

    def _lbl(self, cid: str, attn: dict) -> str:
        e = self._kb.get_concept(cid).epistemic_error
        a = attn.get(cid, 0.0)
        return f"{cid}\nE={e:.2f}  A={a:.2f}"

    # --- public API ---

    def set_title(self, text: str) -> None:
        self._title.set_text(text)

    def trigger_highlight(self, cid: str) -> None:
        self._highlighted[cid] = time.time() + _HI_DUR

    def update(self) -> None:
        """Mutate all dynamic artists in-place. Called once per animation frame."""
        attn = self._am.attention()
        now  = time.time()

        # Node colours + sizes
        colors = [_CMAP(self._kb.get_concept(c).epistemic_error) for c in self._concept_ids]
        sizes  = [_NODE_MIN + (_NODE_MAX - _NODE_MIN) * attn.get(c, 0)
                  for c in self._concept_ids]
        self._nodes.set_facecolor(colors)
        self._nodes.set_sizes(sizes)

        # Labels
        for cid in self._concept_ids:
            self._texts[cid].set_text(self._lbl(cid, attn))

        # Highlight rings: fade over _HI_DUR seconds
        for cid, h in self._hi.items():
            if cid in self._highlighted:
                expiry = self._highlighted[cid]
                if now >= expiry:
                    del self._highlighted[cid]
                    h.set_alpha(0.0)
                else:
                    frac = (expiry - now) / _HI_DUR
                    hi_sz = _NODE_MIN + (_NODE_MAX - _NODE_MIN) * attn.get(cid, 0) + _HI_BONUS
                    h.set_sizes([hi_sz])
                    h.set_alpha(frac * 0.85)
            else:
                h.set_alpha(0.0)

    def weighted_e(self) -> float:
        non_task = [c for c in self._concept_ids
                    if self._kb.get_concept(c).concept_type != 'task']
        errors = [self._kb.get_concept(c).epistemic_error for c in non_task]
        attns  = [self._am.attention().get(c, 0.0)         for c in non_task]
        denom  = sum(attns) or 1.0
        return sum(e * a for e, a in zip(errors, attns)) / denom


# ---------------------------------------------------------------------------
# ComparisonViz — top-level controller
# ---------------------------------------------------------------------------

class ComparisonViz:

    def __init__(self, config_name: str) -> None:
        fc_ablated   = _NAMED_CONFIGS[config_name]
        side_label   = _LABELS[config_name]
        formula_name = _FORMULA_NAMES[config_name]
        test_f4      = (config_name == 'no_budget')

        # Build two independent KB + AM pairs with the same graph structure
        kb_l = build_birdhouse_kb_extended()
        kb_r = build_birdhouse_kb_extended()

        shared_pos = nx.spring_layout(kb_l._graph, seed=42)

        am_l = AwarenessManager(
            kb_l, goal_id=_INITIAL_GOAL,
            alpha=_ALPHA, max_distance=_MAX_DIST,
            budget=_BUDGET, observation_interval=_F3_INTERVAL,
            memory_budget=_MEMORY_BUDGET if test_f4 else None,
            feature_config=FeatureConfig.all_on(),
        )
        am_r = AwarenessManager(
            kb_r, goal_id=_INITIAL_GOAL,
            alpha=_ALPHA, max_distance=_MAX_DIST,
            budget=_BUDGET, observation_interval=_F3_INTERVAL,
            memory_budget=_MEMORY_BUDGET if test_f4 else None,
            feature_config=fc_ablated,
        )

        # Queue the disconnected future goal in both AMs
        am_l.queue_goal(_QUEUED_GOAL, eta=_SWITCH_ETA, level='task')
        am_r.queue_goal(_QUEUED_GOAL, eta=_SWITCH_ETA, level='task')

        self._elapsed   = 0.0
        self._last_obs  = -_OBS_SIM_S   # observe on the first eligible tick
        self._side_label = side_label
        self._formula    = formula_name

        # Rolling metric history
        self._hist_t    = deque(maxlen=_METRIC_MAXLEN)
        self._hist_l    = deque(maxlen=_METRIC_MAXLEN)
        self._hist_r    = deque(maxlen=_METRIC_MAXLEN)

        # ------------------------------------------------------------------
        # Figure layout
        # ------------------------------------------------------------------
        self._fig = plt.figure(figsize=(18, 9))
        self._fig.patch.set_facecolor(_BG)

        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3, 1],
            hspace=0.38, wspace=0.04,
            left=0.02, right=0.98, top=0.91, bottom=0.07,
        )

        ax_l = self._fig.add_subplot(gs[0, 0])
        ax_r = self._fig.add_subplot(gs[0, 1])
        ax_m = self._fig.add_subplot(gs[1, :])

        self._panel_l = _Panel(ax_l, kb_l, am_l, shared_pos)
        self._panel_r = _Panel(ax_r, kb_r, am_r, shared_pos)

        # Metric strip
        ax_m.set_facecolor(_BG)
        for spine in ax_m.spines.values():
            spine.set_edgecolor('#555566')
        ax_m.tick_params(colors='#cccccc', labelsize=8)
        ax_m.set_xlabel('Simulated time  (s)', color='#cccccc', fontsize=9)
        ax_m.set_ylabel('Σ(E·A) / Σ(A)', color='#cccccc', fontsize=9)
        ax_m.set_title('Attention-weighted epistemic error', color='#aaaacc', fontsize=9)
        ax_m.set_xlim(0, _SWITCH_ETA * 2.5)
        ax_m.set_ylim(0, 0.45)

        self._line_l, = ax_m.plot([], [], color='#2196F3', lw=1.8, label='Full system (all ON)')
        self._line_r, = ax_m.plot([], [], color='#FF5722', lw=1.8, ls='--', label=side_label)
        ax_m.axvline(_SWITCH_ETA, color='#666688', lw=0.9, ls=':', label='goal switch')
        leg = ax_m.legend(facecolor='#2a2a3e', edgecolor='#444455',
                          labelcolor='#ddddee', fontsize=8, loc='upper left')
        self._ax_m = ax_m

        # Super-title
        self._fig.suptitle(
            f'Live Comparison  —  {formula_name}\n'
            f'Left: full system  |  Right: {side_label}  '
            f'(budget={_BUDGET}, α={_ALPHA}, obs every {_OBS_SIM_S}s sim)',
            color='white', fontsize=11,
        )

        # Start animation
        self._anim = animation.FuncAnimation(
            self._fig, self._frame, interval=_FRAME_MS, cache_frame_data=False,
        )

        print('=' * 70)
        print(f'  Comparison: Full System  vs.  {side_label}')
        print(f'  Sim runs at 2× real time. Goal switches at t={_SWITCH_ETA}s sim.')
        print(f'  Node colour: green=fresh (E≈0), red=stale (E≈1)')
        print(f'  Node size:   larger = more attention')
        print(f'  Yellow ring: concept just observed by the AM')
        print('=' * 70)

        plt.show()

    # ------------------------------------------------------------------
    # Animation callback — called every _FRAME_MS real milliseconds
    # ------------------------------------------------------------------

    def _frame(self, _fn) -> None:
        # Advance simulation by a fixed step each frame (2× real-time)
        self._elapsed   += _SIM_STEP
        self._panel_l._am.tick(_SIM_STEP)
        self._panel_r._am.tick(_SIM_STEP)

        # Observation events (tied to sim time)
        if self._elapsed - self._last_obs >= _OBS_SIM_S:
            self._last_obs = self._elapsed
            for panel in (self._panel_l, self._panel_r):
                for cid in panel._am._top_n():
                    panel._am.observe(cid)
                    panel.trigger_highlight(cid)

        # Update titles with current goal + sim time
        self._panel_l.set_title(
            f'Full System  |  goal: {self._panel_l._am.goal_id}  '
            f't = {self._elapsed:.1f}s'
        )
        self._panel_r.set_title(
            f'{self._side_label}  |  goal: {self._panel_r._am.goal_id}'
        )

        # Update graph artists
        self._panel_l.update()
        self._panel_r.update()

        # Record metrics
        self._hist_t.append(self._elapsed)
        self._hist_l.append(self._panel_l.weighted_e())
        self._hist_r.append(self._panel_r.weighted_e())

        # Update metric lines
        ts = list(self._hist_t)
        self._line_l.set_data(ts, list(self._hist_l))
        self._line_r.set_data(ts, list(self._hist_r))

        # Autoscale Y on metric chart
        all_we = list(self._hist_l) + list(self._hist_r)
        if all_we:
            y_max = max(0.12, max(all_we) * 1.2)
            self._ax_m.set_ylim(0, y_max)

        self._fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Side-by-side live comparison: Full System vs. formula disabled.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            'Available configs:',
            *[f'  {k:<18} — {v}' for k, v in _LABELS.items()],
        ]),
    )
    p.add_argument(
        '--config',
        choices=list(_NAMED_CONFIGS),
        default='no_drift',
        help='Which formula to disable on the right panel (default: no_drift).',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ComparisonViz(args.config)


if __name__ == '__main__':
    main()
