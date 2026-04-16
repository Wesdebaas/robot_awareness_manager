"""
Demo: PV inspection knowledge base live visualiser — PyQtGraph version.

Run from the workspace root:
    python3 src/awareness_manager/demos/run_pv_inspection_viz_qt.py

Identical scenario to run_pv_inspection_viz.py but uses PyQtGraph instead of
matplotlib for rendering. Typical render cost: ~5ms vs ~135ms per frame,
allowing smooth 20fps updates.

Controls:
    Click a non-task node  — manual observation (refresh)
    Click a task node      — switch to that mission goal
    RadioButtons (right)   — switch active goal from the list
"""

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb
from awareness_manager.visualizer_pyqtgraph import KBVisualizerQt

_REFRESH_INTERVAL = 2.0    # real seconds between observations
_FORMULA3_INTERVAL = 10.0  # simulated seconds each observation compensates for


def main() -> None:
    kb = build_pv_inspection_kb()
    am = AwarenessManager(
        kb,
        goal_id='inspect_pv_field',
        budget=2,
        observation_interval=_FORMULA3_INTERVAL,
        lambda_horizon=0.1,
    )
    am.queue_goal('emergency_landing', eta=30.0)

    viz = KBVisualizerQt(
        kb,
        goal_id='inspect_pv_field',
        sim_interval=0.1,
        frame_interval=0.05,   # 20 fps target — achievable with PyQtGraph
        awareness_manager=am,
        refresh_interval=_REFRESH_INTERVAL,
        highlight_duration=1.5,
    )

    print("=" * 70)
    print("  Robot Awareness — PV Inspection Demo  (PyQtGraph)")
    print("  Goal: inspect_pv_field → emergency_landing queued at t=30s")
    print("  Formula 2: watch battery/landing nodes grow as ETA approaches 0.")
    print("=" * 70)

    viz.start()


if __name__ == '__main__':
    main()
