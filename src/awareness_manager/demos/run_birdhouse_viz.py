"""
Demo: birdhouse knowledge base live visualiser with Awareness Manager.

Run from the workspace root:
    python3 src/awareness_manager/demos/run_birdhouse_viz.py

The AM has a budget of 1: it can only observe one concept every 2 real seconds.
The refresh amount per observation is computed via Formula 3 (Utility Saturation):
    refresh(n) = 1 − e^(−δ(n) × observation_interval)
so each observation exactly compensates for the drift accumulated since the last one.

Watch the AM prioritise whichever node has the highest E×A. The yellow ring shows
which concept was just observed and fades out over 1.5 seconds.

Controls:
    Click a tool/object/location node  — manual observation (bypasses budget)
    Click a task node                  — switch to that mission goal
    RadioButtons (right panel)         — switch active goal from the list
"""

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.visualizer import KBVisualizer

_OBSERVATION_INTERVAL = 2.0   # seconds between auto-observations


def main() -> None:
    kb = build_birdhouse_kb()
    am = AwarenessManager(
        kb,
        goal_id='build_birdhouse',
        budget=1,
        observation_interval=_OBSERVATION_INTERVAL,
    )
    viz = KBVisualizer(
        kb,
        goal_id='build_birdhouse',
        sim_interval=0.1,
        frame_interval=0.1,
        awareness_manager=am,
        refresh_interval=_OBSERVATION_INTERVAL,
        highlight_duration=1.5,
    )

    print("=" * 70)
    print("  Robot Awareness — Birdhouse Demo")
    print("  Budget: 1 observation every 2 seconds (Formula 3 refresh values).")
    print("  Yellow ring = node just observed by the AM.")
    print("  Watch it choose the highest-priority (E×A) concept each time.")
    print("=" * 70)

    viz.start()


if __name__ == '__main__':
    main()
