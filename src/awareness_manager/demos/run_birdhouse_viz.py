"""
Demo: birdhouse knowledge base live visualiser.

Run from the workspace root:
    python3 src/awareness_manager/demos/run_birdhouse_viz.py

Controls:
    Click a tool/object/location node  — refresh it (simulate an observation)
    Click a task node                  — switch to that mission goal
    RadioButtons (right panel)         — switch active goal from the list
"""

from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.visualizer import KBVisualizer


def main() -> None:
    kb = build_birdhouse_kb()
    viz = KBVisualizer(kb, goal_id='build_birdhouse', sim_interval=0.1, frame_interval=0.1)

    print("=" * 70)
    print("  Robot Awareness — Birdhouse Demo")
    print("  Watch nodes turn red as epistemic error grows.")
    print("  Click a node to refresh it. Click a task node to change the goal.")
    print("=" * 70)

    viz.start()


if __name__ == '__main__':
    main()
