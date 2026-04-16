"""
Demo: PV inspection knowledge base live visualiser with Awareness Manager.

Run from the workspace root:
    python3 src/awareness_manager/demos/run_pv_inspection_viz.py

Two task nodes demonstrate top-down goal switching:
    inspect_pv_field    — normal inspection: attends to panels, camera, light, wind
    emergency_landing   — battery emergency: attends to battery, landing zone, airspace

Switch goals by clicking a task node or using the RadioButtons panel on the right.
Watch how the size distribution of nodes completely reshuffles on goal change.

Controls:
    Click a non-task node  — manual observation (refresh)
    Click a task node      — switch to that mission goal
    RadioButtons (right)   — switch active goal from the list
"""

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb
from awareness_manager.visualizer import KBVisualizer

_REFRESH_INTERVAL = 2.0    # real seconds between observations (observation cadence)
_FORMULA3_INTERVAL = 10.0  # simulated seconds each observation compensates for (Formula 3)


def main() -> None:
    kb = build_pv_inspection_kb()
    am = AwarenessManager(
        kb,
        goal_id='inspect_pv_field',
        budget=2,
        observation_interval=_FORMULA3_INTERVAL,
    )
    viz = KBVisualizer(
        kb,
        goal_id='inspect_pv_field',
        sim_interval=0.1,
        frame_interval=0.1,
        awareness_manager=am,
        refresh_interval=_REFRESH_INTERVAL,
        highlight_duration=1.5,
    )

    print("=" * 70)
    print("  Robot Awareness — PV Inspection Demo  (CoreSense D7.1)")
    print("  Goal: inspect_pv_field → switch to emergency_landing")
    print("  Watch node sizes reshuffle when the goal changes.")
    print("=" * 70)

    viz.start()


if __name__ == '__main__':
    main()
