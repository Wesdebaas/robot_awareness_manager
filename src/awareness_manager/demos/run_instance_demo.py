"""
Demo: class/instance two-layer awareness on the birdhouse scenario.

Prints a live terminal table showing class attention, instance attention,
and the merged schedule. Run from the workspace root:

    python3 src/awareness_manager/demos/run_instance_demo.py

What to watch for:
  - Instances of *attended* classes inherit attention (e.g. hammer_bench inherits
    from hammer).
  - Instances of *unattended* classes stay at zero (the class gate).
  - After switching the goal to 'workbench', the instance attention reshuffles:
    workbench_north/workbench_south become relevant, hammer instances drop.
  - The merged schedule (class + instance) never exceeds the budget.
"""

import os
import time

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.birdhouse import (
    build_birdhouse_instance_kb,
    build_birdhouse_kb,
)

# ── ANSI colour helpers ──────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + _DIM + "░" * (width - filled) + _RESET

def _attention_colour(a: float) -> str:
    if a > 0.6:
        return _GREEN
    if a > 0.2:
        return _YELLOW
    return _DIM

# ── Main demo ────────────────────────────────────────────────────────────────

def _print_table(
    am: AwarenessManager,
    goal: str,
    schedule: list[str],
    sim_time: float,
) -> None:
    attn = am.attention()
    prio = am.priorities()
    ikb = am.instance_kb

    print(f"\n{_BOLD}t={sim_time:6.1f}s  goal={goal}  schedule={schedule}{_RESET}")

    # Class-level
    print(f"\n  {_CYAN}{'CLASS CONCEPTS':30s} {'ATTENTION':>8}  {'PRIORITY':>8}{_RESET}")
    print(f"  {'─'*54}")
    for cid in sorted(attn.keys()):
        if ikb is not None and cid in ikb.instance_ids():
            continue  # skip instances in this section
        a = attn.get(cid, 0.0)
        p = prio.get(cid, 0.0)
        marker = " ◀" if cid in schedule else ""
        col = _attention_colour(a)
        print(f"  {col}{cid:30s}{_RESET}  {a:7.3f}  {p:8.4f}{marker}")

    # Instance-level
    if ikb is not None:
        print(f"\n  {_CYAN}{'INSTANCE CONCEPTS':30s} {'ATTENTION':>8}  {'PRIORITY':>8}  CLASS{_RESET}")
        print(f"  {'─'*64}")
        for iid in sorted(ikb.instance_ids()):
            inst = ikb.get_instance(iid)
            a = attn.get(iid, 0.0)
            p = prio.get(iid, 0.0)
            marker = " ◀" if iid in schedule else ""
            col = _attention_colour(a)
            print(f"  {col}{iid:30s}{_RESET}  {a:7.3f}  {p:8.4f}  {_DIM}{inst.class_id}{_RESET}{marker}")


def main() -> None:
    BUDGET = 4
    OBS_INTERVAL = 2.0
    DT = 0.5

    kb  = build_birdhouse_kb()
    ikb = build_birdhouse_instance_kb()
    am  = AwarenessManager(
        kb,
        goal_id='build_birdhouse',
        budget=BUDGET,
        observation_interval=OBS_INTERVAL,
        instance_kb=ikb,
        instance_relational_weight=0.3,
    )

    print("=" * 70)
    print("  Robot Awareness - Class/Instance Two-Layer Demo")
    print(f"  Budget: {BUDGET} concepts (class + instance combined)")
    print("  ◀ = in schedule this tick")
    print("=" * 70)
    print("\nPhase 1: goal=build_birdhouse  (5 ticks, 0.5s each)\n")

    sim_time = 0.0
    for _ in range(5):
        schedule = am.tick(dt=DT)
        # Auto-observe top concept
        if schedule:
            am.observe(schedule[0])
        _print_table(am, am.goal_id, schedule, sim_time)
        sim_time += DT
        time.sleep(0.3)

    print("\n" + "─" * 70)
    print("  Switching goal → 'workbench'")
    print("  Expected: workbench_north, workbench_south gain attention.")
    print("            hammer, nail, etc. lose attention.")
    print("─" * 70)
    am.set_goal('workbench')

    for _ in range(5):
        schedule = am.tick(dt=DT)
        if schedule:
            am.observe(schedule[0])
        _print_table(am, am.goal_id, schedule, sim_time)
        sim_time += DT
        time.sleep(0.3)

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("  Key observations:")
    print("  1. Instances inherit attention from their class (class gate).")
    print("  2. On goal switch, instance attention reshuffles with classes.")
    print("  3. Budget={} enforced across class + instance concepts.".format(BUDGET))
    print("=" * 70)


if __name__ == '__main__':
    main()
