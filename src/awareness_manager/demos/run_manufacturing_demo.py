"""
Demo: Cross-class instance attention — CoreSense D6.1 Manufacturing Testbed.

Run from the workspace root:
    python3 src/awareness_manager/demos/run_manufacturing_demo.py

Scenario (CORESENSE D6.1, MT-1 Agile Manufacturing, IMR Mullingar)
-------------------------------------------------------------------
A collaborative UR5 robot assembles plastic spur gear units from parts
delivered by a KUKA KMR mobile manipulator.  The class graph encodes the
*semantic* relationships for the assembly task:

    assemble_gear_unit → gear_part, shaft, fastener, assembly_robot, assembly_station
    assembly_robot     → gripper

The following classes have NO semantic path to assemble_gear_unit:

    mobile_robot       (KMR — intralogistics, not assembly)
    human_operator     (Alice — loading/handover, not modelled as an assembly tool)
    laser_scanner      (Pilz SSM — safety, not an assembly resource)
    inspection_camera  (Zivid One+S — quality control, not assembly)

BUT at the instance level, these physical co-location edges exist:

    kmr_1          (mobile_robot)      --parkedAt-->  station_cell_1  [1-hop]
    operator_alice (human_operator)    --workingAt--> station_cell_1  [1-hop]
    pilz_scanner_1 (laser_scanner)     --monitors-->  station_cell_1  [1-hop]
    zivid_camera_1 (inspection_camera) --mountedOn--> kmr_1           [2-hop]

The demo prints a 'CROSS-CLASS SPOTLIGHT' table each tick showing which
instances gain attention purely through relational proximity — their parent
class has A=0 in the class graph, yet the instance has A>0.

What to watch for
-----------------
Phase 1  goal=assemble_gear_unit
  • gear/shaft/fastener instances → high attention (class gate)
  • kmr_1, operator_alice, pilz_scanner_1 → A>0 despite class A=0  (1-hop)
  • zivid_camera_1 → smaller A>0 (2-hop through kmr_1)

Phase 2  goal=transport_parts
  • kmr_1 gains CLASS-level attention (mobile_robot is now semantically relevant)
  • zivid_camera_1 boost STRENGTHENS: it rides on kmr_1's now-high base attention
  • gear/shaft instances DROP to near-zero
  • operator_alice and pilz_scanner_1 retain relational boosts (still at station)
"""

import os
import time

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.manufacturing_d61 import (
    build_manufacturing_d61_kb,
    build_manufacturing_d61_instance_kb,
)

# ── ANSI colour helpers ──────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _bar(value: float, width: int = 16) -> str:
    filled = int(round(value * width))
    return "█" * filled + _DIM + "░" * (width - filled) + _RESET

def _col(a: float) -> str:
    if a > 0.5:  return _GREEN
    if a > 0.15: return _YELLOW
    return _DIM

# ── Table helpers ─────────────────────────────────────────────────────────────

def _print_class_table(am: AwarenessManager, schedule: list[str]) -> None:
    kb = am._kb
    attn = am.attention()
    print(f"\n  {_CYAN}{'CLASS':<22} {'A':>6}  {'BAR':<18}{_RESET}")
    print(f"  {'─'*50}")
    for cid in sorted(kb.concept_ids()):
        a = attn.get(cid, 0.0)
        marker = " ◀" if cid in schedule else ""
        c = _col(a)
        print(f"  {c}{cid:<22}{_RESET}  {a:5.3f}  {_bar(a)}{marker}")


def _print_instance_table(am: AwarenessManager, schedule: list[str]) -> None:
    ikb = am.instance_kb
    if ikb is None:
        return
    attn = am.attention()
    kb_ids = set(am._kb.concept_ids())

    print(f"\n  {_CYAN}{'INSTANCE':<22} {'CLASS':<20} {'A':>6}  {'BAR':<18}{_RESET}")
    print(f"  {'─'*72}")
    for iid in sorted(ikb.instance_ids()):
        inst = ikb.get_instance(iid)
        a = attn.get(iid, 0.0)
        marker = " ◀" if iid in schedule else ""
        c = _col(a)
        print(f"  {c}{iid:<22}{_RESET}  {_DIM}{inst.class_id:<20}{_RESET}  {a:5.3f}  {_bar(a)}{marker}")


def _print_crossclass_spotlight(am: AwarenessManager) -> None:
    """
    Highlight instances whose class has A=0 in the class graph but whose
    own instance attention is > 0 (purely from relational proximity).
    """
    ikb = am.instance_kb
    if ikb is None:
        return
    attn = am.attention()

    cross = []
    for iid in ikb.instance_ids():
        inst = ikb.get_instance(iid)
        class_a = attn.get(inst.class_id, 0.0)
        inst_a  = attn.get(iid, 0.0)
        if class_a < 0.001 and inst_a > 0.001:
            cross.append((iid, inst.class_id, class_a, inst_a))

    if not cross:
        print(f"\n  {_DIM}(no cross-class instances with attention this tick){_RESET}")
        return

    print(f"\n  {_RED}{_BOLD}CROSS-CLASS SPOTLIGHT{_RESET}  "
          f"{_DIM}— class A=0, but instance A>0 via relational proximity{_RESET}")
    print(f"  {_RED}{'─'*70}{_RESET}")
    header = f"  {'INSTANCE':<22} {'CLASS':<22} {'cls A':>6}  {'inst A':>6}  BAR"
    print(f"  {_CYAN}{header.strip()}{_RESET}")
    print(f"  {'─'*70}")
    for iid, cid, class_a, inst_a in sorted(cross, key=lambda x: -x[3]):
        neighbors = ikb.neighbors_of(iid)
        rel_strs = []
        for nb in neighbors:
            rt = ikb.get_relation_type(iid, nb)
            nb_a = attn.get(nb, 0.0)
            if nb_a > 0.001:
                rel_strs.append(f"{rt}→{nb}(A={nb_a:.2f})")
        via = "  " + _DIM + " | ".join(rel_strs) + _RESET if rel_strs else ""
        print(f"  {_RED}{iid:<22}{_RESET}  {_DIM}{cid:<22}{_RESET}  "
              f"{class_a:5.3f}   {_YELLOW}{inst_a:5.3f}{_RESET}  {_bar(inst_a)}{via}")


# ── Main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    BUDGET          = 5
    OBS_INTERVAL    = 2.0
    DT              = 0.5
    TICKS_PHASE_1   = 6
    TICKS_PHASE_2   = 6

    kb  = build_manufacturing_d61_kb()
    ikb = build_manufacturing_d61_instance_kb()
    am  = AwarenessManager(
        kb,
        goal_id='assemble_gear_unit',
        budget=BUDGET,
        observation_interval=OBS_INTERVAL,
        instance_kb=ikb,
        instance_relational_weight=0.3,
    )

    print("=" * 72)
    print("  CoreSense D6.1 — Agile Manufacturing Testbed")
    print("  Cross-class instance attention demo")
    print()
    print("  Class graph for 'assemble_gear_unit' covers:")
    print("    gear_part · shaft · fastener · assembly_robot · gripper · assembly_station")
    print()
    print("  These classes have ZERO semantic link to the goal:")
    print("    mobile_robot · human_operator · laser_scanner · inspection_camera")
    print()
    print("  Instance-level co-location creates relational boosts:")
    print("    kmr_1          (mobile_robot)     --parkedAt-->  station_cell_1  [1-hop]")
    print("    operator_alice (human_operator)   --workingAt--> station_cell_1  [1-hop]")
    print("    pilz_scanner_1 (laser_scanner)    --monitors-->  station_cell_1  [1-hop]")
    print("    zivid_camera_1 (inspection_camera)--mountedOn--> kmr_1           [2-hop]")
    print("=" * 72)

    sim_time = 0.0

    # ── Phase 1: assemble_gear_unit ───────────────────────────────────────
    print(f"\n{_BOLD}Phase 1: goal=assemble_gear_unit  ({TICKS_PHASE_1} ticks){_RESET}\n")
    for tick in range(TICKS_PHASE_1):
        schedule = am.tick(dt=DT)
        if schedule:
            am.observe(schedule[0])

        print(f"\n{'─'*72}")
        print(f"  {_BOLD}t={sim_time:5.1f}s  goal=assemble_gear_unit  schedule={schedule}{_RESET}")

        _print_class_table(am, schedule)
        _print_instance_table(am, schedule)
        _print_crossclass_spotlight(am)

        sim_time += DT
        time.sleep(0.4)

    # ── Phase 2: transport_parts ──────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Switching goal → 'transport_parts'")
    print(f"  Expected changes:")
    print(f"    • kmr_1 GAINS class-level base attention (mobile_robot now relevant)")
    print(f"    • zivid_camera_1 boost STRENGTHENS (rides on kmr_1's higher seed attention)")
    print(f"    • gear/shaft instances DROP to near-zero")
    print(f"    • operator_alice / pilz_scanner_1 retain relational boosts (still at station)")
    print(f"{'═'*72}\n")
    am.set_goal('transport_parts')

    for tick in range(TICKS_PHASE_2):
        schedule = am.tick(dt=DT)
        if schedule:
            am.observe(schedule[0])

        print(f"\n{'─'*72}")
        print(f"  {_BOLD}t={sim_time:5.1f}s  goal=transport_parts  schedule={schedule}{_RESET}")

        _print_class_table(am, schedule)
        _print_instance_table(am, schedule)
        _print_crossclass_spotlight(am)

        sim_time += DT
        time.sleep(0.4)

    print(f"\n{'='*72}")
    print("  Demo complete.  Key observations:")
    print("  1. Cross-class instances (kmr_1, operator_alice, pilz_scanner_1) gain")
    print("     attention purely from physical co-location with the active assembly")
    print("     station — their parent classes have A=0 in the class graph.")
    print("  2. zivid_camera_1 gains a 2-hop boost: camera → KMR → assembly station.")
    print("  3. On goal switch, kmr_1 gains CLASS-level attention and its relational")
    print("     boost to zivid_camera_1 strengthens proportionally.")
    print("  4. This demonstrates that instance links can cross class boundaries —")
    print("     physical reality can differ from the semantic ontology.")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
