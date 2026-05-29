"""
drink_serving_gazebo.py — Scenario constants and KB builder for the Gazebo drink-serving demo.

The physical setup is the KRR_Course_Small_house world with five items placed on the
kitchen counter (y ≈ -1.0):
    Drinks:      kitchen_beer (−1.125, −1.0)  kitchen_wine (−2.0, −1.0)
    Distractors: kitchen_book (−0.25, −1.0)   kitchen_cube (0.625, −1.0)  kitchen_ball (1.5, −1.0)

Robot waypoints:
    kitchen:     (0.0,  0.0)   — centre of kitchen floor space
    living_room: (−8.0, 0.35)  — open area in front of sofas

ROOM_BOUNDS are approximate and may need tuning after a manual run that logs AMCL pose.
"""

from pathlib import Path

from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.loader import load_kb_from_ttl

_TTL_PATH = Path(__file__).parent / 'drink_serving_gazebo.ttl'

# ── Instance concept IDs matching Gazebo model names ─────────────────────────

DRINK_INSTANCE_IDS: list[str] = ['kitchen_beer', 'kitchen_wine']
DISTRACTOR_INSTANCE_IDS: list[str] = ['kitchen_book', 'kitchen_cube', 'kitchen_ball']

# Decay rate for all kitchen items (s⁻¹).
# 0.02 ensures 15 s travel adds only 0.3 E — enough for AM (E≈0.3) to stay
# below the 0.5 threshold while reactive-missed items (E≈0.9) exceed it.
ITEM_DECAY_RATE: float = 0.02

# ── Spatial constants ─────────────────────────────────────────────────────────

ZONE_TRAVEL_TIMES: dict[tuple[str, str], float] = {
    ('kitchen', 'living_room'): 15.0,
    ('living_room', 'kitchen'): 15.0,
}

# Approximate AMCL map-frame bounding boxes (x_min, x_max, y_min, y_max).
# Derived from:
#   robot spawn  (1.05, 0.51)
#   kitchen waypoint  (0.0, 0.0)  — near counter at y=−1.0
#   living_room waypoint  (−8.0, 0.35)
# Tune these after logging AMCL pose during a manual patrol run.
ROOM_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    'kitchen':     (-3.0,  3.5,  -3.5,  2.5),
    'living_room': (-13.0, -3.5, -3.0,  4.0),
}


def get_room_for_pose(x: float, y: float) -> str | None:
    """Return zone name for AMCL (x, y), or None if outside all defined bounds."""
    for zone, (x_min, x_max, y_min, y_max) in ROOM_BOUNDS.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return None


def build_drink_serving_kb() -> KnowledgeBase:
    """Load the class-level knowledge base from the TTL ontology."""
    return load_kb_from_ttl(_TTL_PATH)
