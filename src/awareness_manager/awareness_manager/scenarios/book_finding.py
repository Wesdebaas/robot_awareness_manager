"""
book_finding.py — "Find My Book" awareness scenario.

A domestic robot navigates a multi-room house to find a book.
The AM pre-tunes awareness for room-specific book locations as the
robot moves, demonstrating F1 (spreading activation across rooms),
F2 (anticipatory pre-tuning when a nav goal is set), F5 (epistemic
error drives which room to check first), and F6 (spatial opportunity
cost — books in the current room are cheapest to observe).

Room topology (KRR_Course_Small_house):
    living_room ── kitchen
         │
    bathroom
         │
    bedroom ── study

Zone travel time table (seconds) — symmetric, approximate.
Used as F6 zone-to-zone transit cost.
"""

from pathlib import Path

from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.scenarios.loader import (
    load_kb_from_ttl,
    load_instance_kb_from_ttl,
    load_zone_assignment_from_ttl,
)

# ── Spatial constants ─────────────────────────────────────────────────────────

# Approximate transit times between room zones (seconds at walking speed).
# Robot starts in living_room.
ZONE_TRAVEL_TIMES: dict[tuple[str, str], float] = {
    ("living_room", "living_room"): 0.0,
    ("living_room", "kitchen"):     8.0,
    ("living_room", "bathroom"):    6.0,
    ("living_room", "bedroom"):    10.0,
    ("living_room", "study"):      15.0,
    ("kitchen",     "living_room"): 8.0,
    ("kitchen",     "kitchen"):     0.0,
    ("kitchen",     "bathroom"):   10.0,
    ("kitchen",     "bedroom"):    14.0,
    ("kitchen",     "study"):      18.0,
    ("bathroom",    "living_room"): 6.0,
    ("bathroom",    "kitchen"):    10.0,
    ("bathroom",    "bathroom"):    0.0,
    ("bathroom",    "bedroom"):     8.0,
    ("bathroom",    "study"):      12.0,
    ("bedroom",     "living_room"):10.0,
    ("bedroom",     "kitchen"):    14.0,
    ("bedroom",     "bathroom"):    8.0,
    ("bedroom",     "bedroom"):     0.0,
    ("bedroom",     "study"):       5.0,
    ("study",       "living_room"):15.0,
    ("study",       "kitchen"):    18.0,
    ("study",       "bathroom"):   12.0,
    ("study",       "bedroom"):     5.0,
    ("study",       "study"):       0.0,
}

# Room bounding boxes in map frame (x_min, x_max, y_min, y_max).
# Used by the ROS node to map amcl_pose → current zone.
# Approximate values for KRR_Course_Small_house; tune after first launch.
ROOM_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "living_room": (-3.0,  2.5, -0.5,  5.0),
    "kitchen":     ( 1.0,  4.5, -0.5,  4.0),
    "bathroom":    (-1.5,  2.5, -5.0, -0.5),
    "bedroom":     (-5.5, -1.5, -0.5,  4.5),
    "study":       (-8.5, -5.0, -0.5,  4.5),
}

_TTL = Path(__file__).with_name("book_finding.ttl")


def build_book_finding_kb() -> KnowledgeBase:
    """Load the class-level knowledge base from the TTL ontology."""
    return load_kb_from_ttl(_TTL)


def build_book_finding_instance_kb() -> InstanceKnowledgeBase:
    """Load the instance-level KB (one book per room + owner)."""
    return load_instance_kb_from_ttl(_TTL)


def load_zone_assignment() -> dict[str, str]:
    """Return mapping of concept/instance ID → zone name (from am:zone in TTL)."""
    return load_zone_assignment_from_ttl(_TTL)


def get_room_for_pose(x: float, y: float) -> str | None:
    """Return the zone name for a map-frame (x, y) pose, or None if unknown."""
    for room, (xmin, xmax, ymin, ymax) in ROOM_BOUNDS.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return room
    return None
