from pathlib import Path

from awareness_manager.concept import Concept
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.loader import (
    load_instance_kb_from_ttl,
    load_kb_from_ttl,
    load_zone_assignment_from_ttl,
)

_TTL = Path(__file__).parent / 'social_serving.ttl'

# Drink preferences per person class (used by the scenario controller)
PREFERRED_DRINKS: dict[str, list[str]] = {
    'child': ['juice', 'cola'],
    'adult': ['beer', 'wine'],
    'VIP':   ['champagne', 'wine'],
}

# Cross-zone travel times in seconds (symmetric).
# Within-zone travel costs are zero (only observation_cost applies).
ZONE_TRAVEL_TIMES: dict[tuple[str, str], float] = {
    ('table_area', 'restock_zone'): 4.0,
    ('restock_zone', 'table_area'): 4.0,
}


def build_social_serving_kb() -> KnowledgeBase:
    return load_kb_from_ttl(_TTL)


def build_social_serving_instance_kb() -> InstanceKnowledgeBase:
    return load_instance_kb_from_ttl(_TTL)


def load_zone_assignment() -> dict[str, str]:
    """Return {concept_id: zone_name} for all instances that carry am:zone."""
    return load_zone_assignment_from_ttl(_TTL)


def preferred_drinks_for(all_class_ids: list[str]) -> list[str]:
    """Return the union of preferred drink classes for a multi-class person."""
    seen: set[str] = set()
    drinks: list[str] = []
    for cls in all_class_ids:
        for d in PREFERRED_DRINKS.get(cls, []):
            if d not in seen:
                seen.add(d)
                drinks.append(d)
    return drinks


def create_serve_goal(kb: KnowledgeBase, person_id: str, drink_class: str) -> str:
    """Create or update a parameterised serve goal for a specific person and drink class.

    The goal node ``serve_<person_id>`` is added to the class KB if it does not exist,
    with edges to:
      - ``person`` class (weight 2.0) — keeps general person awareness alive
      - ``drink_class`` (weight 1.0) — focuses attention on this specific drink

    The drink-class edge accumulates: calling with a different drink_class adds a new
    edge so the AM can consider all attempted drinks. This is correct — if beer is gone,
    the wine edge that was added earlier surfaces naturally via (E × A) / cost.

    Returns the goal_id to pass to ``am.set_goal()``.
    """
    goal_id = f'serve_{person_id}'
    if goal_id not in kb.concept_ids():
        kb.add_concept(Concept(
            concept_id=goal_id,
            concept_type='task',
            decay_rate=0.0,
        ))
        # weight=3.0 → A=0.125 for persons during serving, low enough that
        # drink instances (A=0.5) win after zone-travel cost is factored in.
        kb.add_relation(goal_id, 'person', weight=3.0)
    if drink_class in kb.concept_ids():
        kb.add_relation(goal_id, drink_class, weight=1.0)
    return goal_id
