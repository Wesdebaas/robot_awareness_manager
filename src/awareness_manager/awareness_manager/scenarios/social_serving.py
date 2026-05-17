from pathlib import Path

from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.loader import load_instance_kb_from_ttl, load_kb_from_ttl

_TTL = Path(__file__).parent / 'social_serving.ttl'

# Drink preferences per person class (used by the scenario controller)
PREFERRED_DRINKS: dict[str, list[str]] = {
    'child': ['juice', 'cola'],
    'adult': ['beer', 'wine'],
    'VIP':   ['champagne', 'wine'],
}


def build_social_serving_kb() -> KnowledgeBase:
    return load_kb_from_ttl(_TTL)


def build_social_serving_instance_kb() -> InstanceKnowledgeBase:
    return load_instance_kb_from_ttl(_TTL)


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
