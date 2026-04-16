from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase


def build_birdhouse_kb() -> KnowledgeBase:
    """
    Manufacturing scenario: building a birdhouse.

    Based on the CoreSense Manufacturing Testbed (D6.1). Provides a concrete
    semantic graph for developing and evaluating the awareness manager.

    Decay rates (delta) reflect how quickly each concept's validity degrades:
        0.0   /s  task node        — abstract goal, does not decay
        0.001 /s  furniture        — workbench/tool_rack rarely moves
        0.005 /s  large materials  — wood plank moved infrequently
        0.01  /s  tools            — may be picked up or relocated
        0.02  /s  small fasteners  — nails/screws frequently repositioned
        0.1   /s  human hand       — position changes continuously

    Edge weights represent initial semantic distance (lower = closer):
        1.0  — direct task dependency or tight functional coupling
               (e.g. build_birdhouse→hammer, hammer↔nail)
        1.5  — human agent to tool: indirect, depends on human action
        2.0  — tool or material to its storage location: present there
               but not functionally bound

    Weight rationale: initial values are set by domain knowledge. The
    architecture is designed so these can be updated from co-occurrence
    statistics without structural changes to the KB.
    """
    kb = KnowledgeBase()

    # --- Task (goal node) ---
    kb.add_concept(Concept('build_birdhouse', 'task',     decay_rate=0.0))

    # --- Tools ---
    kb.add_concept(Concept('hammer',          'object',   decay_rate=0.01))
    kb.add_concept(Concept('saw',             'object',   decay_rate=0.01))
    kb.add_concept(Concept('drill',           'object',   decay_rate=0.01))

    # --- Fasteners / consumables ---
    kb.add_concept(Concept('nail',            'object',   decay_rate=0.02))
    kb.add_concept(Concept('screw',           'object',   decay_rate=0.02))

    # --- Materials ---
    kb.add_concept(Concept('wood_plank',      'object',   decay_rate=0.005))

    # --- Locations ---
    kb.add_concept(Concept('workbench',       'location', decay_rate=0.001))
    kb.add_concept(Concept('tool_rack',       'location', decay_rate=0.001))

    # --- Human ---
    kb.add_concept(Concept('human_hand',      'state',    decay_rate=0.1))

    # Task → primary components (direct task dependencies)
    kb.add_relation('build_birdhouse', 'hammer',     weight=1.0)
    kb.add_relation('build_birdhouse', 'nail',       weight=1.0)
    kb.add_relation('build_birdhouse', 'wood_plank', weight=1.0)
    kb.add_relation('build_birdhouse', 'saw',        weight=1.0)
    kb.add_relation('build_birdhouse', 'screw',      weight=1.0)
    kb.add_relation('build_birdhouse', 'drill',      weight=1.0)

    # Tool ↔ fastener/material (tight functional coupling)
    kb.add_relation('hammer',     'nail',       weight=1.0)
    kb.add_relation('drill',      'screw',      weight=1.0)
    kb.add_relation('saw',        'wood_plank', weight=1.0)

    # Tool/material ↔ location (present at location, not functionally bound)
    kb.add_relation('hammer',     'workbench',  weight=2.0)
    kb.add_relation('saw',        'workbench',  weight=2.0)
    kb.add_relation('nail',       'workbench',  weight=2.0)
    kb.add_relation('wood_plank', 'workbench',  weight=2.0)
    kb.add_relation('drill',      'workbench',  weight=2.0)
    kb.add_relation('screw',      'workbench',  weight=2.0)
    kb.add_relation('hammer',     'tool_rack',  weight=2.0)
    kb.add_relation('saw',        'tool_rack',  weight=2.0)
    kb.add_relation('drill',      'tool_rack',  weight=2.0)

    # Human ↔ tools (indirect coupling via human action)
    kb.add_relation('human_hand', 'hammer',     weight=1.5)
    kb.add_relation('human_hand', 'saw',        weight=1.5)
    kb.add_relation('human_hand', 'drill',      weight=1.5)

    return kb
