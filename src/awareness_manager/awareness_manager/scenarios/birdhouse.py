from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
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


def build_birdhouse_kb_extended() -> KnowledgeBase:
    """
    Extended birdhouse KB with a *disconnected* store_tools subgraph.

    The store_tools task and its three satellite concepts share no edges with
    the build_birdhouse subgraph. This means:
        - With goal='build_birdhouse': store_tools concepts have A=0 (unreachable
          via spreading activation from build_birdhouse).
        - With Anticipatory Horizon (F2) ON and store_tools queued: these concepts
          gradually receive pre-tuned attention as ETA decreases.
        - With F2 OFF: they remain at A=0 until the goal switches, demonstrating
          the value of anticipatory lookahead.

    New concepts (not connected to existing graph):
        store_tools      — task, decay_rate=0
        storage_bin      — location, decay_rate=0.001  (rarely moved)
        cleaning_cloth   — object,   decay_rate=0.05   (frequently repositioned)
        tool_organizer   — object,   decay_rate=0.01   (occasionally moved)
    """
    kb = build_birdhouse_kb()

    kb.add_concept(Concept('store_tools',    'task',     decay_rate=0.0))
    kb.add_concept(Concept('storage_bin',    'location', decay_rate=0.001))
    kb.add_concept(Concept('cleaning_cloth', 'object',   decay_rate=0.05))
    kb.add_concept(Concept('tool_organizer', 'object',   decay_rate=0.01))

    kb.add_relation('store_tools',    'storage_bin',    weight=1.0)
    kb.add_relation('store_tools',    'cleaning_cloth', weight=1.0)
    kb.add_relation('store_tools',    'tool_organizer', weight=1.0)
    kb.add_relation('tool_organizer', 'storage_bin',    weight=1.5)
    kb.add_relation('cleaning_cloth', 'storage_bin',    weight=1.5)

    return kb


def build_birdhouse_instance_kb() -> InstanceKnowledgeBase:
    """
    Instance-level knowledge base for the birdhouse manufacturing scenario.

    Adds specific physical individuals for several birdhouse classes. This
    demonstrates the class/instance distinction: the class 'hammer' defines
    what a hammer is; instances 'hammer_rack' and 'hammer_bench' are the two
    specific hammers present in the workshop.

    Decay rates mirror those of the parent class — instances of a class
    degrade at roughly the same rate as the class itself.

    Instance relations use typed edges reflecting actual spatial / causal
    structure:
        locatedAt  — instance is physically at a location instance
        uses       — an agent instance actively uses a tool instance
        locatedOn  — a material/tool rests on a surface instance

    Semantic coverage (8 instances):
        hammer_rack, hammer_bench  — two hammers (class: hammer)
        workbench_north, workbench_south — two workbenches (class: workbench)
        alice_hand, bob_hand        — two workers' hands (class: human_hand)
        nail_box_A                  — fastener supply (class: nail)
        plank_1, plank_2            — two wood planks (class: wood_plank)
    """
    ikb = InstanceKnowledgeBase()

    # --- Hammer instances ---
    ikb.add_instance(InstanceConcept('hammer_rack',  'object', decay_rate=0.01, class_id='hammer'))
    ikb.add_instance(InstanceConcept('hammer_bench', 'object', decay_rate=0.01, class_id='hammer'))

    # --- Workbench instances ---
    ikb.add_instance(InstanceConcept('workbench_north', 'location', decay_rate=0.001, class_id='workbench'))
    ikb.add_instance(InstanceConcept('workbench_south', 'location', decay_rate=0.001, class_id='workbench'))

    # --- Human hand instances ---
    ikb.add_instance(InstanceConcept('alice_hand', 'state', decay_rate=0.1, class_id='human_hand'))
    ikb.add_instance(InstanceConcept('bob_hand',   'state', decay_rate=0.1, class_id='human_hand'))

    # --- Fastener instance ---
    ikb.add_instance(InstanceConcept('nail_box_A', 'object', decay_rate=0.02, class_id='nail'))

    # --- Material instances ---
    ikb.add_instance(InstanceConcept('plank_1', 'object', decay_rate=0.005, class_id='wood_plank'))
    ikb.add_instance(InstanceConcept('plank_2', 'object', decay_rate=0.005, class_id='wood_plank'))

    # --- Instance relations ---
    # hammer_bench is at the workbench (the one Alice is working at)
    ikb.add_instance_relation('hammer_bench', 'workbench_north', weight=1.0, relation_type='locatedAt')
    # hammer_rack is stored on the rack (off-task, tied to workbench_south)
    ikb.add_instance_relation('hammer_rack',  'workbench_south', weight=2.0, relation_type='locatedAt')
    # Alice is actively using hammer_bench
    ikb.add_instance_relation('alice_hand',   'hammer_bench',    weight=1.0, relation_type='uses')
    # Bob is at the south workbench
    ikb.add_instance_relation('bob_hand',     'workbench_south', weight=1.5, relation_type='locatedAt')
    # Materials are at the north workbench
    ikb.add_instance_relation('plank_1',      'workbench_north', weight=1.0, relation_type='locatedOn')
    ikb.add_instance_relation('plank_2',      'workbench_south', weight=1.5, relation_type='locatedOn')
    # Nails are at the north workbench
    ikb.add_instance_relation('nail_box_A',   'workbench_north', weight=1.0, relation_type='locatedAt')

    return ikb
