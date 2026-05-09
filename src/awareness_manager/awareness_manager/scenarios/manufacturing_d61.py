"""
CoreSense D6.1 - Agile Manufacturing Testbed scenario.

Based on the Manufacturing Testbed described in CORESENSE D6.1 (CS-025-v1.0),
specifically the MT-1 agile production use case at IMR Mullingar:

    Mobile robot (KUKA KMR iiwa)  - intralogistics, machine tending
    Assembly robot (UR5)           - collaborative gear unit assembly
    Inspection camera (Zivid One+S) - quality control
    Pilz laser scanner             - Speed & Separation Monitoring (SSM)
    Human operator                 - loading, inspection, handover

Process: Manufacturing → Inspection → Assembly
Goal objects: plastic spur gears (SLA), metallic base plates, cantilever shafts,
              collars, washers, nuts assembled into gear units.

Cross-class instance attention (the key feature this scenario demonstrates)
---------------------------------------------------------------------------
When the goal is `assemble_gear_unit`, the class graph attends to:
    gear_part, shaft, fastener, assembly_robot, gripper, assembly_station

These classes have NO semantic path to assemble_gear_unit:
    mobile_robot, human_operator, laser_scanner, inspection_camera

BUT the following instance-level co-location relations exist:

    kmr_1           (mobile_robot)      --parkedAt-->  station_cell_1  (assembly_station)
    operator_alice  (human_operator)    --workingAt--> station_cell_1
    pilz_scanner_1  (laser_scanner)     --monitors-->  station_cell_1
    zivid_camera_1  (inspection_camera) --mountedOn--> kmr_1

Result: kmr_1, operator_alice, and pilz_scanner_1 receive a 1-hop relational
boost from the active assembly station instance.  zivid_camera_1 receives a
2-hop boost via kmr_1 → station_cell_1.  None of these are reachable from
assemble_gear_unit through the class graph alone.

On goal switch to `transport_parts`, kmr_1 gains CLASS-level attention
(mobile_robot is now semantically relevant), and zivid_camera_1's boost
strengthens because it rides on a now-high-attention seed.
"""

from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase


# ── Class-level knowledge base ───────────────────────────────────────────────

def build_manufacturing_d61_kb() -> KnowledgeBase:
    """
    Class-level KB for the D6.1 agile manufacturing testbed.

    Two task goals with intentionally disjoint semantic neighbourhoods:
        assemble_gear_unit - attends to parts, assembly robot, station
        transport_parts    - attends to mobile robot, machines, station

    mobile_robot, human_operator, laser_scanner, inspection_camera have
    NO semantic path to assemble_gear_unit - making them ideal candidates
    to demonstrate cross-class instance attention.
    """
    kb = KnowledgeBase()

    # ── Tasks ──────────────────────────────────────────────────────────
    kb.add_concept(Concept('assemble_gear_unit', concept_type='task',     decay_rate=0.0))
    kb.add_concept(Concept('transport_parts',    concept_type='task',     decay_rate=0.0))

    # ── Classes attended by assemble_gear_unit ─────────────────────────
    kb.add_concept(Concept('gear_part',        concept_type='object',   decay_rate=0.10))
    kb.add_concept(Concept('shaft',            concept_type='object',   decay_rate=0.08))
    kb.add_concept(Concept('fastener',         concept_type='object',   decay_rate=0.06))  # collars/washers/nuts
    kb.add_concept(Concept('assembly_robot',   concept_type='tool',     decay_rate=0.04))  # UR5
    kb.add_concept(Concept('gripper',          concept_type='tool',     decay_rate=0.05))  # Schunk EGP 50
    kb.add_concept(Concept('assembly_station', concept_type='location', decay_rate=0.02))  # 900x1400x1000 mm cell

    # ── Classes attended by transport_parts (NOT assemble_gear_unit) ───
    kb.add_concept(Concept('mobile_robot',     concept_type='tool',     decay_rate=0.03))  # KUKA KMR iiwa
    kb.add_concept(Concept('production_machine', concept_type='location', decay_rate=0.02))  # 3D printers, CNC

    # ── Classes with NO semantic link to either task ────────────────────
    # These are physically present in the workspace and will demonstrate
    # cross-class instance attention.
    kb.add_concept(Concept('human_operator',    concept_type='agent',   decay_rate=0.01))
    kb.add_concept(Concept('laser_scanner',     concept_type='safety',  decay_rate=0.01))  # Pilz SSM scanners
    kb.add_concept(Concept('inspection_camera', concept_type='tool',    decay_rate=0.05))  # Zivid One+ S

    # ── Semantic edges for assemble_gear_unit ──────────────────────────
    kb.add_relation('assemble_gear_unit', 'gear_part',        weight=1.0)
    kb.add_relation('assemble_gear_unit', 'shaft',            weight=1.0)
    kb.add_relation('assemble_gear_unit', 'fastener',         weight=1.0)
    kb.add_relation('assemble_gear_unit', 'assembly_robot',   weight=1.0)
    kb.add_relation('assemble_gear_unit', 'assembly_station', weight=1.2)
    kb.add_relation('assembly_robot',     'gripper',          weight=1.0)
    kb.add_relation('gear_part',          'assembly_station', weight=1.0)
    kb.add_relation('shaft',              'assembly_station', weight=1.0)

    # ── Semantic edges for transport_parts ─────────────────────────────
    # Deliberately NO connection to assembly_station - that node is only
    # in the assemble_gear_unit sub-graph so that mobile_robot is unreachable
    # from assemble_gear_unit via the class graph alone.  Physical co-location
    # at the assembly cell is expressed only in the instance graph.
    kb.add_relation('transport_parts',   'mobile_robot',       weight=1.0)
    kb.add_relation('transport_parts',   'production_machine', weight=1.0)

    return kb


# ── Instance-level knowledge base ────────────────────────────────────────────

def build_manufacturing_d61_instance_kb() -> InstanceKnowledgeBase:
    """
    Instance KB for the D6.1 agile manufacturing testbed.

    Three groups of instances:

    GROUP A - Standard (class gate applies, normal attention inheritance):
        gear_small_1, gear_large_1  → gear_part
        shaft_1, shaft_2            → shaft
        fastener_set_1              → fastener
        ur5_robot_1                 → assembly_robot
        egp50_gripper_1             → gripper
        station_cell_1              → assembly_station

    GROUP B - Cross-class (1-hop relational boost from station_cell_1):
        kmr_1           (mobile_robot)     parkedAt     → station_cell_1
        operator_alice  (human_operator)   workingAt    → station_cell_1
        pilz_scanner_1  (laser_scanner)    monitors     → station_cell_1

    GROUP C - Cross-class (2-hop boost via kmr_1 → station_cell_1):
        zivid_camera_1  (inspection_camera) mountedOn   → kmr_1

    When goal=assemble_gear_unit:
        Group A: high attention (class gate)
        Group B: zero CLASS attention, non-zero INSTANCE attention (1-hop boost)
        Group C: zero CLASS attention, small INSTANCE attention (2-hop boost)

    When goal switches to transport_parts:
        kmr_1 gains class-level base attention (mobile_robot now relevant)
        zivid_camera_1 boost strengthens (rides on high-attention kmr_1 seed)
        Group A gear/shaft instances lose attention
    """
    ikb = InstanceKnowledgeBase()

    # ── Group A: standard instances ─────────────────────────────────────
    ikb.add_instance(InstanceConcept('gear_small_1',   'object',   decay_rate=0.10, class_id='gear_part'))
    ikb.add_instance(InstanceConcept('gear_large_1',   'object',   decay_rate=0.10, class_id='gear_part'))
    ikb.add_instance(InstanceConcept('shaft_1',        'object',   decay_rate=0.08, class_id='shaft'))
    ikb.add_instance(InstanceConcept('shaft_2',        'object',   decay_rate=0.08, class_id='shaft'))
    ikb.add_instance(InstanceConcept('fastener_set_1', 'object',   decay_rate=0.06, class_id='fastener'))
    ikb.add_instance(InstanceConcept('ur5_robot_1',    'tool',     decay_rate=0.04, class_id='assembly_robot'))
    ikb.add_instance(InstanceConcept('egp50_gripper_1','tool',     decay_rate=0.05, class_id='gripper'))
    ikb.add_instance(InstanceConcept('station_cell_1', 'location', decay_rate=0.02, class_id='assembly_station'))

    # ── Group B: cross-class, 1-hop from station_cell_1 ─────────────────
    # mobile_robot - NOT semantically linked to assemble_gear_unit
    ikb.add_instance(InstanceConcept('kmr_1',          'tool',     decay_rate=0.03, class_id='mobile_robot'))
    # human_operator - NOT semantically linked to assemble_gear_unit
    ikb.add_instance(InstanceConcept('operator_alice', 'agent',    decay_rate=0.01, class_id='human_operator'))
    # laser_scanner - NOT semantically linked to assemble_gear_unit
    ikb.add_instance(InstanceConcept('pilz_scanner_1', 'safety',   decay_rate=0.01, class_id='laser_scanner'))

    # ── Group C: cross-class, 2-hop via kmr_1 → station_cell_1 ──────────
    # inspection_camera - NOT semantically linked to assemble_gear_unit
    ikb.add_instance(InstanceConcept('zivid_camera_1', 'tool',     decay_rate=0.05, class_id='inspection_camera'))

    # ── Intra-class relations (within semantically attended classes) ─────
    ikb.add_instance_relation('gear_small_1',    'station_cell_1', weight=1.0, relation_type='locatedAt')
    ikb.add_instance_relation('gear_large_1',    'station_cell_1', weight=1.0, relation_type='locatedAt')
    ikb.add_instance_relation('shaft_1',         'station_cell_1', weight=1.0, relation_type='locatedAt')
    ikb.add_instance_relation('shaft_2',         'station_cell_1', weight=1.0, relation_type='locatedAt')
    ikb.add_instance_relation('fastener_set_1',  'station_cell_1', weight=1.0, relation_type='locatedAt')
    ikb.add_instance_relation('ur5_robot_1',     'station_cell_1', weight=1.0, relation_type='operatesAt')
    ikb.add_instance_relation('egp50_gripper_1', 'ur5_robot_1',    weight=1.0, relation_type='attachedTo')

    # ── Cross-class relations (GROUP B: 1-hop from active instance) ──────
    ikb.add_instance_relation('kmr_1',          'station_cell_1', weight=1.0, relation_type='parkedAt')
    ikb.add_instance_relation('operator_alice', 'station_cell_1', weight=1.0, relation_type='workingAt')
    ikb.add_instance_relation('pilz_scanner_1', 'station_cell_1', weight=1.0, relation_type='monitors')

    # ── Cross-class relations (GROUP C: 2-hop via kmr_1) ─────────────────
    ikb.add_instance_relation('zivid_camera_1', 'kmr_1',          weight=1.0, relation_type='mountedOn')

    return ikb
