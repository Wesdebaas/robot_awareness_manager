from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase


def build_pv_inspection_kb() -> KnowledgeBase:
    """
    Inspection testbed scenario: autonomous drone inspecting a PV solar plant.

    Based on CoreSense D7.1.  Two operational goals with deliberately disjoint
    1-hop neighborhoods, enabling the Anticipatory Horizon (F2) to demonstrate
    measurable pre-tuning value:

        inspect_pv_field  → {solar_panel, drone_camera, image_quality, light_conditions}
        emergency_landing → {drone_battery, landing_zone, wind_speed, airspace}

    Zero neighborhood overlap ensures that F2's pre-allocation to the emergency
    cluster is unambiguously measurable: a Reactive baseline assigns zero attention
    to emergency concepts until the goal fires, while the AM pre-allocates budget
    as the ETA of emergency_landing decreases.

    Decay rates (δ, s⁻¹):
        0.0   task nodes            - abstract goals, no decay
        0.001 solar_panel           - static infrastructure
        0.001 panel_row             - fixed grid location
        0.005 landing_zone          - usually clear, can be obstructed
        0.01  drone_camera          - calibration drift, vibration
        0.02  airspace              - other drones/birds move unpredictably
        0.05  drone_battery         - drains continuously during flight
        0.08  light_conditions      - clouds and sun angle change over minutes
        0.08  image_quality         - focus, vibration, shadow/blur artifacts (D7.1 UC4)
        0.1   wind_speed            - gusts change on a seconds timescale

    Grounding in D7.1:
        solar_panel, drone_camera, light_conditions - Inspection mode captures optical
          and thermal images; light directly affects image quality.
        image_quality - UC4 (Valid images): online analysis checks shadows, reflections,
          blur.  Justifies a fast-decaying state concept in the inspection cluster.
        drone_battery - Table 7.1 disturbance: battery discharge triggers emergency.
        landing_zone  - Emergency mode: robot identifies a clear landing site.
        wind_speed    - Emergency mode: must be below threshold for safe landing.
        airspace      - UC2: airspace invasion (bird, other drone) triggers emergency.
    """
    kb = KnowledgeBase()

    # --- Task nodes (decay=0, always fresh) ---
    kb.add_concept(Concept('inspect_pv_field',  'task',     decay_rate=0.0))
    kb.add_concept(Concept('emergency_landing', 'task',     decay_rate=0.0))

    # --- Inspection cluster ---
    kb.add_concept(Concept('solar_panel',       'object',   decay_rate=0.001))
    kb.add_concept(Concept('drone_camera',      'object',   decay_rate=0.01))
    kb.add_concept(Concept('image_quality',     'state',    decay_rate=0.08))
    kb.add_concept(Concept('light_conditions',  'state',    decay_rate=0.08))

    # --- Emergency cluster ---
    kb.add_concept(Concept('drone_battery',     'state',    decay_rate=0.05))
    kb.add_concept(Concept('landing_zone',      'location', decay_rate=0.005))
    kb.add_concept(Concept('wind_speed',        'state',    decay_rate=0.1))
    kb.add_concept(Concept('airspace',          'state',    decay_rate=0.02))

    # --- Structural bridge node (not in either 1-hop hood) ---
    kb.add_concept(Concept('panel_row',         'location', decay_rate=0.001))

    # --- Inspection goal: 1-hop = {solar_panel, drone_camera, image_quality, light_conditions} ---
    kb.add_relation('inspect_pv_field', 'solar_panel',      weight=1.0)
    kb.add_relation('inspect_pv_field', 'drone_camera',     weight=1.0)
    kb.add_relation('inspect_pv_field', 'image_quality',    weight=1.0)
    kb.add_relation('inspect_pv_field', 'light_conditions', weight=1.0)

    # --- Emergency goal: 1-hop = {drone_battery, landing_zone, wind_speed, airspace} ---
    kb.add_relation('emergency_landing', 'drone_battery',   weight=1.0)
    kb.add_relation('emergency_landing', 'landing_zone',    weight=1.0)
    kb.add_relation('emergency_landing', 'wind_speed',      weight=1.0)
    kb.add_relation('emergency_landing', 'airspace',        weight=1.0)

    # --- Inspection cluster internal edges ---
    kb.add_relation('solar_panel',    'panel_row',          weight=1.0)   # panels in rows
    kb.add_relation('drone_camera',   'solar_panel',        weight=1.5)   # camera targets panels
    kb.add_relation('drone_camera',   'image_quality',      weight=1.0)   # camera → quality signal
    kb.add_relation('image_quality',  'light_conditions',   weight=1.5)   # light → image quality

    # --- Cross-cluster structural bridges (2-hop, not in either goal's 1-hop hood) ---
    kb.add_relation('panel_row',      'landing_zone',       weight=2.0)   # spatial co-location
    kb.add_relation('drone_camera',   'drone_battery',      weight=2.5)   # camera power draw
    kb.add_relation('light_conditions', 'wind_speed',       weight=2.0)   # atmospheric coupling

    return kb


def build_pv_inspection_instance_kb() -> InstanceKnowledgeBase:
    """
    Instance-level knowledge base for the PV inspection scenario.

    Adds specific physical individuals for the drone inspection domain.
    Demonstrates goal-dependent instance relevance:
        - During inspect_pv_field:   panel instances and camera_main are critical.
        - During emergency_landing:  battery_main and landing zone instances become critical.

    Instance relations use typed edges:
        partOf     - instance belongs to a larger structure
        monitors   - sensor instance is targeted at an object instance
        locatedAt  - instance is physically at a location

    Instances (7):
        panel_A1, panel_A2, panel_B1  - solar panels (class: solar_panel)
        battery_main                   - primary drone battery (class: drone_battery)
        lz_north, lz_south             - two landing zones (class: landing_zone)
        camera_main                    - primary inspection camera (class: drone_camera)
    """
    ikb = InstanceKnowledgeBase()

    # --- Solar panel instances ---
    ikb.add_instance(InstanceConcept('panel_A1', 'object', decay_rate=0.001, class_id='solar_panel'))
    ikb.add_instance(InstanceConcept('panel_A2', 'object', decay_rate=0.001, class_id='solar_panel'))
    ikb.add_instance(InstanceConcept('panel_B1', 'object', decay_rate=0.001, class_id='solar_panel'))

    # --- Battery instance ---
    ikb.add_instance(InstanceConcept('battery_main', 'state', decay_rate=0.05, class_id='drone_battery'))

    # --- Landing zone instances ---
    ikb.add_instance(InstanceConcept('lz_north', 'location', decay_rate=0.005, class_id='landing_zone'))
    ikb.add_instance(InstanceConcept('lz_south', 'location', decay_rate=0.005, class_id='landing_zone'))

    # --- Camera instance ---
    ikb.add_instance(InstanceConcept('camera_main', 'object', decay_rate=0.01, class_id='drone_camera'))

    # --- Instance relations ---
    ikb.add_instance_relation('panel_A1', 'panel_A2', weight=1.0, relation_type='partOf')
    ikb.add_instance_relation('panel_A2', 'panel_B1', weight=2.0, relation_type='partOf')
    ikb.add_instance_relation('camera_main', 'panel_A1', weight=1.0, relation_type='monitors')
    ikb.add_instance_relation('lz_north', 'lz_south', weight=1.5, relation_type='locatedAt')

    return ikb
