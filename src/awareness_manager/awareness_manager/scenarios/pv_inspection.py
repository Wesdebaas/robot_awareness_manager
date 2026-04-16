from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase


def build_pv_inspection_kb() -> KnowledgeBase:
    """
    Inspection testbed scenario: autonomous drone inspecting a PV solar plant.

    Based on the CoreSense Inspection Testbed (D7.1). A drone swarm inspects
    photovoltaic power plants, collecting optical and thermal images of panels.
    The key awareness challenge is goal-dependent reprioritisation: during normal
    inspection the robot attends to panels, camera quality and light; when a
    battery emergency is declared the robot must immediately shift full attention
    to battery, landing zone and airspace.

    Two task nodes allow this contrast to be demonstrated by switching goals:
        inspect_pv_field    — normal inspection mission
        emergency_landing   — triggered by low battery or critical fault

    Decay rates (δ) reflect how fast each concept becomes stale:
        0.0   /s  task nodes         — abstract goals, no decay
        0.001 /s  solar_panel        — panels are static infrastructure
        0.001 /s  panel_row          — fixed grid location, highly stable
        0.005 /s  landing_zone       — usually clear, can be obstructed
        0.01  /s  drone_camera       — vibration / calibration drift
        0.02  /s  drone_gps          — signal can fluctuate in the field
        0.02  /s  airspace           — other drones/birds move unpredictably
        0.05  /s  drone_battery      — drains continuously during flight
        0.08  /s  light_conditions   — clouds and sun angle change over minutes
        0.1   /s  wind_speed         — gusts change on a seconds timescale

    Edge weights (semantic distance, lower = tighter coupling):
        1.0  — direct dependency (task ↔ primary concept, sensor ↔ signal)
        1.5  — functional coupling (camera ↔ target panels, gps ↔ airspace)
        2.0  — loose coupling (spatial proximity, system-level co-dependency)
        2.5  — peripheral awareness (battery known but not primary concern)
    """
    kb = KnowledgeBase()

    # --- Task nodes (goal = 0 decay, always stay fresh) ---
    kb.add_concept(Concept('inspect_pv_field',  'task',     decay_rate=0.0))
    kb.add_concept(Concept('emergency_landing', 'task',     decay_rate=0.0))

    # --- Physical plant ---
    kb.add_concept(Concept('solar_panel',       'object',   decay_rate=0.001))
    kb.add_concept(Concept('panel_row',         'location', decay_rate=0.001))
    kb.add_concept(Concept('landing_zone',      'location', decay_rate=0.005))

    # --- Drone subsystems ---
    kb.add_concept(Concept('drone_camera',      'object',   decay_rate=0.01))
    kb.add_concept(Concept('drone_battery',     'state',    decay_rate=0.05))
    kb.add_concept(Concept('drone_gps',         'state',    decay_rate=0.02))

    # --- Environment ---
    kb.add_concept(Concept('wind_speed',        'state',    decay_rate=0.1))
    kb.add_concept(Concept('light_conditions',  'state',    decay_rate=0.08))
    kb.add_concept(Concept('airspace',          'state',    decay_rate=0.02))

    # --- Inspection goal: attends to panels, camera, light, wind ---
    kb.add_relation('inspect_pv_field', 'solar_panel',      weight=1.0)
    kb.add_relation('inspect_pv_field', 'drone_camera',     weight=1.0)
    kb.add_relation('inspect_pv_field', 'light_conditions', weight=1.0)
    kb.add_relation('inspect_pv_field', 'wind_speed',       weight=1.5)
    kb.add_relation('inspect_pv_field', 'drone_gps',        weight=2.0)
    kb.add_relation('inspect_pv_field', 'drone_battery',    weight=2.5)

    # --- Emergency landing goal: attends to battery, landing zone, airspace, wind ---
    kb.add_relation('emergency_landing', 'drone_battery',   weight=1.0)
    kb.add_relation('emergency_landing', 'landing_zone',    weight=1.0)
    kb.add_relation('emergency_landing', 'wind_speed',      weight=1.0)
    kb.add_relation('emergency_landing', 'airspace',        weight=1.0)

    # --- Physical plant structure ---
    kb.add_relation('solar_panel',  'panel_row',            weight=1.0)
    kb.add_relation('panel_row',    'landing_zone',         weight=2.0)

    # --- Sensor ↔ target / environment ---
    kb.add_relation('drone_camera', 'solar_panel',          weight=1.5)
    kb.add_relation('drone_camera', 'light_conditions',     weight=1.0)

    # --- Drone subsystem cross-links ---
    kb.add_relation('drone_battery', 'drone_gps',           weight=2.0)
    kb.add_relation('drone_gps',     'airspace',            weight=1.5)

    return kb
