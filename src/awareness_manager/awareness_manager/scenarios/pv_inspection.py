from pathlib import Path

from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.loader import load_instance_kb_from_ttl, load_kb_from_ttl

_TTL = Path(__file__).parent / 'pv_inspection.ttl'


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
    return load_kb_from_ttl(_TTL)


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
    return load_instance_kb_from_ttl(_TTL)
