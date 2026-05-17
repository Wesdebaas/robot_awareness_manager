"""
snapshot.py - serialise AwarenessManager state into Cytoscape elements.

compute_positions() is called ONCE at startup to compute fixed node positions
from NetworkX spring_layout (class nodes) and orbital placement (instances).
Passing the same positions dict on every poll keeps the graph completely stable.

get_snapshot() is called on every dashboard poll (250 ms) under a threading
lock held by SimulationRunner. It is a pure function: no side effects.

Phase 2 additions:
  - _channel_color(): hue-by-dominant-channel + lightness-by-E fill color.
  - Per-node channel fields in element data: ch_mission, ch_anticipatory,
    ch_relational, ch_surprise - read by the inspector hover card.
  - is_active_goal: distinguishes current goal from queued task nodes.
  - has_instances: allows empty class containers to be visually de-emphasised.
"""

import colorsys
import math
from collections import defaultdict

import networkx as nx

from awareness_manager.baselines.strategy import AttentionStrategy

# Canvas coordinate space for preset layout positions
_CANVAS_W = 1000.0
_CANVAS_H = 750.0
_ORBIT_RADIUS_BASE = 68.0
_ORBIT_RADIUS_STEP = 18.0  # extra radius per instance beyond 2
# Margin reserved on each edge for instance orbits (class nodes must stay inside)
_LAYOUT_MARGIN = _ORBIT_RADIUS_BASE + _ORBIT_RADIUS_STEP * 3 + 20  # ≈ 122 px


def _fit_to_canvas(pos_raw: dict) -> dict[str, dict]:
    """
    Scale and centre spring-layout positions to fill the usable canvas area.

    spring_layout returns coordinates in roughly [-1, 1].  The naive approach
    of multiplying by a fixed scale fails when the canvas is not square (H=750
    vs W=1000) or when the graph's bounding box is smaller than ±1 — nodes end
    up clipped off-screen and Cytoscape auto-fits everything into a tiny pile.

    This function:
      1. Measures the actual bounding box of the raw positions.
      2. Computes a uniform scale so the bounding box fills the effective canvas
         area (canvas minus _LAYOUT_MARGIN on all sides, reserving room for
         orbiting instance nodes).
      3. Centres the scaled positions on the canvas.

    The effective canvas for class nodes is
      x ∈ [_LAYOUT_MARGIN, _CANVAS_W - _LAYOUT_MARGIN]
      y ∈ [_LAYOUT_MARGIN, _CANVAS_H - _LAYOUT_MARGIN]
    so instance orbits stay fully visible.
    """
    if not pos_raw:
        return {}

    xs = [p[0] for p in pos_raw.values()]
    ys = [p[1] for p in pos_raw.values()]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    eff_w = _CANVAS_W - 2 * _LAYOUT_MARGIN
    eff_h = _CANVAS_H - 2 * _LAYOUT_MARGIN

    # Uniform scale: fit the larger dimension; fall back to eff_w if graph is a point
    if x_range < 1e-6 and y_range < 1e-6:
        scale = eff_w
    elif x_range < 1e-6:
        scale = eff_h / y_range
    elif y_range < 1e-6:
        scale = eff_w / x_range
    else:
        scale = min(eff_w / x_range, eff_h / y_range)

    # Centre of the raw bounding box → canvas centre
    raw_cx = (min(xs) + max(xs)) / 2.0
    raw_cy = (min(ys) + max(ys)) / 2.0
    canvas_cx = _CANVAS_W / 2.0
    canvas_cy = _CANVAS_H / 2.0

    return {
        cid: {
            "x": float(round((x - raw_cx) * scale + canvas_cx, 2)),
            "y": float(round((y - raw_cy) * scale + canvas_cy, 2)),
        }
        for cid, (x, y) in pos_raw.items()
    }

# Hue for each dominant activation channel (used in _channel_color and in the UI legend)
CHANNEL_COLORS = {
    'mission':     '#4a8af4',   # blue - goal-driven (F1 + F2)
    'anticipatory':'#9a6af4',   # purple-blue - F2 anticipatory only (breakdown table)
    'relational':  '#44bb88',   # teal - instance-graph propagation
    'surprise':    '#f4944a',   # orange - prediction-error violation
}

_HUES = {
    'mission':    0.605,   # blue
    'relational': 0.468,   # teal
    'surprise':   0.083,   # orange
}


def _channel_color(
    mission: float,
    anticipatory: float,
    relational: float,
    surprise: float,
    e: float,
) -> str:
    """
    Compute a node's fill colour.

    Hue: determined by the dominant activation source.
        mission + anticipatory → blue   (top-down, goal-driven)
        relational             → teal   (instance-graph propagation)
        surprise               → orange (prediction-error violation)

    Lightness: modulated by epistemic error.
        E=0 (fresh) → bright (l≈0.62)
        E=1 (stale) → dark   (l≈0.30)

    Saturation: scaled by how decisively one channel dominates over the others.
    """
    m_total = mission + anticipatory
    total = m_total + relational + surprise

    e_c = max(0.0, min(1.0, e))
    l = 0.62 - 0.32 * e_c  # lightness: bright when fresh, dark when stale

    if total < 0.005:
        # No meaningful activation - neutral dark grey, darkness by E
        v = int((l * 0.7) * 255)
        return f"#{v:02x}{v:02x}{v+8:02x}"

    if m_total >= relational and m_total >= surprise:
        h = _HUES['mission']
        dominant_val = m_total
    elif relational >= surprise:
        h = _HUES['relational']
        dominant_val = relational
    else:
        h = _HUES['surprise']
        dominant_val = surprise

    dominance = dominant_val / total   # [0.33 … 1.0]
    s = 0.45 + 0.55 * dominance        # saturation: 0.45 … 1.0

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def compute_positions_from_structure(structure: dict) -> dict[str, dict]:
    """
    Compute preset-layout positions from a serialised structure block (meta.json).

    Produces positions identical to compute_positions() for the same graph
    topology - uses the same spring_layout seed and orbital placement logic.
    Used by ReplayReader when positions are not stored in the trace.
    """
    G = nx.Graph()
    for cls in structure["classes"]:
        G.add_node(cls["id"])
    for id_a, id_b, weight in structure["semantic_edges"]:
        G.add_edge(id_a, id_b, weight=weight)

    pos_raw = nx.spring_layout(G, seed=42)
    positions: dict[str, dict] = _fit_to_canvas(pos_raw)

    class_instances: dict[str, list[str]] = defaultdict(list)
    for inst in structure.get("instances", []):
        class_instances[inst["class_id"]].append(inst["id"])

    for class_id, instance_ids in class_instances.items():
        if class_id not in positions:
            continue
        pcx = positions[class_id]["x"]
        pcy = positions[class_id]["y"]
        n = len(instance_ids)
        radius = _ORBIT_RADIUS_BASE + _ORBIT_RADIUS_STEP * max(0, n - 2)
        for i, iid in enumerate(sorted(instance_ids)):
            angle = 2.0 * math.pi * i / n + math.pi / 6.0
            positions[iid] = {
                "x": float(round(pcx + radius * math.cos(angle), 2)),
                "y": float(round(pcy + radius * math.sin(angle), 2)),
            }

    return positions


def compute_positions(am: AttentionStrategy) -> dict[str, dict]:
    """
    Compute fixed Cytoscape preset-layout positions for every node.

    Class nodes: NetworkX spring_layout on a graph reconstructed from the
    public semantic_edges() API - no private attributes accessed.
    Instance nodes: evenly distributed on a circle centred on their parent
    class node. Radius grows with instance count to avoid overlap.

    Returns {node_id: {"x": float, "y": float}}.
    """
    kb = am.kb

    G = nx.Graph()
    G.add_nodes_from(kb.concept_ids())
    for id_a, id_b, weight in kb.semantic_edges():
        G.add_edge(id_a, id_b, weight=weight)

    pos_raw = nx.spring_layout(G, seed=42)
    positions: dict[str, dict] = _fit_to_canvas(pos_raw)

    if am.instance_kb is not None:
        class_instances: dict[str, list[str]] = defaultdict(list)
        for iid in am.instance_kb.instance_ids():
            inst = am.instance_kb.get_instance(iid)
            class_instances[inst.class_id].append(iid)

        for class_id, instance_ids in class_instances.items():
            if class_id not in positions:
                continue
            pcx = positions[class_id]["x"]
            pcy = positions[class_id]["y"]
            n = len(instance_ids)
            radius = _ORBIT_RADIUS_BASE + _ORBIT_RADIUS_STEP * max(0, n - 2)
            for i, iid in enumerate(sorted(instance_ids)):
                angle = 2.0 * math.pi * i / n + math.pi / 6.0
                positions[iid] = {
                    "x": float(round(pcx + radius * math.cos(angle), 2)),
                    "y": float(round(pcy + radius * math.sin(angle), 2)),
                }

    return positions


def get_snapshot(
    am: AttentionStrategy,
    schedule: list[str],
    elapsed: float,
    positions: dict[str, dict],
    threshold: float = 0.0,
) -> dict:
    """
    Build a serialisable snapshot of AM state for the dashboard callback.

    Node data fields (Phase 2 additions marked *):
        id, label, type, concept_type / class_id
        attention, epistemic_error, prediction_error, decay_rate
        in_schedule, below_threshold
        color            - fill hex, hue-by-dominant-channel + lightness-by-E (*)
        ch_mission       - F1 attention from current goal (*)
        ch_anticipatory  - F2 attention from queued goals (*)
        ch_relational    - relational boost above class-gate (instances only) (*)
        ch_surprise      - prediction-error violation boost (*)
        is_active_goal   - 1 if this node is the current active goal (*)
        has_instances    - 1 if this class node has ≥1 instance (*)
    """
    kb = am.kb
    ikb = am.instance_kb
    attn = am.attention()
    ch = am.attention_channels()
    schedule_set = set(schedule)
    active_goal = am.goal_id
    queued_ids = {gid for gid, _, _ in am.mission_queue}

    # Classes that have at least one instance (for empty-container de-emphasis)
    populated_classes: set[str] = set()
    if ikb is not None:
        for iid in ikb.instance_ids():
            populated_classes.add(ikb.get_instance(iid).class_id)

    elements: list[dict] = []

    # ── Class nodes (compound containers) ────────────────────────────────
    for cid in kb.concept_ids():
        concept = kb.get_concept(cid)
        a = attn.get(cid, 0.0)
        e = concept.epistemic_error
        pe = concept.prediction_error
        pos = positions.get(cid, {"x": 0.0, "y": 0.0})

        # Per-channel contributions for this class node
        cm = ch['mission'].get(cid, 0.0)
        ca = ch['anticipatory'].get(cid, 0.0)
        cr = 0.0  # class nodes never receive relational boost
        cs = ch['surprise'].get(cid, 0.0)

        label = f"{cid}\nA={a:.2f}  E={e:.2f}"

        elements.append({
            "data": {
                "id": cid,
                "label": label,
                "type": "class",
                "concept_type": concept.concept_type,
                "attention": round(a, 4),
                "epistemic_error": round(e, 4),
                "prediction_error": round(pe, 4),
                "decay_rate": concept.decay_rate,
                "in_schedule": 1 if cid in schedule_set else 0,
                "below_threshold": 1 if a < threshold else 0,
                "is_active_goal": 1 if cid == active_goal else 0,
                "has_instances": 1 if cid in populated_classes else 0,
                "ch_mission": round(cm, 4),
                "ch_anticipatory": round(ca, 4),
                "ch_relational": round(cr, 4),
                "ch_surprise": round(cs, 4),
                "color": _channel_color(cm, ca, cr, cs, e),
            },
            "position": pos,
        })

    # ── Instance nodes (children inside class compound containers) ────────
    if ikb is not None:
        for iid in ikb.instance_ids():
            inst = ikb.get_instance(iid)
            a = attn.get(iid, 0.0)
            e = inst.epistemic_error
            pe = inst.prediction_error
            pos = positions.get(iid, {"x": 0.0, "y": 0.0})

            # Instance mission = class-gate baseline (F1 of class + F2 of class)
            cm = ch['mission'].get(inst.class_id, 0.0)
            ca = ch['anticipatory'].get(inst.class_id, 0.0)
            cr = ch['relational'].get(iid, 0.0)
            cs = ch['surprise'].get(iid, 0.0)

            label = f"{iid}\nA={a:.2f}\nE={e:.2f}"

            elements.append({
                "data": {
                    "id": iid,
                    "label": label,
                    "parent": inst.class_id,
                    "type": "instance",
                    "class_id": inst.class_id,
                    "attention": round(a, 4),
                    "epistemic_error": round(e, 4),
                    "prediction_error": round(pe, 4),
                    "decay_rate": inst.decay_rate,
                    "in_schedule": 1 if iid in schedule_set else 0,
                    "below_threshold": 1 if a < threshold else 0,
                    "is_active_goal": 0,
                    "has_instances": 0,
                    "ch_mission": round(cm, 4),
                    "ch_anticipatory": round(ca, 4),
                    "ch_relational": round(cr, 4),
                    "ch_surprise": round(cs, 4),
                    "color": _channel_color(cm, ca, cr, cs, e),
                },
                "position": pos,
            })

    # ── Semantic edges (class–class, dashed light grey) ───────────────────
    for id_a, id_b, weight in kb.semantic_edges():
        elements.append({
            "data": {
                "id": f"sem__{id_a}__{id_b}",
                "source": id_a,
                "target": id_b,
                "edge_type": "semantic",
                "weight": weight,
                "label": f"{weight:.1f}",
            }
        })

    # ── Relational edges (instance–instance, solid teal) ─────────────────
    if ikb is not None:
        for id_a, id_b, weight, relation_type in ikb.relational_edges():
            rtype = relation_type or "related"
            elements.append({
                "data": {
                    "id": f"rel__{id_a}__{id_b}",
                    "source": id_a,
                    "target": id_b,
                    "edge_type": "relational",
                    "weight": weight,
                    "relation_type": rtype,
                    "label": rtype,
                }
            })

    # ── Top-bar scalars ───────────────────────────────────────────────────
    queue = am.mission_queue
    if queue:
        next_goal, next_eta, next_level = queue[0]
        queue_text = f"→ {next_goal} in {next_eta:.1f}s ({next_level})"
    else:
        queue_text = "no queued goals"

    b = am.budget
    return {
        "elements": elements,
        "goal_id": am.goal_id,
        "queue_text": queue_text,
        "n_scheduled": len(schedule),
        "budget": b,
        "budget_unlimited": b < 0,
        "strategy": am.strategy_name(),
        "elapsed": round(elapsed, 1),
        "schedule": list(schedule),
    }
