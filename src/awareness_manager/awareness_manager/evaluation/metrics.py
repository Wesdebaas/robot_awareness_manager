"""
metrics.py - Pure-function metric library for trace-based replay comparison.

Used by the dashboard compare view to compute M1-M5 from saved trace files.
All functions take a loaded trace dict (from load_trace()) and return scalars
or timeseries — no scenario re-running required.

Metrics returned by all_metrics():
    M1  m1_e_at_transition     - max E in new-goal's 1-hop hood at the transition tick
    M2  m2_pre_transition_attn - mean attention to incoming-goal hood before the switch
    M3  m3_lag_seconds         - seconds until E_max < 0.1 after the transition
    M4  m4_e_relevant          - run-mean epistemic error over goal-relevant concepts
    M5  m5_budget_util         - schedule slots used / budget per tick

Neighborhood convention: 1-hop semantic class neighbours of the active goal,
filtered to decay_rate > 0. Strategy-agnostic.

Trace format (schema v2)
    meta: {schema_version, scenario, strategy, strategy_params, budget, dt, structure}
    tick: {t, goal, queue, schedule, events, concepts: {cid: {a, e, pe, ...}}}
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Trace loader
# ---------------------------------------------------------------------------

def load_trace(trace_dir: str | Path) -> dict:
    """
    Load a trace directory into a dict with keys 'meta' and 'ticks'.

    Returns:
        {
            "meta":  dict   - parsed meta.json
            "ticks": list   - list of tick dicts from ticks.jsonl
        }
    """
    trace_dir = Path(trace_dir)
    meta = json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    ticks: list[dict] = []
    with open(trace_dir / "ticks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ticks.append(json.loads(line))
    return {"meta": meta, "ticks": ticks}


# ---------------------------------------------------------------------------
# Neighbourhood helpers
# ---------------------------------------------------------------------------

def _class_decay_rates(meta: dict) -> dict[str, float]:
    """Map class concept_id → decay_rate."""
    return {c["id"]: c["decay_rate"] for c in meta["structure"]["classes"]}


def _goal_neighborhood(meta: dict, goal_id: str) -> frozenset[str]:
    """
    1-hop semantic class neighbours of goal_id, filtered to decay_rate > 0.

    Includes the goal itself if decay_rate > 0 (task nodes have decay_rate=0
    so they are always excluded in practice).

    Returns frozenset of class concept IDs.
    """
    decay = _class_decay_rates(meta)
    edges = meta["structure"]["semantic_edges"]
    hood: set[str] = set()
    for a, b, _ in edges:
        if a == goal_id and decay.get(b, 0.0) > 0.0:
            hood.add(b)
        elif b == goal_id and decay.get(a, 0.0) > 0.0:
            hood.add(a)
    if decay.get(goal_id, 0.0) > 0.0:
        hood.add(goal_id)
    return frozenset(hood)


def _mission_neighborhood(meta: dict, ticks: list[dict]) -> frozenset[str]:
    """
    Union of all goal neighborhoods that appear as the active goal at any tick.

    Used for M1/M2: a concept is "goal-relevant over the run" if it was in
    the active goal's neighbourhood at any point during the trace.
    """
    goals_seen: set[str] = {r["goal"] for r in ticks if r.get("goal")}
    union: set[str] = set()
    for g in goals_seen:
        union |= _goal_neighborhood(meta, g)
    return frozenset(union)


def _instance_class_map(meta: dict) -> dict[str, str]:
    """Map instance_id → class_id."""
    return {i["id"]: i["class_id"] for i in meta["structure"].get("instances", [])}


# ---------------------------------------------------------------------------
# M4 — Mean epistemic error over goal-relevant concepts  (m4_e_relevant)
# ---------------------------------------------------------------------------

def mean_epistemic_error_relevant(
    trace: dict,
    *,
    include_instances: bool = True,
) -> dict:
    """
    Mean epistemic error over goal-relevant concepts.

    Goal-relevant = concepts in the 1-hop semantic neighbourhood of any goal
    that appears as the active goal during the trace (decay_rate > 0, class
    concepts; instances inherit their class's membership).

    Formula:
        E_rel(t) = mean(E_c for c ∈ hood_t ∩ concepts(t))
        E_rel     = mean over t of E_rel(t)

    Returns:
        {
            "scalar":    float - run-mean E over relevant concepts
            "timeseries": [(t, E_rel_t), ...] - per-tick values
            "neighborhood": sorted list of class IDs in the relevant set
        }

    Caveats:
        If no goal-relevant concept appears in a tick (e.g. a gap tick),
        that tick is excluded from the scalar mean.
    """
    meta, ticks = trace["meta"], trace["ticks"]
    hood = _mission_neighborhood(meta, ticks)
    inst_class = _instance_class_map(meta) if include_instances else {}

    timeseries: list[tuple[float, float]] = []
    for rec in ticks:
        vals = [
            c["e"] for cid, c in rec["concepts"].items()
            if cid in hood or (include_instances and inst_class.get(cid) in hood)
        ]
        if vals:
            timeseries.append((rec["t"], statistics.mean(vals)))

    scalar = statistics.mean(v for _, v in timeseries) if timeseries else float("nan")
    return {
        "scalar": scalar,
        "timeseries": timeseries,
        "neighborhood": sorted(hood),
    }




# ---------------------------------------------------------------------------
# M5 — Budget utilisation  (m5_budget_util)
# Sanity check only — always returns 1.0 in well-formed experiments because
# _top_n() fills the full budget whenever len(schedulable) >= budget, which
# is structurally guaranteed by the KB/budget sizing in all experiments.
# Not used in thesis evaluation results.
# ---------------------------------------------------------------------------

def budget_utilisation(trace: dict) -> dict:
    """
    Per-tick refresh budget utilisation: |schedule| / budget.

    For unlimited-budget strategies (budget < 0), returns 1.0 for every tick
    with a non-empty schedule (the budget is never the limiting factor).

    Formula:
        util(t) = |schedule_t| / budget      if budget ≥ 0
                = 1.0                         if budget < 0 and schedule non-empty
                = 0.0                         if schedule empty

    Returns:
        {
            "scalar":    float - mean utilisation over all ticks
            "timeseries": [(t, util_t), ...]
            "budget":    int   - budget value from meta (-1 = unlimited)
            "unlimited": bool
        }
    """
    meta, ticks = trace["meta"], trace["ticks"]
    budget = meta.get("budget", -1)
    unlimited = budget < 0

    timeseries: list[tuple[float, float]] = []
    for rec in ticks:
        n = len(rec.get("schedule", []))
        if unlimited:
            util = 1.0 if n > 0 else 0.0
        else:
            util = n / budget if budget > 0 else 0.0
        timeseries.append((rec["t"], util))

    scalar = statistics.mean(v for _, v in timeseries) if timeseries else float("nan")
    return {
        "scalar": scalar,
        "timeseries": timeseries,
        "budget": budget,
        "unlimited": unlimited,
    }


# ---------------------------------------------------------------------------
# M1/M3 — Cognitive lag at goal transition  (m1_e_at_transition, m3_lag_seconds)
# ---------------------------------------------------------------------------

def cognitive_lag(
    trace: dict,
    *,
    fresh_threshold: float = 0.1,
    max_lookahead_s: float | None = None,
) -> dict:
    """
    Cognitive lag at each goal transition.

    At each goal_transition event (old_goal → new_goal), measures how long
    it takes until all class concepts in the 1-hop neighbourhood of new_goal
    (decay_rate > 0) fall below `fresh_threshold`.

    Formula:
        hood(g) = 1-hop class neighbours of g with decay_rate > 0
        E_max(t) = max(E_c for c ∈ hood(new_goal)) at tick t
        lag = first t ≥ t_transition such that E_max(t) < fresh_threshold
              else None  (didn't recover within trace / max_lookahead_s)

    Returns:
        {
            "transitions": [
                {
                    "t_transition": float,
                    "from_goal":    str,
                    "to_goal":      str,
                    "neighborhood": list[str],
                    "e_max_at_transition": float,
                    "lag_seconds":  float | None,  - None if not recovered
                    "recovered":    bool,
                }
            ],
            "mean_lag_seconds": float | None,  - mean over recovered transitions
        }

    Caveats:
        Undefined (empty transitions list) if no goal transitions in trace.
        lag_seconds=None means recovery was not observed in the remaining
        trace (or max_lookahead_s window).  Report as ">T" in prose.
    """
    meta, ticks = trace["meta"], trace["ticks"]
    tick_by_index: list[dict] = ticks

    # Build index: t → tick record (approximate - use first tick at or after t)
    t_to_idx: dict[float, int] = {round(r["t"], 6): i for i, r in enumerate(ticks)}

    transitions = []
    for i, rec in enumerate(ticks):
        for ev in rec.get("events", []):
            if ev["type"] != "goal_transition":
                continue

            new_goal = ev["to"]
            hood = _goal_neighborhood(meta, new_goal)
            if not hood:
                continue  # no scoreable neighbours

            t_g = rec["t"]
            e_at_transition = max(
                (ticks[i]["concepts"].get(c, {}).get("e", 0.0) for c in hood),
                default=0.0,
            )

            lag_seconds: float | None = None
            for j in range(i, len(ticks)):
                t_j = ticks[j]["t"]
                if max_lookahead_s is not None and t_j - t_g > max_lookahead_s:
                    break
                e_max = max(
                    ticks[j]["concepts"].get(c, {}).get("e", 0.0) for c in hood
                )
                if e_max < fresh_threshold:
                    lag_seconds = round(t_j - t_g, 3)
                    break

            transitions.append({
                "t_transition": t_g,
                "from_goal": ev["from"],
                "to_goal": new_goal,
                "neighborhood": sorted(hood),
                "e_max_at_transition": round(e_at_transition, 4),
                "lag_seconds": lag_seconds,
                "recovered": lag_seconds is not None,
            })

    recovered_lags = [tr["lag_seconds"] for tr in transitions if tr["recovered"]]
    mean_lag = statistics.mean(recovered_lags) if recovered_lags else None

    return {
        "transitions": transitions,
        "mean_lag_seconds": mean_lag,
    }


# ---------------------------------------------------------------------------
# M2 — Pre-transition attention  (m2_pre_transition_attn)
# ---------------------------------------------------------------------------

def pre_transition_attention(
    trace: dict,
    *,
    lookback_seconds: float | None = None,
    lookback_ticks: int = 50,
) -> dict:
    """
    Mean attention assigned to the incoming goal's 1-hop neighborhood
    in the window immediately before each goal transition.

    Measures the AM's *recommendation quality* independently of budget
    sufficiency or decay rates: did the strategy signal that the upcoming
    goal's concepts were important before the switch fired?

    This is the direct test of F2 (Anticipatory Horizon): the AM should
    pre-allocate attention to the incoming goal's neighborhood as its ETA
    decreases, even before the goal becomes active.  ReactiveBaseline assigns
    zero attention to concepts outside the current goal's 1-hop set, so its
    score here is a structural baseline determined by overlap between the
    two goal neighborhoods.

    Window: the `lookback_seconds` (= observation_interval) immediately
    before each transition tick (exclusive: [t_trans − window, t_trans)).

    Concepts: class-level 1-hop neighbors of the incoming goal with
    decay_rate > 0 (same neighborhood definition as M4/M5).

    Aggregation:
        For each transition: mean attention over all (tick, concept) pairs
        in the window.  Transitions with no ticks in the window or an empty
        neighborhood are excluded.
        Final scalar: mean over per-transition values.

    Returns:
        {
            "scalar":      float | None  - mean over all transitions; None if
                                          no scoreable transitions
            "transitions": [
                {
                    "t_transition":   float,
                    "to_goal":        str,
                    "neighborhood":   list[str],
                    "mean_attention": float | None,
                }
            ],
            "lookback_ticks": int  - actual window size used
        }

    Caveats:
        AlwaysOnBaseline will score 1.0 trivially (attention is always 1.0).
        The meaningful contrast is AM vs Reactive.
        A Reactive score > 0 indicates overlap between the two goal hoods.
    """
    meta, ticks = trace["meta"], trace["ticks"]
    dt = meta.get("dt", 0.1)
    if lookback_seconds is not None:
        window_ticks = max(1, int(round(lookback_seconds / dt)))
    else:
        window_ticks = lookback_ticks

    results = []
    for i, rec in enumerate(ticks):
        for ev in rec.get("events", []):
            if ev["type"] != "goal_transition":
                continue
            new_goal = ev["to"]
            hood = _goal_neighborhood(meta, new_goal)
            if not hood:
                continue

            lo = max(0, i - window_ticks)
            attention_vals: list[float] = []
            for j in range(lo, i):
                concepts = ticks[j].get("concepts", {})
                for cid in hood:
                    if cid in concepts:
                        attention_vals.append(concepts[cid].get("a", 0.0))

            mean_att = statistics.mean(attention_vals) if attention_vals else None
            results.append({
                "t_transition":   rec["t"],
                "to_goal":        new_goal,
                "neighborhood":   sorted(hood),
                "mean_attention": round(mean_att, 4) if mean_att is not None else None,
            })

    valid = [r["mean_attention"] for r in results if r["mean_attention"] is not None]
    scalar = statistics.mean(valid) if valid else None

    return {
        "scalar":       round(scalar, 4) if scalar is not None else None,
        "transitions":  results,
        "lookback_ticks": window_ticks,
    }


# ---------------------------------------------------------------------------
# Convenience: compute all metrics for one trace
# ---------------------------------------------------------------------------

def all_metrics(trace: dict, *, params: dict | None = None) -> dict:
    """
    Compute M1-M5 for a single trace and return them in a flat dict.

    Used by the dashboard compare view. All metrics are computed from the
    saved trace — no scenario re-running required.

    Args:
        trace:  Loaded trace dict (from load_trace()).
        params: Optional run-level param dict. Used to derive observation_interval
                for the M2 pre-attention window.

    Returns:
        {
            "m1_e_at_transition":     float | None,
            "m2_pre_transition_attn": float | None,
            "m3_lag_seconds":         float | None,
            "m4_e_relevant":          float,
            "m5_budget_util":         float,
            "m_cognitive_lag":        {...},
            "m_pre_attn":             {...},
            "m_e_relevant":           {...},
            "m_budget_util":          {...},
        }
    """
    meta = trace["meta"]
    oi = (
        (params or {}).get("observation_interval")
        or meta.get("strategy_params", {}).get("observation_interval")
    )

    m_e_rel  = mean_epistemic_error_relevant(trace)
    m_budget = budget_utilisation(trace)
    m_lag    = cognitive_lag(trace)
    m_pre    = pre_transition_attention(trace, lookback_seconds=oi)

    m1_e_at = (
        statistics.mean(tr["e_max_at_transition"] for tr in m_lag["transitions"])
        if m_lag["transitions"] else None
    )

    return {
        "m1_e_at_transition":     round(m1_e_at, 4) if m1_e_at is not None else None,
        "m2_pre_transition_attn": m_pre["scalar"],
        "m3_lag_seconds":         round(m_lag["mean_lag_seconds"], 3) if m_lag["mean_lag_seconds"] is not None else None,
        "m4_e_relevant":          round(m_e_rel["scalar"], 6),
        "m5_budget_util":         round(m_budget["scalar"], 6),
        "m_cognitive_lag":        m_lag,
        "m_pre_attn":             m_pre,
        "m_e_relevant":           m_e_rel,
        "m_budget_util":          m_budget,
    }
