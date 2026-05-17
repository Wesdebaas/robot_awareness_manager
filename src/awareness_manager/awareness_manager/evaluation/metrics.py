"""
metrics.py - Pure-function metric library for trace-based evaluation.

Each function takes a loaded trace (dict with 'meta' and 'ticks' keys, as
returned by load_trace()) and returns a scalar or timeseries.  All metrics
read from the trace file only - no scenario re-running required.

Metric set (primary → secondary)
---------------------------------
M1  e_at_transition     - max E across new-goal's 1-hop hood at the transition tick.
                          Primary readiness metric: did the strategy prepare before the switch?
M2  pre_transition_attn - mean attention to incoming-goal hood in the window before switch.
                          Tests F2 (Anticipatory Horizon) directly: did the AM predict the need?
M3  lag_seconds         - seconds until E_max < 0.1 after the transition.
                          Recovery speed; complements M1 (low M1 → low M3 almost by definition).
M4  e_relevant          - run-mean epistemic error over goal-relevant concepts.
                          Ongoing knowledge quality; useful for regime comparisons.
M5  budget_util         - schedule slots used / budget per tick.
                          Sanity check; confirms budget is the binding constraint.

Additional functions available but not included in all_metrics() output:
    mean_epistemic_error_irrelevant() - E over concepts outside the mission neighbourhood.
    anticipatory_cache_hit_rate()     - binary pre-cache hit rate (redundant with M2).
    refresh_count_by_relevance()      - relevant vs irrelevant schedule slots.

Neighborhood convention
-----------------------
The "goal neighborhood" of goal g is the set of class concepts reachable in
exactly 1 hop in the semantic_edges graph, filtered to decay_rate > 0 (task
nodes are excluded — their E is always 0 by design).

Strategy-agnostic: does not depend on the strategy's attention values.
Matches ReactiveBaseline's active set exactly — the fairest common ruler.

Class concepts only.  Instance concepts inherit their class's membership for
mean-E metrics; excluded from transition metrics (class-level only).

Trace format (schema v2)
------------------------
meta: {schema_version, scenario, strategy, strategy_params, budget, dt,
       sample_rate, structure: {classes, instances, semantic_edges,
       relational_edges}, history_maxlen, truncated}

tick: {t, goal, queue, schedule, events, concepts:
        {concept_id: {a, e, pe, ch_m, ch_an, ch_r, ch_s}}}

"in_schedule" is NOT a stored per-concept field; reconstruct as
    concept_id in tick["schedule"]
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

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
# M1 - Mean epistemic error over goal-relevant concepts
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
# M2 - Mean epistemic error over goal-irrelevant concepts
# ---------------------------------------------------------------------------

def mean_epistemic_error_irrelevant(
    trace: dict,
    *,
    include_instances: bool = True,
) -> dict:
    """
    Mean epistemic error over goal-irrelevant concepts.

    Irrelevant = all concepts with decay_rate > 0 that are NOT in the
    goal-relevant neighbourhood. Measures "wasted staleness" - concepts
    ignored by the strategy that might have been useful.

    Formula: complement of M1, same averaging logic.

    Returns:
        {
            "scalar":    float
            "timeseries": [(t, E_irrel_t), ...]
            "irrelevant_set": sorted list of class IDs in the irrelevant set
        }
    """
    meta, ticks = trace["meta"], trace["ticks"]
    hood = _mission_neighborhood(meta, ticks)
    decay = _class_decay_rates(meta)
    inst_class = _instance_class_map(meta) if include_instances else {}

    # All class concepts with decay_rate > 0 that are NOT in hood
    irrel_classes = frozenset(
        cid for cid, dr in decay.items() if dr > 0.0 and cid not in hood
    )
    irrel_instances = frozenset(
        iid for iid, cls in inst_class.items() if cls in irrel_classes
    ) if include_instances else frozenset()

    timeseries: list[tuple[float, float]] = []
    for rec in ticks:
        vals = [
            c["e"] for cid, c in rec["concepts"].items()
            if cid in irrel_classes or cid in irrel_instances
        ]
        if vals:
            timeseries.append((rec["t"], statistics.mean(vals)))

    scalar = statistics.mean(v for _, v in timeseries) if timeseries else float("nan")
    return {
        "scalar": scalar,
        "timeseries": timeseries,
        "irrelevant_set": sorted(irrel_classes),
    }


# ---------------------------------------------------------------------------
# M3 - Refresh budget utilisation
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
# M4 - Cognitive lag at task transition
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
# M5 - Anticipatory cache hit rate
# ---------------------------------------------------------------------------

def anticipatory_cache_hit_rate(
    trace: dict,
    *,
    lookback_ticks: int = 50,
    lookback_seconds: float | None = None,
) -> dict:
    """
    At each goal transition, was the new goal's neighbourhood already being
    refreshed (in the schedule) before the transition occurred?

    A "hit" on transition i is defined as: ≥1 concept in the new goal's
    1-hop neighbourhood appeared in the schedule during the lookback window
    immediately preceding the transition tick.

    The lookback window is determined by (in priority order):
        1. lookback_seconds (if given) → int(lookback_seconds / dt) ticks
        2. lookback_ticks (default 50)
    The intended recency window is observation_interval / 2, which is passed
    by all_metrics when the trace's observation_interval is known.

    This is the metric that quantifies F2 (Anticipatory Horizon): the AM
    pre-loads the upcoming goal's neighbourhood before the transition fires;
    reactive and always_on do not (or do so only coincidentally).

    Formula:
        hit(i) = |{c ∈ hood(new_i) : c ∈ union(schedule[t-lookback..t-1])}| > 0
        hit_rate = sum(hit) / len(transitions)

    Returns:
        {
            "hit_rate":       float  - fraction of transitions with ≥1 pre-cached concept
            "hits":           int
            "total":          int
            "transitions":    [{t_transition, to_goal, hit, pre_cached_concepts}, ...]
            "lookback_ticks": int    - actual window used
        }

    Caveats:
        AlwaysOnBaseline hits 1.0 trivially (observes everything every tick).
        The meaningful contrast is AM vs Reactive.
    """
    meta, ticks = trace["meta"], trace["ticks"]
    dt = meta.get("dt", 0.1)
    if lookback_seconds is not None:
        lookback_ticks = max(1, int(lookback_seconds / dt))

    results = []

    for i, rec in enumerate(ticks):
        for ev in rec.get("events", []):
            if ev["type"] != "goal_transition":
                continue
            new_goal = ev["to"]
            hood = _goal_neighborhood(meta, new_goal)
            lo = max(0, i - lookback_ticks)
            pre_scheduled: set[str] = set()
            for j in range(lo, i):
                pre_scheduled |= set(ticks[j].get("schedule", []))
            pre_cached = hood & pre_scheduled
            hit = len(pre_cached) > 0
            results.append({
                "t_transition": rec["t"],
                "to_goal": new_goal,
                "hit": hit,
                "pre_cached_concepts": sorted(pre_cached),
            })

    hits = sum(1 for r in results if r["hit"])
    total = len(results)
    hit_rate = hits / total if total > 0 else float("nan")
    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "total": total,
        "transitions": results,
        "lookback_ticks": lookback_ticks,
    }


# ---------------------------------------------------------------------------
# M6 - Total refresh count by relevance bucket
# ---------------------------------------------------------------------------

def refresh_count_by_relevance(
    trace: dict,
    *,
    include_instances: bool = True,
) -> dict:
    """
    Total number of refresh schedule slots spent on relevant vs irrelevant
    concepts over the entire run.

    "In the schedule" is reconstructed as concept_id ∈ record["schedule"].
    Each slot counts once - if schedule has budget 2, a relevant concept
    being scheduled contributes 1 to relevant_count.

    Formula:
        relevant_count   = Σ_t |schedule_t ∩ relevant_set|
        irrelevant_count = Σ_t |schedule_t ∩ irrelevant_set|
        ratio = relevant_count / max(1, irrelevant_count)

    Returns:
        {
            "relevant_count":   int
            "irrelevant_count": int
            "total_count":      int
            "relevant_fraction": float  - relevant_count / total_count
            "ratio":            float   - relevant / irrelevant (inf if irrelevant=0)
        }

    Caveats:
        Concepts that are neither relevant nor irrelevant (e.g. task nodes
        with decay_rate=0) are not counted in either bucket.  Their schedule
        appearances contribute to total_count but not to either sub-count.
    """
    meta, ticks = trace["meta"], trace["ticks"]
    hood = _mission_neighborhood(meta, ticks)
    decay = _class_decay_rates(meta)
    inst_class = _instance_class_map(meta) if include_instances else {}

    irrel_classes = frozenset(
        cid for cid, dr in decay.items() if dr > 0.0 and cid not in hood
    )
    irrel_instances = frozenset(
        iid for iid, cls in inst_class.items() if cls in irrel_classes
    ) if include_instances else frozenset()

    inst_relevant = frozenset(
        iid for iid, cls in inst_class.items() if cls in hood
    ) if include_instances else frozenset()

    rel_count = 0
    irrel_count = 0
    total_count = 0
    for rec in ticks:
        for cid in rec.get("schedule", []):
            total_count += 1
            if cid in hood or cid in inst_relevant:
                rel_count += 1
            elif cid in irrel_classes or cid in irrel_instances:
                irrel_count += 1

    ratio = rel_count / max(1, irrel_count)
    rel_frac = rel_count / max(1, total_count)
    return {
        "relevant_count": rel_count,
        "irrelevant_count": irrel_count,
        "total_count": total_count,
        "relevant_fraction": round(rel_frac, 4),
        "ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# M7 - Pre-transition attention (AM signal quality)
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
    Compute all metrics for a single trace and return them in a flat dict.

    Scalar values are pulled to the top level for easy CSV export.  Full
    timeseries and transition details are kept nested.

    Args:
        trace:  Loaded trace dict (from load_trace()).
        params: Optional run-level param dict (e.g. from manifest.json).
                Used to derive observation_interval for the M2 pre-attention window.
                Falls back to meta["strategy_params"]["observation_interval"]
                for AM traces; uses a 50-tick fallback for others.

    Returns:
        {
            # Primary metrics (transition quality):
            "m1_e_at_transition":     float | None,  - max E in new-goal hood at switch tick
            "m2_pre_transition_attn": float | None,  - mean attention to incoming hood before switch
            "m3_lag_seconds":         float | None,  - seconds until E_max < 0.1 after switch

            # Secondary metrics (run-level quality):
            "m4_e_relevant":          float,  - run-mean E over goal-relevant concepts
            "m5_budget_util":         float,  - mean schedule utilisation (slots / budget)

            # Full nested results:
            "m_cognitive_lag":  {...},   - full cognitive_lag() result
            "m_pre_attn":       {...},   - full pre_transition_attention() result
            "m_e_relevant":     {...},   - full mean_epistemic_error_relevant() result
            "m_budget_util":    {...},   - full budget_utilisation() result
        }
    """
    meta = trace["meta"]
    oi = (
        (params or {}).get("observation_interval")
        or meta.get("strategy_params", {}).get("observation_interval")
    )
    m2_lookback_s = oi if oi else None

    m_e_rel = mean_epistemic_error_relevant(trace)
    m_budget = budget_utilisation(trace)
    m_lag = cognitive_lag(trace)
    m_pre = pre_transition_attention(trace, lookback_seconds=m2_lookback_s)

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
        "m_cognitive_lag": m_lag,
        "m_pre_attn":      m_pre,
        "m_e_relevant":    m_e_rel,
        "m_budget_util":   m_budget,
    }
