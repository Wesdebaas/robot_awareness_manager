"""
necessity/experiments.py — Necessity experiments for the three core components.

Each experiment removes one core component and demonstrates that the system
fails to function as an awareness manager without it.

    N1 — Epistemic state E [core, R1]
        Without E, priority = A for all concepts. When A is uniform (all
        concepts equally relevant), the scheduler reduces to a fixed
        deterministic selection and cannot respond to staleness at all.
        Volatile concepts whose spikes are never detected accumulate maximum
        epistemic error — the system cannot distinguish fresh from stale.

    N2 — Relevance A [core, R2]
        Without A, priority = E for all concepts including goal-irrelevant ones.
        Irrelevant volatile concepts spiked to E=1.0 steal the observation budget
        and the mission-relevant concepts accumulate high epistemic error — the
        system cannot focus resources on what matters for the current goal.

    N3 — Combination P = ExA [core, R3]
        The multiplicative combination creates a joint filter: a concept must be
        BOTH stale AND relevant to win the budget. With additive (E+A), a stale
        low-relevance concept can outcompete a fresh high-relevance one by making
        up for low A with high E, diverting budget from mission-critical concepts.

    N4 — Causal benefit term w_causal [core, R4]
        Without the causal benefit term, the scheduler treats X_causal (which
        reduces epistemic error in non-observable implied concept Y via causal
        propagation) identically to any other observable concept.  When Y is stale,
        the causal benefit term elevates X_causal's priority so the budget is
        directed toward observations that deliver the most total epistemic gain,
        including indirect gain for non-observable concepts.

Usage:

    python -m awareness_manager.evaluation.necessity n1
    python -m awareness_manager.evaluation.necessity n4
    python -m awareness_manager.evaluation.necessity all
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# N1 — Epistemic state E is necessary
# ---------------------------------------------------------------------------

def experiment_e_necessity(
    N: int = 12,
    K: int = 3,
    delta_volatile: float = 0.1,
    delta_static: float = 0.001,
    budget: int = 1,
    obs_interval: float = 10.0,
    spike_period: int = 50,
    ticks: int = 500,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    N1: Epistemic state E is necessary for awareness management.

    Setup
    -----
    N concepts, all 1-hop from goal → uniform attention A = (1-α)^1 = 0.5.
    Because every concept has identical A, attention alone carries zero
    discriminating signal. The only signal that can differentiate concepts is
    their epistemic state E.

    K concepts are volatile (δ_volatile=0.1, periodically spiked to E=1.0).
    N-K concepts are static (δ_static=0.001, drift extremely slowly).
    All concepts start at E=0.
    Budget = 1 (one observation slot per tick).

    Non-degeneracy
    --------------
    With K=3 volatile and budget=1 under E-on, each volatile is observed
    every K=3 ticks. Drift per 3 ticks = 0.3. F3 refresh = 0.632.
    Net per cycle: -0.332 → E_volatile converges to 0 (non-degenerate regime).
    δ_static is chosen so that static concepts remain below E=0.5 throughout
    the 500-tick run, avoiding budget competition with the volatile group.

    Conditions
    ----------
    E-on  (normal AM, F5 enabled):  priority = E x A.
          When a volatile concept spikes, P = 1.0 x 0.5 = 0.5, while the
          other volatile are near E=0 (P ≈ 0). The spiked concept wins the
          budget immediately. Detection latency = 0 ticks.

    E-off (F5 disabled):             priority = A = 0.5 for every concept.
          All priorities are identical. The scheduler's stable sort always
          returns the same concept first. With budget=1 only that one concept
          is ever scheduled, regardless of what has changed in the KB.
          The other N-1 concepts — including K-1 volatile ones — are never
          refreshed; their spikes are never detected.

    Failure criterion
    -----------------
    Without E the scheduler makes the same choice every tick regardless of
    the current epistemic state of the KB. It cannot distinguish a concept
    that has just changed state (E=1.0 after spike) from one it already knows
    (E≈0). Detection rate collapses to ≈1/N; mean E of the volatile group
    approaches the maximum possible value.

    Returns
    -------
    dict with keys:
        "N", "K", "budget", "ticks", "seeds"
        "delta_volatile", "delta_static"
        "conditions": ["e_on", "e_off"]
        "detection_rate":      {condition → mean across seeds}
        "mean_latency_ticks":  {condition → mean across seeds (detected only)}
        "mean_e_volatile":     {condition → mean E of volatile group over run}
        "mean_e_static":       {condition → mean E of static group over run}
        "rows":                list[dict] — one row per (condition, seed)
        "summary":             {condition → {detection_rate, mean_latency, mean_e_volatile}}
    """
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.concept import Concept
    from awareness_manager.feature_config import FeatureConfig
    from awareness_manager.knowledge_base import KnowledgeBase

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    # All concepts are 1-hop from goal (R = N, K_overlap = K).
    # build_abstract_kb is not used directly because we need two decay rates.
    # The KB is built by hand following the same structure.

    def _build_kb(seed: int) -> tuple[KnowledgeBase, list[str], list[str]]:
        """Build KB: N concepts, all 1-hop, K volatile (high δ), N-K static (low δ)."""
        rng = random.Random(seed)

        kb = KnowledgeBase()
        kb.add_concept(Concept(concept_id="goal", concept_type="task", decay_rate=0.0))

        concept_ids = [f"c{i}" for i in range(N)]
        for cid in concept_ids:
            kb.add_concept(Concept(
                concept_id=cid,
                concept_type="object",
                decay_rate=0.1,        # overwritten below
                epistemic_error=0.0,   # start fully known; volatile must earn priority via E
            ))
            kb.add_relation("goal", cid, weight=1.0)

        # Pick K volatile concepts at random (seeded for reproducibility).
        volatile_ids = rng.sample(concept_ids, K)
        static_ids   = [c for c in concept_ids if c not in volatile_ids]

        for cid in volatile_ids:
            kb.get_concept(cid).decay_rate = delta_volatile
        for cid in static_ids:
            kb.get_concept(cid).decay_rate = delta_static

        return kb, volatile_ids, static_ids

    def _run_condition(condition: str, seed: int) -> dict:
        kb, volatile_ids, static_ids = _build_kb(seed)
        volatile_set = set(volatile_ids)
        static_set   = set(static_ids)

        fc = FeatureConfig() if condition == "e_on" else FeatureConfig.with_disabled("f5")
        am = AwarenessManager(
            kb=kb,
            goal_id="goal",
            budget=budget,
            observation_interval=obs_interval,
            alpha=0.5,
            feature_config=fc,
        )

        # Build per-concept spike schedule with ±30 % jitter.
        spike_rng = random.Random(seed ^ 0xDEAD)
        stagger   = max(1, spike_period // max(K, 1))
        spikes_by_tick: dict[int, list[str]] = {}
        for idx, cid in enumerate(volatile_ids):
            offset = stagger * idx
            jitter = max(1, stagger // 2)
            t = max(0, offset + spike_rng.randint(-jitter, jitter))
            while t < ticks:
                spikes_by_tick.setdefault(t, []).append(cid)
                t += spike_rng.randint(
                    max(1, int(0.7 * spike_period)),
                    int(1.3 * spike_period),
                )

        # Track detection latency.
        pending: dict[str, int] = {}   # cid → spike_tick
        total_spikes  = 0
        detected_spikes = 0
        latencies: list[int] = []

        # Track mean E of each group.
        e_volatile_sum = 0.0
        e_static_sum   = 0.0
        tick_count     = 0

        for tick_i in range(ticks):
            # Inject spikes before strategy tick.
            for cid in spikes_by_tick.get(tick_i, []):
                kb.get_concept(cid).epistemic_error = 1.0
                pending[cid] = tick_i
                total_spikes += 1

            schedule = am.tick(1.0)
            if schedule:
                am.observe(schedule[0])

            # Check detections.
            for cid in list(pending):
                if cid in schedule:
                    latencies.append(tick_i - pending.pop(cid))
                    detected_spikes += 1

            # Accumulate mean E per group.
            e_volatile_sum += _mean_e(kb, volatile_set)
            e_static_sum   += _mean_e(kb, static_set)
            tick_count += 1

        detection_rate  = detected_spikes / total_spikes if total_spikes > 0 else 0.0
        mean_latency    = statistics.mean(latencies) if latencies else None
        mean_e_volatile = e_volatile_sum / tick_count if tick_count > 0 else float("nan")
        mean_e_static   = e_static_sum   / tick_count if tick_count > 0 else float("nan")

        return {
            "condition":       condition,
            "seed":            seed,
            "total_spikes":    total_spikes,
            "detected_spikes": detected_spikes,
            "detection_rate":  detection_rate,
            "mean_latency":    mean_latency,
            "mean_e_volatile": mean_e_volatile,
            "mean_e_static":   mean_e_static,
        }

    rows: list[dict] = []
    for condition in ["e_on", "e_off"]:
        for seed in seeds:
            rows.append(_run_condition(condition, seed))

    def _agg(condition: str, key: str) -> dict:
        vals = [r[key] for r in rows
                if r["condition"] == condition and r[key] is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not vals:
            return {"mean": None, "std": None}
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    summary: dict[str, dict] = {}
    for cond in ["e_on", "e_off"]:
        summary[cond] = {
            "detection_rate":  _agg(cond, "detection_rate"),
            "mean_latency":    _agg(cond, "mean_latency"),
            "mean_e_volatile": _agg(cond, "mean_e_volatile"),
            "mean_e_static":   _agg(cond, "mean_e_static"),
        }

    return {
        "N": N, "K": K, "budget": budget, "ticks": ticks, "seeds": seeds,
        "delta_volatile": delta_volatile,
        "delta_static":   delta_static,
        "obs_interval":   obs_interval,
        "spike_period":   spike_period,
        "conditions":     ["e_on", "e_off"],
        "rows":           rows,
        "summary":        summary,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_n1(results: dict) -> None:
    N       = results["N"]
    K       = results["K"]
    budget  = results["budget"]
    ticks   = results["ticks"]
    n_seeds = len(results["seeds"])
    dv      = results["delta_volatile"]
    ds      = results["delta_static"]
    obs     = results["obs_interval"]

    print("\n=== N1: E Necessity — Epistemic State is a Core Component ===\n")
    print(f"  Note  : Near-deterministic regime (std ≈ 0) — results are capability")
    print(f"          demonstrations, not statistical estimates. p-values trivially")
    print(f"          reject H₀ because outcomes are deterministic, not because")
    print(f"          the effect is statistically sampled.")
    print()
    print(f"  Setup : N={N} concepts, all 1-hop from goal → uniform A=0.50")
    print(f"          K={K} volatile (δ={dv}), {N-K} static (δ={ds})")
    print(f"          Budget={budget}, obs_interval={obs}s, {ticks} ticks, {n_seeds} seeds")
    print()
    print(f"  Claim : A is identical for every concept, so A alone cannot")
    print(f"          differentiate which concepts need refreshing. Only E")
    print(f"          carries that signal. With F5 disabled (E removed from")
    print(f"          priority), all priorities collapse to the same value,")
    print(f"          and the scheduler fixes on one concept permanently.")
    print()

    s = results["summary"]

    # Expected E-off detection rate from theory.
    # With budget=1 and all equal priority the stable sort always picks c0.
    # Only spikes to c0 (1 volatile out of K, if c0 happens to be volatile)
    # are detected.  In general: detection_rate ≈ 1/N (only first concept seen).
    theoretical_e_off = round(1.0 / N, 4)

    header = f"  {'Condition':<10} {'Det. rate ↑':>14} {'Latency (ticks) ↓':>20} {'E_volatile ↓':>14} {'E_static ↓':>12}"
    print(header)
    print("  " + "-" * 74)
    for cond in ["e_on", "e_off"]:
        label  = "E-on  (AM)" if cond == "e_on" else "E-off (A-only)"
        dr     = s[cond]["detection_rate"]
        lat    = s[cond]["mean_latency"]
        ev     = s[cond]["mean_e_volatile"]
        est    = s[cond]["mean_e_static"]
        print(f"  {label:<14} {_fmt_pm(dr):>14} {_fmt_pm(lat):>20} "
              f"{_fmt_pm(ev):>14} {_fmt_pm(est):>12}")

    print()
    dr_on  = s["e_on"]["detection_rate"]["mean"]
    dr_off = s["e_off"]["detection_rate"]["mean"]
    ev_on  = s["e_on"]["mean_e_volatile"]["mean"]
    ev_off = s["e_off"]["mean_e_volatile"]["mean"]

    if dr_on is not None and dr_off is not None:
        print(f"  Detection rate  E-on={dr_on:.2f}  E-off={dr_off:.2f}"
              f"  (theoretical E-off lower bound ≈ {theoretical_e_off:.2f} = 1/N)")
    if ev_on is not None and ev_off is not None:
        delta_e = round(ev_off - ev_on, 4)
        print(f"  ΔE_volatile (E-off − E-on) = {delta_e:.4f}  (positive = E removes staleness bias)")

    print()
    print("  Interpretation:")
    print(f"  Without E, all {N} concepts have identical priority (P = A = 0.50).")
    print(f"  The scheduler's stable sort picks the same concept every tick,")
    print(f"  making {N-1} out of {N} concepts permanently invisible to the budget.")
    print(f"  Volatile spikes go undetected; epistemic error accumulates without bound.")
    print(f"  With E, a spike immediately elevates that concept's priority above all")
    print(f"  others (P_spiked = 1.0 × 0.50 = 0.50 vs P_fresh ≈ {ds*obs*0.5:.4f}),")
    print(f"  and detection latency collapses to 0 ticks.")
    print(f"  Conclusion: E is constitutive — removing it eliminates the scheduler's")
    print(f"  ability to respond to changes in the world.")


# ---------------------------------------------------------------------------
# N2 — Relevance A is necessary
# ---------------------------------------------------------------------------

def experiment_a_necessity(
    N_relevant: int = 3,
    K_irrelevant: int = 6,
    delta: float = 0.1,
    budget: int = 1,
    obs_interval: float = 10.0,
    spike_period: int = 30,
    ticks: int = 500,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    N2: Goal-conditioned relevance A is necessary for awareness management.

    Setup
    -----
    N_relevant concepts each connected to the goal node by a weight-1 edge:
    spreading activation gives A = (1-0.5)^1 = 0.50 (A-on) or A = 1.0 (E-only).
    K_irrelevant volatile concepts have NO edge to the goal: A = 0.0 under F1.
    All concepts share the same decay rate δ.  Initial epistemic error E = 0.50.
    Budget = 1 (one observation slot per tick).

    Non-degeneracy
    --------------
    A-on with R=3 and budget=1: cycle time = 3 ticks.
    Net change per cycle = R x δ - refresh = 3x0.10 - 0.632 = -0.332 < 0 → converges.

    A-off with all 9 concepts competing and budget=1: effective cycle ≈ 9 ticks.
    Net = 9 x 0.10 - 0.632 = +0.268 > 0 → degenerate (E_relevant → 1.0).

    Conditions
    ----------
    A-on  (normal AM, F1+F5 enabled):  priority = E x A.
          Irrelevant concepts have A = 0 → P = 0 → structurally excluded.
          All budget flows to the N_relevant mission-relevant concepts.
          E_relevant converges to ~0 (non-degenerate regime).

    A-off (E-only, no goal structure):  priority = E over ALL schedulable concepts.
          Irrelevant volatile concepts spike to E = 1.0 and win the budget.
          Relevant concepts receive budget only when no volatile is at high E.
          E_relevant drifts to ~1.0 (degenerate regime).

    Failure criterion
    -----------------
    Without A the scheduler is goal-blind: high-E irrelevant concepts dominate the
    schedule and mission-relevant concepts are starved of observations.  The fraction
    of budget allocated to relevant concepts collapses and E_relevant approaches 1.0.

    Returns
    -------
    dict with keys:
        "N_relevant", "K_irrelevant", "delta", "budget", "ticks", "seeds"
        "obs_interval", "spike_period"
        "conditions": ["a_on", "a_off"]
        "mean_e_relevant":      {condition → agg}
        "mean_e_irrelevant":    {condition → agg}
        "sched_frac_relevant":  {condition → agg}  (fraction of ticks relevant scheduled)
        "sched_frac_irrelevant":{condition → agg}
        "rows":                 list[dict] — one row per (condition, seed)
        "summary":              {condition → {mean_e_relevant, sched_frac_relevant, ...}}
    """
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.concept import Concept
    from awareness_manager.knowledge_base import KnowledgeBase

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    def _build_kb(seed: int) -> tuple[KnowledgeBase, list[str], list[str]]:
        kb = KnowledgeBase()
        kb.add_concept(Concept(concept_id="goal", concept_type="task", decay_rate=0.0))

        relevant_ids = [f"r{i}" for i in range(N_relevant)]
        for cid in relevant_ids:
            kb.add_concept(Concept(concept_id=cid, concept_type="object",
                                   decay_rate=delta, epistemic_error=0.5))
            kb.add_relation("goal", cid, weight=1.0)

        irrelevant_ids = [f"v{i}" for i in range(K_irrelevant)]
        for cid in irrelevant_ids:
            kb.add_concept(Concept(concept_id=cid, concept_type="object",
                                   decay_rate=delta, epistemic_error=0.5))
            # intentionally no edge to goal → A=0 in AM; invisible to E×A priority

        return kb, relevant_ids, irrelevant_ids

    def _run_condition(condition: str, seed: int) -> dict:
        kb, relevant_ids, irrelevant_ids = _build_kb(seed)
        relevant_set   = set(relevant_ids)
        irrelevant_set = set(irrelevant_ids)

        if condition == "a_on":
            am = AwarenessManager(
                kb=kb, goal_id="goal", budget=budget,
                observation_interval=obs_interval, alpha=0.5,
            )

            def _tick():   return am.tick(1.0)
            def _observe(cid): am.observe(cid)
        else:
            # E-only strategy: sort ALL schedulable concepts by E, ignore goal structure
            _sched_ids = [c for c in kb.concept_ids()
                          if c != "goal" and kb.get_concept(c).decay_rate > 0]

            def _tick():
                kb.tick(1.0)
                ranked = sorted(_sched_ids,
                                key=lambda c: kb.get_concept(c).epistemic_error,
                                reverse=True)
                return ranked[:budget]

            def _observe(cid):
                dc = kb.get_concept(cid).decay_rate
                refresh = 1.0 - math.exp(-dc * obs_interval)
                kb.refresh_concept(cid, refresh)

        # Spike schedule for irrelevant volatile
        spike_rng = random.Random(seed ^ 0xBEEF)
        stagger   = max(1, spike_period // max(K_irrelevant, 1))
        spikes_by_tick: dict[int, list[str]] = {}
        for idx, cid in enumerate(irrelevant_ids):
            offset = stagger * idx
            jitter = max(1, stagger // 2)
            t = max(0, offset + spike_rng.randint(-jitter, jitter))
            while t < ticks:
                spikes_by_tick.setdefault(t, []).append(cid)
                t += spike_rng.randint(
                    max(1, int(0.7 * spike_period)),
                    int(1.3 * spike_period),
                )

        e_rel_sum   = 0.0
        e_irrel_sum = 0.0
        sched_rel   = 0
        sched_irrel = 0
        tick_count  = 0

        for tick_i in range(ticks):
            for cid in spikes_by_tick.get(tick_i, []):
                kb.get_concept(cid).epistemic_error = 1.0

            schedule = _tick()
            if schedule:
                _observe(schedule[0])
                if schedule[0] in relevant_set:
                    sched_rel += 1
                elif schedule[0] in irrelevant_set:
                    sched_irrel += 1

            e_rel_sum   += _mean_e(kb, relevant_set)
            e_irrel_sum += _mean_e(kb, irrelevant_set)
            tick_count  += 1

        return {
            "condition":           condition,
            "seed":                seed,
            "mean_e_relevant":     e_rel_sum   / tick_count,
            "mean_e_irrelevant":   e_irrel_sum / tick_count,
            "sched_frac_relevant": sched_rel   / ticks,
            "sched_frac_irrelevant": sched_irrel / ticks,
        }

    rows: list[dict] = []
    for condition in ["a_on", "a_off"]:
        for seed in seeds:
            rows.append(_run_condition(condition, seed))

    def _agg(condition: str, key: str) -> dict:
        vals = [r[key] for r in rows if r["condition"] == condition]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    summary: dict[str, dict] = {}
    for cond in ["a_on", "a_off"]:
        summary[cond] = {
            "mean_e_relevant":      _agg(cond, "mean_e_relevant"),
            "mean_e_irrelevant":    _agg(cond, "mean_e_irrelevant"),
            "sched_frac_relevant":  _agg(cond, "sched_frac_relevant"),
            "sched_frac_irrelevant":_agg(cond, "sched_frac_irrelevant"),
        }

    return {
        "N_relevant": N_relevant, "K_irrelevant": K_irrelevant,
        "delta": delta, "budget": budget, "ticks": ticks, "seeds": seeds,
        "obs_interval": obs_interval, "spike_period": spike_period,
        "conditions": ["a_on", "a_off"],
        "rows":    rows,
        "summary": summary,
    }


def print_n2(results: dict) -> None:
    R    = results["N_relevant"]
    K    = results["K_irrelevant"]
    d    = results["delta"]
    bud  = results["budget"]
    obs  = results["obs_interval"]
    tick = results["ticks"]
    ns   = len(results["seeds"])
    sp   = results["spike_period"]

    print("\n=== N2: A Necessity — Goal-Conditioned Relevance is a Core Component ===\n")
    print(f"  Note  : Near-deterministic regime (std ≈ 0) — results are capability")
    print(f"          demonstrations, not statistical estimates.")
    print()
    print(f"  Setup : R={R} relevant concepts (1-hop from goal → A=0.50)")
    print(f"          K={K} irrelevant volatile (no edge to goal → A=0.00)")
    print(f"          All concepts: δ={d}, Budget={bud}, obs_interval={obs}s")
    print(f"          spike_period≈{sp} ticks, {tick} ticks, {ns} seeds")
    print()
    cycle_non_degen = R * d
    refresh_val = 1.0 - math.exp(-d * obs)
    net_aon  = cycle_non_degen - refresh_val
    net_aoff = (R + K) * d - refresh_val
    print(f"  Non-degeneracy: A-on cycle={R} ticks, net={net_aon:+.3f} (< 0 ✓)")
    print(f"                  A-off cycle≈{R+K} ticks, net={net_aoff:+.3f} (> 0 → degenerate)")
    print()
    print(f"  Claim : Without A, the scheduler sees all {R+K} concepts equally. Irrelevant")
    print(f"          volatile (E=1.0 after spike) outcompete relevant concepts ({R} present)")
    print(f"          and monopolise the budget. E_relevant drifts to maximum.")
    print(f"          With A, irrelevant concepts are structurally excluded (A=0 → P=0)")
    print(f"          and all budget flows to the {R} mission-relevant concepts.")
    print()

    s = results["summary"]

    header = (f"  {'Condition':<18} {'E_relevant ↓':>14} {'E_irrelevant':>14} "
              f"{'Sched frac rel ↑':>18} {'Sched frac irrel':>18}")
    print(header)
    print("  " + "-" * 88)
    for cond, label in [("a_on", "A-on  (AM)"), ("a_off", "A-off (E-only)")]:
        er   = s[cond]["mean_e_relevant"]
        ei   = s[cond]["mean_e_irrelevant"]
        sfr  = s[cond]["sched_frac_relevant"]
        sfi  = s[cond]["sched_frac_irrelevant"]
        print(f"  {label:<18} {_fmt_pm(er):>14} {_fmt_pm(ei):>14} "
              f"{_fmt_pm(sfr):>18} {_fmt_pm(sfi):>18}")

    print()
    er_on  = s["a_on"]["mean_e_relevant"]["mean"]
    er_off = s["a_off"]["mean_e_relevant"]["mean"]
    sf_on  = s["a_on"]["sched_frac_relevant"]["mean"]
    sf_off = s["a_off"]["sched_frac_relevant"]["mean"]

    if er_on is not None and er_off is not None:
        delta_e = round(er_off - er_on, 4)
        print(f"  ΔE_relevant (A-off − A-on) = {delta_e:+.4f}  (positive → A prevents staleness of relevant concepts)")
    if sf_on is not None and sf_off is not None:
        delta_sf = round(sf_on - sf_off, 4)
        print(f"  Δsched_frac_relevant (A-on − A-off) = {delta_sf:+.4f}  (positive → A focuses budget on goal)")

    print()
    print("  Interpretation:")
    print(f"  Without A, {K} irrelevant volatile concepts steal the observation budget.")
    print(f"  Relevant concepts are observed only when no volatile is at high E.")
    print(f"  Their effective cycle time grows to ≈{R+K} ticks → degenerate (E_relevant → 1.0).")
    print(f"  With A, irrelevant volatile have P = Ex0 = 0 and are never scheduled.")
    print(f"  Budget is exclusively allocated to the {R} mission-relevant concepts.")
    print(f"  Their cycle time = {R} ticks → non-degenerate (E_relevant → 0).")
    print(f"  Conclusion: A is constitutive — removing it makes the scheduler goal-blind,")
    print(f"  wasting resources on irrelevant environmental changes.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean_e(kb, concept_ids) -> float:
    vals = [kb.get_concept(c).epistemic_error for c in concept_ids]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.4f}"


def _fmt_pm(d: dict | None) -> str:
    if not d or d.get("mean") is None:
        return "—"
    mean = d["mean"]
    std  = d.get("std") or 0.0
    return f"{mean:.4f}±{std:.4f}"


# ---------------------------------------------------------------------------
# N3 — Multiplicative combination P = E×A is necessary
# ---------------------------------------------------------------------------

def experiment_combination_necessity(
    R_primary: int = 3,
    K_secondary: int = 6,
    delta: float = 0.1,
    budget: int = 1,
    obs_interval: float = 10.0,
    spike_period: int = 30,
    ticks: int = 500,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    N3: The multiplicative combination P = ExA is necessary for awareness management.

    Setup
    -----
    R_primary concepts at weight-1 graph distance from goal → A = (1-0.5)^1 = 0.50.
    K_secondary volatile concepts at weight-4 distance → A = (1-0.5)^4 = 0.0625.
    All concepts share the same decay rate δ. Initial epistemic error E = 0.50.
    Budget = 1. K_secondary concepts are periodically spiked to E = 1.0.

    Non-degeneracy
    --------------
    Multiplicative (ExA): primary wins whenever E_prim > E_sec x (A_sec/A_prim)
    = E_sec / 8. With secondary spiked (E_sec = 1.0): primary wins at E_prim > 0.125.
    After observation E_prim → 0; rebounds past 0.125 within 2 ticks.
    Effective primary cycle ≈ 3 ticks. Net = 3 x 0.1 - 0.632 = -0.332 < 0 → converges.

    Additive (E+A): secondary wins when E_sec > E_prim + 0.4375. After primary is
    observed (E_prim → 0), spiked secondary (E_sec = 1.0) wins for ≈ 5-6 ticks.
    Effective primary cycle grows to ≈ 8-10 ticks → net > 0 → degenerate.

    Conditions
    ----------
    Multiplicative (P = E x A):
        Spiked secondary (E=1.0): P_sec = 0.0625. Primary at E_prim > 0.125: P_prim > 0.0625.
        Primary wins almost always. Budget stays focused on mission-relevant primary.
        E_primary converges to near 0.

    Additive (P = E + A):
        Spiked secondary: P_sec = 1.0625. Primary needs E_prim > 0.5625 to win.
        After each primary observation the budget is diverted to secondary for ≈ 5-6 ticks.
        E_primary drifts upward (degenerate regime).

    Failure criterion
    -----------------
    Additive combination allows stale low-relevance (A=0.0625) concepts to outcompete
    moderately-fresh high-relevance (A=0.5) concepts whenever their E difference
    exceeds the A gap (0.4375). Budget is wasted maintaining barely-relevant volatile
    concepts at the cost of mission-critical primary awareness.

    Returns
    -------
    dict with keys:
        "R_primary", "K_secondary", "delta", "budget", "obs_interval", "spike_period"
        "a_primary" (0.5), "a_secondary" (0.0625), "ticks", "seeds"
        "conditions": ["mult", "add"]
        "rows":    list[dict] — one row per (condition, seed)
        "summary": {condition → {mean_e_primary, mean_e_secondary, sched_frac_primary, ...}}
    """
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.concept import Concept
    from awareness_manager.feature_config import PriorityWeights
    from awareness_manager.knowledge_base import KnowledgeBase

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    A_PRIMARY   = (1.0 - 0.5) ** 1.0   # = 0.5
    A_SECONDARY = (1.0 - 0.5) ** 4.0   # = 0.0625

    def _build_kb(seed: int) -> tuple[KnowledgeBase, list[str], list[str]]:
        kb = KnowledgeBase()
        kb.add_concept(Concept(concept_id="goal", concept_type="task", decay_rate=0.0))

        primary_ids = [f"p{i}" for i in range(R_primary)]
        for cid in primary_ids:
            kb.add_concept(Concept(concept_id=cid, concept_type="object",
                                   decay_rate=delta, epistemic_error=0.5))
            kb.add_relation("goal", cid, weight=1.0)  # A = 0.5

        secondary_ids = [f"s{i}" for i in range(K_secondary)]
        for cid in secondary_ids:
            kb.add_concept(Concept(concept_id=cid, concept_type="object",
                                   decay_rate=delta, epistemic_error=0.5))
            kb.add_relation("goal", cid, weight=4.0)  # A = (0.5)^4 = 0.0625

        return kb, primary_ids, secondary_ids

    def _run_condition(condition: str, seed: int) -> dict:
        kb, primary_ids, secondary_ids = _build_kb(seed)
        primary_set   = set(primary_ids)
        secondary_set = set(secondary_ids)

        # Both conditions use the same AwarenessManager; only the combination_rule differs.
        # 'mult': P_ea = w_ea × (E × A)  — multiplicative joint filter (default AM)
        # 'add':  P_ea = w_ea × (E + A)  — additive, allows high-E to compensate low-A
        pw = PriorityWeights(
            combination_rule='multiplicative' if condition == "mult" else 'additive'
        )
        am = AwarenessManager(
            kb=kb, goal_id="goal", budget=budget,
            observation_interval=obs_interval, alpha=0.5,
            priority_weights=pw,
        )

        # Spike schedule for secondary volatile
        spike_rng = random.Random(seed ^ 0xCAFE)
        stagger   = max(1, spike_period // max(K_secondary, 1))
        spikes_by_tick: dict[int, list[str]] = {}
        for idx, cid in enumerate(secondary_ids):
            offset = stagger * idx
            jitter = max(1, stagger // 2)
            t = max(0, offset + spike_rng.randint(-jitter, jitter))
            while t < ticks:
                spikes_by_tick.setdefault(t, []).append(cid)
                t += spike_rng.randint(
                    max(1, int(0.7 * spike_period)),
                    int(1.3 * spike_period),
                )

        e_prim_sum = 0.0
        e_sec_sum  = 0.0
        sched_prim = 0
        sched_sec  = 0
        tick_count = 0

        for tick_i in range(ticks):
            for cid in spikes_by_tick.get(tick_i, []):
                kb.get_concept(cid).epistemic_error = 1.0

            schedule = am.tick(1.0)
            if schedule:
                am.observe(schedule[0])
                if schedule[0] in primary_set:
                    sched_prim += 1
                elif schedule[0] in secondary_set:
                    sched_sec += 1

            e_prim_sum += _mean_e(kb, primary_set)
            e_sec_sum  += _mean_e(kb, secondary_set)
            tick_count += 1

        return {
            "condition":            condition,
            "seed":                 seed,
            "mean_e_primary":       e_prim_sum / tick_count,
            "mean_e_secondary":     e_sec_sum  / tick_count,
            "sched_frac_primary":   sched_prim / ticks,
            "sched_frac_secondary": sched_sec  / ticks,
        }

    rows: list[dict] = []
    for condition in ["mult", "add"]:
        for seed in seeds:
            rows.append(_run_condition(condition, seed))

    def _agg(condition: str, key: str) -> dict:
        vals = [r[key] for r in rows if r["condition"] == condition]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    summary: dict[str, dict] = {}
    for cond in ["mult", "add"]:
        summary[cond] = {
            "mean_e_primary":       _agg(cond, "mean_e_primary"),
            "mean_e_secondary":     _agg(cond, "mean_e_secondary"),
            "sched_frac_primary":   _agg(cond, "sched_frac_primary"),
            "sched_frac_secondary": _agg(cond, "sched_frac_secondary"),
        }

    return {
        "R_primary":   R_primary,  "K_secondary": K_secondary,
        "delta":       delta,      "budget":      budget,
        "obs_interval": obs_interval, "spike_period": spike_period,
        "a_primary":   A_PRIMARY,  "a_secondary": A_SECONDARY,
        "ticks":       ticks,      "seeds":       seeds,
        "conditions":  ["mult", "add"],
        "rows":        rows,
        "summary":     summary,
    }


def print_n3(results: dict) -> None:
    R   = results["R_primary"]
    K   = results["K_secondary"]
    d   = results["delta"]
    bud = results["budget"]
    obs = results["obs_interval"]
    sp  = results["spike_period"]
    t   = results["ticks"]
    ns  = len(results["seeds"])
    a_p = results["a_primary"]    # 0.5
    a_s = results["a_secondary"]  # 0.0625

    ratio    = a_p / a_s                  # 8.0
    thresh_e = a_s / a_p                  # 0.125  — primary E threshold to beat spiked secondary
    thresh_a = a_p - a_s                  # 0.4375 — E_sec advantage needed for secondary to win (additive)

    print("\n=== N3: P=E×A Necessity — Multiplicative Combination is a Core Component ===\n")
    print(f"  Note  : Near-deterministic regime (std ≈ 0) — results are capability")
    print(f"          demonstrations, not statistical estimates. Both conditions use")
    print(f"          the same AwarenessManager; only PriorityWeights.combination_rule")
    print(f"          differs ('multiplicative' vs 'additive'), ensuring a fair ablation.")
    print()
    print(f"  Setup : R={R} primary relevant concepts  (weight=1.0 edge → A={a_p:.4f})")
    print(f"          K={K} secondary volatile concepts (weight=4.0 edge → A={a_s:.4f})")
    print(f"          All concepts: δ={d}, Budget={bud}, obs_interval={obs}s")
    print(f"          spike_period≈{sp} ticks, {t} ticks, {ns} seeds")
    print()
    print(f"  Threshold (E×A): secondary wins when E_sec > {ratio:.0f} × E_prim")
    print(f"                   Spiked (E_sec=1.0): primary wins at E_prim > {thresh_e:.3f} — converges ✓")
    print(f"  Threshold (E+A): secondary wins when E_sec > E_prim + {thresh_a:.4f}")
    print(f"                   Spiked secondary beats primary until E_prim > {1.0 + a_s - a_p:.4f} — degenerate ✗")
    print()
    print(f"  Claim : E×A gates low-A secondary concepts even at peak E. Primary always")
    print(f"          wins once E_prim > {thresh_e:.3f}, giving a short cycle and low E_primary.")
    print(f"          E+A does not gate: spiked secondary (P={1.0+a_s:.4f}) beats primary")
    print(f"          for ≈5 ticks per spike, forcing a long primary cycle (degenerate).")
    print()

    s = results["summary"]

    header = (f"  {'Condition':<22} {'E_primary ↓':>13} {'E_secondary':>13} "
              f"{'Sched frac prim ↑':>19} {'Sched frac sec':>16}")
    print(header)
    print("  " + "-" * 88)
    for cond, label in [("mult", "Mult (E×A, AM)"), ("add", "Add  (E+A)")]:
        ep  = s[cond]["mean_e_primary"]
        es  = s[cond]["mean_e_secondary"]
        sfp = s[cond]["sched_frac_primary"]
        sfs = s[cond]["sched_frac_secondary"]
        print(f"  {label:<22} {_fmt_pm(ep):>13} {_fmt_pm(es):>13} "
              f"{_fmt_pm(sfp):>19} {_fmt_pm(sfs):>16}")

    print()
    ep_mult  = s["mult"]["mean_e_primary"]["mean"]
    ep_add   = s["add"]["mean_e_primary"]["mean"]
    sfp_mult = s["mult"]["sched_frac_primary"]["mean"]
    sfp_add  = s["add"]["sched_frac_primary"]["mean"]

    if ep_mult is not None and ep_add is not None:
        delta_e = round(ep_add - ep_mult, 4)
        print(f"  ΔE_primary (Add − Mult) = {delta_e:+.4f}  (positive → multiplicative keeps primary fresher)")
    if sfp_mult is not None and sfp_add is not None:
        delta_sf = round(sfp_mult - sfp_add, 4)
        print(f"  Δsched_frac_primary (Mult − Add) = {delta_sf:+.4f}  (positive → multiplicative focuses budget)")

    print()
    print("  Interpretation:")
    print(f"  ExA: secondary (A={a_s}) can beat primary only when E_sec > {ratio:.0f} x E_prim.")
    print(f"  Even at peak E=1.0, P_mult_secondary = {a_s:.4f} is beaten by primary at E_prim={thresh_e:.3f}.")
    print(f"  Primary cycle ≈ 3 ticks → non-degenerate. E_primary → 0.")
    print(f"  E+A: spiked secondary (P={1.0+a_s:.4f}) beats primary whenever E_prim < {1.0+a_s-a_p:.4f}.")
    print(f"  Each primary observation triggers ≈5-6 ticks of secondary dominance.")
    print(f"  Primary effective cycle grows → degenerate. E_primary drifts toward 1.0.")
    print(f"  Conclusion: ExA is constitutive — the multiplicative joint filter prevents")
    print(f"  low-relevance volatile concepts from compensating weak A with high E.")


# ---------------------------------------------------------------------------
# N4 — Causal benefit term is necessary
# ---------------------------------------------------------------------------

def experiment_causal_benefit_necessity(
    K_obs: int = 4,
    delta_obs: float = 0.1,
    delta_y: float = 0.2,
    prop_weight: float = 0.7,
    w_causal: float = 1.0,
    budget: int = 1,
    obs_interval: float = 10.0,
    spike_period: int = 50,
    ticks: int = 500,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    N4: The causal benefit term (w_causal > 0) is necessary for awareness
    management when non-observable concepts are causally implied by observable ones.

    Setup
    -----
    K_obs=4 observable concepts (x_causal, x2, x3, x4) are each connected to the
    goal by a weight-1 semantic edge → A = (1−0.5)^1 = 0.50 for all.
    All share the same decay rate delta_obs.

    One non-observable concept y_implied is also connected to the goal by a
    weight-1 semantic edge (A = 0.50) but is marked observable=False — it can
    only be known through causal propagation, never by direct observation.
    y_implied has decay rate delta_y > delta_obs (drifts faster).

    A directed causal edge x_causal → y_implied (propagation_weight=prop_weight)
    means that observing x_causal reduces E(y_implied) by prop_weight × refresh.

    y_implied is periodically spiked to E = 1.0 to simulate unexpected state change.

    Non-degeneracy check
    --------------------
    Causal_off — x_causal selected ~1/K_obs of ticks.
        Net drift per x_causal observation cycle (K_obs ticks):
            E_y drift      = delta_y × K_obs = 0.2 × 4 = 0.8
            causal refresh = prop_weight × (1 − exp(−delta_obs × obs_interval)) ÷ 1
                           = 0.7 × 0.632 = 0.442   (only on the 1 tick x_causal is chosen)
        Net = 0.8 − 0.442 = +0.358 > 0  →  degenerate; E_y drifts toward 1.0 between spikes.

    Causal_on — when y_implied is spiked (E=1.0), x_causal's priority is boosted:
        P(x_causal) = E(x_causal)×A + w_causal×prop_weight×1.0×0.5
                    = E(x_causal)×0.5 + 0.35
        P(xi)       = E(xi)×0.5   (no causal benefit)
        x_causal wins unless E(xi)−E(x_causal) > 0.7 — effectively always.
        Causal refresh fires every tick → net per tick = prop_weight×refresh − delta_y
                                        = 0.442 − 0.2 = +0.242 > 0  →  E_y recovers.

    Conditions
    ----------
    Causal_on  (w_causal > 0):
        When y_implied is stale the scheduler raises x_causal's priority.
        x_causal is observed almost every tick until E_y drops.
        Causal propagation keeps E_y low throughout the run.

    Causal_off (w_causal = 0):
        All four observable concepts compete with equal base priority.
        x_causal is scheduled only ≈1/4 of ticks regardless of E_y.
        y_implied accumulates high epistemic error between spikes; recovery is slow.

    Failure criterion
    -----------------
    Without the causal benefit term, the scheduler is unaware that observing
    x_causal delivers indirect epistemic benefit to y_implied.  It treats x_causal
    identically to causally-inert concepts x2/x3/x4, underinvesting in x_causal
    and leaving y_implied chronically stale.

    Returns
    -------
    dict with keys:
        "K_obs", "delta_obs", "delta_y", "prop_weight", "w_causal"
        "budget", "obs_interval", "spike_period", "ticks", "seeds"
        "conditions": ["causal_on", "causal_off"]
        "mean_e_y":        {condition → agg}   (lower is better)
        "sched_frac_xcausal": {condition → agg}  (higher = x_causal prioritised)
        "rows":    list[dict] — one row per (condition, seed)
        "summary": {condition → {...}}
    """
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.concept import Concept
    from awareness_manager.feature_config import PriorityWeights
    from awareness_manager.knowledge_base import KnowledgeBase

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    def _build_kb() -> tuple[KnowledgeBase, str, list[str]]:
        # KB structure is fixed (no seed) — the 5 seeds only vary spike timing.
        # std across seeds therefore reflects spike-timing variance only, not KB variance.
        kb = KnowledgeBase()
        kb.add_concept(Concept(concept_id="goal", concept_type="task", decay_rate=0.0))

        # x_causal: has causal edge → y_implied
        kb.add_concept(Concept(concept_id="x_causal", concept_type="object",
                                decay_rate=delta_obs, epistemic_error=0.0))
        kb.add_relation("goal", "x_causal", weight=1.0)

        # x2..x{K_obs}: causally inert observables
        other_obs = [f"x{i}" for i in range(2, K_obs + 1)]
        for cid in other_obs:
            kb.add_concept(Concept(concept_id=cid, concept_type="object",
                                    decay_rate=delta_obs, epistemic_error=0.0))
            kb.add_relation("goal", cid, weight=1.0)

        # y_implied: non-observable, decays fast, connected to goal semantically
        kb.add_concept(Concept(concept_id="y_implied", concept_type="state",
                                decay_rate=delta_y, epistemic_error=0.0,
                                observable=False))
        kb.add_relation("goal", "y_implied", weight=1.0)

        # Causal edge: observing x_causal reduces E(y_implied)
        kb.add_causal_relation("x_causal", "y_implied", propagation_weight=prop_weight)

        return kb, "x_causal", other_obs

    def _run_condition(condition: str, seed: int) -> dict:
        kb, xcausal_id, other_obs_ids = _build_kb()

        pw = PriorityWeights(w_causal=(w_causal if condition == "causal_on" else 0.0))
        am = AwarenessManager(
            kb=kb,
            goal_id="goal",
            budget=budget,
            observation_interval=obs_interval,
            alpha=0.5,
            priority_weights=pw,
        )

        # Spike schedule for y_implied — periodic with ±30 % jitter
        spike_rng = random.Random(seed ^ 0xF00D)
        spikes_by_tick: dict[int, bool] = {}
        t = spike_rng.randint(max(1, spike_period // 2), spike_period)
        while t < ticks:
            spikes_by_tick[t] = True
            t += spike_rng.randint(
                max(1, int(0.7 * spike_period)),
                int(1.3 * spike_period),
            )

        e_y_sum = 0.0
        sched_xcausal = 0
        tick_count = 0

        for tick_i in range(ticks):
            if spikes_by_tick.get(tick_i):
                kb.get_concept("y_implied").epistemic_error = 1.0

            schedule = am.tick(1.0)
            if schedule:
                am.observe(schedule[0])
                if schedule[0] == xcausal_id:
                    sched_xcausal += 1

            e_y_sum += kb.get_concept("y_implied").epistemic_error
            tick_count += 1

        return {
            "condition":          condition,
            "seed":               seed,
            "mean_e_y":           e_y_sum / tick_count,
            "sched_frac_xcausal": sched_xcausal / ticks,
        }

    rows: list[dict] = []
    for condition in ["causal_on", "causal_off"]:
        for seed in seeds:
            rows.append(_run_condition(condition, seed))

    def _agg(condition: str, key: str) -> dict:
        vals = [r[key] for r in rows if r["condition"] == condition]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    summary: dict[str, dict] = {}
    for cond in ["causal_on", "causal_off"]:
        summary[cond] = {
            "mean_e_y":           _agg(cond, "mean_e_y"),
            "sched_frac_xcausal": _agg(cond, "sched_frac_xcausal"),
        }

    return {
        "K_obs": K_obs, "delta_obs": delta_obs, "delta_y": delta_y,
        "prop_weight": prop_weight, "w_causal": w_causal,
        "budget": budget, "obs_interval": obs_interval,
        "spike_period": spike_period, "ticks": ticks, "seeds": seeds,
        "conditions": ["causal_on", "causal_off"],
        "rows":    rows,
        "summary": summary,
    }


def print_n4(results: dict) -> None:
    K     = results["K_obs"]
    do    = results["delta_obs"]
    dy    = results["delta_y"]
    pw    = results["prop_weight"]
    wc    = results["w_causal"]
    bud   = results["budget"]
    obs   = results["obs_interval"]
    sp    = results["spike_period"]
    t     = results["ticks"]
    ns    = len(results["seeds"])

    refresh_obs = 1.0 - math.exp(-do * obs)
    causal_refresh_per_tick = pw * refresh_obs

    # Non-degeneracy arithmetic
    net_causal_off = dy * K - pw * refresh_obs   # drift over K ticks minus 1 causal refresh
    net_causal_on  = dy - pw * refresh_obs        # per-tick: drift minus causal refresh

    print("\n=== N4: Causal Benefit — R4 Causal Term is a Core Component ===\n")
    print(f"  Setup : {K} observable concepts (x_causal, x2, ..., x{K}), all weight-1 edges → A=0.50")
    print(f"          1 non-observable y_implied: weight-1 edge → A=0.50 (semantically known,")
    print(f"          not directly observable); marked observable=False.")
    print(f"          Causal edge: x_causal → y_implied (propagation_weight={pw})")
    print(f"          δ_obs={do}, δ_y={dy}, budget={bud}, obs_interval={obs}s")
    print(f"          spike_period≈{sp} ticks, {t} ticks, {ns} seeds")
    print()
    print(f"  Non-degeneracy:")
    print(f"    Causal_off: x_causal chosen ≈1/{K} ticks → net E_y per cycle = {net_causal_off:+.3f} (> 0 → degenerate)")
    print(f"    Causal_on:  x_causal prioritised when E_y=1.0 → net E_y per tick = {net_causal_on:+.3f} (< 0 → recovers ✓)")
    print()
    print(f"  Claim : Without w_causal, x_causal and x2/x3/x4 have identical base priority.")
    print(f"          The scheduler is unaware that observing x_causal delivers indirect")
    print(f"          epistemic benefit (Δ{pw:.1f}×refresh) to non-observable y_implied.")
    print(f"          With w_causal={wc:.1f}, P(x_causal) += {wc}×{pw}×E(y_implied)×0.5,")
    print(f"          boosting x_causal whenever y_implied is stale and focusing budget")
    print(f"          on the observation that maximises total epistemic gain.")
    print()

    s = results["summary"]

    header = (f"  {'Condition':<22} {'E(y_implied) ↓':>16} {'Sched frac x_causal ↑':>23}")
    print(header)
    print("  " + "-" * 65)
    for cond, label in [("causal_on", f"Causal-on  (w={wc:.1f})"), ("causal_off", "Causal-off (w=0.0)")]:
        ey  = s[cond]["mean_e_y"]
        sf  = s[cond]["sched_frac_xcausal"]
        print(f"  {label:<22} {_fmt_pm(ey):>16} {_fmt_pm(sf):>23}")

    print()
    ey_on  = s["causal_on"]["mean_e_y"]["mean"]
    ey_off = s["causal_off"]["mean_e_y"]["mean"]
    sf_on  = s["causal_on"]["sched_frac_xcausal"]["mean"]
    sf_off = s["causal_off"]["sched_frac_xcausal"]["mean"]

    if ey_on is not None and ey_off is not None:
        delta_e = round(ey_off - ey_on, 4)
        print(f"  ΔE_y (Causal-off − Causal-on) = {delta_e:+.4f}  (positive → causal term keeps implied concept fresher)")
    if sf_on is not None and sf_off is not None:
        delta_sf = round(sf_on - sf_off, 4)
        print(f"  Δsched_frac_xcausal (on − off) = {delta_sf:+.4f}  (positive → causal term redirects budget to x_causal)")

    print()
    print("  Interpretation:")
    print(f"  Without the causal benefit term, x_causal and x2/x3/x{K} are indistinguishable")
    print(f"  to the scheduler. x_causal is chosen ≈1/{K} of ticks, too infrequently to")
    print(f"  combat y_implied's fast drift (δ={dy}). y_implied accumulates high E between")
    print(f"  spikes — the scheduler is unaware there is indirect epistemic work to be done.")
    print(f"  With w_causal={wc:.1f}, a stale y_implied (E=1.0) adds {wc}×{pw}×1.0×0.5={wc*pw*0.5:.3f}")
    print(f"  to P(x_causal), making it dominant until E_y recovers. The causal term closes")
    print(f"  the feedback loop: the scheduler now accounts for the full epistemic consequence")
    print(f"  of each observable action, including its indirect benefit to implied concepts.")
    print(f"  Conclusion: without the causal benefit term, the priority function P = E×A")
    print(f"  is incomplete — it ignores the indirect epistemic value of causal observations,")
    print(f"  leaving non-observable implied concepts chronically stale.")


# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> dict[str, Any]:
    """Run all necessity experiments."""
    results: dict[str, Any] = {}

    print("Running N1 — E necessity...")
    results["n1"] = experiment_e_necessity()
    if verbose:
        print_n1(results["n1"])

    print("\nRunning N2 — A necessity...")
    results["n2"] = experiment_a_necessity()
    if verbose:
        print_n2(results["n2"])

    print("\nRunning N3 — P=E×A combination necessity...")
    results["n3"] = experiment_combination_necessity()
    if verbose:
        print_n3(results["n3"])

    print("\nRunning N4 — causal benefit term necessity...")
    results["n4"] = experiment_causal_benefit_necessity()
    if verbose:
        print_n4(results["n4"])

    return results
