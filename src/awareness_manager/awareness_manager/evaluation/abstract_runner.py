"""
abstract_runner.py — Abstract N-variable simulation for Telogenesis comparison.

Runs four attention strategies on the abstract N-variable scenario
(see scenarios/abstract_n.py) and returns per-run detection latency statistics.

Strategies
----------
PRIORITY-AM       Full AwarenessManager with goal-conditioning (F1 spreading
                  activation + F5 epistemic priority). Comparable to Telogenesis
                  Experiment 1 with goal-conditioned relevance weighting.

PRIORITY-EPISTEMIC Pure epistemic priority: sort all N concepts by E, take top-b.
                  No goal-conditioning. Direct analog of Telogenesis priority
                  strategy (Ignorance + Staleness without mission relevance).

ROTATION          Round-robin over all N schedulable concepts (budget-capped).
                  Matches ReactiveBaseline behaviour in the abstract setting.

RANDOM            Uniform random selection from all N schedulable concepts.

Volatility mechanism
--------------------
Every `volatility_period` ticks (staggered by concept index) a volatile concept
switches regime: its epistemic_error and prediction_error are set to 1.0.
Detection = first tick where the concept appears in the returned schedule.
Detection latency = (detection_tick - spike_tick).

Usage
-----
    from awareness_manager.evaluation.abstract_runner import run_abstract_experiment

    results = run_abstract_experiment(
        N=24, K=3, R=6, budget=1, ticks=500, volatility_period=50, seed=42
    )
    # results: dict keyed by strategy name → {"mean_latency_ticks": ..., ...}
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import Concept
from awareness_manager.feature_config import FeatureConfig, PriorityWeights
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.abstract_n import build_abstract_kb

StrategyName = Literal[
    "PRIORITY-AM", "PRIORITY-EPISTEMIC", "REACTIVE", "ROTATION", "RANDOM"
]

_DT: float = 1.0  # tick size (seconds) — latency in ticks equals latency in seconds


@dataclass
class _VolatilityEvent:
    concept_id: str
    spike_tick: int
    detection_tick: int | None = None

    @property
    def detected(self) -> bool:
        return self.detection_tick is not None

    @property
    def latency_ticks(self) -> int | None:
        if self.detection_tick is None:
            return None
        return self.detection_tick - self.spike_tick


# ---------------------------------------------------------------------------
# Lightweight strategy implementations
# ---------------------------------------------------------------------------

def _f3_refresh(decay_rate: float, obs_interval: float) -> float:
    """Formula 3: refresh = 1 - exp(-δ × T)."""
    return 1.0 - math.exp(-decay_rate * obs_interval)


class _EpistemicStrategy:
    """PRIORITY-EPISTEMIC: sort all N concepts by E, take top-b."""

    def __init__(self, kb: KnowledgeBase, budget: int) -> None:
        self._kb = kb
        self._budget = budget
        self._ids = [c for c in kb.concept_ids() if kb.get_concept(c).decay_rate > 0]

    def tick(self, dt: float) -> list[str]:
        self._kb.tick(dt)
        ranked = sorted(self._ids, key=lambda c: self._kb.get_concept(c).epistemic_error, reverse=True)
        return ranked[: self._budget]

    def observe(self, cid: str, obs_interval: float) -> None:
        decay_rate = self._kb.get_concept(cid).decay_rate
        self._kb.refresh_concept(cid, _f3_refresh(decay_rate, obs_interval))


class _ReactiveStrategy:
    """REACTIVE: round-robin over the goal's 1-hop neighbourhood only (budget-capped).

    Approximates ReactiveBaseline: goal-aware scheduling without epistemic priority.
    Shows that goal-awareness alone beats blind rotation, while PRIORITY-AM adds
    epistemic prioritisation on top.
    """

    def __init__(self, kb: KnowledgeBase, goal_id: str, budget: int) -> None:
        self._kb = kb
        self._budget = budget
        relevant: list[str] = []
        for a, b, _ in kb.semantic_edges():
            if a == goal_id and kb.get_concept(b).decay_rate > 0:
                relevant.append(b)
            elif b == goal_id and kb.get_concept(a).decay_rate > 0:
                relevant.append(a)
        self._ids = relevant
        self._idx = 0

    def tick(self, dt: float) -> list[str]:
        self._kb.tick(dt)
        n = len(self._ids)
        if n == 0:
            return []
        schedule = []
        for _ in range(self._budget):
            schedule.append(self._ids[self._idx % n])
            self._idx += 1
        return schedule

    def observe(self, cid: str, obs_interval: float) -> None:
        decay_rate = self._kb.get_concept(cid).decay_rate
        self._kb.refresh_concept(cid, _f3_refresh(decay_rate, obs_interval))


class _RotationStrategy:
    """ROTATION: round-robin over all N schedulable concepts."""

    def __init__(self, kb: KnowledgeBase, budget: int) -> None:
        self._kb = kb
        self._budget = budget
        self._ids = [c for c in kb.concept_ids() if kb.get_concept(c).decay_rate > 0]
        self._idx = 0

    def tick(self, dt: float) -> list[str]:
        self._kb.tick(dt)
        schedule = []
        n = len(self._ids)
        for _ in range(self._budget):
            schedule.append(self._ids[self._idx % n])
            self._idx += 1
        return schedule

    def observe(self, cid: str, obs_interval: float) -> None:
        decay_rate = self._kb.get_concept(cid).decay_rate
        self._kb.refresh_concept(cid, _f3_refresh(decay_rate, obs_interval))


class _RandomStrategy:
    """RANDOM: uniform random selection."""

    def __init__(self, kb: KnowledgeBase, budget: int, seed: int = 0) -> None:
        self._kb = kb
        self._budget = budget
        self._ids = [c for c in kb.concept_ids() if kb.get_concept(c).decay_rate > 0]
        self._rng = random.Random(seed)

    def tick(self, dt: float) -> list[str]:
        self._kb.tick(dt)
        return self._rng.sample(self._ids, min(self._budget, len(self._ids)))

    def observe(self, cid: str, obs_interval: float) -> None:
        decay_rate = self._kb.get_concept(cid).decay_rate
        self._kb.refresh_concept(cid, _f3_refresh(decay_rate, obs_interval))


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _run_single(
    strategy_name: StrategyName,
    N: int,
    K: int,
    R: int,
    K_overlap: int,
    budget: int,
    ticks: int,
    volatility_period: int,
    obs_interval: float,
    seed: int,
    w_surprise: float = 0.0,
    feature_config=None,  # FeatureConfig | None — only used for PRIORITY-AM
    priority_weights=None,  # PriorityWeights | None — only used for PRIORITY-AM
) -> dict:
    """
    Run one abstract scenario with the given strategy.

    Returns a dict with detection latency statistics and raw event list.
    """
    kb, volatile_ids, relevant_ids = build_abstract_kb(
        N=N, K=K, R=R, K_overlap=K_overlap, seed=seed
    )

    # Build strategy
    if strategy_name == "PRIORITY-AM":
        fc = feature_config if feature_config is not None else FeatureConfig()
        pw = priority_weights if priority_weights is not None else PriorityWeights(w_surprise=w_surprise)
        strategy = AwarenessManager(
            kb=kb,
            goal_id="goal",
            budget=budget,
            observation_interval=obs_interval,
            alpha=0.5,
            feature_config=fc,
            priority_weights=pw,
        )
    elif strategy_name == "PRIORITY-EPISTEMIC":
        strategy = _EpistemicStrategy(kb, budget)
    elif strategy_name == "REACTIVE":
        strategy = _ReactiveStrategy(kb, "goal", budget)
    elif strategy_name == "ROTATION":
        strategy = _RotationStrategy(kb, budget)
    elif strategy_name == "RANDOM":
        strategy = _RandomStrategy(kb, budget, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Build volatility schedule: volatile concepts spike every ~volatility_period ticks,
    # staggered and jittered so each seed produces a distinct schedule.
    # spike_rng is seeded independently from the KB-construction seed so the two sources
    # of randomness don't alias each other.
    spike_rng = random.Random(seed ^ 0xDEAD)
    stagger = max(1, volatility_period // max(K, 1))
    spike_times: list[tuple[str, int]] = []
    for idx, cid in enumerate(volatile_ids):
        offset = stagger * idx
        jitter = max(1, stagger // 2)
        t = max(0, offset + spike_rng.randint(-jitter, jitter))
        while t < ticks:
            spike_times.append((cid, t))
            t += spike_rng.randint(
                max(1, int(0.7 * volatility_period)),
                int(1.3 * volatility_period),
            )

    # Index spike times by tick
    spikes_by_tick: dict[int, list[str]] = {}
    for cid, t in spike_times:
        spikes_by_tick.setdefault(t, []).append(cid)

    # Active events awaiting detection: concept_id → _VolatilityEvent
    pending: dict[str, _VolatilityEvent] = {}
    completed_events: list[_VolatilityEvent] = []

    for tick_i in range(ticks):
        # Inject volatility spikes before the strategy tick
        for cid in spikes_by_tick.get(tick_i, []):
            concept = kb.get_concept(cid)
            concept.epistemic_error = 1.0
            concept.prediction_error = 1.0
            # Overwrite any unresolved pending event (new spike before detection)
            if cid in pending:
                completed_events.append(pending[cid])  # undetected
            pending[cid] = _VolatilityEvent(concept_id=cid, spike_tick=tick_i)

        schedule = strategy.tick(_DT)

        # Observe top concept
        if schedule:
            top = schedule[0]
            if strategy_name == "PRIORITY-AM":
                strategy.observe(top)
            else:
                strategy.observe(top, obs_interval)

        # Check detections
        for cid in list(pending):
            if cid in schedule:
                pending[cid].detection_tick = tick_i
                completed_events.append(pending.pop(cid))

    # Remaining undetected
    for ev in pending.values():
        completed_events.append(ev)

    # Aggregate
    detected = [ev for ev in completed_events if ev.detected]
    latencies = [ev.latency_ticks for ev in detected if ev.latency_ticks is not None]

    return {
        "strategy": strategy_name,
        "N": N, "K": K, "R": R, "K_overlap": K_overlap, "budget": budget,
        "ticks": ticks,
        "total_events": len(completed_events),
        "detected_events": len(detected),
        "mean_latency_ticks": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_ticks": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "detection_rate": len(detected) / len(completed_events) if completed_events else None,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_abstract_experiment(
    N: int = 24,
    K: int = 3,
    R: int = 6,
    K_overlap: int | None = None,
    budget: int = 1,
    ticks: int = 500,
    volatility_period: int = 50,
    obs_interval: float = 1.0,
    seed: int = 42,
    strategies: list[StrategyName] | None = None,
    w_surprise: float = 0.0,
    priority_weights=None,  # PriorityWeights | None — forwarded to PRIORITY-AM only
) -> dict[str, dict]:
    """
    Run all four strategies on the abstract N-variable scenario.

    Args:
        N:                  Total schedulable concepts.
        K:                  Volatile concepts (periodically spiked).
        R:                  Mission-relevant concepts (1-hop from goal).
        K_overlap:          Volatile concepts that are also relevant (default=min(K,R)).
        budget:             Attention budget (concepts per tick).
        ticks:              Simulation length in ticks.
        volatility_period:  Ticks between successive spikes of each volatile concept.
        obs_interval:       Observation interval (T) for Formula 3.
        seed:               Random seed.
        strategies:         Which strategies to run (default: all four).
        w_surprise:         w_surprise weight for PRIORITY-AM (0=off, 1=on).
        priority_weights:   PriorityWeights override for PRIORITY-AM (takes precedence over
                            w_surprise when provided).

    Returns:
        Dict keyed by strategy name → result dict with detection latency stats.
    """
    if K_overlap is None:
        K_overlap = min(K, R)

    if strategies is None:
        strategies = ["PRIORITY-AM", "PRIORITY-EPISTEMIC", "REACTIVE", "ROTATION", "RANDOM"]

    results = {}
    for name in strategies:
        results[name] = _run_single(
            strategy_name=name,
            N=N, K=K, R=R, K_overlap=K_overlap,
            budget=budget, ticks=ticks,
            volatility_period=volatility_period,
            obs_interval=obs_interval,
            seed=seed,
            w_surprise=w_surprise,
            priority_weights=priority_weights,
        )
    return results


def learnable_decay_verification(
    N_stable: int = 8,
    N_volatile: int = 8,
    N_start_high: int = 4,
    budget: int = 2,
    ticks: int = 500,
    volatility_period: int = 30,
    obs_interval: float = 20.0,
    tau_decay: float = 0.05,
    seed: int = 42,
    seeds: list[int] | None = None,
) -> dict:
    """
    Verification experiment for learnable δ (Telogenesis Experiment 3 analog).

    Three groups of concepts, all observed by the AM:
      - volatile:        periodically spiked → δ should rise
      - baseline_stable: never spiked, start at δ=0.1 → δ should stay low
      - start_high_stable: never spiked, start at δ=0.5 → δ should decrease

    The start_high_stable group demonstrates bidirectional adaptation: the
    mechanism self-organises toward the true volatility level whether it starts
    too high or too low.

    When `seeds` is provided (default: [42,43,44,45,46]) the experiment runs once
    per seed and reports mean ± std of the volatile/stable δ ratio.

    Returns a dict with:
        "volatile_delta_mean":      mean δ of volatile group (last seed, or mean across seeds)
        "baseline_stable_delta_mean": mean δ of baseline stable group
        "start_high_stable_delta_mean": mean δ of start-high group (should be < 0.5 at end)
        "volatile_delta_all":       list of final δ for volatile (last seed)
        "baseline_stable_delta_all":list of final δ for baseline stable (last seed)
        "start_high_stable_delta_all": list of final δ for start-high group (last seed)
        "ticks":                    simulation length
        "u_stat":                   Mann-Whitney U (volatile vs baseline_stable)
        "p_value":                  one-sided p-value
        "ratio_mean":               mean of (volatile_mean / stable_mean) across seeds
        "ratio_std":                std of that ratio
        "per_seed":                 list of per-seed result dicts
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    per_seed_results: list[dict] = []

    for run_seed in seeds:
        N = N_volatile + N_stable
        K = N_volatile
        R = N  # all concepts are relevant so AM schedules them

        kb, volatile_ids, _ = build_abstract_kb(
            N=N, K=K, R=R, K_overlap=K, seed=run_seed
        )

        all_stable = [cid for cid in kb.concept_ids()
                      if cid != "goal" and cid not in volatile_ids]
        # First N_start_high stable concepts form the start-high group.
        start_high_ids = all_stable[:N_start_high]
        baseline_stable_ids = all_stable[N_start_high:]

        # Give start-high-stable concepts an inflated initial δ so we can observe
        # the mechanism pulling it back down toward the true (low) volatility level.
        for cid in start_high_ids:
            kb.get_concept(cid).decay_rate = 0.5

        fc = FeatureConfig(use_learnable_decay=True)
        strategy = AwarenessManager(
            kb=kb,
            goal_id="goal",
            budget=budget,
            observation_interval=obs_interval,
            alpha=0.5,
            feature_config=fc,
            priority_weights=PriorityWeights(w_surprise=1.0),
            tau_decay=tau_decay,
        )

        spike_rng = random.Random(run_seed ^ 0xDEAD)
        stagger = max(1, volatility_period // max(K, 1))
        spikes_by_tick: dict[int, list[str]] = {}
        for idx, cid in enumerate(volatile_ids):
            offset = stagger * idx
            jitter = max(1, stagger // 2)
            t = max(0, offset + spike_rng.randint(-jitter, jitter))
            while t < ticks:
                spikes_by_tick.setdefault(t, []).append(cid)
                t += spike_rng.randint(
                    max(1, int(0.7 * volatility_period)),
                    int(1.3 * volatility_period),
                )

        for tick_i in range(ticks):
            for cid in spikes_by_tick.get(tick_i, []):
                concept = kb.get_concept(cid)
                concept.epistemic_error = 1.0
                concept.prediction_error = 1.0
            schedule = strategy.tick(_DT)
            for cid in schedule:
                strategy.observe(cid)

        v_deltas  = [kb.get_concept(c).decay_rate for c in volatile_ids]
        bs_deltas = [kb.get_concept(c).decay_rate for c in baseline_stable_ids]
        sh_deltas = [kb.get_concept(c).decay_rate for c in start_high_ids]

        v_mean  = sum(v_deltas)  / len(v_deltas)  if v_deltas  else float("nan")
        bs_mean = sum(bs_deltas) / len(bs_deltas) if bs_deltas else float("nan")
        ratio   = v_mean / bs_mean if bs_mean > 0 else float("nan")

        per_seed_results.append({
            "seed": run_seed,
            "volatile_delta_mean": v_mean,
            "baseline_stable_delta_mean": bs_mean,
            "start_high_stable_delta_mean": sum(sh_deltas) / len(sh_deltas) if sh_deltas else float("nan"),
            "ratio": ratio,
            "volatile_delta_all": v_deltas,
            "baseline_stable_delta_all": bs_deltas,
            "start_high_stable_delta_all": sh_deltas,
        })

    # Aggregate across seeds
    ratios = [s["ratio"] for s in per_seed_results if not math.isnan(s["ratio"])]
    ratio_mean = sum(ratios) / len(ratios) if ratios else float("nan")
    ratio_std  = (
        math.sqrt(sum((r - ratio_mean) ** 2 for r in ratios) / (len(ratios) - 1))
        if len(ratios) > 1 else 0.0
    )

    # Use last seed's data for per-concept lists and Mann-Whitney U
    last = per_seed_results[-1]
    from scipy import stats as _stats
    u_stat, p_value = _stats.mannwhitneyu(
        last["volatile_delta_all"], last["baseline_stable_delta_all"],
        alternative="greater",
    )

    return {
        "volatile_delta_mean":             last["volatile_delta_mean"],
        "baseline_stable_delta_mean":      last["baseline_stable_delta_mean"],
        "start_high_stable_delta_mean":    last["start_high_stable_delta_mean"],
        "volatile_delta_all":              last["volatile_delta_all"],
        "baseline_stable_delta_all":       last["baseline_stable_delta_all"],
        "start_high_stable_delta_all":     last["start_high_stable_delta_all"],
        "ticks":                           ticks,
        "u_stat":                          u_stat,
        "p_value":                         p_value,
        "ratio_mean":                      ratio_mean,
        "ratio_std":                       ratio_std,
        "per_seed":                        per_seed_results,
        # Legacy keys for backward compatibility
        "stable_delta_mean":   last["baseline_stable_delta_mean"],
        "stable_delta_all":    last["baseline_stable_delta_all"],
    }


