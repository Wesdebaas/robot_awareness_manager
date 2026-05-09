"""
batch.py - Automated multi-run experiment driver.

Runs (strategy, param) combinations for a named scenario, writing one trace
directory per run and a manifest.json summarising the experiment.

Usage
-----
    from awareness_manager.evaluation.batch import run_experiment
    from pathlib import Path

    run_experiment(
        scenario="pv_inspection",
        strategies=["awareness_manager", "reactive", "always_on"],
        param_grid=[
            {"budget": b, "observation_interval": oi}
            for b in [1, 2, 4]
            for oi in [1.0, 5.0, 10.0]
        ],
        output_dir=Path("experiments/budget_obsint_sweep"),
        duration_s=70.0,
    )

Or via CLI:
    python -m awareness_manager.evaluation batch \\
        --scenario pv_inspection \\
        --strategies awareness_manager,reactive,always_on \\
        --budget 1,2,4 \\
        --obs-interval 1.0,5.0,10.0 \\
        --output experiments/budget_obsint_sweep/

param_grid notes
----------------
budget         - effective for awareness_manager and reactive; ignored by
                 always_on (which has no budget cap).  always_on runs are
                 still generated for every param_grid entry so the manifest
                 is uniform, but their results will only vary with
                 observation_interval (which affects Formula 3 refresh amount).

observation_interval - effective for all strategies (governs Formula 3:
                 refresh = 1 - exp(-δ x T)).  Changing this is the primary
                 way to vary the "information freshness" regime.

alpha          - AwarenessManager-only (spreading-activation decay).
                 Ignored by baselines.  Defaults to 0.5 if not in param entry.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIO_BUILDERS: dict[str, str] = {
    "pv_inspection": "awareness_manager.scenarios.pv_inspection",
}

_SCENARIO_GOAL: dict[str, str] = {
    "pv_inspection": "inspect_pv_field",
}

_SCENARIO_QUEUED_GOALS: dict[str, list[tuple[str, float, str]]] = {
    # (goal_id, eta_s, level)
    "pv_inspection": [("emergency_landing", 30.0, "global")],
}


def _load_scenario(scenario: str):
    """Return (kb_builder, ikb_builder) callables for a named scenario."""
    if scenario not in _SCENARIO_BUILDERS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {sorted(_SCENARIO_BUILDERS)}"
        )
    mod_name = _SCENARIO_BUILDERS[scenario]
    import importlib
    mod = importlib.import_module(mod_name)
    # Conventions: build_<scenario>_kb() and build_<scenario>_instance_kb()
    scenario_snake = scenario  # already snake_case
    kb_fn = getattr(mod, f"build_{scenario_snake}_kb")
    try:
        ikb_fn = getattr(mod, f"build_{scenario_snake}_instance_kb")
    except AttributeError:
        ikb_fn = lambda: None  # noqa: E731
    return kb_fn, ikb_fn


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def _build_strategy(
    strategy_name: str,
    kb,
    ikb,
    goal_id: str,
    params: dict,
):
    """Instantiate a strategy object from a name and param dict."""
    budget = params.get("budget", 2)
    obs_interval = params.get("observation_interval", 10.0)
    alpha = params.get("alpha", 0.5)

    if strategy_name == "awareness_manager":
        from awareness_manager.awareness_manager import AwarenessManager
        return AwarenessManager(
            kb,
            goal_id=goal_id,
            budget=budget,
            observation_interval=obs_interval,
            lambda_horizon=params.get("lambda_horizon", 0.1),
            instance_kb=ikb,
            instance_relational_weight=params.get("instance_relational_weight", 0.3),
            alpha=alpha,
        ), True  # observe_top=True

    if strategy_name == "reactive":
        from awareness_manager.baselines.reactive import ReactiveBaseline
        return ReactiveBaseline(
            kb,
            goal_id=goal_id,
            budget=budget,
            observation_interval=obs_interval,
            ikb=ikb,
            hop_distance=params.get("hop_distance", 1),
        ), True  # observe_top=True - runner must call observe(); round-robin is selection only

    if strategy_name == "always_on":
        from awareness_manager.baselines.always_on import AlwaysOnBaseline
        return AlwaysOnBaseline(
            kb,
            goal_id=goal_id,
            observation_interval=obs_interval,
            ikb=ikb,
        ), False  # observe_top=False - AlwaysOn self-observes inside tick()

    raise ValueError(
        f"Unknown strategy '{strategy_name}'. "
        "Available: awareness_manager, reactive, always_on"
    )


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------

def _run_single(
    scenario: str,
    strategy_name: str,
    params: dict,
    run_dir: Path,
    duration_s: float,
    dt: float,
) -> dict:
    """
    Execute one (strategy, params) run and write a trace to run_dir.

    Returns a run-record dict suitable for inclusion in manifest.json.
    """
    from awareness_manager.visualization.snapshot import compute_positions
    from awareness_manager.visualization.trace_logger import TraceLogger

    kb_fn, ikb_fn = _load_scenario(scenario)
    kb = kb_fn()
    ikb = ikb_fn()

    goal_id = _SCENARIO_GOAL[scenario]
    strategy, observe_top = _build_strategy(strategy_name, kb, ikb, goal_id, params)

    for gid, eta, level in _SCENARIO_QUEUED_GOALS.get(scenario, []):
        strategy.queue_goal(gid, eta=eta, level=level)

    positions = compute_positions(strategy)
    n_ticks = int(round(duration_s / dt))
    sample_rate = params.get("sample_rate", 1)

    logger = TraceLogger(
        strategy,
        scenario=scenario,
        output_dir=run_dir,
        dt=dt,
        sample_rate=sample_rate,
        history_maxlen=n_ticks,
    )
    logger.open(positions=positions)

    t0 = time.perf_counter()
    for i in range(n_ticks):
        schedule = strategy.tick(dt)
        if observe_top and schedule:
            strategy.observe(schedule[0])
        logger.record((i + 1) * dt, schedule)
    elapsed_wall = time.perf_counter() - t0

    logger.close()

    return {
        "strategy": strategy_name,
        "params": params,
        "trace_dir": str(run_dir),
        "n_ticks": n_ticks,
        "duration_s": duration_s,
        "wall_seconds": round(elapsed_wall, 3),
        "completed": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    scenario: str,
    strategies: list[str],
    param_grid: list[dict],
    output_dir: str | Path,
    duration_s: float = 70.0,
    dt: float = 0.1,
    resume: bool = True,
) -> Path:
    """
    Run the full experiment: every (strategy, params) combination.

    Args:
        scenario:    Name of the scenario to run (must be in _SCENARIO_BUILDERS).
        strategies:  List of strategy names.
        param_grid:  List of param dicts.  Each entry is one configuration;
                     the cross-product of strategies x param_grid is enumerated.
        output_dir:  Root output directory.  Each run goes into a subdirectory
                     named <strategy>__<param_slug>/
        duration_s:  Simulated duration per run in seconds.
        dt:          Simulation timestep in seconds.
        resume:      If True, skip runs whose trace_dir already contains a
                     valid meta.json (allows interrupted experiments to continue).

    Returns:
        Path to the written manifest.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    total = len(strategies) * len(param_grid)
    done = 0

    for strategy_name in strategies:
        for params in param_grid:
            done += 1
            slug = _param_slug(params)
            run_dir = output_dir / f"{strategy_name}__{slug}"

            # Resume: skip if already completed
            if resume and (run_dir / "meta.json").exists():
                print(f"[{done:3d}/{total}] SKIP (exists) {run_dir.name}")
                runs.append({
                    "strategy": strategy_name,
                    "params": params,
                    "trace_dir": str(run_dir),
                    "completed": True,
                    "skipped": True,
                })
                continue

            print(f"[{done:3d}/{total}] {strategy_name}  {params}", end="  ", flush=True)
            try:
                record = _run_single(
                    scenario, strategy_name, params, run_dir,
                    duration_s=duration_s, dt=dt,
                )
                print(f"✓  ({record['wall_seconds']:.1f}s wall)")
            except Exception as exc:
                print(f"✗  {exc}")
                record = {
                    "strategy": strategy_name,
                    "params": params,
                    "trace_dir": str(run_dir),
                    "completed": False,
                    "error": str(exc),
                }
            runs.append(record)

    manifest = {
        "scenario": scenario,
        "experiment_dir": str(output_dir),
        "strategies": strategies,
        "param_grid": param_grid,
        "duration_s": duration_s,
        "dt": dt,
        "total_runs": total,
        "completed_runs": sum(1 for r in runs if r.get("completed")),
        "runs": runs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest → {manifest_path}  ({manifest['completed_runs']}/{total} runs)")
    return manifest_path


def _param_slug(params: dict) -> str:
    """
    Short filesystem-safe identifier for a param dict.

    Example: {"budget": 2, "observation_interval": 5.0} → "b2_oi5.0"
    """
    parts = []
    if "budget" in params:
        parts.append(f"b{params['budget']}")
    if "observation_interval" in params:
        oi = params["observation_interval"]
        parts.append(f"oi{oi:g}")
    if "alpha" in params:
        parts.append(f"a{params['alpha']}")
    # Fall back to sorted key=value pairs for any remaining keys
    extras = {
        k: v for k, v in params.items()
        if k not in ("budget", "observation_interval", "alpha", "sample_rate")
    }
    for k, v in sorted(extras.items()):
        parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "default"
