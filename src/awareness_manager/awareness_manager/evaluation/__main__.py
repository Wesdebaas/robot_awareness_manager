"""
CLI entry point for the evaluation package.

    python -m awareness_manager.evaluation batch --help
    python -m awareness_manager.evaluation report --help

Examples
--------
Run the standard budget x observation_interval sweep:

    python -m awareness_manager.evaluation batch \\
        --scenario pv_inspection \\
        --strategies awareness_manager,reactive,always_on \\
        --budget 1,2,4 \\
        --obs-interval 1.0,5.0,10.0 \\
        --output experiments/budget_obsint_sweep/

Generate the report:

    python -m awareness_manager.evaluation report \\
        --experiment experiments/budget_obsint_sweep/ \\
        --output reports/budget_obsint_sweep/

Alpha sweep (secondary experiment):

    python -m awareness_manager.evaluation batch \\
        --scenario pv_inspection \\
        --strategies awareness_manager \\
        --budget 2 \\
        --obs-interval 10.0 \\
        --alpha 0.3,0.5,0.7 \\
        --output experiments/alpha_sweep/
"""

import argparse
import sys
from pathlib import Path


def _cmd_batch(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.batch import run_experiment

    strategies = [s.strip() for s in args.strategies.split(",")]
    budgets = [int(b) for b in args.budget.split(",")]
    obs_intervals = [float(t) for t in args.obs_interval.split(",")]
    alphas = [float(a) for a in args.alpha.split(",")] if args.alpha else [0.5]

    # Build param_grid as cross-product of budget x obs_interval x alpha
    param_grid = [
        {"budget": b, "observation_interval": oi, "alpha": a}
        for b in budgets
        for oi in obs_intervals
        for a in alphas
    ]
    # Deduplicate if alpha is a single value (avoids confusing run IDs)
    if len(alphas) == 1:
        for p in param_grid:
            del p["alpha"]

    print(f"Scenario:   {args.scenario}")
    print(f"Strategies: {strategies}")
    print(f"Param grid: {len(param_grid)} combinations")
    print(f"Total runs: {len(strategies) * len(param_grid)}")
    print(f"Output:     {args.output}")
    print()

    run_experiment(
        scenario=args.scenario,
        strategies=strategies,
        param_grid=param_grid,
        output_dir=Path(args.output),
        duration_s=args.duration,
        dt=args.dt,
        resume=not args.no_resume,
    )


def _cmd_report(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.report import generate_report
    generate_report(Path(args.experiment), Path(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m awareness_manager.evaluation",
        description="Awareness Manager - batch experiment and report CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── batch ─────────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Run a batch experiment")
    p_batch.add_argument("--scenario", default="pv_inspection",
                         help="Scenario name (default: pv_inspection)")
    p_batch.add_argument("--strategies",
                         default="awareness_manager,reactive,always_on",
                         help="Comma-separated strategy names")
    p_batch.add_argument("--budget", default="1,2,4",
                         help="Comma-separated budget values (default: 1,2,4)")
    p_batch.add_argument("--obs-interval", default="1.0,5.0,10.0",
                         help="Comma-separated observation intervals (default: 1.0,5.0,10.0)")
    p_batch.add_argument("--alpha", default=None,
                         help="Comma-separated alpha values for AM (default: 0.5, fixed)")
    p_batch.add_argument("--output", default="experiments/batch/",
                         help="Output directory")
    p_batch.add_argument("--duration", type=float, default=70.0,
                         help="Simulated duration per run in seconds (default: 70)")
    p_batch.add_argument("--dt", type=float, default=0.1,
                         help="Simulation timestep in seconds (default: 0.1)")
    p_batch.add_argument("--no-resume", action="store_true",
                         help="Re-run completed runs instead of skipping them")
    p_batch.set_defaults(func=_cmd_batch)

    # ── report ────────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Generate report from experiment")
    p_report.add_argument("--experiment", required=True,
                          help="Experiment directory containing manifest.json")
    p_report.add_argument("--output", required=True,
                          help="Output directory for report files")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
