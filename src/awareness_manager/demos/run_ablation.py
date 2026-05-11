"""
run_ablation.py - Standalone entry point for the 5-step component ablation study.

Runs the ablation and optionally generates the report in one shot.

Usage
-----
    python3 src/awareness_manager/demos/run_ablation.py
    python3 src/awareness_manager/demos/run_ablation.py --output experiments/ablation/
    python3 src/awareness_manager/demos/run_ablation.py --report --report-output reports/ablation/
    python3 src/awareness_manager/demos/run_ablation.py --no-resume

Ablation steps
--------------
    Step 0  Reactive           - 1-hop cutoff, round-robin (no spreading activation)
    Step 1  AM [F1]            - Graded spreading activation only
    Step 2  AM [F1+F2]         - + Anticipatory horizon (pre-tuning for future goals)
    Step 3  AM [F1+F2+F5]      - + Epistemic priority scheduling (E x A)
    Step 4  AM [full]          - + Quadratic depth constraint (full Awareness Manager)

Each step adds exactly one component over the previous.  F3 (utility saturation)
is always active for all configurations — it is shared with both baselines and is
not an AM-specific component.

Equivalent CLI:
    python3 -m awareness_manager.evaluation ablation --output experiments/ablation/
    python3 -m awareness_manager.evaluation report \\
        --experiment experiments/ablation/ --output reports/ablation/
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 5-step AM component ablation study.",
    )
    parser.add_argument(
        "--scenario", default="pv_inspection",
        help="Scenario name (default: pv_inspection)",
    )
    parser.add_argument(
        "--budget", type=int, default=2,
        help="Refresh budget per tick (default: 2)",
    )
    parser.add_argument(
        "--obs-interval", type=float, default=10.0,
        help="Observation interval T in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--output", default="experiments/ablation_study/",
        help="Output directory for traces and manifest (default: experiments/ablation_study/)",
    )
    parser.add_argument(
        "--duration", type=float, default=70.0,
        help="Simulated duration per run in seconds (default: 70.0)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Simulation timestep in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-run completed runs instead of skipping them",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Also generate the report after running",
    )
    parser.add_argument(
        "--report-output", default=None,
        help="Output directory for the report (default: <output>/../reports/<experiment_name>/)",
    )
    args = parser.parse_args()

    from awareness_manager.evaluation.batch import run_ablation_experiment

    manifest_path = run_ablation_experiment(
        scenario=args.scenario,
        budget=args.budget,
        obs_interval=args.obs_interval,
        output_dir=Path(args.output),
        duration_s=args.duration,
        dt=args.dt,
        resume=not args.no_resume,
    )

    if args.report:
        from awareness_manager.evaluation.report import generate_report
        experiment_dir = manifest_path.parent
        if args.report_output:
            report_dir = Path(args.report_output)
        else:
            report_dir = experiment_dir.parent / "reports" / experiment_dir.name
        print(f"\nGenerating report → {report_dir}/")
        generate_report(experiment_dir, report_dir)


if __name__ == "__main__":
    main()
