"""
CLI entry point for the evaluation package.

Run all six experiments:

    python -m awareness_manager.evaluation experiments

Run individual experiments:

    python -m awareness_manager.evaluation exp1              # scaling (Telogenesis replication)
    python -m awareness_manager.evaluation exp2              # goal conditioning (E×A)
    python -m awareness_manager.evaluation exp3              # anticipatory horizon (F2 on/off, ETA sweep)
    python -m awareness_manager.evaluation exp4              # instance KB vs class-only (multi-instance)
    python -m awareness_manager.evaluation exp5              # formula ablation (F1 spreading activation)
    python -m awareness_manager.evaluation exp6              # F6 spatial opportunity cost

Run the abstract N-variable simulation directly (quick smoke test):

    python -m awareness_manager.evaluation abstract --quick

Learnable decay verification:

    python -m awareness_manager.evaluation learnable-decay
"""

import argparse
import sys
from pathlib import Path


def _cmd_experiments(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import run_all
    run_all(verbose=True)


def _cmd_exp1(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_scaling, print_exp1
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_scaling(seeds=seeds)
    print_exp1(results)


def _cmd_exp2(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_goal_conditioning, print_exp2
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_goal_conditioning(seeds=seeds)
    print_exp2(results)


def _cmd_exp3(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_anticipatory_horizon, print_exp3
    results = experiment_anticipatory_horizon(
        budget_values=args.budgets,
        eta_values=args.etas,
        obs_interval=args.obs_interval,
    )
    print_exp3(results)


def _cmd_exp4(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_instance_kb, print_exp4
    instances = args.instances if args.instances else None
    results = experiment_instance_kb(
        instances=instances,
        budget=args.budget,
        obs_interval=args.obs_interval,
    )
    print_exp4(results)


def _cmd_exp5(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_formula_ablation, print_exp_ablation
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_formula_ablation(
        seeds=seeds,
        obs_interval=args.obs_interval,
    )
    print_exp_ablation(results)


def _cmd_exp6(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.experiments import experiment_f6_spatial_cost, print_exp6
    results = experiment_f6_spatial_cost(
        ticks=args.ticks,
        budget=args.budget,
        obs_interval=args.obs_interval,
    )
    print_exp6(results)


def _cmd_abstract(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.abstract_runner import run_abstract_experiment
    if args.quick:
        N, K, R, ticks, seeds = 12, 2, 4, 100, [42]
    else:
        N, K, R, ticks, seeds = 24, 3, 6, 500, [42, 43, 44]
    for seed in seeds:
        results = run_abstract_experiment(
            N=N, K=K, R=R, K_overlap=K,
            budget=args.budget,
            ticks=ticks,
            obs_interval=args.obs_interval,
            seed=seed,
        )
        print(f"\n--- seed={seed}, N={N}, K={K}, R={R}, budget={args.budget} ---")
        for strat, d in results.items():
            lat = d["mean_latency_ticks"]
            rate = d["detection_rate"]
            print(f"  {strat:<22}  latency={lat:.1f} ticks   detection_rate={rate:.2f}")


def _cmd_learnable_decay(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.abstract_runner import learnable_decay_verification
    result = learnable_decay_verification(
        N_volatile=8,
        N_stable=8,
        N_start_high=4,
        budget=2,
        ticks=500,
        tau_decay=0.05,
        seeds=[42, 43, 44, 45, 46],
    )
    print("\n=== Learnable Decay Verification (5 seeds) ===")
    print()
    print(f"  Group                    Mean δ (last seed)")
    print(f"  {'Volatile (spiked)':<30} {result['volatile_delta_mean']:.4f}")
    print(f"  {'Baseline stable (δ=0.1 start)':<30} {result['baseline_stable_delta_mean']:.4f}")
    print(f"  {'Start-high stable (δ=0.5 start)':<30} {result['start_high_stable_delta_mean']:.4f}")
    print()
    ratio_last = result["volatile_delta_mean"] / max(result["baseline_stable_delta_mean"], 1e-9)
    print(f"  Volatile / baseline_stable ratio (last seed): {ratio_last:.2f}×")
    print(f"  Ratio across {len(result['per_seed'])} seeds: {result['ratio_mean']:.2f} ± {result['ratio_std']:.2f}")
    print(f"  Mann-Whitney U (volatile > baseline_stable): U={result['u_stat']:.0f}, p={result['p_value']:.2e}")
    print()
    if ratio_last > 2.0:
        print("  ✓ Volatile concepts adapted to higher decay rates")
    else:
        print("  ✗ Ratio unexpectedly low — check tau_decay or ticks")
    sh_mean = result["start_high_stable_delta_mean"]
    if sh_mean < 0.45:
        print(f"  ✓ Start-high-stable group decreased from δ=0.5 → {sh_mean:.4f} (downward adaptation)")
    else:
        print(f"  ✗ Start-high-stable δ={sh_mean:.4f} — downward adaptation weaker than expected")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m awareness_manager.evaluation",
        description="Awareness Manager — evaluation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── experiments ────────────────────────────────────────────────────────
    p_all = sub.add_parser("experiments", help="Run all six experiments")
    p_all.set_defaults(func=_cmd_experiments)

    # ── exp1 ───────────────────────────────────────────────────────────────
    p1 = sub.add_parser("exp1", help="Exp 1: scaling — latency vs N and budget")
    p1.add_argument("--seeds", type=int, default=5,
                    help="Number of random seeds (default: 5)")
    p1.set_defaults(func=_cmd_exp1)

    # ── exp2 ───────────────────────────────────────────────────────────────
    p2 = sub.add_parser("exp2", help="Exp 2: goal conditioning — K_overlap sweep")
    p2.add_argument("--seeds", type=int, default=5,
                    help="Number of random seeds (default: 5)")
    p2.set_defaults(func=_cmd_exp2)

    # ── exp3 ───────────────────────────────────────────────────────────────
    p3 = sub.add_parser("exp3", help="Exp 3: anticipatory horizon — F2 on vs off, ETA sweep")
    p3.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4],
                    metavar="B", help="Budget values to sweep (default: 1 2 4)")
    p3.add_argument("--etas", type=float, nargs="+", default=[20.0, 25.0, 30.0, 35.0, 40.0],
                    metavar="ETA", help="ETA values to sweep in seconds (default: 20 25 30 35 40)")
    p3.add_argument("--obs-interval", type=float, default=10.0)
    p3.set_defaults(func=_cmd_exp3)

    # ── exp4 ───────────────────────────────────────────────────────────────
    p4 = sub.add_parser("exp4", help="Exp 4: instance KB vs class-only — multi-instance sweep")
    p4.add_argument("--budget", type=int, default=2)
    p4.add_argument("--obs-interval", type=float, default=10.0)
    p4.add_argument("--instances", type=str, nargs="+", default=None,
                    metavar="ID",
                    help="Instance IDs to sweep (default: all 7 PV inspection instances)")
    p4.set_defaults(func=_cmd_exp4)

    # ── exp5 ───────────────────────────────────────────────────────────────
    p5 = sub.add_parser("exp5", help="Exp 5: formula ablation — F1 spreading activation")
    p5.add_argument("--seeds", type=int, default=5,
                    help="Number of random seeds (default: 5)")
    p5.add_argument("--obs-interval", type=float, default=10.0)
    p5.set_defaults(func=_cmd_exp5)

    # ── exp6 ───────────────────────────────────────────────────────────────
    p6 = sub.add_parser("exp6", help="Exp 6: F6 spatial opportunity cost — social serving")
    p6.add_argument("--ticks", type=int, default=50)
    p6.add_argument("--budget", type=int, default=2)
    p6.add_argument("--obs-interval", type=float, default=10.0)
    p6.set_defaults(func=_cmd_exp6)

    # ── abstract ──────────────────────────────────────────────────────────
    p_abs = sub.add_parser("abstract",
                            help="Run abstract N-variable scenario (smoke test)")
    p_abs.add_argument("--budget", type=int, default=1)
    p_abs.add_argument("--obs-interval", type=float, default=1.0)
    p_abs.add_argument("--quick", action="store_true",
                       help="Small parameters for fast smoke test")
    p_abs.set_defaults(func=_cmd_abstract)

    # ── learnable-decay ───────────────────────────────────────────────────
    p_ld = sub.add_parser("learnable-decay",
                           help="Verify learnable δ adapts volatile concepts")
    p_ld.set_defaults(func=_cmd_learnable_decay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
