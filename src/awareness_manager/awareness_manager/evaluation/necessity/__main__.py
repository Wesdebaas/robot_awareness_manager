"""
CLI entry point for the necessity evaluation package.

Proves that each core component of the awareness manager is necessary:
removing it causes the system to fail as an awareness manager.

    python -m awareness_manager.evaluation.necessity n1     # E necessity
    python -m awareness_manager.evaluation.necessity n2     # A necessity
    python -m awareness_manager.evaluation.necessity n3     # P=ExA combination necessity
    python -m awareness_manager.evaluation.necessity all    # all experiments
"""

import argparse


def _cmd_n1(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.necessity.experiments import (
        experiment_e_necessity,
        print_n1,
    )
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_e_necessity(
        N=args.N,
        K=args.K,
        budget=args.budget,
        ticks=args.ticks,
        seeds=seeds,
        spike_period=args.spike_period,
        obs_interval=args.obs_interval,
    )
    print_n1(results)


def _cmd_n2(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.necessity.experiments import (
        experiment_a_necessity,
        print_n2,
    )
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_a_necessity(
        N_relevant=args.N_relevant,
        K_irrelevant=args.K_irrelevant,
        budget=args.budget,
        ticks=args.ticks,
        seeds=seeds,
        spike_period=args.spike_period,
        obs_interval=args.obs_interval,
    )
    print_n2(results)


def _cmd_n3(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.necessity.experiments import (
        experiment_combination_necessity,
        print_n3,
    )
    seeds = list(range(42, 42 + args.seeds))
    results = experiment_combination_necessity(
        R_primary=args.R_primary,
        K_secondary=args.K_secondary,
        budget=args.budget,
        ticks=args.ticks,
        seeds=seeds,
        spike_period=args.spike_period,
        obs_interval=args.obs_interval,
    )
    print_n3(results)


def _cmd_all(args: argparse.Namespace) -> None:
    from awareness_manager.evaluation.necessity.experiments import run_all
    run_all(verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m awareness_manager.evaluation.necessity",
        description="Awareness Manager — necessity experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── n1 — E necessity ──────────────────────────────────────────────────
    p1 = sub.add_parser("n1", help="N1: epistemic state E is necessary")
    p1.add_argument("--N",    type=int,   default=12,   help="Total concepts (default 12)")
    p1.add_argument("--K",    type=int,   default=3,    help="Volatile concepts (default 3)")
    p1.add_argument("--budget",      type=int,   default=1,    help="Observation budget (default 1)")
    p1.add_argument("--ticks",       type=int,   default=500,  help="Simulation ticks (default 500)")
    p1.add_argument("--seeds",       type=int,   default=5,    help="Number of seeds (default 5)")
    p1.add_argument("--spike-period",type=int,   default=50,   help="Mean ticks between spikes (default 50)")
    p1.add_argument("--obs-interval",type=float, default=10.0, help="Observation interval s (default 10)")
    p1.set_defaults(func=_cmd_n1)

    # ── n2 — A necessity ──────────────────────────────────────────────────
    p2 = sub.add_parser("n2", help="N2: goal-conditioned relevance A is necessary")
    p2.add_argument("--N-relevant",   type=int,   default=3,    help="Relevant concepts (default 3)")
    p2.add_argument("--K-irrelevant", type=int,   default=6,    help="Irrelevant volatile (default 6)")
    p2.add_argument("--budget",       type=int,   default=1,    help="Observation budget (default 1)")
    p2.add_argument("--ticks",        type=int,   default=500,  help="Simulation ticks (default 500)")
    p2.add_argument("--seeds",        type=int,   default=5,    help="Number of seeds (default 5)")
    p2.add_argument("--spike-period", type=int,   default=30,   help="Mean ticks between spikes (default 30)")
    p2.add_argument("--obs-interval", type=float, default=10.0, help="Observation interval s (default 10)")
    p2.set_defaults(func=_cmd_n2)

    # ── n3 — P=E×A combination necessity ─────────────────────────────────
    p3 = sub.add_parser("n3", help="N3: multiplicative combination P=E×A is necessary")
    p3.add_argument("--R-primary",    type=int,   default=3,    help="Primary relevant concepts (default 3)")
    p3.add_argument("--K-secondary",  type=int,   default=6,    help="Secondary volatile (default 6)")
    p3.add_argument("--budget",       type=int,   default=1,    help="Observation budget (default 1)")
    p3.add_argument("--ticks",        type=int,   default=500,  help="Simulation ticks (default 500)")
    p3.add_argument("--seeds",        type=int,   default=5,    help="Number of seeds (default 5)")
    p3.add_argument("--spike-period", type=int,   default=30,   help="Mean ticks between spikes (default 30)")
    p3.add_argument("--obs-interval", type=float, default=10.0, help="Observation interval s (default 10)")
    p3.set_defaults(func=_cmd_n3)

    # ── all ───────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="Run all necessity experiments")
    p_all.set_defaults(func=_cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
