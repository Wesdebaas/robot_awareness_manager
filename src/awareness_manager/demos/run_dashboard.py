"""
Awareness Manager dashboard - PV inspection scenario (CoreSense D7.1).

Usage (live mode):
    python src/awareness_manager/demos/run_dashboard.py
    python src/awareness_manager/demos/run_dashboard.py --strategy always_on
    python src/awareness_manager/demos/run_dashboard.py --strategy reactive
    python src/awareness_manager/demos/run_dashboard.py --log
    python src/awareness_manager/demos/run_dashboard.py --log --log-path traces/run_001 --log-rate 2

Usage (replay mode):
    python src/awareness_manager/demos/run_dashboard.py --replay traces/run_001

Usage (compare mode):
    python src/awareness_manager/demos/run_dashboard.py --replay traces/am_run --compare traces/reactive_run

Flags:
    --strategy NAME  Live-mode strategy: awareness_manager (default), always_on, reactive.
    --replay PATH    Load a trace directory and run in offline replay mode.
    --compare PATH   Second trace for A/B comparison (requires --replay for first trace).
    --log            Enable trace logging to disk in live mode.
    --log-path PATH  Trace output directory (default: traces/run_<timestamp>).
    --log-rate N     Record every Nth tick (default: 1 = every tick).
    --port PORT      Dashboard port (default: 8050).
    --debug          Enable Dash debug mode.
"""

import argparse
import datetime
from pathlib import Path

SCENARIO = "pv_inspection"


def _build_am(args: argparse.Namespace):
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.scenarios.pv_inspection import (
        build_pv_inspection_instance_kb,
        build_pv_inspection_kb,
    )

    kb = build_pv_inspection_kb()
    ikb = build_pv_inspection_instance_kb()
    am = AwarenessManager(
        kb,
        goal_id='inspect_pv_field',
        budget=2,
        observation_interval=10.0,
        lambda_horizon=0.1,
        instance_kb=ikb,
        instance_relational_weight=0.3,
    )
    am.queue_goal('emergency_landing', eta=30.0, level='global')
    return am, True  # observe_top=True


def _build_always_on(args: argparse.Namespace):
    from awareness_manager.baselines.always_on import AlwaysOnBaseline
    from awareness_manager.scenarios.pv_inspection import (
        build_pv_inspection_instance_kb,
        build_pv_inspection_kb,
    )

    kb = build_pv_inspection_kb()
    ikb = build_pv_inspection_instance_kb()
    strategy = AlwaysOnBaseline(
        kb,
        goal_id='inspect_pv_field',
        observation_interval=10.0,
        ikb=ikb,
    )
    strategy.queue_goal('emergency_landing', eta=30.0, level='global')
    return strategy, False  # observe_top=False - AlwaysOn self-manages in tick()


def _build_reactive(args: argparse.Namespace):
    from awareness_manager.baselines.reactive import ReactiveBaseline
    from awareness_manager.scenarios.pv_inspection import (
        build_pv_inspection_instance_kb,
        build_pv_inspection_kb,
    )

    kb = build_pv_inspection_kb()
    ikb = build_pv_inspection_instance_kb()
    strategy = ReactiveBaseline(
        kb,
        goal_id='inspect_pv_field',
        budget=2,
        observation_interval=10.0,
        ikb=ikb,
    )
    strategy.queue_goal('emergency_landing', eta=30.0, level='global')
    return strategy, True  # observe_top=True - runner must call observe(); round-robin is selection only


def _build_live(args: argparse.Namespace):
    from awareness_manager.visualization.runner import SimulationRunner

    strategy_name = args.strategy or "awareness_manager"
    builders = {
        "awareness_manager": _build_am,
        "always_on": _build_always_on,
        "reactive": _build_reactive,
    }
    if strategy_name not in builders:
        raise SystemExit(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {', '.join(builders)}"
        )

    strategy, observe_top = builders[strategy_name](args)
    print(f"  Strategy: {strategy_name}  observe_top={observe_top}")

    logger = None
    if args.log:
        from awareness_manager.visualization.trace_logger import TraceLogger
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = (Path(args.log_path) if args.log_path
                    else Path("traces") / f"{strategy_name}_{ts}")
        logger = TraceLogger(
            strategy,
            scenario=SCENARIO,
            output_dir=log_path,
            dt=0.1,
            sample_rate=args.log_rate,
            history_maxlen=300,
        )
        print(f"  Logging trace → {log_path}")

    runner = SimulationRunner(
        strategy, dt=0.1, observe_top=observe_top,
        history_maxlen=300, logger=logger,
    )
    return runner


def _build_replay(path_str: str):
    from awareness_manager.visualization.replay_reader import ReplayReader
    trace_dir = Path(path_str)
    if not trace_dir.exists():
        raise SystemExit(f"Trace directory not found: {trace_dir}")
    reader = ReplayReader(trace_dir)
    print(f"  Loaded {reader.tick_count} ticks  "
          f"({reader.total_duration():.1f}s)  "
          f"scenario={reader.meta.get('scenario', '?')}")
    return reader


def main() -> None:
    parser = argparse.ArgumentParser(description="Awareness Manager Dashboard")
    parser.add_argument(
        "--strategy", metavar="NAME", default="awareness_manager",
        help="Live-mode strategy: awareness_manager (default), always_on, reactive",
    )
    parser.add_argument("--replay", metavar="PATH",
                        help="Replay a trace directory (offline mode)")
    parser.add_argument("--compare", metavar="PATH",
                        help="Second trace for A/B comparison (requires --replay)")
    parser.add_argument("--log", action="store_true",
                        help="Log trace to disk in live mode")
    parser.add_argument("--log-path", metavar="PATH",
                        help="Trace output directory (default: traces/<strategy>_<timestamp>)")
    parser.add_argument("--log-rate", type=int, default=1, metavar="N",
                        help="Record every Nth tick (default: 1)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.compare and not args.replay:
        raise SystemExit("--compare requires --replay to specify the first trace.")

    from awareness_manager.visualization.dashboard import run

    print("=" * 60)
    source_b = None
    if args.compare:
        print("  Robot Awareness Dashboard - Compare mode")
        source = _build_replay(args.replay)
        source_b = _build_replay(args.compare)
    elif args.replay:
        print("  Robot Awareness Dashboard - Replay mode")
        source = _build_replay(args.replay)
    else:
        print(f"  Robot Awareness Dashboard - {SCENARIO} (live)")
        source = _build_live(args)
    print(f"  Open http://localhost:{args.port} in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    run(source, source_b=source_b, scenario=SCENARIO,
        debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
