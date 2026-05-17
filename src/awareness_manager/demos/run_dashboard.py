"""
Awareness Manager dashboard.

Usage (live mode):
    python src/awareness_manager/demos/run_dashboard.py
    python src/awareness_manager/demos/run_dashboard.py --strategy always_on
    python src/awareness_manager/demos/run_dashboard.py --strategy reactive
    python src/awareness_manager/demos/run_dashboard.py --scenario social_serving
    python src/awareness_manager/demos/run_dashboard.py --log
    python src/awareness_manager/demos/run_dashboard.py --log --log-path traces/run_001 --log-rate 2

Usage (replay mode):
    python src/awareness_manager/demos/run_dashboard.py --replay traces/run_001

Usage (compare mode):
    python src/awareness_manager/demos/run_dashboard.py --replay traces/am_run --compare traces/reactive_run

Channel colour demo (shows all four node colours):
    python src/awareness_manager/demos/run_dashboard.py --demo

    What to watch:
      blue   (mission)    - all nodes driven by the active goal via spreading activation
      teal   (relational) - battery_main: attended because it powers camera_main
                            (relational instance edge, not directly goal-relevant).
                            Turns blue once the emergency goal fires and drone_battery
                            becomes mission-relevant.
      orange (surprise)   - image_quality: flashes orange every ~10 s when a simulated
                            sensor reading violates the current expected value.
      darker shade        - higher epistemic error (concept is stale / unobserved)

Flags:
    --strategy NAME  Live-mode strategy: awareness_manager (default), always_on, reactive.
    --demo           Run the channel-colour demo (overrides --strategy).
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
import random
from pathlib import Path


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


# ===========================================================================
# Social serving scenario builders
# ===========================================================================

_DRINK_CLASSES = {'juice', 'cola', 'beer', 'wine', 'champagne'}
_DISAPPEAR_RATE = 0.003  # per drink per second → ~1 disappearance every 33s


def _make_social_serving_tick_hook(seed: int = 42):
    """
    Returns a tick_hook that randomly removes drinks from the IKB.

    Drink disappearances are irreversible during a dashboard run — the robot's
    epistemic state degrades for gone drinks and the class E (derived) adjusts
    to reflect only remaining instances.  Visual cue: confirmed-absent drink
    nodes dim to A=0 and stop being scheduled.
    """
    from awareness_manager.concept import PresenceState

    rng = random.Random(seed)
    dt = 0.1  # runner dt

    def _hook(strategy, elapsed: float) -> None:
        ikb = strategy.instance_kb
        if ikb is None:
            return
        for iid in list(ikb.active_instance_ids()):
            inst = ikb.get_instance(iid)
            if not any(c in _DRINK_CLASSES for c in inst.all_class_ids):
                continue
            if rng.random() < _DISAPPEAR_RATE * dt:
                if inst.presence_state != PresenceState.CONFIRMED_ABSENT:
                    ikb.mark_suspected_absent(iid)
                    ikb.confirm_absent(iid)

    return _hook


def _build_social_serving_am(args):
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.scenarios.social_serving import (
        build_social_serving_instance_kb,
        build_social_serving_kb,
    )

    kb  = build_social_serving_kb()
    ikb = build_social_serving_instance_kb()
    am  = AwarenessManager(
        kb, goal_id='serve_drinks', alpha=0.5,
        budget=3, observation_interval=5.0,
        instance_kb=ikb,
        instance_relational_weight=0.3,
    )
    return am, True, _make_social_serving_tick_hook()


def _build_social_serving_always_on(args):
    from awareness_manager.baselines.always_on import AlwaysOnBaseline
    from awareness_manager.scenarios.social_serving import (
        build_social_serving_instance_kb,
        build_social_serving_kb,
    )

    kb  = build_social_serving_kb()
    ikb = build_social_serving_instance_kb()
    strategy = AlwaysOnBaseline(
        kb, goal_id='serve_drinks',
        observation_interval=5.0, ikb=ikb,
    )
    return strategy, False, _make_social_serving_tick_hook()


def _build_social_serving_reactive(args):
    from awareness_manager.baselines.reactive import ReactiveBaseline
    from awareness_manager.scenarios.social_serving import (
        build_social_serving_instance_kb,
        build_social_serving_kb,
    )

    kb  = build_social_serving_kb()
    ikb = build_social_serving_instance_kb()
    strategy = ReactiveBaseline(
        kb, goal_id='serve_drinks',
        budget=3, observation_interval=5.0, ikb=ikb,
    )
    return strategy, True, _make_social_serving_tick_hook()


def _build_channel_demo():
    """
    Build an AM configured to show all four node colour channels.

    Two additions on top of the standard PV scenario:

    1. Relational edge  camera_main → battery_main (weight=0.5, 'powers')
       During inspect_pv_field, camera_main has high mission attention (drone_camera
       is 1-hop from the goal).  The low-weight edge makes battery_main's relational
       boost exceed its class-gate (drone_battery is 2-hop, class attention ≈ 0.25),
       so battery_main shows TEAL.  When emergency_landing fires, drone_battery
       becomes 1-hop and mission attention surges → battery_main turns BLUE.

    2. Surprise tick hook  injects observe_with_feedback('image_quality', ...) every
       10 simulated seconds, alternating observed=0.0 / 1.0.  The first call sets
       the expected value; each subsequent call produces epsilon=1.0 ≫ threshold →
       image_quality flashes ORANGE and the violation propagates to its 1-hop neighbours.

    lambda_horizon=0.5 keeps the anticipatory pre-tuning discount very small for the
    60 s eta, so relational spread (teal) is not overwhelmed by anticipatory (blue)
    during the inspection phase.
    """
    from awareness_manager.awareness_manager import AwarenessManager
    from awareness_manager.scenarios.pv_inspection import (
        build_pv_inspection_instance_kb,
        build_pv_inspection_kb,
    )

    kb = build_pv_inspection_kb()
    ikb = build_pv_inspection_instance_kb()
    # Cross-cluster instance edge: camera draws power from the battery.
    # Low weight (0.5) → short Dijkstra distance → strong relational spread.
    ikb.add_instance_relation('camera_main', 'battery_main', weight=0.5,
                              relation_type='powers')

    am = AwarenessManager(
        kb,
        goal_id='inspect_pv_field',
        budget=2,
        observation_interval=10.0,
        lambda_horizon=0.5,        # high: keeps anticipatory near-zero for far-future goals
        instance_kb=ikb,
        instance_relational_weight=0.8,  # boosted to make relational beat class-gate
    )
    am.queue_goal('emergency_landing', eta=60.0, level='global')

    # Surprise injection: alternates image_quality between 0.0 and 1.0 every 10 s.
    # First call sets the predicted value (no violation); each subsequent call
    # produces epsilon=1.0, triggering orange on image_quality and its neighbours.
    _last_surprise = [0.0]
    _flip = [True]
    _surprise_interval = 10.0

    def _tick_hook(strategy, elapsed: float) -> None:
        if elapsed - _last_surprise[0] >= _surprise_interval:
            observed = 0.0 if _flip[0] else 1.0
            strategy.observe_with_feedback('image_quality', observed_value=observed)
            _flip[0] = not _flip[0]
            _last_surprise[0] = elapsed

    return am, True, _tick_hook  # strategy, observe_top, tick_hook


def _build_live(args: argparse.Namespace) -> tuple:
    from awareness_manager.visualization.runner import SimulationRunner

    tick_hook = None
    scenario = getattr(args, 'scenario', 'pv_inspection') or 'pv_inspection'

    if args.demo:
        strategy, observe_top, tick_hook = _build_channel_demo()
        strategy_name = "channel_demo"
    elif scenario == 'social_serving':
        strategy_name = args.strategy or "awareness_manager"
        builders = {
            "awareness_manager": _build_social_serving_am,
            "always_on":         _build_social_serving_always_on,
            "reactive":          _build_social_serving_reactive,
        }
        if strategy_name not in builders:
            raise SystemExit(
                f"Unknown strategy '{strategy_name}'. "
                f"Choose from: {', '.join(builders)}"
            )
        strategy, observe_top, tick_hook = builders[strategy_name](args)
    else:
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

    print(f"  Strategy: {strategy_name}  scenario={scenario}  observe_top={observe_top}")

    logger = None
    if args.log:
        from awareness_manager.visualization.trace_logger import TraceLogger
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = (Path(args.log_path) if args.log_path
                    else Path("traces") / f"{strategy_name}_{ts}")
        logger = TraceLogger(
            strategy,
            scenario=scenario,
            output_dir=log_path,
            dt=0.1,
            sample_rate=args.log_rate,
            history_maxlen=300,
        )
        print(f"  Logging trace → {log_path}")

    runner = SimulationRunner(
        strategy, dt=0.1, observe_top=observe_top,
        history_maxlen=300, logger=logger, tick_hook=tick_hook,
        observe_interval=2.0,
    )
    return scenario, runner


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
        "--scenario", metavar="NAME", default="pv_inspection",
        choices=["pv_inspection", "social_serving"],
        help="Scenario to run: pv_inspection (default) or social_serving",
    )
    parser.add_argument(
        "--strategy", metavar="NAME", default="awareness_manager",
        help="Live-mode strategy: awareness_manager (default), always_on, reactive",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the channel-colour demo (shows teal relational + orange surprise nodes)",
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
    scenario = args.scenario
    if args.compare:
        print("  Robot Awareness Dashboard - Compare mode")
        source = _build_replay(args.replay)
        source_b = _build_replay(args.compare)
        scenario = source.meta.get("scenario", scenario)
    elif args.replay:
        print("  Robot Awareness Dashboard - Replay mode")
        source = _build_replay(args.replay)
        scenario = source.meta.get("scenario", scenario)
    else:
        label = "channel demo" if args.demo else scenario
        print(f"  Robot Awareness Dashboard - {label} (live)")
        scenario, source = _build_live(args)
    print(f"  Open http://localhost:{args.port} in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    run(source, source_b=source_b, scenario=scenario,
        debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
