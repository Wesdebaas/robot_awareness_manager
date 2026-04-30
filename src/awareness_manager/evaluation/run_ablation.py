"""
Ablation Study: Grounding Formula Contributions
================================================

Runs the birdhouse scenario under different FeatureConfig combinations to show
the measurable contribution of each grounding formula. Each formula is compared
against a degraded baseline where that formula is disabled.

Formula OFF baselines:
    F1 OFF — Uniform attention (A=1 for all reachable concepts). Budget is
             spread equally across all concepts regardless of goal-relevance.
    F2 OFF — No anticipatory lookahead. Queued future goals are ignored until
             they become active, leaving the robot flat-footed on transitions.
    F3 OFF — Full over-refresh (refresh=1.0 always). Slow-decaying concepts
             waste their observation budget driving E near-zero unnecessarily.
    F4 OFF — Fixed max_distance used even when memory_budget is set. No
             resource-driven depth bounding; window size is uncontrolled.
    F5 OFF — Drift still active but priority = A only. Robot schedules by
             goal-relevance alone, ignoring information staleness.

Usage:
    # Compare one formula vs full system (saves figure to evaluation/figures/):
    python3 src/awareness_manager/evaluation/run_ablation.py --config no_anticipatory

    # Run all six ablation comparisons in one figure:
    python3 src/awareness_manager/evaluation/run_ablation.py --all

    # Disable specific formulas manually:
    python3 src/awareness_manager/evaluation/run_ablation.py --disable F1 F3

    # Run full system only (reference):
    python3 src/awareness_manager/evaluation/run_ablation.py --config full
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.feature_config import FeatureConfig
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb, build_birdhouse_kb_extended

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_OUT = Path(__file__).parent / 'figures'
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style — matches run_evaluation.py
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.fontsize':   9,
    'legend.framealpha': 0.8,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

_C_FULL    = '#2196F3'   # blue  — full system
_C_ABLATED = '#FF5722'   # red-orange — formula disabled

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

_DT            = 0.1    # simulated seconds per tick
_SIM_DURATION  = 60.0   # simulated seconds
_OBS_INTERVAL  = 2.0    # simulated seconds between observation events
_F3_INTERVAL   = 10.0   # observation_interval param (Formula 3 calibration)
_BUDGET        = 2      # concepts refreshed per observation event
_ALPHA         = 0.5
_MAX_DIST      = 4.0
_MEMORY_BUDGET = 9      # used when testing F4; depth = √9 - 1 = 2

# Goal transition: future goal is queued at t=0 and promotes at t=_SWITCH_T.
# For F2 the queued goal is 'store_tools' — a disconnected subgraph that has
# zero spreading-activation attention from 'build_birdhouse'. This makes the
# anticipatory pre-tuning effect visible (F2 ON pre-tunes; F2 OFF is blind).
# All other configs also use the extended KB and store_tools queue so the
# scenario is consistent across all ablations.
_INITIAL_GOAL  = 'build_birdhouse'
_QUEUED_GOAL   = 'store_tools'
_SWITCH_T      = 25.0              # ETA from t=0; goal promotes here

# ---------------------------------------------------------------------------
# Named configs
# ---------------------------------------------------------------------------

_NAMED_CONFIGS: dict[str, FeatureConfig] = {
    'full':           FeatureConfig.all_on(),
    'no_spreading':   FeatureConfig.with_disabled('f1'),
    'no_anticipatory':FeatureConfig.with_disabled('f2'),
    'no_saturation':  FeatureConfig.with_disabled('f3'),
    'no_budget':      FeatureConfig.with_disabled('f4'),
    'no_drift':       FeatureConfig.with_disabled('f5'),
    'baseline':       FeatureConfig.all_off(),
}

_LABELS: dict[str, str] = {
    'full':           'Full system',
    'no_spreading':   'F1 OFF — Uniform attention',
    'no_anticipatory':'F2 OFF — No lookahead',
    'no_saturation':  'F3 OFF — Over-refresh',
    'no_budget':      'F4 OFF — No budget constraint',
    'no_drift':       'F5 OFF — Attention-only priority',
    'baseline':       'All formulas OFF',
}

_TITLES: dict[str, str] = {
    'no_spreading':   'F1: Spreading Activation',
    'no_anticipatory':'F2: Anticipatory Horizon',
    'no_saturation':  'F3: Utility Saturation',
    'no_budget':      'F4: Memory Budget Constraint',
    'no_drift':       'F5: Epistemic Drift',
    'baseline':       'Baseline (all OFF)',
}

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run(fc: FeatureConfig, test_f4: bool = False) -> dict:
    """
    Simulate _SIM_DURATION seconds using the extended birdhouse scenario.

    The extended KB includes a disconnected store_tools subgraph so that
    Formula 2 (Anticipatory Horizon) has a measurable pre-tuning effect:
    with F2 ON the robot pre-tunes attention on store_tools concepts before
    the goal switch; with F2 OFF they are invisible until the switch fires.

    Also tracks cumulative observation waste for the F3 metric:
        waste = Σ max(0, actual_refresh − calibrated_refresh)
    F3 ON: waste ≈ 0 (refresh is calibrated to accumulated drift).
    F3 OFF: waste is large (refresh=1.0 over-cleans slow concepts).

    Args:
        fc:      FeatureConfig controlling which formulas are active.
        test_f4: When True, sets memory_budget=_MEMORY_BUDGET so the F4
                 depth constraint is active (or absent when F4 is off).

    Returns:
        dict with keys:
            times          — array of simulated timestamps
            weighted_error — Σ(E·A)/Σ(A) at each tick
            mean_error     — unweighted mean E at each tick
            window_size    — number of concepts with A > 0 at each tick
            obs_count      — cumulative observations executed
            cum_waste      — cumulative observation waste (F3 metric)
            switch_t       — time of goal transition
    """
    kb = build_birdhouse_kb_extended()
    non_task = [c for c in kb.concept_ids()
                if kb.get_concept(c).concept_type != 'task']

    am = AwarenessManager(
        kb,
        goal_id=_INITIAL_GOAL,
        alpha=_ALPHA,
        max_distance=_MAX_DIST,
        budget=_BUDGET,
        observation_interval=_F3_INTERVAL,
        memory_budget=_MEMORY_BUDGET if test_f4 else None,
        feature_config=fc,
    )

    am.queue_goal(_QUEUED_GOAL, eta=_SWITCH_T, level='task')

    times           = []
    weighted_errors = []
    mean_errors     = []
    window_sizes    = []
    total_obs       = 0
    obs_counts      = []
    total_waste     = 0.0
    cum_wastes      = []
    last_obs        = -_OBS_INTERVAL

    t = 0.0
    while t <= _SIM_DURATION + 1e-9:
        am.tick(dt=_DT)

        if t - last_obs >= _OBS_INTERVAL:
            for cid in am._top_n():
                calibrated = am.observation_refresh_value(cid)
                actual     = am.observe(cid)
                total_obs += 1
                total_waste += max(0.0, actual - calibrated)
            last_obs = t

        errors = [kb.get_concept(c).epistemic_error for c in non_task]
        attns  = [am.attention().get(c, 0.0)         for c in non_task]
        denom  = sum(attns) or 1.0
        w_e    = sum(e * a for e, a in zip(errors, attns)) / denom
        m_e    = float(np.mean(errors))
        win    = sum(1 for a in am.attention().values() if a > 0.0)

        times.append(t)
        weighted_errors.append(w_e)
        mean_errors.append(m_e)
        window_sizes.append(win)
        obs_counts.append(total_obs)
        cum_wastes.append(total_waste)

        t = round(t + _DT, 10)

    return {
        'times':          np.array(times),
        'weighted_error': np.array(weighted_errors),
        'mean_error':     np.array(mean_errors),
        'window_size':    np.array(window_sizes),
        'obs_count':      np.array(obs_counts),
        'cum_waste':      np.array(cum_wastes),
        'switch_t':       _SWITCH_T,
    }


# ---------------------------------------------------------------------------
# Single-comparison plot (full vs. one ablated config)
# ---------------------------------------------------------------------------

def _plot_comparison(name: str, fc: FeatureConfig) -> tuple[Path, dict, dict]:
    """Run full system and one ablated config; produce a two-panel comparison."""
    test_f4 = (name == 'no_budget')
    print(f'[ablation]  running  full  …')
    r_full   = _run(FeatureConfig.all_on(), test_f4=test_f4)
    print(f'[ablation]  running  {name}  …')
    r_ablated = _run(fc, test_f4=test_f4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    title = _TITLES.get(name, name)

    # Left: weighted epistemic error over time
    ax = axes[0]
    ax.plot(r_full['times'],    r_full['weighted_error'],
            color=_C_FULL,    lw=1.5, label=_LABELS['full'])
    ax.plot(r_ablated['times'], r_ablated['weighted_error'],
            color=_C_ABLATED, lw=1.5, label=_LABELS[name], linestyle='--')
    ax.axvline(_SWITCH_T, color='#666', lw=0.8, linestyle=':', label='goal switch')
    ax.set_xlabel('Simulated time  (s)')
    ax.set_ylabel('Attention-weighted epistemic error  Σ(E·A)/Σ(A)')
    ax.set_title(f'{title} — weighted epistemic error')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Right: secondary metric — chosen per formula to best highlight its contribution
    ax2 = axes[1]
    if name == 'no_budget':
        ax2.plot(r_full['times'],    r_full['window_size'],
                 color=_C_FULL,    lw=1.5, label=_LABELS['full'])
        ax2.plot(r_ablated['times'], r_ablated['window_size'],
                 color=_C_ABLATED, lw=1.5, label=_LABELS[name], linestyle='--')
        ax2.set_ylabel('Attention window size  |{c : A(c) > 0}|')
        ax2.set_title(f'{title} — window size (budget B = {_MEMORY_BUDGET})')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    elif name == 'no_saturation':
        # F3: show cumulative observation waste = Σ max(0, actual_refresh - calibrated)
        ax2.plot(r_full['times'],    r_full['cum_waste'],
                 color=_C_FULL,    lw=1.5, label=_LABELS['full'])
        ax2.plot(r_ablated['times'], r_ablated['cum_waste'],
                 color=_C_ABLATED, lw=1.5, label=_LABELS[name], linestyle='--')
        ax2.axvline(_SWITCH_T, color='#666', lw=0.8, linestyle=':', label='goal switch')
        ax2.set_ylabel('Cumulative observation waste  Σ(refresh − calibrated)')
        ax2.set_title(f'{title} — observation waste (F3 OFF over-refreshes slow concepts)')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    else:
        ax2.plot(r_full['times'],    r_full['mean_error'],
                 color=_C_FULL,    lw=1.5, label=_LABELS['full'])
        ax2.plot(r_ablated['times'], r_ablated['mean_error'],
                 color=_C_ABLATED, lw=1.5, label=_LABELS[name], linestyle='--')
        ax2.axvline(_SWITCH_T, color='#666', lw=0.8, linestyle=':', label='goal switch')
        ax2.set_ylabel('Mean epistemic error  Ē')
        ax2.set_title(f'{title} — mean epistemic error')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Simulated time  (s)')
    ax2.legend()

    fig.suptitle(
        f'Ablation: {title}\n'
        f'(birdhouse scenario, budget={_BUDGET}, α={_ALPHA}, '
        f'obs every {_OBS_INTERVAL}s, goal switch at t={_SWITCH_T}s)',
        fontsize=10,
    )
    fig.tight_layout()

    path = _OUT / f'ablation_{name}.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[ablation]  saved → {path}')
    return path, r_full, r_ablated


# ---------------------------------------------------------------------------
# All-formulas comparison (2×3 grid)
# ---------------------------------------------------------------------------

def _plot_all() -> Path:
    """Run all six ablation configs and produce a 2×3 comparison grid."""
    ablation_names = ['no_spreading', 'no_anticipatory', 'no_saturation',
                      'no_budget', 'no_drift']

    # Run full system once (reused as reference in every subplot)
    print('[ablation]  running  full  …')
    r_full = _run(FeatureConfig.all_on(), test_f4=False)
    r_full_f4 = _run(FeatureConfig.all_on(), test_f4=True)

    results: dict[str, dict] = {}
    for name in ablation_names:
        print(f'[ablation]  running  {name}  …')
        test_f4 = (name == 'no_budget')
        results[name] = _run(_NAMED_CONFIGS[name], test_f4=test_f4)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(ablation_names):
        ax = axes_flat[idx]
        ref = r_full_f4 if name == 'no_budget' else r_full

        if name == 'no_budget':
            y_full    = ref['window_size']
            y_ablated = results[name]['window_size']
            ylabel    = 'Window size'
        elif name == 'no_saturation':
            y_full    = ref['cum_waste']
            y_ablated = results[name]['cum_waste']
            ylabel    = 'Cum. obs. waste'
        else:
            y_full    = ref['weighted_error']
            y_ablated = results[name]['weighted_error']
            ylabel    = 'Weighted E'

        ax.plot(ref['times'],              y_full,
                color=_C_FULL,    lw=1.4, label='Full')
        ax.plot(results[name]['times'],    y_ablated,
                color=_C_ABLATED, lw=1.4, label='OFF', linestyle='--')
        ax.axvline(_SWITCH_T, color='#888', lw=0.7, linestyle=':')
        ax.set_title(_TITLES[name], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Hide unused 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle(
        'Ablation Study: Grounding Formula Contributions\n'
        f'(birdhouse, budget={_BUDGET}, α={_ALPHA}, '
        f'obs every {_OBS_INTERVAL}s, goal switch at t={_SWITCH_T}s)',
        fontsize=11,
    )
    fig.tight_layout()

    path = _OUT / 'ablation_all.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[ablation]  saved → {path}')
    return path


# ---------------------------------------------------------------------------
# Terminal summary table
# ---------------------------------------------------------------------------

def _print_table(name: str, r_full: dict, r_ablated: dict) -> None:
    full_final_e  = r_full['weighted_error'][-1]
    abl_final_e   = r_ablated['weighted_error'][-1]
    full_obs      = r_full['obs_count'][-1]
    abl_obs       = r_ablated['obs_count'][-1]
    full_win      = float(np.mean(r_full['window_size']))
    abl_win       = float(np.mean(r_ablated['window_size']))

    print()
    print(f'  {"Config":<28}  {"Final weighted E":>16}  {"Total obs":>10}  {"Avg window":>10}')
    print(f'  {"-"*28}  {"-"*16}  {"-"*10}  {"-"*10}')
    print(f'  {"full":<28}  {full_final_e:>16.4f}  {full_obs:>10}  {full_win:>10.1f}')
    print(f'  {name:<28}  {abl_final_e:>16.4f}  {abl_obs:>10}  {abl_win:>10.1f}')
    delta = abl_final_e - full_final_e
    sign  = '+' if delta >= 0 else ''
    print(f'  {"delta":<28}  {sign}{delta:>15.4f}')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Ablation study for grounding formula contributions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            'Named configs:',
            *[f'  {k:<20} — {v}' for k, v in _LABELS.items()],
        ]),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config', choices=list(_NAMED_CONFIGS),
        help='Run a named ablation config vs. full system.',
    )
    group.add_argument(
        '--disable', nargs='+', metavar='FN',
        help='Disable specific formulas by name, e.g. --disable F1 F3.',
    )
    group.add_argument(
        '--all', action='store_true',
        help='Run all five single-formula ablations in one 2×3 figure.',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.all:
        _plot_all()
        return

    if args.disable:
        try:
            fc = FeatureConfig.with_disabled(*args.disable)
        except ValueError as exc:
            print(f'Error: {exc}')
            return
        name = 'custom_' + '_'.join(f.lower() for f in args.disable)
        _LABELS[name]  = f'Custom: {", ".join(f.upper() for f in args.disable)} OFF'
        _TITLES[name]  = _LABELS[name]
    else:
        name = args.config
        fc   = _NAMED_CONFIGS[name]

    if name == 'full':
        print('[ablation]  running full system only …')
        test_f4 = False
        r = _run(fc, test_f4=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(r['times'], r['weighted_error'], color=_C_FULL, lw=1.5)
        ax.axvline(_SWITCH_T, color='#666', lw=0.8, linestyle=':', label='goal switch')
        ax.set_xlabel('Simulated time (s)')
        ax.set_ylabel('Attention-weighted epistemic error Σ(E·A)/Σ(A)')
        ax.set_title('Full system — weighted epistemic error')
        ax.legend()
        fig.tight_layout()
        path = _OUT / 'ablation_full.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'[ablation]  saved → {path}')
        return

    path, r_full, r_ablated = _plot_comparison(name, fc)
    _print_table(name, r_full, r_ablated)


if __name__ == '__main__':
    main()
