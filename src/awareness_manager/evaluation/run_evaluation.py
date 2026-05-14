"""
Evaluation: Robot Awareness Management - Grounding Formulas
===========================================================

Produces seven thesis figures saved to evaluation/figures/:

    fig1_attention_reshuffle.png  - Formula 1: attention reshuffles on goal switch
    fig2_scheduling_comparison.png- Formula 3: AM-guided vs random vs no-refresh
    fig3_anticipatory_horizon.png - Formula 2: pre-tuning as future goal ETA → 0
    fig4_memory_budget.png        - Formula 4: attention window size vs budget B
    fig8_certainty_threshold.png  - Phase 4: refresh count vs epistemic error trade-off
    fig9_hierarchical_horizons.png- Phase 5: global / phase / task blending over time

Run from the workspace root:
    python3 src/awareness_manager/evaluation/run_evaluation.py
"""

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_OUT = Path(__file__).parent / 'figures'
_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'legend.fontsize':  9,
    'legend.framealpha':0.8,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

_C = {
    'am':       '#2196F3',   # blue
    'random':   '#FF5722',   # red-orange
    'noop':     '#9E9E9E',   # grey
    'battery':  '#F44336',   # red
    'landing':  '#FF9800',   # amber
    'solar':    '#4CAF50',   # green
    'camera':   '#9C27B0',   # purple
    'inspect':  '#2196F3',   # blue
    'emergency':'#F44336',   # red
}

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------

_DT            = 0.1    # simulated seconds per tick
_SIM_DURATION  = 60.0   # simulated seconds per experiment
_OBS_INTERVAL  = 2.0    # simulated seconds between observations
_F3_INTERVAL   = 10.0   # Formula 3 observation_interval parameter
_BUDGET        = 2      # concepts refreshed per observation event
_ALPHA         = 0.5
_MAX_DIST      = 4.0
_SEED          = 42


# ===========================================================================
# Figure 1 - Attention reshuffle on goal switch  (Formula 1)
# ===========================================================================

def fig1_attention_reshuffle() -> None:
    """
    Bar chart: attention per concept under inspect_pv_field vs emergency_landing.
    Demonstrates that spreading activation completely reshuffles priorities when
    the mission goal changes.
    """
    kb = build_pv_inspection_kb()
    a_inspect   = kb.compute_attention('inspect_pv_field',  alpha=_ALPHA, max_distance=_MAX_DIST)
    a_emergency = kb.compute_attention('emergency_landing', alpha=_ALPHA, max_distance=_MAX_DIST)

    # Concepts that appear in either attention window, sorted by inspect attention
    concepts = sorted(
        set(a_inspect) | set(a_emergency),
        key=lambda c: a_inspect.get(c, 0.0),
        reverse=True,
    )

    x = np.arange(len(concepts))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    bars_i = ax.bar(x - w/2, [a_inspect.get(c, 0.0)   for c in concepts],
                    width=w, color=_C['inspect'],   label='inspect_pv_field',   alpha=0.85)
    bars_e = ax.bar(x + w/2, [a_emergency.get(c, 0.0) for c in concepts],
                    width=w, color=_C['emergency'],  label='emergency_landing', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Attention  A(c)')
    ax.set_ylim(0, 1.12)
    ax.set_title('Fig 1 - Attention redistribution on goal switch  (Formula 1: Spreading Activation)')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Annotate the biggest swings
    for i, c in enumerate(concepts):
        diff = a_emergency.get(c, 0.0) - a_inspect.get(c, 0.0)
        if abs(diff) > 0.1:
            arrow_y = max(a_inspect.get(c, 0.0), a_emergency.get(c, 0.0)) + 0.04
            ax.annotate(
                f'Δ{diff:+.2f}',
                xy=(i, arrow_y), ha='center', va='bottom',
                fontsize=7.5, color='#444444',
            )

    fig.tight_layout()
    path = _OUT / 'fig1_attention_reshuffle.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 1]  saved → {path}')

    # Console summary
    print('         concept              inspect   emergency   delta')
    for c in concepts:
        ai = a_inspect.get(c, 0.0)
        ae = a_emergency.get(c, 0.0)
        print(f'         {c:<22}  {ai:.3f}     {ae:.3f}     {ae-ai:+.3f}')


# ===========================================================================
# Figure 2 - Scheduling comparison  (Formula 3)
# ===========================================================================

def _run_scheduling_strategy(strategy: str, seed: int) -> dict:
    """
    Simulate _SIM_DURATION seconds with one of three scheduling strategies:
        'am'      - priority = E x A, top-BUDGET concepts observed each cycle
        'random'  - BUDGET randomly selected non-task concepts each cycle
        'noop'    - no observations; purely passive epistemic drift

    Returns dict with keys: times, mean_error, weighted_error, final_errors
    """
    rng = random.Random(seed)

    kb = build_pv_inspection_kb()
    am = AwarenessManager(
        kb, goal_id='inspect_pv_field',
        alpha=_ALPHA, max_distance=_MAX_DIST,
        budget=_BUDGET, observation_interval=_F3_INTERVAL,
    )

    non_task = [c for c in kb.concept_ids()
                if kb.get_concept(c).concept_type != 'task']

    times, mean_errors, weighted_errors = [], [], []
    t = 0.0
    last_obs = -_OBS_INTERVAL  # observe on first eligible tick

    while t <= _SIM_DURATION + 1e-9:
        # --- Tick ---
        am.tick(dt=_DT)

        # --- Observe ---
        if t - last_obs >= _OBS_INTERVAL:
            if strategy == 'am':
                for cid in am._top_n():
                    am.observe(cid)
            elif strategy == 'random':
                for cid in rng.sample(non_task, min(_BUDGET, len(non_task))):
                    am.observe(cid)
            # 'noop': nothing
            last_obs = t

        # --- Record ---
        errors  = [kb.get_concept(c).epistemic_error for c in non_task]
        attns   = [am.attention().get(c, 0.0)         for c in non_task]
        mean_e  = float(np.mean(errors))
        denom   = sum(attns) or 1.0
        w_e     = sum(e * a for e, a in zip(errors, attns)) / denom

        times.append(t)
        mean_errors.append(mean_e)
        weighted_errors.append(w_e)

        t = round(t + _DT, 10)  # avoid float drift

    return {
        'times':          np.array(times),
        'mean_error':     np.array(mean_errors),
        'weighted_error': np.array(weighted_errors),
        'final_errors':   {c: kb.get_concept(c).epistemic_error for c in non_task},
    }


def fig2_scheduling_comparison() -> None:
    """
    Line plot: mean epistemic error and attention-weighted epistemic error over
    simulated time for three scheduling strategies.
    """
    print('[Fig 2]  running simulations …')
    results = {
        'am':     _run_scheduling_strategy('am',     _SEED),
        'random': _run_scheduling_strategy('random', _SEED),
        'noop':   _run_scheduling_strategy('noop',   _SEED),
    }

    labels = {
        'am':     'AM-guided  (priority = E x A)',
        'random': 'Random selection',
        'noop':   'No refresh  (passive drift)',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(
        'Fig 2 - Scheduling strategies  (Formula 3: Utility Saturation)',
        fontsize=11,
    )

    for strategy, res in results.items():
        t = res['times']
        ax1.plot(t, res['mean_error'],     color=_C[strategy], label=labels[strategy], linewidth=1.6)
        ax2.plot(t, res['weighted_error'], color=_C[strategy], label=labels[strategy], linewidth=1.6)

    ax1.set_xlabel('Simulated time  (s)')
    ax1.set_ylabel('Mean epistemic error  E̅')
    ax1.set_title('Mean E across all non-task concepts')
    ax1.set_ylim(bottom=0)
    ax1.legend()

    ax2.set_xlabel('Simulated time  (s)')
    ax2.set_ylabel('Attention-weighted epistemic error')
    ax2.set_title('Σ(E·A) / Σ(A)  - penalises errors on attended concepts')
    ax2.set_ylim(bottom=0)
    ax2.legend()

    fig.tight_layout()
    path = _OUT / 'fig2_scheduling_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 2]  saved → {path}')

    # Console summary - final E values
    print('         Final epistemic errors at t={:.0f}s:'.format(_SIM_DURATION))
    print(f'         {"concept":<22}  {"AM":>8}  {"random":>8}  {"noop":>8}')
    all_concepts = list(results['am']['final_errors'])
    for c in sorted(all_concepts):
        row = '  '.join(f'{results[s]["final_errors"][c]:8.3f}' for s in ('am', 'random', 'noop'))
        print(f'         {c:<22}  {row}')


# ===========================================================================
# Figure 3 - Anticipatory horizon  (Formula 2)
# ===========================================================================

def fig3_anticipatory_horizon() -> None:
    """
    Line plot: attention on emergency-goal concepts (drone_battery, landing_zone,
    airspace, wind_speed) over simulated time as the queued goal ETA counts down
    from 30 → 0 and the goal auto-promotes.

    The growing attention before t=30 is the anticipatory pre-tuning signal.
    The vertical dotted line marks goal promotion.
    """
    eta_start   = 30.0
    lambda_h    = 0.1
    sim_end     = 40.0   # a few seconds past promotion to show the jump

    kb = build_pv_inspection_kb()
    am = AwarenessManager(
        kb, goal_id='inspect_pv_field',
        alpha=_ALPHA, max_distance=_MAX_DIST,
        budget=_BUDGET, observation_interval=_F3_INTERVAL,
        lambda_horizon=lambda_h,
    )
    am.queue_goal('emergency_landing', eta=eta_start, level='global')

    # Baseline: attention without any queued goal (pure Formula 1)
    kb_base = build_pv_inspection_kb()
    am_base = AwarenessManager(
        kb_base, goal_id='inspect_pv_field',
        alpha=_ALPHA, max_distance=_MAX_DIST,
        budget=_BUDGET, observation_interval=_F3_INTERVAL,
    )

    watch = ['drone_battery', 'landing_zone', 'airspace', 'wind_speed']
    watch_colors = [_C['battery'], _C['landing'], '#03A9F4', '#8BC34A']

    times = []
    attn_horizon  = {c: [] for c in watch}
    attn_baseline = {c: [] for c in watch}
    goal_promoted_at = None

    t = 0.0
    while t <= sim_end + 1e-9:
        am.tick(dt=_DT)
        am_base.tick(dt=_DT)

        if am.goal_id == 'emergency_landing' and goal_promoted_at is None:
            goal_promoted_at = t

        for c in watch:
            attn_horizon[c].append(am.attention().get(c, 0.0))
            attn_baseline[c].append(am_base.attention().get(c, 0.0))

        times.append(t)
        t = round(t + _DT, 10)

    times = np.array(times)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title(
        f'Fig 3 - Anticipatory horizon  (Formula 2: e^{{-λΔt}},  λ={lambda_h})\n'
        f'Emergency goal queued at t=0, ETA={eta_start}s  →  promotes at t≈{eta_start}s'
    )

    for c, col in zip(watch, watch_colors):
        ax.plot(times, attn_horizon[c],  color=col, linewidth=1.8, label=f'{c}  (with queue)')
        ax.plot(times, attn_baseline[c], color=col, linewidth=1.0, linestyle='--', alpha=0.45)

    # Vertical line at promotion
    if goal_promoted_at is not None:
        ax.axvline(goal_promoted_at, color='#555', linewidth=1.2, linestyle=':')
        ax.text(goal_promoted_at + 0.3, 0.96, 'goal promoted',
                va='top', fontsize=8.5, color='#555', transform=ax.get_xaxis_transform())

    # Legend: explain dashed = baseline
    from matplotlib.lines import Line2D
    legend_extras = [
        Line2D([0], [0], color='grey', linewidth=1.8, label='solid = with queued goal'),
        Line2D([0], [0], color='grey', linewidth=1.0, linestyle='--', alpha=0.6,
               label='dashed = inspect goal only (baseline)'),
    ]
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extras, fontsize=8.5, ncol=2)

    ax.set_xlabel('Simulated time  (s)')
    ax.set_ylabel('Attention  A(c)')
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, sim_end)

    fig.tight_layout()
    path = _OUT / 'fig3_anticipatory_horizon.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 3]  saved → {path}')

    # Console summary: attention at t=0, t=15, t=29, t=30, t=35
    checkpoints = [0.0, 10.0, 20.0, 29.0, eta_start, eta_start + 5.0]
    print(f'         Attention on drone_battery at key timepoints (λ={lambda_h}):')
    print(f'         {"t":>6}   {"with queue":>12}   {"baseline":>10}   {"boost":>8}')
    for cp in checkpoints:
        idx = int(round(cp / _DT))
        idx = min(idx, len(times) - 1)
        h = attn_horizon['drone_battery'][idx]
        b = attn_baseline['drone_battery'][idx]
        print(f'         {cp:>6.1f}   {h:>12.4f}   {b:>10.4f}   {h-b:>+8.4f}')


# ===========================================================================
# Figure 4 - Memory budget vs attention window  (Formula 4)
# ===========================================================================

def fig4_memory_budget() -> None:
    """
    Two-panel plot:
        Left  - effective_max_distance = sqrt(B) - 1 vs budget B
        Right - number of concepts in attention window vs budget B

    Confirms Formula 4: depth = √B - 1 and that the window grows predictably.
    """
    budgets = list(range(1, 51))

    kb = build_pv_inspection_kb()
    concept_ids = kb.concept_ids()

    depths, window_sizes = [], []
    for B in budgets:
        am = AwarenessManager(
            kb, goal_id='inspect_pv_field',
            alpha=_ALPHA, memory_budget=B,
            budget=_BUDGET, observation_interval=_F3_INTERVAL,
        )
        d = am.effective_max_distance
        attn = am.attention()
        n_in_window = sum(1 for c in concept_ids if attn.get(c, 0.0) > 0.0)
        depths.append(d)
        window_sizes.append(n_in_window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        'Fig 4 - Memory budget constraint  (Formula 4: depth = √B - 1)',
        fontsize=11,
    )

    # Left: depth vs B
    ax1.plot(budgets, depths, color='#2196F3', linewidth=2, marker='o', markersize=3)
    # Overlay theoretical curve
    b_cont = np.linspace(1, 50, 200)
    ax1.plot(b_cont, np.sqrt(b_cont) - 1, color='#2196F3', linewidth=0.8,
             linestyle='--', alpha=0.5, label='√B - 1  (theoretical)')
    ax1.set_xlabel('Memory budget  B')
    ax1.set_ylabel('Effective max distance  (depth)')
    ax1.set_title('Search depth derived from budget')
    ax1.legend(fontsize=8.5)
    ax1.set_xlim(0, 51)
    ax1.set_ylim(bottom=0)

    # Right: window size vs B
    ax2.plot(budgets, window_sizes, color='#4CAF50', linewidth=2, marker='o', markersize=3)
    ax2.axhline(len(concept_ids), color='#9E9E9E', linewidth=1, linestyle='--',
                label=f'total concepts = {len(concept_ids)}')
    ax2.set_xlabel('Memory budget  B')
    ax2.set_ylabel('Concepts in attention window')
    ax2.set_title('Attention window size grows with budget')
    ax2.legend(fontsize=8.5)
    ax2.set_xlim(0, 51)
    ax2.set_ylim(0, len(concept_ids) + 1)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()
    path = _OUT / 'fig4_memory_budget.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 4]  saved → {path}')

    # Console table
    print(f'         {"B":>4}  {"depth":>7}  {"in window":>10}')
    for B, d, w in zip(budgets[::5], depths[::5], window_sizes[::5]):
        print(f'         {B:>4}  {d:>7.2f}  {w:>10}')


# ===========================================================================
# Figure 8 - Certainty threshold: refresh count vs epistemic error  (Phase 4)
# ===========================================================================

def fig8_certainty_threshold() -> None:
    """
    Two-panel figure illustrating the trade-off controlled by certainty_threshold.

    The Probabilistic Forgetting gate suppresses scheduling for concepts whose
    epistemic error E ≤ threshold - i.e. concepts that are already known
    well enough. A tighter threshold allows more budget to be freed; a looser
    one behaves like the vanilla AM.

    Simulation:
        PV-inspection scenario, goal=inspect_pv_field, 60 s at 0.1 s/tick.
        Budget=2, observation_interval=10 s (Formula 3 calibrated to drift rate).
        An observation event fires every 2 real ticks (0.2 sim-seconds).

    Left panel:  cumulative refresh count over time for each threshold.
    Right panel: mean epistemic error of attended concepts over time.

    Thresholds compared:
        τ = 0.00  (baseline - no gate)
        τ = 0.03
        τ = 0.07
        τ = 0.12
        τ = 0.20  (aggressive forgetting)
    """
    from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb

    kb = build_pv_inspection_kb()

    thresholds  = [0.00, 0.03, 0.07, 0.12, 0.20]
    colors      = ['#607D8B', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
    labels      = [f'τ = {τ:.2f}' for τ in thresholds]

    sim_duration    = 60.0
    dt              = 0.1
    obs_every_ticks = 2    # fire observation every N ticks
    n_ticks         = int(sim_duration / dt)
    times           = np.arange(n_ticks) * dt

    fig, (ax_count, ax_err) = plt.subplots(1, 2, figsize=(10, 4))

    for τ, color, label in zip(thresholds, colors, labels):
        am = AwarenessManager(
            kb,
            goal_id='inspect_pv_field',
            budget=_BUDGET,
            alpha=_ALPHA,
            observation_interval=_F3_INTERVAL,
            certainty_threshold=τ,
        )

        cum_refreshes: list[int] = []
        mean_errors:   list[float] = []
        total_refreshes = 0

        for tick in range(n_ticks):
            schedule = am.tick(dt=dt)

            if tick % obs_every_ticks == 0:
                for cid in schedule:
                    am.observe(cid)
                    total_refreshes += 1

            cum_refreshes.append(total_refreshes)

            # Mean E of currently attended concepts (A > 0.05)
            attn = am.attention()
            attended_errors = [
                am._kb.get_concept(cid).epistemic_error
                for cid in kb.concept_ids()
                if attn.get(cid, 0.0) > 0.05 and am._kb.get_concept(cid).decay_rate > 0
            ]
            mean_errors.append(float(np.mean(attended_errors)) if attended_errors else 0.0)

        ax_count.plot(times, cum_refreshes, color=color, label=label, linewidth=1.8)
        ax_err.plot(  times, mean_errors,   color=color, label=label, linewidth=1.8)

    ax_count.set_xlabel('Simulation time  (s)')
    ax_count.set_ylabel('Cumulative refresh count')
    ax_count.set_title('Budget usage: refreshes over time')
    ax_count.legend(fontsize=8.5)
    ax_count.set_xlim(0, sim_duration)

    ax_err.set_xlabel('Simulation time  (s)')
    ax_err.set_ylabel('Mean epistemic error  E')
    ax_err.set_title('Information quality: mean E of attended concepts')
    ax_err.legend(fontsize=8.5)
    ax_err.set_xlim(0, sim_duration)
    ax_err.set_ylim(bottom=0)

    fig.suptitle(
        'Probabilistic Forgetting - certainty threshold trade-off\n'
        'Higher τ skips already-known concepts, freeing budget for uncertain ones',
        fontsize=10,
    )
    fig.tight_layout()
    path = _OUT / 'fig8_certainty_threshold.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 8]  saved → {path}')

    # Console summary: final refresh counts
    print(f'         {"threshold":>10}  {"total refreshes":>16}')
    for τ, label in zip(thresholds, labels):
        am = AwarenessManager(
            kb, goal_id='inspect_pv_field', budget=_BUDGET,
            alpha=_ALPHA, observation_interval=_F3_INTERVAL,
            certainty_threshold=τ,
        )
        count = 0
        for tick in range(n_ticks):
            schedule = am.tick(dt=dt)
            if tick % obs_every_ticks == 0:
                for cid in schedule:
                    am.observe(cid)
                    count += 1
        print(f'         {τ:>10.2f}  {count:>16}')


# ===========================================================================
# Figure 9 - Hierarchical attention blending over time  (Phase 5)
# ===========================================================================

def fig9_hierarchical_horizons() -> None:
    """
    Shows how three hierarchy levels (global / phase / task) blend differently
    into current attention as a queued goal's ETA counts down to zero.

    Scenario: PV inspection is active. 'emergency_landing' is queued at
    three different ETAs (60 s) using each level in turn. We plot the
    attention on 'drone_battery' (a concept that gains attention under
    emergency_landing) as a function of ETA ticking down from 60 s to 0.

    At ETA=0 the goal promotes; before that the anticipatory boost is purely
    from the queued entry at each level's λ rate.

    Left panel:  attention on 'drone_battery' vs ETA remaining (for each level)
    Right panel: attention on 'drone_battery' vs wall-clock sim time (same run)
    """
    from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb

    concept_of_interest = 'drone_battery'
    eta_start = 60.0
    levels = ['global', 'phase', 'task']
    level_colors = {'global': '#607D8B', 'phase': '#FF9800', 'task': '#F44336'}
    lambdas     = {'global': 0.05, 'phase': 0.20, 'task': 0.50}

    # For the ETA-sweep panel: compute analytically
    # A_boost(eta) = e^{-λ x eta} x A_emergency(drone_battery)
    kb_ref = build_pv_inspection_kb()
    am_ref = AwarenessManager(kb_ref, goal_id='emergency_landing', budget=5, alpha=_ALPHA)
    a_emergency = am_ref.attention().get(concept_of_interest, 0.0)

    kb_base = build_pv_inspection_kb()
    am_base = AwarenessManager(kb_base, goal_id='inspect_pv_field', budget=5, alpha=_ALPHA)
    a_current = am_base.attention().get(concept_of_interest, 0.0)

    etas = np.linspace(eta_start, 0.0, 300)

    fig, (ax_eta, ax_time) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left panel: analytic attention vs ETA ─────────────────────────────
    for lvl in levels:
        lam = lambdas[lvl]
        discounts = np.exp(-lam * etas)
        attn_vals = np.minimum(1.0, a_current + discounts * a_emergency)
        ax_eta.plot(etas, attn_vals, color=level_colors[lvl],
                    label=f"level='{lvl}'  (λ={lam})", linewidth=2)

    ax_eta.axhline(a_current,   color='#2196F3', linestyle='--', linewidth=1,
                   label=f'current goal baseline  (A={a_current:.2f})')
    ax_eta.axhline(a_emergency, color='#4CAF50', linestyle='--', linewidth=1,
                   label=f'target goal maximum  (A={a_emergency:.2f})')
    ax_eta.set_xlabel('ETA to emergency_landing  (s)')
    ax_eta.set_ylabel(f'Attention on {concept_of_interest}')
    ax_eta.set_title('Anticipatory boost vs ETA (analytic)')
    ax_eta.invert_xaxis()   # time flows left → right: far future on left
    ax_eta.legend(fontsize=8.5)
    ax_eta.set_xlim(eta_start, 0)
    ax_eta.set_ylim(0, 1.05)

    # ── Right panel: simulated attention over wall-clock time ─────────────
    dt        = 0.1
    n_ticks   = int((eta_start + 5.0) / dt)  # 5 s extra post-promotion
    times_all = np.arange(n_ticks) * dt

    for lvl in levels:
        kb = build_pv_inspection_kb()
        am = AwarenessManager(kb, goal_id='inspect_pv_field', budget=5, alpha=_ALPHA)
        am.queue_goal('emergency_landing', eta=eta_start, level=lvl)

        attn_series = []
        for _ in range(n_ticks):
            am.tick(dt=dt)
            attn_series.append(am.attention().get(concept_of_interest, 0.0))

        ax_time.plot(times_all, attn_series, color=level_colors[lvl],
                     label=f"level='{lvl}'", linewidth=1.8)

    ax_time.axvline(eta_start, color='#555', linestyle=':', linewidth=1,
                    label='goal promoted (t=60 s)')
    ax_time.set_xlabel('Simulation time  (s)')
    ax_time.set_ylabel(f'Attention on {concept_of_interest}')
    ax_time.set_title('Simulated attention over time')
    ax_time.legend(fontsize=8.5)
    ax_time.set_xlim(0, eta_start + 5.0)
    ax_time.set_ylim(0, 1.05)

    fig.suptitle(
        'Hierarchical Mission Horizons - global / phase / task level comparison\n'
        f'Queued goal: emergency_landing at ETA={eta_start} s   '
        f'Concept tracked: {concept_of_interest}',
        fontsize=10,
    )
    fig.tight_layout()
    path = _OUT / 'fig9_hierarchical_horizons.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'[Fig 9]  saved → {path}')

    # Console: attention on drone_battery at selected ETAs for each level
    print(f'         {"ETA (s)":>8}  ' + '  '.join(f'{lvl:>8}' for lvl in levels))
    for eta_sample in [60.0, 30.0, 10.0, 5.0, 1.0]:
        row = f'         {eta_sample:>8.1f}  '
        for lvl in levels:
            lam = lambdas[lvl]
            a = min(1.0, a_current + math.exp(-lam * eta_sample) * a_emergency)
            row += f'{a:>10.4f}'
        print(row)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print('=' * 60)
    print('  Awareness Manager - Evaluation')
    print(f'  Output directory: {_OUT}')
    print('=' * 60)

    fig1_attention_reshuffle()
    print()
    fig2_scheduling_comparison()
    print()
    fig3_anticipatory_horizon()
    print()
    fig4_memory_budget()
    print()
    fig8_certainty_threshold()
    print()
    fig9_hierarchical_horizons()

    print()
    print('=' * 60)
    print(f'  All figures saved to {_OUT}')
    print('=' * 60)


if __name__ == '__main__':
    main()
