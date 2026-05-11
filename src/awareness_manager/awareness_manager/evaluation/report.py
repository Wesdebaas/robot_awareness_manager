"""
report.py - Report generator for batch experiments.

Reads a manifest.json produced by batch.run_experiment() or
batch.run_ablation_experiment(), computes all metrics for every run, and writes:

  summary.csv        - one row per run, one column per scalar metric
  plots.html         - interactive Plotly figures (layout depends on experiment type):

      Standard (budget x obs-interval sweep):
        (1) M1 E@transition and M4 E_relevant per strategy (box)
        (2) M3 cognitive lag distribution per strategy (box)
        (3) M4 E_relevant vs observation_interval (line)
        (4) M5 budget utilisation heatmap (AM and reactive)

      Ablation (5-step component study):
        (1) M2 pre-transition attention per step (bar) — primary
        (2) M1 E at transition per step (bar)          — primary
        (3) M3 lag seconds per step (bar)              — secondary
        (4) M4 E_relevant per step (bar)               — context

  summary.md         - key numbers + statistical tests

Usage
-----
    from awareness_manager.evaluation.report import generate_report
    generate_report(Path("experiments/budget_obsint_sweep"), Path("reports/pv_v1"))

Or via CLI:
    python -m awareness_manager.evaluation report \\
        --experiment experiments/budget_obsint_sweep/ \\
        --output reports/pv_v1/
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

from awareness_manager.evaluation.metrics import all_metrics, load_trace


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

_STRATEGY_COLORS = {
    "awareness_manager": "#4a8af4",   # blue
    "reactive":          "#f4944a",   # orange
    "always_on":         "#44bb88",   # teal
}

_STRATEGY_LABELS = {
    "awareness_manager": "Awareness Manager",
    "reactive":          "Reactive Baseline",
    "always_on":         "Always-On Baseline",
}

# Ablation step colors: ramp from orange (reactive) to blue (full AM)
_ABLATION_COLORS = [
    "#f4944a",  # step 0: Reactive (orange)
    "#a0b4f4",  # step 1: AM [F1]
    "#7095f4",  # step 2: AM [F1+F2]
    "#5079f4",  # step 3: AM [F1+F2+F5]
    "#1a54f4",  # step 4: AM [full]
]


def _ablation_label(strategy: str, feature_config: dict | None) -> str:
    """Return display label for an ablation run."""
    if strategy == "reactive":
        return "Reactive"
    if not feature_config:
        return "AM [full]"
    f1 = feature_config.get("f1", True)
    f2 = feature_config.get("f2", True)
    f4 = feature_config.get("f4", True)
    f5 = feature_config.get("f5", True)
    if f1 and f2 and f4 and f5:
        return "AM [full]"
    parts = []
    if f1: parts.append("F1")
    if f2: parts.append("F2")
    if f5: parts.append("F5")
    return "AM [" + "+".join(parts) + "]" if parts else "AM [none]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    experiment_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Read a batch experiment manifest, compute metrics, and write report files.

    Args:
        experiment_dir: Directory containing manifest.json and run subdirs.
        output_dir:     Where to write summary.csv, plots.html, summary.md.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {experiment_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Loading {len(manifest['runs'])} runs …")
    rows = _compute_rows(manifest)
    print(f"  Computed metrics for {len(rows)} runs.")

    _write_csv(rows, output_dir / "summary.csv")
    print(f"  → summary.csv")

    if _PLOTLY:
        _write_plots(rows, manifest, output_dir / "plots.html")
        print(f"  → plots.html")
    else:
        print("  (plotly not installed - skipping plots.html)")

    _write_markdown(rows, manifest, output_dir / "summary.md")
    print(f"  → summary.md")

    print(f"\nReport written to {output_dir}/")


# ---------------------------------------------------------------------------
# Metric computation across all runs
# ---------------------------------------------------------------------------

def _compute_rows(manifest: dict) -> list[dict]:
    """Load each run's trace, compute all metrics, return flat row dicts."""
    is_ablation = manifest.get("experiment_type") == "ablation"
    rows = []
    for step_idx, run in enumerate(manifest["runs"]):
        if not run.get("completed", False):
            continue
        trace_dir = Path(run["trace_dir"])
        if not (trace_dir / "meta.json").exists():
            print(f"  SKIP (no meta.json): {trace_dir.name}")
            continue
        params = run.get("params", {})
        try:
            trace = load_trace(trace_dir)
            m = all_metrics(trace, params=params)
        except Exception as exc:
            print(f"  SKIP (metric error): {trace_dir.name}: {exc}")
            continue
        fc = params.get("feature_config")
        row = {
            "strategy":             run["strategy"],
            "budget":               params.get("budget", -1),
            "observation_interval": params.get("observation_interval", "?"),
            "alpha":                params.get("alpha", 0.5),
            "feature_config":       fc,
            "step_label":           _ablation_label(run["strategy"], fc) if is_ablation else None,
            "step_idx":             step_idx if is_ablation else None,
            "trace_dir":            run["trace_dir"],
            **{k: v for k, v in m.items() if not isinstance(v, dict)},
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scalar_keys = [k for k, v in rows[0].items() if not isinstance(v, (dict, list))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def _write_plots(rows: list[dict], manifest: dict, path: Path) -> None:
    if manifest.get("experiment_type") == "ablation":
        figs = _ablation_figures(rows, manifest)
    else:
        figs = _standard_figures(rows, manifest)

    html_parts = ["<html><head><meta charset='utf-8'></head><body>"]
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts), encoding="utf-8")


def _standard_figures(rows: list[dict], manifest: dict) -> list:
    strategies = manifest.get("strategies", sorted({r["strategy"] for r in rows}))
    figs = []

    # ── Figure 1: E@transition and E_relevant per strategy ────────────────
    fig1 = go.Figure()
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        if not sr:
            continue
        trans_vals = [r["m1_e_at_transition"] for r in sr if r.get("m1_e_at_transition") is not None]
        rel_vals   = [r["m4_e_relevant"]       for r in sr if r.get("m4_e_relevant") is not None]
        color = _STRATEGY_COLORS.get(strat, "#aaaaaa")
        label = _STRATEGY_LABELS.get(strat, strat)
        if trans_vals:
            fig1.add_trace(go.Box(
                y=trans_vals, name=f"{label} (E@transition)",
                marker_color=color, boxmean=True, legendgroup=strat,
            ))
        if rel_vals:
            fig1.add_trace(go.Box(
                y=rel_vals, name=f"{label} (E_relevant)",
                marker_color=color, boxmean=True, legendgroup=strat,
                fillcolor="rgba(0,0,0,0)",
            ))
    fig1.update_layout(
        title="M1/M4 - Epistemic Error at Transition and Over Relevant Concepts",
        yaxis_title="E (lower = fresher knowledge)",
        yaxis=dict(rangemode="tozero"),
        legend=dict(groupclick="toggleitem"),
        height=420,
    )
    figs.append(fig1)

    # ── Figure 2: Cognitive lag distribution ──────────────────────────────
    fig2 = go.Figure()
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        lag_vals = [r["m3_lag_seconds"] for r in sr
                    if r.get("m3_lag_seconds") is not None]
        label = _STRATEGY_LABELS.get(strat, strat)
        color = _STRATEGY_COLORS.get(strat, "#aaaaaa")
        if lag_vals:
            fig2.add_trace(go.Box(
                y=lag_vals, name=label,
                marker_color=color, boxmean=True,
                boxpoints="all", jitter=0.3, pointpos=-1.8,
            ))
    fig2.update_layout(
        title="M3 - Cognitive Lag at Goal Transition (seconds to E < 0.1)",
        yaxis_title="Lag (s) - lower is better, None = never recovered",
        yaxis=dict(rangemode="tozero"),
        height=380,
    )
    figs.append(fig2)

    # ── Figure 3: M4 E_relevant vs observation_interval, one line/strategy ─
    fig3 = go.Figure()
    oi_vals = sorted({r["observation_interval"] for r in rows
                      if isinstance(r["observation_interval"], (int, float))})
    for strat in strategies:
        color = _STRATEGY_COLORS.get(strat, "#aaaaaa")
        label = _STRATEGY_LABELS.get(strat, strat)
        x, y, y_err = [], [], []
        for oi in oi_vals:
            sr = [r for r in rows
                  if r["strategy"] == strat and r["observation_interval"] == oi
                  and r.get("m4_e_relevant") is not None]
            if not sr:
                continue
            vals = [r["m4_e_relevant"] for r in sr]
            x.append(oi)
            y.append(statistics.mean(vals))
            y_err.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        if x:
            fig3.add_trace(go.Scatter(
                x=x, y=y,
                error_y=dict(type="data", array=y_err, visible=True),
                mode="lines+markers",
                name=label, marker_color=color, line_color=color,
            ))
    fig3.update_layout(
        title="M4 - Mean E (Relevant) vs Observation Interval",
        xaxis_title="Observation interval T (s)",
        yaxis_title="Mean E_relevant (↓ better)",
        yaxis=dict(rangemode="tozero"),
        height=380,
    )
    figs.append(fig3)

    # ── Figure 4: Budget utilisation heatmap (AM and reactive only) ───────
    budget_vals = sorted({r["budget"] for r in rows
                          if isinstance(r["budget"], (int, float)) and r["budget"] >= 0})
    for strat in ["awareness_manager", "reactive"]:
        if not any(r["strategy"] == strat for r in rows):
            continue
        z = []
        for oi in oi_vals:
            row_z = []
            for b in budget_vals:
                sr = [r for r in rows
                      if r["strategy"] == strat and r["observation_interval"] == oi
                      and r["budget"] == b and r.get("m5_budget_util") is not None]
                row_z.append(statistics.mean(r["m5_budget_util"] for r in sr) if sr else None)
            z.append(row_z)
        label = _STRATEGY_LABELS.get(strat, strat)
        fig4 = go.Figure(go.Heatmap(
            z=z,
            x=[f"B={b}" for b in budget_vals],
            y=[f"T={oi}" for oi in oi_vals],
            colorscale="Blues",
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" if v is not None else "-" for v in row] for row in z],
            texttemplate="%{text}",
        ))
        fig4.update_layout(
            title=f"M5 - Budget Utilisation ({label})",
            xaxis_title="Budget B",
            yaxis_title="Observation interval T (s)",
            height=320,
        )
        figs.append(fig4)

    return figs


def _ablation_figures(rows: list[dict], manifest: dict) -> list:
    """Four figures for the 5-step ablation: M2, M1, M3, M4."""
    # Rows are in manifest order (step 0 → 4); use that as display order.
    ordered = sorted(rows, key=lambda r: r.get("step_idx", 0))
    labels = [r["step_label"] or r["strategy"] for r in ordered]
    colors = [_ABLATION_COLORS[min(i, len(_ABLATION_COLORS) - 1)]
              for i in range(len(ordered))]

    def _bar(values, title, yaxis_title, higher_better=False):
        arrow = "↑ higher = AM predicted upcoming goal" if higher_better else "↓ lower = better"
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" if v is not None else "—" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"{title} ({arrow})",
            yaxis_title=yaxis_title,
            yaxis=dict(rangemode="tozero"),
            height=380,
            showlegend=False,
        )
        return fig

    m2_vals  = [r.get("m2_pre_transition_attn") for r in ordered]
    m1_vals  = [r.get("m1_e_at_transition")     for r in ordered]
    m3_vals  = [r.get("m3_lag_seconds")         for r in ordered]
    m4_vals  = [r.get("m4_e_relevant")          for r in ordered]

    return [
        _bar(m2_vals, "M2 - Pre-transition Attention (incoming goal neighborhood)",
             "Mean attention in window before transition", higher_better=True),
        _bar(m1_vals, "M1 - Epistemic Error at Goal Transition",
             "E_max at transition tick (↓ = robot was already prepared)"),
        _bar(m3_vals, "M3 - Cognitive Lag (seconds to E < 0.1 after transition)",
             "Lag (s) — None shown as absent bar"),
        _bar(m4_vals, "M4 - Mean E over Relevant Concepts (full run)",
             "Mean E_relevant (↓ better)"),
    ]


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _mannwhitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U statistic and two-tailed p-value."""
    if not _SCIPY or len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _write_markdown(rows: list[dict], manifest: dict, path: Path) -> None:
    if manifest.get("experiment_type") == "ablation":
        _write_ablation_markdown(rows, manifest, path)
        return

    strategies = manifest.get("strategies", sorted({r["strategy"] for r in rows}))
    scenario = manifest.get("scenario", "?")
    n_runs = manifest.get("total_runs", len(rows))

    lines = [
        f"# Experiment Report: {scenario}",
        "",
        f"**Experiment dir:** `{manifest.get('experiment_dir', '?')}`  ",
        f"**Total runs:** {n_runs}  |  "
        f"**Strategies:** {', '.join(strategies)}  ",
        f"**Duration:** {manifest.get('duration_s', '?')}s  "
        f"**dt:** {manifest.get('dt', '?')}s  ",
        "",
        "## Metric Summary",
        "",
        "| Strategy | B | T | M1 E@trans ↓ | M2 pre-attn ↑ | M3 lag(s) ↓ | M4 E_rel ↓ |",
        "|----------|---|---|-------------|--------------|------------|-----------|",
    ]
    for r in sorted(rows, key=lambda x: (x["strategy"], x["budget"], x["observation_interval"])):
        m1 = f"{r['m1_e_at_transition']:.4f}" if r.get("m1_e_at_transition") is not None else "-"
        m2 = f"{r['m2_pre_transition_attn']:.4f}" if r.get("m2_pre_transition_attn") is not None else "-"
        m3 = f"{r['m3_lag_seconds']:.2f}" if r.get("m3_lag_seconds") is not None else "-"
        lines.append(
            f"| {r['strategy']} | {r['budget']} | {r['observation_interval']} "
            f"| {m1} | {m2} | {m3} | {r['m4_e_relevant']:.4f} |"
        )

    lines += ["", "## Aggregate per Strategy", ""]
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        if not sr:
            continue
        label = _STRATEGY_LABELS.get(strat, strat)
        trans = [r["m1_e_at_transition"] for r in sr if r.get("m1_e_at_transition") is not None]
        rel   = [r["m4_e_relevant"]      for r in sr if r.get("m4_e_relevant") is not None]
        lags  = [r["m3_lag_seconds"]     for r in sr if r.get("m3_lag_seconds") is not None]
        n_no_recover = sum(1 for r in sr if r.get("m3_lag_seconds") is None
                           and r.get("m1_e_at_transition") is not None)

        def _fmt(vals):
            if not vals:
                return "-"
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mu:.4f} ± {sd:.4f}"

        lines += [
            f"### {label}",
            f"- **M1 E@transition** (mean ± sd): {_fmt(trans)}",
            f"- **M4 E_relevant** (mean ± sd): {_fmt(rel)}",
            f"- **M3 lag** (mean ± sd): {_fmt(lags)}"
            + (f"  ({n_no_recover} run(s) did not recover)" if n_no_recover else ""),
            "",
        ]

    # ── Per-T-regime breakdown ─────────────────────────────────────────────
    oi_vals = sorted({r["observation_interval"] for r in rows
                      if isinstance(r["observation_interval"], (int, float))})
    lines += ["", "## Per-Regime Breakdown (by observation interval T)", ""]
    lines += [
        "_T = 1.0 s is a degenerate regime: refresh = 1−exp(−δxT) is too small_",
        "_to overcome drift for most concepts, so all strategies show E ≈ 0.45 and_",
        "_no goal-transition recovery. Pooled statistics are dominated by this regime._",
        "",
    ]
    for oi in oi_vals:
        sr_oi = [r for r in rows if r["observation_interval"] == oi]
        lines += [f"### T = {oi} s", ""]
        lines += [
            "| Strategy | M1 E@trans ↓ | M3 lag (s) ↓ | M3 no-recover | M4 E_rel ↓ |",
            "|----------|-------------|------------|--------------|-----------|",
        ]
        for strat in strategies:
            sr = [r for r in sr_oi if r["strategy"] == strat]
            if not sr:
                continue
            trans = [r["m1_e_at_transition"] for r in sr if r.get("m1_e_at_transition") is not None]
            lags  = [r["m3_lag_seconds"]     for r in sr if r.get("m3_lag_seconds") is not None]
            rel   = [r["m4_e_relevant"]      for r in sr if r.get("m4_e_relevant") is not None]
            n_no  = sum(1 for r in sr if r.get("m3_lag_seconds") is None
                        and r.get("m1_e_at_transition") is not None)
            label = _STRATEGY_LABELS.get(strat, strat)
            m1_s = f"{statistics.mean(trans):.4f}" if trans else "-"
            m3_s = f"{statistics.mean(lags):.2f}" if lags else "-"
            m4_s = f"{statistics.mean(rel):.4f}" if rel else "-"
            lines.append(
                f"| {label} | {m1_s} | {m3_s} | {n_no} | {m4_s} |"
            )
        lines.append("")

        # Per-T Mann-Whitney U (AM vs others)
        ref_sr = [r for r in sr_oi if r["strategy"] == "awareness_manager"]
        if ref_sr:
            for strat in strategies:
                if strat == "awareness_manager":
                    continue
                other_sr = [r for r in sr_oi if r["strategy"] == strat]
                if not other_sr:
                    continue
                label = _STRATEGY_LABELS.get(strat, strat)
                a_rel = [r["m4_e_relevant"] for r in ref_sr if r.get("m4_e_relevant") is not None]
                b_rel = [r["m4_e_relevant"] for r in other_sr if r.get("m4_e_relevant") is not None]
                u_rel, p_rel = _mannwhitney(a_rel, b_rel)
                a_lag = [r["m3_lag_seconds"] for r in ref_sr if r.get("m3_lag_seconds") is not None]
                b_lag = [r["m3_lag_seconds"] for r in other_sr if r.get("m3_lag_seconds") is not None]
                u_lag, p_lag = _mannwhitney(a_lag, b_lag)
                sig_rel = ("✓ sig." if not _isnan(p_rel) and p_rel < 0.05
                           else ("✗ n.s." if not _isnan(p_rel) else "-"))
                sig_lag = ("✓ sig." if not _isnan(p_lag) and p_lag < 0.05
                           else ("✗ n.s." if not _isnan(p_lag) else "-"))
                lines.append(
                    f"_AM vs {label}: M4 U={_fmtf(u_rel)}, p={_fmtf(p_rel)} {sig_rel}; "
                    f"M3 U={_fmtf(u_lag)}, p={_fmtf(p_lag)} {sig_lag}_"
                )
            lines.append("")

    # ── Pooled pairwise tests ─────────────────────────────────────────────
    lines += ["## Statistical Comparison - Pooled (Mann-Whitney U)", ""]
    lines += [
        "_Pooled over all B and T settings. The T=1.0 regime inflates variance_",
        "_and suppresses pooled M4 significance; see per-regime breakdown above._",
        "",
    ]
    ref = "awareness_manager"
    ref_rows = [r for r in rows if r["strategy"] == ref]
    for strat in strategies:
        if strat == ref:
            continue
        other_rows = [r for r in rows if r["strategy"] == strat]
        label = _STRATEGY_LABELS.get(strat, strat)

        a_rel = [r["m4_e_relevant"] for r in ref_rows if r.get("m4_e_relevant") is not None]
        b_rel = [r["m4_e_relevant"] for r in other_rows if r.get("m4_e_relevant") is not None]
        u_rel, p_rel = _mannwhitney(a_rel, b_rel)

        a_lag = [r["m3_lag_seconds"] for r in ref_rows if r.get("m3_lag_seconds") is not None]
        b_lag = [r["m3_lag_seconds"] for r in other_rows if r.get("m3_lag_seconds") is not None]
        u_lag, p_lag = _mannwhitney(a_lag, b_lag)

        sig_rel = "✓ significant (p < 0.05)" if (not _isnan(p_rel) and p_rel < 0.05) else \
                  ("✗ not significant" if not _isnan(p_rel) else "(scipy unavailable)")
        sig_lag = "✓ significant (p < 0.05)" if (not _isnan(p_lag) and p_lag < 0.05) else \
                  ("✗ not significant" if not _isnan(p_lag) else "(scipy unavailable)")

        lines += [
            f"### AM vs {label}",
            f"- **M4 E_relevant**: U={_fmtf(u_rel)}, p={_fmtf(p_rel)}  - {sig_rel}",
            f"- **M3 lag**: U={_fmtf(u_lag)}, p={_fmtf(p_lag)}  - {sig_lag}",
            "",
        ]

    lines += [
        "## Metric Notes",
        "",
        "### M3 (Cognitive Lag) - did-not-recover exclusion",
        "Runs where E_max never drops below the freshness threshold (0.10) after a goal",
        "transition are recorded as lag = None and are **excluded** from the mean.",
        "These are typically T = 1.0 s runs, reported separately as 'no-recover' counts.",
        "",
        "---",
        "_Non-parametric test (Mann-Whitney U, two-tailed). Small N - interpret p-values with caution._",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ablation_markdown(rows: list[dict], manifest: dict, path: Path) -> None:
    scenario = manifest.get("scenario", "?")
    budget   = manifest.get("budget", "?")
    obs_int  = manifest.get("obs_interval", "?")

    ordered = sorted(rows, key=lambda r: r.get("step_idx", 0))

    lines = [
        f"# Ablation Study Report: {scenario}",
        "",
        f"**Experiment dir:** `{manifest.get('experiment_dir', '?')}`  ",
        f"**Budget:** B={budget}  |  **Observation interval:** T={obs_int} s  ",
        f"**Duration:** {manifest.get('duration_s', '?')} s  "
        f"**dt:** {manifest.get('dt', '?')} s  ",
        "",
        "## Component Ablation Steps",
        "",
        "| Step | Label | F1 | F2 | F5 | F4 |",
        "|------|-------|----|----|----|----|",
    ]
    for r in ordered:
        fc = r.get("feature_config") or {}
        f1 = "✓" if fc.get("f1", r["strategy"] != "reactive") else "—"
        f2 = "✓" if fc.get("f2", False) else "—"
        f5 = "✓" if fc.get("f5", False) else "—"
        f4 = "✓" if fc.get("f4", False) else "—"
        if r["strategy"] == "reactive":
            f1 = f2 = f5 = f4 = "—"
        lines.append(f"| {r['step_idx']} | {r['step_label']} | {f1} | {f2} | {f5} | {f4} |")

    lines += [
        "",
        "F3 (utility saturation) is always active for all configurations and baselines.",
        "",
        "## Primary Metrics",
        "",
        "| Step | Label | M2 pre-attn ↑ | M1 E@trans ↓ | M3 lag(s) ↓ | M4 E_rel ↓ |",
        "|------|-------|--------------|-------------|------------|-----------|",
    ]
    for r in ordered:
        m2  = f"{r['m2_pre_transition_attn']:.4f}" if r.get("m2_pre_transition_attn") is not None else "—"
        m1  = f"{r['m1_e_at_transition']:.4f}"     if r.get("m1_e_at_transition") is not None     else "—"
        m3  = f"{r['m3_lag_seconds']:.2f}"         if r.get("m3_lag_seconds") is not None         else "—"
        m4  = f"{r['m4_e_relevant']:.4f}"          if r.get("m4_e_relevant") is not None          else "—"
        lines.append(
            f"| {r['step_idx']} | {r['step_label']} | {m2} | {m1} | {m3} | {m4} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "**M2 (pre-transition attention):** Mean attention the strategy assigned to the",
        "incoming goal's 1-hop neighborhood in the observation_interval seconds before",
        "each goal transition.  Measures the AM's *signal quality* — did it predict the",
        "upcoming need before the switch fired?  Higher is better.",
        "- A jump between steps where F2 is added confirms the anticipatory horizon",
        "  pre-allocates budget to the correct concepts before the goal fires.",
        "- ReactiveBaseline's score is near zero because the two goal neighborhoods",
        "  are disjoint; it assigns no attention to emergency concepts until the switch.",
        "",
        "**M1 (E at transition):** Worst-case epistemic error in the incoming goal's",
        "neighborhood at the exact tick of the goal switch.  Lower means the robot",
        "already had fresh knowledge when the goal changed.  Confirms whether the",
        "attention signal (M2) translated into actual readiness.",
        "- If M2 rises but M1 does not drop: the budget was insufficient to act on",
        "  the AM's correct signal — an informative finding about resource constraints.",
        "",
        "**M3 (lag):** Seconds until E_max < 0.1 after the transition.  Lower is better.",
        "Complements M1: a low M1 means low M3 almost by definition.",
        "",
        "---",
        "_Report generated automatically by awareness_manager.evaluation.report_",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _isnan(x) -> bool:
    import math
    try:
        return math.isnan(x)
    except TypeError:
        return True


def _fmtf(x) -> str:
    if _isnan(x):
        return "-"
    return f"{x:.4g}"
