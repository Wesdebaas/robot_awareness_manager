"""
report.py - Report generator for batch experiments.

Reads a manifest.json produced by batch.run_experiment(), computes all
six metrics for every run, and writes:

  summary.csv        - one row per run, one column per scalar metric
  plots.html         - interactive Plotly figures:
                         (1) mean E relevant/irrelevant per strategy (bar + scatter)
                         (2) cognitive lag distribution per strategy (box/strip)
                         (3) per-tick E timeseries averaged over param settings
                         (4) budget utilisation vs observation_interval (heatmap)
  summary.md         - key numbers + pairwise Mann-Whitney U tests (AM vs reactive,
                       AM vs always_on) for M1 and M4

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
# Strategy display config
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
    rows = []
    for run in manifest["runs"]:
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
        row = {
            "strategy":            run["strategy"],
            "budget":              params.get("budget", -1),
            "observation_interval": params.get("observation_interval", "?"),
            "alpha":               params.get("alpha", 0.5),
            "trace_dir":           run["trace_dir"],
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
    strategies = manifest.get("strategies", sorted({r["strategy"] for r in rows}))
    figs = []

    # ── Figure 1: Mean E relevant vs irrelevant per strategy ──────────────
    fig1 = go.Figure()
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        if not sr:
            continue
        rel_vals  = [r["m1_e_relevant"]   for r in sr if r["m1_e_relevant"]   is not None]
        irr_vals  = [r["m2_e_irrelevant"] for r in sr if r["m2_e_irrelevant"] is not None]
        color = _STRATEGY_COLORS.get(strat, "#aaaaaa")
        label = _STRATEGY_LABELS.get(strat, strat)
        if rel_vals:
            fig1.add_trace(go.Box(
                y=rel_vals, name=f"{label} (relevant)",
                marker_color=color, boxmean=True, legendgroup=strat,
            ))
        if irr_vals:
            fig1.add_trace(go.Box(
                y=irr_vals, name=f"{label} (irrelevant)",
                marker_color=color, boxmean=True, legendgroup=strat,
                fillcolor="rgba(0,0,0,0)",
            ))
    fig1.update_layout(
        title="M1/M2 - Mean Epistemic Error: Relevant vs Irrelevant Concepts",
        yaxis_title="Mean E (lower = fresher knowledge)",
        yaxis=dict(rangemode="tozero"),
        legend=dict(groupclick="toggleitem"),
        height=420,
    )
    figs.append(fig1)

    # ── Figure 2: Cognitive lag distribution ──────────────────────────────
    fig2 = go.Figure()
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        lag_vals = [r["m4_lag_seconds"] for r in sr
                    if r.get("m4_lag_seconds") is not None]
        label = _STRATEGY_LABELS.get(strat, strat)
        color = _STRATEGY_COLORS.get(strat, "#aaaaaa")
        if lag_vals:
            fig2.add_trace(go.Box(
                y=lag_vals, name=label,
                marker_color=color, boxmean=True,
                boxpoints="all", jitter=0.3, pointpos=-1.8,
            ))
    fig2.update_layout(
        title="M4 - Cognitive Lag at Goal Transition (seconds to E < 0.1)",
        yaxis_title="Lag (s) - lower is better, None = never recovered",
        yaxis=dict(rangemode="tozero"),
        height=380,
    )
    figs.append(fig2)

    # ── Figure 3: M1 E_relevant vs observation_interval, one line/strategy ─
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
                  and r["m1_e_relevant"] is not None]
            if not sr:
                continue
            vals = [r["m1_e_relevant"] for r in sr]
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
        title="M1 - Mean E (Relevant) vs Observation Interval",
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
        z, oi_labels, b_labels = [], [], []
        for oi in oi_vals:
            row_z = []
            for b in budget_vals:
                sr = [r for r in rows
                      if r["strategy"] == strat and r["observation_interval"] == oi
                      and r["budget"] == b and r["m3_budget_util"] is not None]
                row_z.append(statistics.mean(r["m3_budget_util"] for r in sr) if sr else None)
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
            title=f"M3 - Budget Utilisation ({label})",
            xaxis_title="Budget B",
            yaxis_title="Observation interval T (s)",
            height=320,
        )
        figs.append(fig4)

    # ── Combine into single HTML ──────────────────────────────────────────
    html_parts = ["<html><head><meta charset='utf-8'></head><body>"]
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts), encoding="utf-8")


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
        "| Strategy | B | T | M1 E_rel ↓ | M2 E_irrel | M4 lag(s) ↓ | M5 cache | M6 rel% |",
        "|----------|---|---|-----------|-----------|------------|---------|--------|",
    ]
    for r in sorted(rows, key=lambda x: (x["strategy"], x["budget"], x["observation_interval"])):
        m4 = f"{r['m4_lag_seconds']:.2f}" if r.get("m4_lag_seconds") is not None else "-"
        m5 = f"{r['m5_cache_hit_rate']:.2f}" if r.get("m5_cache_hit_rate") is not None else "-"
        lines.append(
            f"| {r['strategy']} | {r['budget']} | {r['observation_interval']} "
            f"| {r['m1_e_relevant']:.4f} | {r['m2_e_irrelevant']:.4f} "
            f"| {m4} | {m5} | {r['m6_relevant_fraction']:.2%} |"
        )

    lines += ["", "## Aggregate per Strategy", ""]
    for strat in strategies:
        sr = [r for r in rows if r["strategy"] == strat]
        if not sr:
            continue
        label = _STRATEGY_LABELS.get(strat, strat)
        rel  = [r["m1_e_relevant"]   for r in sr if r["m1_e_relevant"]  is not None]
        irr  = [r["m2_e_irrelevant"] for r in sr if r["m2_e_irrelevant"] is not None]
        lags = [r["m4_lag_seconds"]  for r in sr if r["m4_lag_seconds"]  is not None]
        n_no_recover = sum(1 for r in sr if r.get("m4_lag_seconds") is None
                           and r.get("m4_e_at_transition") is not None)

        def _fmt(vals):
            if not vals:
                return "-"
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mu:.4f} ± {sd:.4f}"

        lines += [
            f"### {label}",
            f"- **M1 E_relevant** (mean ± sd): {_fmt(rel)}",
            f"- **M2 E_irrelevant** (mean ± sd): {_fmt(irr)}",
            f"- **M4 lag** (mean ± sd): {_fmt(lags)}"
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
            "| Strategy | M1 E_rel ↓ | M2 E_irrel | M4 lag (s) ↓ | M4 no-recover |",
            "|----------|-----------|-----------|------------|--------------|",
        ]
        for strat in strategies:
            sr = [r for r in sr_oi if r["strategy"] == strat]
            if not sr:
                continue
            rel  = [r["m1_e_relevant"]  for r in sr if r["m1_e_relevant"]  is not None]
            lags = [r["m4_lag_seconds"] for r in sr if r["m4_lag_seconds"] is not None]
            n_no = sum(1 for r in sr if r.get("m4_lag_seconds") is None
                       and r.get("m4_e_at_transition") is not None)
            label = _STRATEGY_LABELS.get(strat, strat)
            m1_s = f"{statistics.mean(rel):.4f}" if rel else "-"
            m2_vals = [r["m2_e_irrelevant"] for r in sr if r["m2_e_irrelevant"] is not None]
            m2_s = f"{statistics.mean(m2_vals):.4f}" if m2_vals else "-"
            m4_s = f"{statistics.mean(lags):.2f}" if lags else "-"
            lines.append(
                f"| {label} | {m1_s} | {m2_s} | {m4_s} | {n_no} |"
            )
        lines.append("")

        # Per-T Mann-Whitney U (AM vs others), only for T > 1.0
        ref_sr = [r for r in sr_oi if r["strategy"] == "awareness_manager"]
        if ref_sr:
            for strat in strategies:
                if strat == "awareness_manager":
                    continue
                other_sr = [r for r in sr_oi if r["strategy"] == strat]
                if not other_sr:
                    continue
                label = _STRATEGY_LABELS.get(strat, strat)
                a_rel = [r["m1_e_relevant"] for r in ref_sr if r["m1_e_relevant"] is not None]
                b_rel = [r["m1_e_relevant"] for r in other_sr if r["m1_e_relevant"] is not None]
                u_rel, p_rel = _mannwhitney(a_rel, b_rel)
                a_lag = [r["m4_lag_seconds"] for r in ref_sr if r["m4_lag_seconds"] is not None]
                b_lag = [r["m4_lag_seconds"] for r in other_sr if r["m4_lag_seconds"] is not None]
                u_lag, p_lag = _mannwhitney(a_lag, b_lag)
                sig_rel = ("✓ sig." if not _isnan(p_rel) and p_rel < 0.05
                           else ("✗ n.s." if not _isnan(p_rel) else "-"))
                sig_lag = ("✓ sig." if not _isnan(p_lag) and p_lag < 0.05
                           else ("✗ n.s." if not _isnan(p_lag) else "-"))
                lines.append(
                    f"_AM vs {label}: M1 U={_fmtf(u_rel)}, p={_fmtf(p_rel)} {sig_rel}; "
                    f"M4 U={_fmtf(u_lag)}, p={_fmtf(p_lag)} {sig_lag}_"
                )
            lines.append("")

    # ── Pooled pairwise tests ─────────────────────────────────────────────
    lines += ["## Statistical Comparison - Pooled (Mann-Whitney U)", ""]
    lines += [
        "_Pooled over all B and T settings. The T=1.0 regime inflates variance_",
        "_and suppresses pooled M1 significance; see per-regime breakdown above._",
        "",
    ]
    ref = "awareness_manager"
    ref_rows = [r for r in rows if r["strategy"] == ref]
    for strat in strategies:
        if strat == ref:
            continue
        other_rows = [r for r in rows if r["strategy"] == strat]
        label = _STRATEGY_LABELS.get(strat, strat)

        a_rel = [r["m1_e_relevant"] for r in ref_rows if r["m1_e_relevant"] is not None]
        b_rel = [r["m1_e_relevant"] for r in other_rows if r["m1_e_relevant"] is not None]
        u_rel, p_rel = _mannwhitney(a_rel, b_rel)

        a_lag = [r["m4_lag_seconds"] for r in ref_rows if r["m4_lag_seconds"] is not None]
        b_lag = [r["m4_lag_seconds"] for r in other_rows if r["m4_lag_seconds"] is not None]
        u_lag, p_lag = _mannwhitney(a_lag, b_lag)

        sig_rel = "✓ significant (p < 0.05)" if (not _isnan(p_rel) and p_rel < 0.05) else \
                  ("✗ not significant" if not _isnan(p_rel) else "(scipy unavailable)")
        sig_lag = "✓ significant (p < 0.05)" if (not _isnan(p_lag) and p_lag < 0.05) else \
                  ("✗ not significant" if not _isnan(p_lag) else "(scipy unavailable)")

        lines += [
            f"### AM vs {label}",
            f"- **M1 E_relevant**: U={_fmtf(u_rel)}, p={_fmtf(p_rel)}  - {sig_rel}",
            f"- **M4 lag**: U={_fmtf(u_lag)}, p={_fmtf(p_lag)}  - {sig_lag}",
            "",
        ]

    # ── M5 and M6 interpretation notes ───────────────────────────────────
    lines += [
        "## Metric Notes",
        "",
        "### M5 (Anticipatory Cache Hit Rate)",
        "The M5 recency window is `observation_interval / 2` seconds (converted to ticks).",
        "AM and reactive both score 1.00 because `inspect_pv_field` and `emergency_landing`",
        "share several concepts (drone_battery, wind_speed) that are scheduled throughout",
        "the trace by both strategies.  M5 cannot discriminate in this scenario because",
        "the two goal neighbourhoods overlap significantly.",
        "",
        "### M6 (Relevant-Fraction of Schedule Slots)",
        "M6 = 100% for AM and reactive is **correct and expected**, not a metric artefact.",
        "The PV inspection scenario has two goals whose 1-hop neighbourhoods together cover",
        "8 of 9 decaying class concepts.  Only `panel_row` lies outside the mission",
        "neighbourhood - and it is never scheduled by any strategy because it sits 2+ hops",
        "from both goals.  M6 would only discriminate in scenarios with a larger fraction",
        "of irrelevant nodes (sparser graphs, or goals covering a small fraction of the KB).",
        "",
        "### M4 (Cognitive Lag) - did-not-recover exclusion",
        "Runs where E_max never drops below the freshness threshold (0.10) after a goal",
        "transition are recorded as lag = None and are **excluded** from the mean_lag_seconds",
        "computation (they are not counted as 0 or as any finite value).  These are the",
        "T = 1.0 s runs for all strategies, reported separately as 'no-recover' counts.",
        "",
        "---",
        "_Non-parametric test (Mann-Whitney U, two-tailed). Small N - interpret p-values with caution._",
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
