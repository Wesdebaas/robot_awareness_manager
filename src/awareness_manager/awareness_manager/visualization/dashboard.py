"""
dashboard.py - Dash + dash-cytoscape awareness-manager visualisation dashboard.

Supports two modes, selected by the type of source passed to build_app():
  Live mode   (SimulationRunner) - real-time AM simulation with rolling history.
  Replay mode (ReplayReader)     - offline trace playback with scrubber.

Layout:
  ┌─ top bar ─────────────────────────────────────────────────────────┐
  │  Goal badge │ Queue text           │ Scheduled N/B ██░ │ t=...s   │
  ├─ threshold slider ────────────────────────────────────────────────┤
  ├─ graph ──────────────────────────── info panel (240px) ───────────┤
  │                                     │  Node inspector             │
  │  Cytoscape (preset layout)          │  ─ data table               │
  │                                     │  ─ channel stacked bar      │
  │  [legend overlay]                   │  ─ channel breakdown table  │
  │                                     │  ─ sparkline (A over time)  │
  ├─ [replay: controls + scrubber + events strip] ────────────────────┤
  └─────────────────────────────────────────────────────────────────  ┘

Node visual encoding:
  Fill color  - hue by dominant activation channel; lightness by epistemic error
                  blue   = mission-driven   (F1 + F2 combined)
                  teal   = relational-driven (instance-graph propagation)
                  orange = surprise-driven  (prediction-error violation)
                  darker shade = higher epistemic error (staler knowledge)
  Size        - instance nodes: proportional to attention
  Opacity     - dim when attention < threshold slider value
  Gold border - task node: SOLID = active goal, DASHED = queued goal
  Yellow border - in current tick's refresh schedule
  Red border    - prediction_error > 0.1

Sparkline:
  Live mode   - rolling window (last history_maxlen ticks), right edge = now
  Replay mode - centred window (±half_window ticks), vertical cursor at current t
"""

import datetime
from pathlib import Path

import dash
import dash_cytoscape as cyto
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, dcc, html

from awareness_manager.visualization.runner import SimulationRunner
from awareness_manager.visualization.snapshot import CHANNEL_COLORS


def _strip_positions(elements: list) -> list:
    """Return elements with 'position' keys removed (preserves Cytoscape user drags)."""
    return [{k: v for k, v in e.items() if k != "position"} for e in elements]

# ── Colour palette (dark theme) ──────────────────────────────────────────────
_BG     = "#1e1e2e"
_PANEL  = "#2a2a3e"
_ACCENT = "#4a8af4"
_GOLD   = "#f4c44a"
_TEXT   = "#e0e0ff"
_DIM    = "#7788aa"
_EDGE_SEM = "#4a5566"
_EDGE_REL = "#44aa88"

# Channel colour aliases (shared with snapshot.py via CHANNEL_COLORS import)
_COL_MISSION      = CHANNEL_COLORS['mission']        # '#4a8af4'
_COL_ANTICIPATORY = CHANNEL_COLORS['anticipatory']   # '#9a6af4'
_COL_RELATIONAL   = CHANNEL_COLORS['relational']     # '#44bb88'
_COL_SURPRISE     = CHANNEL_COLORS['surprise']       # '#f4944a'
_COL_PRIORITY     = '#e8c84a'                        # golden yellow — E × A priority

# ── Static Cytoscape stylesheet ──────────────────────────────────────────────
STYLESHEET = [
    # ── Class compound nodes ─────────────────────────────────────────────
    {
        "selector": "node[type='class']",
        "style": {
            "shape": "roundrectangle",
            "background-color": "data(color)",
            "background-opacity": "mapData(attention, 0, 1, 0.18, 0.80)",
            "border-color": _ACCENT,
            "border-width": "mapData(attention, 0, 1, 1, 6)",
            "border-opacity": "mapData(attention, 0, 1, 0.25, 1.0)",
            "border-style": "solid",
            "label": "data(label)",
            "text-valign": "top",
            "text-halign": "center",
            "color": _TEXT,
            "font-family": "monospace",
            "font-size": "10px",
            "font-weight": "bold",
            "text-background-color": "#0a0a1e",
            "text-background-opacity": 0.72,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "3px",
            "text-wrap": "wrap",
            "text-max-width": "120px",
            "padding": "22px",
            "min-width": "88px",
            "min-height": "52px",
            "compound-sizing-wrt-labels": "include",
        },
    },
    # Empty class containers (no instances) - visually de-emphasised
    {
        "selector": "node[type='class'][has_instances=0]",
        "style": {
            "min-width": "72px",
            "min-height": "40px",
            "font-size": "9px",
            "border-width": "mapData(attention, 0, 1, 0.5, 3)",
            "padding": "14px",
        },
    },
    # All task nodes: dashed gold border (queued / background goal appearance)
    {
        "selector": "node[type='class'][concept_type='task']",
        "style": {
            "border-color": _GOLD,
            "border-style": "dashed",
            "border-width": "mapData(attention, 0, 1, 1.5, 5)",
            "background-color": "data(color)",
        },
    },
    # Active goal: solid gold, slightly thicker - overrides dashed above
    {
        "selector": "node[type='class'][concept_type='task'][is_active_goal=1]",
        "style": {
            "border-style": "solid",
            "border-width": "mapData(attention, 0, 1, 2, 7)",
        },
    },
    # ── Instance nodes (inside class compounds) ───────────────────────────
    {
        "selector": "node[type='instance']",
        "style": {
            "shape": "ellipse",
            "background-color": "data(color)",
            "width": "mapData(attention, 0, 1, 16, 54)",
            "height": "mapData(attention, 0, 1, 16, 54)",
            "opacity": "mapData(attention, 0, 1, 0.22, 0.97)",
            "border-color": "#ccccdd",
            "border-width": 1,
            "border-opacity": 0.45,
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
            "font-family": "monospace",
            "font-size": "7px",
            "font-weight": "bold",
            "text-wrap": "wrap",
            "text-max-width": "52px",
            "text-background-color": _BG,
            "text-background-opacity": 0.55,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "2px",
        },
    },
    # ── State flag overrides - order matters: later rules win ─────────────
    {
        "selector": "node[below_threshold=1]",
        "style": {"opacity": 0.12},
    },
    # Urgency: growing orange-red outer ring on instance nodes.
    # border-width maps 0→0px, 1→7px so the ring only appears when urgency > 0.
    # This sits BELOW in_schedule and prediction_error so those overrides still win.
    {
        "selector": "node[type='instance'][urgency > 0.0]",
        "style": {
            "border-color": "#ff6b35",
            "border-width": "mapData(urgency, 0, 1, 0, 7)",
            "border-opacity": "mapData(urgency, 0.05, 1, 0.4, 1.0)",
            "border-style": "solid",
        },
    },
    {
        "selector": "node[in_schedule=1]",
        "style": {
            "border-color": "#ffdd00",
            "border-width": 4,
            "border-opacity": 1.0,
            "border-style": "solid",
        },
    },
    {
        "selector": "node[prediction_error > 0.1]",
        "style": {
            "border-color": "#ff4444",
            "border-width": 3,
            "border-opacity": 1.0,
            "border-style": "solid",
        },
    },
    # ── Semantic edges (class-class) ──────────────────────────────────────
    {
        "selector": "edge[edge_type='semantic']",
        "style": {
            "line-style": "dashed",
            "line-color": _EDGE_SEM,
            "width": 1.5,
            "opacity": 0.55,
            "curve-style": "bezier",
            "label": "data(label)",
            "font-size": "8px",
            "font-family": "monospace",
            "color": "#7788aa",
            "text-opacity": 0.65,
            "text-background-color": _BG,
            "text-background-opacity": 0.50,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "2px",
        },
    },
    # ── Relational edges (instance-instance) ──────────────────────────────
    {
        "selector": "edge[edge_type='relational']",
        "style": {
            "line-style": "solid",
            "line-color": _EDGE_REL,
            "width": 2,
            "opacity": 0.78,
            "curve-style": "bezier",
            "label": "data(label)",
            "font-size": "8px",
            "font-family": "monospace",
            "color": "#88ddbb",
            "text-opacity": 0.85,
            "text-background-color": _BG,
            "text-background-opacity": 0.50,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "2px",
            "target-arrow-shape": "triangle",
            "target-arrow-color": _EDGE_REL,
        },
    },
]


# ── Legend overlay ────────────────────────────────────────────────────────────

_LEGEND = html.Div(
    style={
        "position": "absolute",
        "bottom": "14px",
        "left": "14px",
        "background": "rgba(18, 18, 38, 0.90)",
        "border": f"1px solid {_PANEL}",
        "border-radius": "6px",
        "padding": "8px 12px",
        "font-size": "10px",
        "font-family": "monospace",
        "pointer-events": "none",
        "line-height": "1.8",
    },
    children=[
        html.Div([html.Span("── ─ ─  ", style={"color": _EDGE_SEM}),
                  html.Span("semantic edge (class-class)", style={"color": _DIM})]),
        html.Div([html.Span("────  ▶ ", style={"color": _EDGE_REL}),
                  html.Span("relational edge (instance-instance)", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": _COL_MISSION}),
                  html.Span("blue = mission-driven (F1 + F2 goal)", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": _COL_RELATIONAL}),
                  html.Span("teal = relational-driven (instance graph)", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": _COL_SURPRISE}),
                  html.Span("orange = surprise (prediction error)", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": "#888866"}),
                  html.Span("darker shade = stale (high epistemic error)", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": "#ffdd00"}),
                  html.Span("yellow border = in refresh schedule", style={"color": _DIM})]),
        html.Div([html.Span("◉  ", style={"color": "#ff4444"}),
                  html.Span("red border = prediction error > 0.1", style={"color": _DIM})]),
        html.Div([html.Span("━━  ", style={"color": _GOLD}),
                  html.Span("solid gold = active goal", style={"color": _DIM})]),
        html.Div([html.Span("┅┅  ", style={"color": _GOLD}),
                  html.Span("dashed gold = queued goal", style={"color": _DIM})]),
    ],
)


# ── Inspector helpers ─────────────────────────────────────────────────────────

_BAR_W_PX = 184


def _channel_bar(
    ch_mission: float,
    ch_anticipatory: float,
    ch_relational: float,
    ch_surprise: float,
    total_attn: float,
) -> html.Div:
    """Horizontal stacked bar. Segments sum to total_attn (max width = 1.0)."""
    ch_total = ch_mission + ch_anticipatory + ch_relational + ch_surprise

    if ch_total < 0.001 or total_attn < 0.001:
        return html.Div(
            style={"width": f"{_BAR_W_PX}px", "height": "10px",
                   "background": "#1a1a2e", "border-radius": "3px"},
        )

    scale = total_attn / ch_total
    segments = [
        (ch_mission      * scale, _COL_MISSION,       "mission"),
        (ch_anticipatory * scale, _COL_ANTICIPATORY,  "anticipatory"),
        (ch_relational   * scale, _COL_RELATIONAL,    "relational"),
        (ch_surprise     * scale, _COL_SURPRISE,       "surprise"),
    ]
    seg_divs = []
    for val, color, label in segments:
        if val < 0.001:
            continue
        seg_divs.append(html.Div(
            title=f"{label}: {val:.4f}",
            style={
                "width": f"{val * _BAR_W_PX:.1f}px",
                "height": "10px",
                "background": color,
                "opacity": 0.85,
                "flex-shrink": "0",
            },
        ))

    return html.Div(
        html.Div(seg_divs, style={"display": "flex", "height": "10px"}),
        style={
            "width": f"{_BAR_W_PX}px",
            "height": "10px",
            "background": "#1a1a2e",
            "border-radius": "3px",
            "overflow": "hidden",
        },
    )


def _hover_card(data: dict | None) -> html.Div:
    if not data:
        return html.Div(
            "Hover over a node to inspect it.",
            style={"color": _DIM, "font-size": "11px", "padding": "6px 2px"},
        )

    node_type  = data.get("type", "-")
    concept_id = data.get("id", "-")
    a    = data.get("attention", 0.0)
    e    = data.get("epistemic_error", 0.0)
    pe   = data.get("prediction_error", 0.0)
    dr   = data.get("decay_rate", 0.0)
    u    = data.get("urgency", 0.0)
    ur   = data.get("urgency_rate", 0.0)
    uc   = data.get("urgency_contribution", 0.0)
    p    = data.get("priority", 0.0)
    rank = data.get("priority_rank", 0)
    n_sc = data.get("n_schedulable", 0)
    sch  = data.get("in_schedule", 0)
    cm   = data.get("ch_mission", 0.0)
    ca   = data.get("ch_anticipatory", 0.0)
    cr   = data.get("ch_relational", 0.0)
    cs   = data.get("ch_surprise", 0.0)

    third_key = "Concept type" if node_type == "class" else "Class"
    third_val = (data.get("concept_type", "-") if node_type == "class"
                 else data.get("class_id", "-"))

    def _row(k, v, vcolor=_TEXT):
        return html.Tr([
            html.Td(k, style={"color": _DIM, "font-size": "10px",
                               "padding": "2px 6px 2px 0", "white-space": "nowrap"}),
            html.Td(v, style={"color": vcolor, "font-size": "10px",
                               "font-weight": "bold"}),
        ])

    rank_str = f"#{rank} / {n_sc}" if rank > 0 else f"— / {n_sc}"
    rank_color = (
        "#f4c44a" if rank == 1
        else "#44cc88" if rank <= 3
        else "#ff6b35" if rank > max(1, n_sc // 2)
        else _TEXT
    )

    rows = [
        _row("ID",            concept_id),
        _row("Type",          node_type),
        _row(third_key,       third_val),
        _row("Priority rank", rank_str, vcolor=rank_color),
        _row("Priority  P",   f"{p:.4f}",
             vcolor=_COL_PRIORITY if p > 0 else _DIM),
        _row("Attention  A",  f"{a:.4f}"),
        _row("Epistemic  E",  f"{e:.4f}"),
        _row("Pred. error ε", f"{pe:.4f}",
             vcolor="#ff8888" if pe > 0.1 else _TEXT),
        _row("Decay rate δ",  f"{dr:.4f}"),
        _row("In schedule",   "Yes ◀" if sch else "No",
             vcolor="#ffdd00" if sch else _TEXT),
    ]
    if ur > 0.0:
        rows.append(_row("Urgency  U",    f"{u:.4f}",
                         vcolor="#ff6b35" if u > 0.3 else _TEXT))
        rows.append(_row("Urgency rate",  f"{ur:.4f}"))

    basic_table = html.Table(rows,
        style={"border-collapse": "collapse", "margin-bottom": "8px"},
    )

    def _prio_row(label, val, color, scale=2.0):
        bar_w = int((val / scale) * (_BAR_W_PX * 0.55))
        return html.Tr([
            html.Td(label, style={"font-size": "9px", "color": color,
                                   "padding": "1px 5px 1px 0",
                                   "white-space": "nowrap"}),
            html.Td(f"{val:.4f}", style={"font-size": "9px", "color": _TEXT,
                                          "font-weight": "bold",
                                          "padding-right": "5px"}),
            html.Td(html.Div(style={
                "width": f"{max(0, bar_w)}px", "height": "6px",
                "background": color, "opacity": 0.8, "border-radius": "2px",
            })),
        ])

    prio_rows = [_prio_row("E × A",  e * a, _COL_PRIORITY)]
    if ur > 0.0:
        prio_rows.append(_prio_row("+ urgency", uc, "#ff6b35"))
    prio_rows.append(_prio_row("= priority", p, _COL_PRIORITY))

    prio_section = html.Div([
        html.Div("Priority breakdown", style={
            "font-size": "9px", "color": _DIM,
            "margin-bottom": "4px", "letter-spacing": "0.05em",
        }),
        html.Table(prio_rows, style={"border-collapse": "collapse"}),
    ], style={"margin-bottom": "8px"})

    bar_section = html.Div([
        html.Div("Attention channels", style={
            "font-size": "9px", "color": _DIM,
            "margin-bottom": "4px", "letter-spacing": "0.05em",
        }),
        _channel_bar(cm, ca, cr, cs, a),
        html.Div(f"total: {a:.3f} / 1.000",
                 style={"font-size": "9px", "color": _DIM, "margin-top": "2px"}),
    ], style={"margin-bottom": "8px"})

    def _ch_row(label, val, color):
        bar_w = int(val * (_BAR_W_PX * 0.55))
        return html.Tr([
            html.Td(label, style={"font-size": "9px", "color": color,
                                   "padding": "1px 5px 1px 0",
                                   "white-space": "nowrap"}),
            html.Td(f"{val:.4f}", style={"font-size": "9px", "color": _TEXT,
                                          "font-weight": "bold",
                                          "padding-right": "5px"}),
            html.Td(html.Div(style={
                "width": f"{bar_w}px", "height": "6px",
                "background": color, "opacity": 0.75, "border-radius": "2px",
            })),
        ])

    ch_table = html.Table([
        _ch_row("mission",      cm, _COL_MISSION),
        _ch_row("anticipatory", ca, _COL_ANTICIPATORY),
        _ch_row("relational",   cr, _COL_RELATIONAL),
        _ch_row("surprise",     cs, _COL_SURPRISE),
    ], style={"border-collapse": "collapse"})

    return html.Div([basic_table, prio_section, bar_section, ch_table])


# ── Sparkline helpers ─────────────────────────────────────────────────────────

_SPARKLINE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,42,0.75)",
    margin=dict(l=4, r=4, t=22, b=4),
    height=115,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               showline=True, linecolor="#333355", linewidth=1),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               range=[0.0, 1.05], showline=False),
    showlegend=False,
    hovermode=False,
)


def _empty_sparkline() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_SPARKLINE_LAYOUT,
        annotations=[dict(
            text="hover a node",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(color=_DIM, size=10, family="monospace"),
        )],
    )
    return fig


def _make_sparkline(history: list, node_id: str, cursor_t: float | None = None) -> go.Figure:
    """
    Small timeseries chart: total attention (background fill) + 3 channel lines.
    In replay mode, cursor_t adds a vertical line at the current scrubber position.
    """
    if not history:
        return _empty_sparkline()

    t     = [h[0] for h in history]
    a     = [h[1] for h in history]
    an    = [h[7] for h in history]                 # anticipatory only (purple)
    m     = [h[2] - h[7] for h in history]         # mission-only / F1 (blue)
    r     = [h[3] for h in history]                # relational (teal)
    s     = [h[4] for h in history]                # surprise (orange)
    p     = [h[8] for h in history]                # scheduling priority (golden)
    p_ref = [h[9] for h in history]                # max-priority across all concepts (reference)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a, fill="tozeroy",
                             fillcolor="rgba(110,110,150,0.18)",
                             line=dict(color="rgba(170,170,210,0.40)", width=1),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=m, line=dict(color=_COL_MISSION, width=1.5),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=an, line=dict(color=_COL_ANTICIPATORY, width=1.5),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=r, line=dict(color=_COL_RELATIONAL, width=1.5),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=s, line=dict(color=_COL_SURPRISE, width=1.5),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=p_ref,
                             line=dict(color="rgba(255,255,255,0.30)", width=1.0, dash="dash"),
                             mode="lines", hoverinfo="none"))
    fig.add_trace(go.Scatter(x=t, y=p, line=dict(color=_COL_PRIORITY, width=2.0, dash="dot"),
                             mode="lines", hoverinfo="none"))

    layout = dict(
        **_SPARKLINE_LAYOUT,
        title=dict(text=f"P · A(t) — {node_id}",
                   font=dict(size=9, color=_DIM, family="monospace"),
                   x=0.02, xanchor="left", y=0.99, yanchor="top"),
    )
    if cursor_t is not None:
        layout["shapes"] = [{
            "type": "line",
            "x0": cursor_t, "x1": cursor_t,
            "y0": 0, "y1": 1,
            "xref": "x", "yref": "paper",
            "line": {"color": "rgba(255,255,255,0.6)", "width": 1.5, "dash": "dot"},
        }]
    fig.update_layout(**layout)
    fig.update_yaxes(rangemode="tozero")
    return fig


# ── Budget bar (live mode) ────────────────────────────────────────────────────

def _budget_bar_html(n: int, total: int) -> html.Div:
    frac = min(1.0, n / max(1, total))
    return html.Div(
        html.Div(style={
            "width": f"{frac * 100:.0f}%",
            "height": "100%",
            "background": "linear-gradient(90deg, #44cc88, #4a8af4)",
            "border-radius": "3px",
            "transition": "width 0.2s ease",
            "min-width": "2px",
        }),
        style={
            "width": "90px",
            "height": "8px",
            "background": "#303055",
            "border-radius": "3px",
            "display": "inline-block",
            "vertical-align": "middle",
        },
    )


# ── Replay events strip ───────────────────────────────────────────────────────

def _make_events_strip(all_events: list[dict], total_duration: float) -> go.Figure:
    """
    Thin Plotly figure showing goal-transition and surprise event markers.
    Computed once at replay startup; cursor updated separately via Patch.
    """
    goal_t = [ev["t"] for ev in all_events if ev["type"] == "goal_transition"]
    surp_t = [ev["t"] for ev in all_events if ev["type"] == "surprise"]

    fig = go.Figure()

    if goal_t:
        fig.add_trace(go.Scatter(
            x=goal_t, y=[0.5] * len(goal_t),
            mode="markers",
            marker=dict(symbol="triangle-up", size=10,
                        color=_COL_MISSION, opacity=0.9),
            name="goal transition",
            hovertemplate="goal → %{customdata}<extra></extra>",
            customdata=[ev.get("to", "?") for ev in all_events
                        if ev["type"] == "goal_transition"],
        ))

    if surp_t:
        fig.add_trace(go.Scatter(
            x=surp_t, y=[0.5] * len(surp_t),
            mode="markers",
            marker=dict(symbol="star", size=9,
                        color=_COL_SURPRISE, opacity=0.85),
            name="surprise",
            hovertemplate="surprise: %{customdata}<extra></extra>",
            customdata=[ev.get("concept", "?") for ev in all_events
                        if ev["type"] == "surprise"],
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,18,42,0.60)",
        margin=dict(l=4, r=4, t=0, b=0),
        height=32,
        xaxis=dict(range=[0, max(total_duration, 1.0)],
                   showticklabels=False, showgrid=False, zeroline=False,
                   showline=False, fixedrange=True),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False,
                   zeroline=False, showline=False, fixedrange=True),
        showlegend=False,
        hovermode="x",
    )
    return fig


# ── Shared graph pane (used in both modes) ────────────────────────────────────

def _graph_pane() -> html.Div:
    return html.Div(
        style={"flex": "1", "position": "relative", "overflow": "hidden"},
        children=[
            cyto.Cytoscape(
                id="awareness-graph",
                elements=[],
                layout={"name": "preset", "fit": True},
                style={"width": "100%", "height": "100%", "background": _BG},
                stylesheet=STYLESHEET,
                minZoom=0.10,
                maxZoom=3.5,
                boxSelectionEnabled=False,
                autoRefreshLayout=False,
            ),
            _LEGEND,
        ],
    )


def _info_panel(header_children: list) -> html.Div:
    return html.Div(
        style={
            "width": "240px",
            "flex-shrink": "0",
            "background": _PANEL,
            "border-left": "1px solid #333355",
            "padding": "12px 10px",
            "overflow-y": "auto",
            "display": "flex",
            "flex-direction": "column",
        },
        children=[
            html.Div(
                header_children,
                style={
                    "font-size": "10px",
                    "font-weight": "bold",
                    "color": _ACCENT,
                    "margin-bottom": "8px",
                    "border-bottom": "1px solid #333355",
                    "padding-bottom": "5px",
                    "flex-shrink": "0",
                },
            ),
            html.Div(id="hover-tooltip", style={"flex-shrink": "0"}),
            html.Div(style={"border-top": "1px solid #333355",
                            "margin": "8px 0 4px 0", "flex-shrink": "0"}),
            html.Div(
                dcc.Graph(id="sparkline-chart", figure=_empty_sparkline(),
                          config={"displayModeBar": False, "staticPlot": True},
                          style={"height": "115px"}),
                style={"flex-shrink": "0", "margin": "0 -4px"},
            ),
        ],
    )


# ── Compare-mode helpers ──────────────────────────────────────────────────────

def _structures_compatible(meta_a: dict, meta_b: dict) -> tuple[bool, str | None]:
    """Deep-compare entire structure blocks from two trace metas."""
    sa, sb = meta_a["structure"], meta_b["structure"]
    if sorted(c["id"] for c in sa["classes"]) != sorted(c["id"] for c in sb["classes"]):
        return False, "class sets differ"
    inst_a = sorted((i["id"], i["class_id"]) for i in sa.get("instances", []))
    inst_b = sorted((i["id"], i["class_id"]) for i in sb.get("instances", []))
    if inst_a != inst_b:
        return False, "instance sets differ"
    edges_a = sorted((e[0], e[1]) for e in sa["semantic_edges"])
    edges_b = sorted((e[0], e[1]) for e in sb["semantic_edges"])
    if edges_a != edges_b:
        return False, "semantic edge topology differs"
    rel_a = sorted((e[0], e[1]) for e in sa.get("relational_edges", []))
    rel_b = sorted((e[0], e[1]) for e in sb.get("relational_edges", []))
    if rel_a != rel_b:
        return False, "relational edge topology differs"
    return True, None


def _meta_label(meta: dict) -> str:
    """Short label for a trace: 'strategy  B=N  [F1+F2]'."""
    schema_v = meta.get("schema_version", 1)
    if schema_v >= 2:
        strat = meta.get("strategy", "?")
        budget = meta.get("budget", "?")
        budget_str = "∞" if (isinstance(budget, int) and budget < 0) else str(budget)
        fc = meta.get("strategy_params", {}).get("feature_config", {})
    else:
        strat = "awareness_manager"
        am = meta.get("am_params", {})
        budget_str = str(am.get("budget", "?"))
        fc = am.get("feature_config", {})
    active_fs = "+".join(
        f"F{i+1}" for i, v in enumerate(
            [fc.get("f1"), fc.get("f2"), fc.get("f3"), fc.get("f4"), fc.get("f5")]
        ) if v
    )
    label = f"{strat}  B={budget_str}"
    if active_fs:
        label += f"  [{active_fs}]"
    return label


def _metrics_panel(source_a: 'ReplayReader', source_b: 'ReplayReader') -> html.Div:
    """
    Static metrics panel computed once at trace-load time.

    Calls all_metrics() for both traces (~80ms total) and renders a compact
    comparison table.  No callbacks - the numbers don't change with the scrubber.
    """
    try:
        from awareness_manager.evaluation.metrics import all_metrics, load_trace
        ma = all_metrics(load_trace(source_a._dir))
        mb = all_metrics(load_trace(source_b._dir))
        metrics_ok = True
    except Exception:
        metrics_ok = False

    strat_a = source_a.meta.get("strategy", "A")
    strat_b = source_b.meta.get("strategy", "B")

    def _pill(text, color):
        return html.Span(text, style={
            "background": color,
            "color": "#fff",
            "border-radius": "4px",
            "padding": "1px 6px",
            "font-size": "9px",
            "font-weight": "bold",
            "font-family": "monospace",
            "margin-right": "4px",
        })

    def _fmt(v, decimals=4):
        if v is None:
            return "-"
        try:
            return f"{v:.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    def _diff_color(diff, lower_is_better=True):
        """Green if A beats B, red otherwise."""
        if diff is None:
            return _DIM
        if lower_is_better:
            return "#66dd88" if diff < -0.0001 else ("#ff7766" if diff > 0.0001 else _DIM)
        return "#66dd88" if diff > 0.0001 else ("#ff7766" if diff < -0.0001 else _DIM)

    if not metrics_ok:
        body = html.Div("(metrics unavailable - install awareness_manager.evaluation)",
                        style={"color": _DIM, "font-size": "10px", "padding": "4px 0"})
    else:
        def _row(label, key, decimals=4, lower_is_better=True, suffix=""):
            va = ma.get(key)
            vb = mb.get(key)
            diff = (va - vb) if (va is not None and vb is not None) else None
            dcol = _diff_color(diff, lower_is_better)
            diff_str = (f"{diff:+.{decimals}f}" if diff is not None else "-")
            return html.Tr([
                html.Td(label,
                        style={"color": _DIM, "font-size": "9px",
                               "padding": "2px 8px 2px 0", "white-space": "nowrap"}),
                html.Td(_fmt(va, decimals) + suffix,
                        style={"color": _GOLD, "font-size": "9px",
                               "font-weight": "bold", "padding": "2px 8px 2px 0",
                               "text-align": "right"}),
                html.Td(_fmt(vb, decimals) + suffix,
                        style={"color": "#88ddff", "font-size": "9px",
                               "font-weight": "bold", "padding": "2px 8px 2px 0",
                               "text-align": "right"}),
                html.Td(diff_str,
                        style={"color": dcol, "font-size": "9px",
                               "font-weight": "bold", "text-align": "right",
                               "padding": "2px 0"}),
            ])

        body = html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Metric", style={"color": _DIM, "font-size": "8px",
                                             "font-weight": "normal",
                                             "padding": "0 8px 3px 0",
                                             "text-align": "left"}),
                    html.Th(_pill(strat_a, _GOLD[:7] + "cc"), style={"padding": "0 8px 3px 0"}),
                    html.Th(_pill(strat_b, "#3388cc"),         style={"padding": "0 8px 3px 0"}),
                    html.Th("A − B", style={"color": _DIM, "font-size": "8px",
                                            "font-weight": "normal",
                                            "text-align": "right"}),
                ])),
                html.Tbody([
                    _row("M1 E@transition ↓",   "m1_e_at_transition",     3, True),
                    _row("M2 pre-attn ↑",        "m2_pre_transition_attn", 3, False),
                    _row("M3 lag (s) ↓",         "m3_lag_seconds",         2, True, "s"),
                    _row("M4 E_relevant ↓",      "m4_e_relevant",          4, True),
                    _row("M5 budget util",        "m5_budget_util",         3, False),
                ]),
            ],
            style={"border-collapse": "collapse"},
        )

    return html.Div(
        style={
            "background": "#12121e",
            "border-top": "1px solid #333355",
            "padding": "5px 16px 4px 16px",
            "display": "flex",
            "align-items": "center",
            "gap": "16px",
            "flex-shrink": "0",
        },
        children=[
            html.Span("Metrics (full run)",
                      style={"color": _ACCENT, "font-size": "9px",
                             "font-weight": "bold", "white-space": "nowrap"}),
            body,
        ],
    )


def _build_compare_app(
    source_a: 'ReplayReader',
    source_b: 'ReplayReader',
) -> dash.Dash:
    """Two-column A/B comparison of two replay traces on the same scenario."""
    app = dash.Dash(
        __name__,
        title="Awareness Manager - Compare",
        suppress_callback_exceptions=True,
    )

    compatible, reason = _structures_compatible(source_a.meta, source_b.meta)
    if not compatible:
        app.layout = html.Div(
            [
                html.H3("Incompatible traces",
                        style={"color": "#ff4444", "font-family": "monospace"}),
                html.P(f"Reason: {reason}",
                       style={"color": _TEXT, "font-family": "monospace"}),
                html.P(
                    "Both traces must be recorded on the same scenario "
                    "(identical class / instance / edge structure).",
                    style={"color": _DIM, "font-family": "monospace",
                           "font-size": "12px"},
                ),
            ],
            style={"background": _BG, "padding": "40px", "min-height": "100vh"},
        )
        return app

    label_a = _meta_label(source_a.meta)
    label_b = _meta_label(source_b.meta)
    max_tick_a = max(0, source_a.tick_count - 1)
    max_tick_b = max(0, source_b.tick_count - 1)
    dt_b = source_b.meta.get("dt", 0.1)

    def _aligned_tick_b(tick_a: int) -> int:
        t = source_a.elapsed_at(tick_a)
        return min(max_tick_b, max(0, round(t / dt_b)))

    def _compare_col(side: str, label: str) -> html.Div:
        accent = _GOLD if side == "a" else "#88ddff"
        return html.Div(
            style={
                "flex": "1",
                "display": "flex",
                "flex-direction": "column",
                "overflow": "hidden",
                **({"border-right": "1px solid #333355"} if side == "a" else {}),
            },
            children=[
                html.Div(
                    label,
                    style={
                        "background": "#181828",
                        "padding": "4px 12px",
                        "font-size": "10px",
                        "color": accent,
                        "font-family": "monospace",
                        "flex-shrink": "0",
                        "border-bottom": "1px solid #2a2a44",
                    },
                ),
                html.Div(
                    style={"flex": "1", "display": "flex", "overflow": "hidden"},
                    children=[
                        html.Div(
                            style={"flex": "1", "position": "relative",
                                   "overflow": "hidden"},
                            children=[
                                cyto.Cytoscape(
                                    id=f"awareness-graph-{side}",
                                    elements=[],
                                    layout={"name": "preset", "fit": True},
                                    style={"width": "100%", "height": "100%",
                                           "background": _BG},
                                    stylesheet=STYLESHEET,
                                    minZoom=0.25,
                                    maxZoom=3.5,
                                    boxSelectionEnabled=False,
                                    autoRefreshLayout=False,
                                ),
                                _LEGEND,
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "200px",
                                "flex-shrink": "0",
                                "background": _PANEL,
                                "border-left": "1px solid #333355",
                                "padding": "10px 8px",
                                "overflow-y": "auto",
                                "display": "flex",
                                "flex-direction": "column",
                            },
                            children=[
                                html.Div(
                                    id=f"hover-tooltip-{side}",
                                    children=[html.Div(
                                        "Hover a node.",
                                        style={"color": _DIM, "font-size": "11px"},
                                    )],
                                    style={"flex-shrink": "0"},
                                ),
                                html.Div(style={"border-top": "1px solid #333355",
                                                "margin": "6px 0 4px 0",
                                                "flex-shrink": "0"}),
                                html.Div(
                                    dcc.Graph(
                                        id=f"sparkline-chart-{side}",
                                        figure=_empty_sparkline(),
                                        config={"displayModeBar": False,
                                                "staticPlot": True},
                                        style={"height": "115px"},
                                    ),
                                    style={"flex-shrink": "0", "margin": "0 -4px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

    _btn_style_cmp = {
        "background": "#252540", "color": _TEXT,
        "border": "1px solid #444466", "border-radius": "6px",
        "padding": "3px 10px", "font-size": "13px",
        "font-family": "monospace", "cursor": "pointer", "line-height": "1",
    }
    total_dur = source_a.total_duration()
    scenario_a = source_a.meta.get("scenario", "?")
    scenario_b = source_b.meta.get("scenario", "?")

    app.layout = html.Div(
        style={"background": _BG, "color": _TEXT, "font-family": "monospace",
               "height": "100vh", "display": "flex",
               "flex-direction": "column", "overflow": "hidden"},
        children=[
            # Top bar
            html.Div(
                style={"background": _PANEL, "padding": "6px 16px",
                       "display": "flex", "align-items": "center", "gap": "12px",
                       "border-bottom": "1px solid #333355", "flex-shrink": "0"},
                children=[
                    html.Span("Compare", style={"color": _ACCENT,
                                                "font-weight": "bold",
                                                "font-size": "12px"}),
                    html.Span(scenario_a, style={"color": _GOLD,
                                                 "font-size": "10px"}),
                    html.Span("vs", style={"color": _DIM, "font-size": "10px"}),
                    html.Span(scenario_b, style={"color": "#88ddff",
                                                 "font-size": "10px"}),
                    html.Div(style={"flex": "1"}),
                    html.Span(id="elapsed-label-cmp",
                              style={"color": _DIM, "font-size": "10px"}),
                ],
            ),
            # Threshold row
            html.Div(
                style={"background": "#1a1a2e", "padding": "4px 16px",
                       "display": "flex", "align-items": "center", "gap": "10px",
                       "border-bottom": "1px solid #2a2a44", "flex-shrink": "0"},
                children=[
                    html.Span("Activation threshold:",
                              style={"color": _DIM, "font-size": "10px",
                                     "white-space": "nowrap"}),
                    html.Div(dcc.Slider(
                        id="threshold-slider",
                        min=0.0, max=1.0, step=0.01, value=0.0,
                        marks={
                            0:   {"label": "0",   "style": {"color": _DIM,
                                                            "font-size": "9px"}},
                            0.5: {"label": "0.5", "style": {"color": _DIM,
                                                            "font-size": "9px"}},
                            1:   {"label": "1",   "style": {"color": _DIM,
                                                            "font-size": "9px"}},
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="drag",
                    ), style={"flex": "1"}),
                    html.Span(id="threshold-label",
                              style={"color": _DIM, "font-size": "10px",
                                     "min-width": "46px", "text-align": "right",
                                     "background": "#2a2a3e", "border-radius": "6px",
                                     "padding": "1px 6px"}),
                ],
            ),
            # Two-column main area
            html.Div(
                style={"flex": "1", "display": "flex", "overflow": "hidden"},
                children=[_compare_col("a", label_a), _compare_col("b", label_b)],
            ),
            # Shared replay controls
            html.Div(
                style={"background": "#181828", "border-top": "1px solid #333355",
                       "padding": "6px 14px 2px 14px", "flex-shrink": "0"},
                children=[
                    html.Div(
                        style={"display": "flex", "align-items": "center",
                               "gap": "6px", "margin-bottom": "4px"},
                        children=[
                            html.Button("⏮", id="btn-start", n_clicks=0,
                                        title="Go to start",
                                        style=_btn_style_cmp),
                            html.Button("◀", id="btn-step-back", n_clicks=0,
                                        title="Step back 1 tick",
                                        style=_btn_style_cmp),
                            html.Button("▶", id="btn-play-pause", n_clicks=0,
                                        title="Play / Pause",
                                        style={**_btn_style_cmp,
                                               "background": "#1a2a3a",
                                               "border-color": _ACCENT,
                                               "color": _ACCENT,
                                               "padding": "3px 14px"}),
                            html.Button("▶", id="btn-step-fwd", n_clicks=0,
                                        title="Step forward 1 tick",
                                        style=_btn_style_cmp),
                            html.Button("⏭", id="btn-end", n_clicks=0,
                                        title="Go to end",
                                        style=_btn_style_cmp),
                            html.Span("Speed:", style={"color": _DIM,
                                                       "font-size": "10px",
                                                       "margin-left": "10px"}),
                            dcc.Dropdown(
                                id="speed-control",
                                options=[
                                    {"label": "0.25x", "value": 0.25},
                                    {"label": "0.5x",  "value": 0.5},
                                    {"label": "1x",    "value": 1.0},
                                    {"label": "4x",    "value": 4.0},
                                    {"label": "16x",   "value": 16.0},
                                ],
                                value=1.0,
                                clearable=False,
                                style={"width": "80px", "background": "#252540",
                                       "color": _TEXT, "font-size": "10px",
                                       "font-family": "monospace", "border": "none"},
                            ),
                            html.Span(id="replay-pos-label",
                                      style={"color": _TEXT, "font-size": "10px",
                                             "font-family": "monospace",
                                             "margin-left": "10px",
                                             "white-space": "nowrap"}),
                        ],
                    ),
                    html.Div(
                        dcc.Slider(
                            id="replay-scrubber",
                            min=0, max=max_tick_a, step=1, value=0,
                            marks={},
                            tooltip={"placement": "top", "always_visible": False},
                            updatemode="drag",
                        ),
                        style={"margin": "0 -4px 2px -4px"},
                    ),
                    html.Div(
                        dcc.Graph(
                            id="events-strip",
                            figure=_make_events_strip(source_a.all_events, total_dur),
                            config={"displayModeBar": False, "staticPlot": False},
                            style={"height": "32px"},
                        ),
                        style={"margin": "0 -4px"},
                    ),
                ],
            ),
            _metrics_panel(source_a, source_b),
            dcc.Store(id="replay-store",
                      data={"tick_index": 0, "playing": False, "speed": 1.0}),
            dcc.Interval(id="update-interval", interval=250, n_intervals=0),
            dcc.Store(id="layout-initialized-a", data=False),
            dcc.Store(id="layout-initialized-b", data=False),
        ],
    )

    # ── Threshold label ───────────────────────────────────────────────────
    @app.callback(
        Output("threshold-label", "children"),
        Input("threshold-slider", "value"),
    )
    def _cmp_threshold_label(value):
        return f"≥ {value:.2f}"

    # ── State manager (mirrors single-replay) ─────────────────────────────
    @app.callback(
        Output("replay-store", "data"),
        Input("update-interval", "n_intervals"),
        Input("replay-scrubber", "value"),
        Input("btn-play-pause", "n_clicks"),
        Input("btn-step-back", "n_clicks"),
        Input("btn-step-fwd", "n_clicks"),
        Input("btn-start", "n_clicks"),
        Input("btn-end", "n_clicks"),
        Input("speed-control", "value"),
        State("replay-store", "data"),
        prevent_initial_call=True,
    )
    def _cmp_state_manager(
        n_intervals, scrubber_val, play_clicks, back_clicks,
        fwd_clicks, start_clicks, end_clicks, speed_val, store,
    ):
        store = store or {"tick_index": 0, "playing": False, "speed": 1.0}
        triggered = dash.callback_context.triggered_id
        if triggered == "replay-scrubber":
            return {**store, "tick_index": int(scrubber_val or 0), "playing": False}
        if triggered == "btn-play-pause":
            playing = not store["playing"]
            if store["tick_index"] >= max_tick_a:
                playing = False
            return {**store, "playing": playing}
        if triggered == "btn-step-back":
            return {**store,
                    "tick_index": max(0, store["tick_index"] - 1),
                    "playing": False}
        if triggered == "btn-step-fwd":
            return {**store,
                    "tick_index": min(max_tick_a, store["tick_index"] + 1),
                    "playing": False}
        if triggered == "btn-start":
            return {**store, "tick_index": 0, "playing": False}
        if triggered == "btn-end":
            return {**store, "tick_index": max_tick_a, "playing": False}
        if triggered == "speed-control":
            return {**store, "speed": float(speed_val or 1.0)}
        if triggered == "update-interval":
            if not store.get("playing", False):
                return store
            dt_a_val = source_a.meta.get("dt", 0.1)
            advance = max(1, round(0.25 * store.get("speed", 1.0) / dt_a_val))
            new_idx = min(max_tick_a, store["tick_index"] + advance)
            return {**store, "tick_index": new_idx,
                    "playing": new_idx < max_tick_a}
        return store

    # ── Graph A ───────────────────────────────────────────────────────────
    @app.callback(
        Output("awareness-graph-a", "elements"),
        Output("elapsed-label-cmp", "children"),
        Output("replay-pos-label", "children"),
        Output("btn-play-pause", "children"),
        Output("layout-initialized-a", "data"),
        Input("replay-store", "data"),
        Input("threshold-slider", "value"),
        State("layout-initialized-a", "data"),
    )
    def _cmp_graph_a(store, threshold, layout_done):
        store = store or {"tick_index": 0}
        tick_a = store.get("tick_index", 0)
        playing = store.get("playing", False)
        snap = source_a.snapshot_at(tick_a, threshold=threshold or 0.0)
        elements = snap["elements"] if not layout_done else _strip_positions(snap["elements"])
        t = snap["elapsed"]
        total_t = source_a.total_duration()
        return (
            elements,
            f"t = {t:.1f}s",
            f"tick {tick_a} / {max_tick_a}  ({t:.1f}s / {total_t:.1f}s)",
            "⏸" if playing else "▶",
            True,
        )

    # ── Graph B ───────────────────────────────────────────────────────────
    @app.callback(
        Output("awareness-graph-b", "elements"),
        Output("layout-initialized-b", "data"),
        Input("replay-store", "data"),
        Input("threshold-slider", "value"),
        State("layout-initialized-b", "data"),
    )
    def _cmp_graph_b(store, threshold, layout_done):
        store = store or {"tick_index": 0}
        tick_b = _aligned_tick_b(store.get("tick_index", 0))
        snap = source_b.snapshot_at(tick_b, threshold=threshold or 0.0)
        elements = snap["elements"] if not layout_done else _strip_positions(snap["elements"])
        return elements, True

    # ── Inspector A ───────────────────────────────────────────────────────
    @app.callback(
        Output("hover-tooltip-a", "children"),
        Output("sparkline-chart-a", "figure"),
        Input("awareness-graph-a", "mouseoverNodeData"),
        Input("replay-store", "data"),
    )
    def _cmp_inspector_a(node_data, store):
        store = store or {"tick_index": 0}
        tick_a = store.get("tick_index", 0)
        cursor_t = source_a.elapsed_at(tick_a)
        if node_data is None:
            return _hover_card(None), _empty_sparkline()
        node_id = node_data.get("id", "")
        snap = source_a.snapshot_at(tick_a, threshold=0.0)
        current_data = next(
            (e["data"] for e in snap["elements"] if e.get("data", {}).get("id") == node_id),
            node_data,
        )
        hist = source_a.history_at(node_id, tick_a) if node_id else []
        return _hover_card(current_data), _make_sparkline(hist, node_id, cursor_t)

    # ── Inspector B ───────────────────────────────────────────────────────
    @app.callback(
        Output("hover-tooltip-b", "children"),
        Output("sparkline-chart-b", "figure"),
        Input("awareness-graph-b", "mouseoverNodeData"),
        Input("replay-store", "data"),
    )
    def _cmp_inspector_b(node_data, store):
        store = store or {"tick_index": 0}
        tick_b = _aligned_tick_b(store.get("tick_index", 0))
        cursor_t = source_b.elapsed_at(tick_b)
        if node_data is None:
            return _hover_card(None), _empty_sparkline()
        node_id = node_data.get("id", "")
        snap = source_b.snapshot_at(tick_b, threshold=0.0)
        current_data = next(
            (e["data"] for e in snap["elements"] if e.get("data", {}).get("id") == node_id),
            node_data,
        )
        hist = source_b.history_at(node_id, tick_b) if node_id else []
        return _hover_card(current_data), _make_sparkline(hist, node_id, cursor_t)

    # ── Events strip cursor ───────────────────────────────────────────────
    @app.callback(
        Output("events-strip", "figure"),
        Input("replay-store", "data"),
    )
    def _cmp_events_cursor(store):
        store = store or {"tick_index": 0}
        t = source_a.elapsed_at(store.get("tick_index", 0))
        patched = Patch()
        patched["layout"]["shapes"] = [{
            "type": "line",
            "x0": t, "x1": t, "y0": 0, "y1": 1,
            "xref": "x", "yref": "paper",
            "line": {"color": "rgba(255,255,255,0.55)",
                     "width": 1.5, "dash": "dot"},
        }]
        return patched

    return app


# ── App factory ───────────────────────────────────────────────────────────────

def build_app(source: 'SimulationRunner | ReplayReader',
              source_b: 'ReplayReader | None' = None,
              scenario: str = "unknown") -> dash.Dash:
    """
    Build and return the Dash app.

    Accepts a SimulationRunner (live), a ReplayReader (replay), or two
    ReplayReaders via source_b (compare mode).
    """
    from awareness_manager.visualization.replay_reader import ReplayReader

    if source_b is not None:
        return _build_compare_app(source, source_b)

    is_replay = isinstance(source, ReplayReader)
    app = dash.Dash(
        __name__,
        title="Awareness Manager" + (" - Replay" if is_replay else ""),
        suppress_callback_exceptions=True,
    )

    # ── Shared top-bar ────────────────────────────────────────────────────
    if is_replay:
        meta = source.meta
        schema_v = meta.get("schema_version", 1)
        if schema_v >= 2:
            strat = meta.get("strategy", "awareness_manager")
            budget_val = meta.get("budget", "?")
            alpha_val = meta.get("strategy_params", {}).get("alpha", "?")
            fc = meta.get("strategy_params", {}).get("feature_config", {})
        else:
            params = meta.get("am_params", {})
            strat = "awareness_manager"
            budget_val = params.get("budget", "?")
            alpha_val = params.get("alpha", "?")
            fc = params.get("feature_config", {})
        budget_display = ("∞" if (isinstance(budget_val, int) and budget_val < 0)
                          else str(budget_val))
        active_fs = "+".join(f"F{i+1}" for i, v in enumerate(
            [fc.get("f1"), fc.get("f2"), fc.get("f3"), fc.get("f4"), fc.get("f5")]
        ) if v) if fc else "-"
        summary_children = [
            html.Span(meta.get("scenario", "?"), style={"color": _GOLD}),
            html.Span(f"  [{strat}]",
                      style={"color": _DIM, "font-size": "9px"}),
            html.Span(f"  B={budget_display}",
                      style={"color": _DIM, "font-size": "9px"}),
            *([] if not alpha_val or alpha_val == "?" else [
                html.Span(f"  α={alpha_val}",
                          style={"color": _DIM, "font-size": "9px"}),
            ]),
            *([] if not active_fs or active_fs == "-" else [
                html.Span(f"  [{active_fs}]",
                          style={"color": _DIM, "font-size": "9px"}),
            ]),
            html.Span(f"  {source.goal_transition_count()} transitions"
                      f"  {source.surprise_event_count()} surprises",
                      style={"color": _DIM, "font-size": "9px"}),
        ]
        topbar_right = html.Div(summary_children,
                                style={"display": "flex", "align-items": "center",
                                       "gap": "0px"})
    else:
        avail_s = int(source._history_maxlen * source._dt)
        topbar_right = html.Div(
            style={"display": "flex", "align-items": "center", "gap": "10px"},
            children=[
                html.Span(id="budget-label",
                          style={"color": _TEXT, "font-size": "10px",
                                 "white-space": "nowrap"}),
                html.Div(id="budget-bar-slot"),
                html.Button(
                    f"Save last {avail_s}s",
                    id="save-btn",
                    n_clicks=0,
                    style={
                        "background": "#1a2a1a",
                        "color": "#88dd88",
                        "border": "1px solid #44aa44",
                        "border-radius": "8px",
                        "padding": "2px 10px",
                        "font-size": "10px",
                        "font-family": "monospace",
                        "cursor": "pointer",
                        "white-space": "nowrap",
                    },
                ),
                html.Span(id="save-status",
                          style={"color": _DIM, "font-size": "9px",
                                 "white-space": "nowrap"}),
            ],
        )

    top_bar = html.Div(
        style={
            "background": _PANEL,
            "padding": "7px 16px",
            "display": "flex",
            "align-items": "center",
            "gap": "14px",
            "border-bottom": "1px solid #333355",
            "flex-shrink": "0",
        },
        children=[
            html.Span("Goal:", style={"color": _DIM, "font-size": "11px"}),
            html.Span(id="goal-badge", style={
                "background": "#1a1800",
                "color": _GOLD,
                "border": f"1px solid {_GOLD}",
                "border-radius": "12px",
                "padding": "2px 11px",
                "font-size": "11px",
                "font-weight": "bold",
                "white-space": "nowrap",
            }),
            html.Span(id="queue-text",
                      style={"color": _DIM, "font-size": "10px",
                             "white-space": "nowrap"}),
            html.Div(style={"flex": "1"}),  # spacer
            topbar_right,
            html.Span(id="elapsed-label",
                      style={"color": _DIM, "font-size": "10px",
                             "white-space": "nowrap"}),
        ],
    )

    threshold_row = html.Div(
        style={
            "background": "#1a1a2e",
            "padding": "4px 16px",
            "display": "flex",
            "align-items": "center",
            "gap": "10px",
            "border-bottom": "1px solid #2a2a44",
            "flex-shrink": "0",
        },
        children=[
            html.Span("Activation threshold:",
                      style={"color": _DIM, "font-size": "10px",
                             "white-space": "nowrap"}),
            html.Div(dcc.Slider(
                id="threshold-slider",
                min=0.0, max=1.0, step=0.01, value=0.0,
                marks={
                    0:   {"label": "0",   "style": {"color": _DIM, "font-size": "9px"}},
                    0.5: {"label": "0.5", "style": {"color": _DIM, "font-size": "9px"}},
                    1:   {"label": "1",   "style": {"color": _DIM, "font-size": "9px"}},
                },
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ), style={"flex": "1"}),
            html.Span(id="threshold-label", style={
                "color": _DIM, "font-size": "10px", "min-width": "30px",
                "text-align": "right", "background": "#2a2a3e",
                "border-radius": "6px", "padding": "1px 6px",
            }),
        ],
    )

    # ── Mode-specific layout ──────────────────────────────────────────────
    if is_replay:
        reader = source
        total_ticks = reader.tick_count
        max_tick = max(0, total_ticks - 1)
        total_dur = reader.total_duration()

        _btn_style = {
            "background": "#252540",
            "color": _TEXT,
            "border": "1px solid #444466",
            "border-radius": "6px",
            "padding": "3px 10px",
            "font-size": "13px",
            "font-family": "monospace",
            "cursor": "pointer",
            "line-height": "1",
        }

        replay_controls = html.Div(
            style={
                "background": "#181828",
                "border-top": "1px solid #333355",
                "padding": "6px 14px 2px 14px",
                "flex-shrink": "0",
            },
            children=[
                # Button row
                html.Div(
                    style={"display": "flex", "align-items": "center", "gap": "6px",
                           "margin-bottom": "4px"},
                    children=[
                        html.Button("⏮", id="btn-start", n_clicks=0,
                                    title="Go to start", style=_btn_style),
                        html.Button("◀", id="btn-step-back", n_clicks=0,
                                    title="Step back 1 tick", style=_btn_style),
                        html.Button("▶", id="btn-play-pause", n_clicks=0,
                                    title="Play / Pause", style={
                                        **_btn_style,
                                        "background": "#1a2a3a",
                                        "border-color": _ACCENT,
                                        "color": _ACCENT,
                                        "padding": "3px 14px",
                                    }),
                        html.Button("▶", id="btn-step-fwd", n_clicks=0,
                                    title="Step forward 1 tick", style=_btn_style),
                        html.Button("⏭", id="btn-end", n_clicks=0,
                                    title="Go to end", style=_btn_style),
                        html.Span("Speed:", style={"color": _DIM, "font-size": "10px",
                                                    "margin-left": "10px"}),
                        dcc.Dropdown(
                            id="speed-control",
                            options=[
                                {"label": "0.25x", "value": 0.25},
                                {"label": "0.5x",  "value": 0.5},
                                {"label": "1x",    "value": 1.0},
                                {"label": "4x",    "value": 4.0},
                                {"label": "16x",   "value": 16.0},
                            ],
                            value=1.0,
                            clearable=False,
                            style={
                                "width": "80px",
                                "background": "#252540",
                                "color": _TEXT,
                                "font-size": "10px",
                                "font-family": "monospace",
                                "border": "none",
                            },
                        ),
                        html.Span(id="replay-pos-label",
                                  style={"color": _TEXT, "font-size": "10px",
                                         "font-family": "monospace",
                                         "margin-left": "10px",
                                         "white-space": "nowrap"}),
                    ],
                ),
                # Scrubber
                html.Div(
                    dcc.Slider(
                        id="replay-scrubber",
                        min=0, max=max_tick, step=1, value=0,
                        marks={},
                        tooltip={"placement": "top", "always_visible": False},
                        updatemode="drag",
                    ),
                    style={"margin": "0 -4px 2px -4px"},
                ),
                # Events strip
                html.Div(
                    dcc.Graph(
                        id="events-strip",
                        figure=_make_events_strip(reader.all_events, total_dur),
                        config={"displayModeBar": False, "staticPlot": False},
                        style={"height": "32px"},
                    ),
                    style={"margin": "0 -4px"},
                ),
            ],
        )

        app.layout = html.Div(
            style={"background": _BG, "color": _TEXT, "font-family": "monospace",
                   "height": "100vh", "display": "flex",
                   "flex-direction": "column", "overflow": "hidden"},
            children=[
                top_bar,
                threshold_row,
                html.Div(
                    style={"flex": "1", "display": "flex", "overflow": "hidden"},
                    children=[
                        _graph_pane(),
                        _info_panel(["Replay inspector"]),
                    ],
                ),
                replay_controls,
                dcc.Store(id="replay-store",
                          data={"tick_index": 0, "playing": False, "speed": 1.0}),
                dcc.Interval(id="update-interval", interval=250, n_intervals=0),
                dcc.Store(id="layout-initialized", data=False),
            ],
        )

    else:
        runner = source

        controller_row = html.Div(
            id="controller-row",
            style={
                "background": "#181828",
                "padding": "5px 16px",
                "display": "none",  # shown dynamically when controller_state present
                "align-items": "center",
                "gap": "10px",
                "border-bottom": "1px solid #2a2a44",
                "flex-shrink": "0",
            },
            children=[
                html.Span("Controller:",
                          style={"color": _DIM, "font-size": "10px",
                                 "white-space": "nowrap"}),
                html.Span(id="ctrl-state-badge",
                          style={"font-size": "10px", "font-weight": "bold",
                                 "font-family": "monospace", "white-space": "nowrap"}),
                html.Span(id="ctrl-target-badge",
                          style={"color": _DIM, "font-size": "10px",
                                 "font-family": "monospace", "white-space": "nowrap"}),
            ],
        )

        app.layout = html.Div(
            style={"background": _BG, "color": _TEXT, "font-family": "monospace",
                   "height": "100vh", "display": "flex",
                   "flex-direction": "column", "overflow": "hidden"},
            children=[
                top_bar,
                threshold_row,
                controller_row,
                html.Div(
                    style={"flex": "1", "display": "flex", "overflow": "hidden"},
                    children=[
                        _graph_pane(),
                        _info_panel(["Node inspector"]),
                    ],
                ),
                dcc.Interval(id="update-interval", interval=250, n_intervals=0),
                dcc.Store(id="layout-initialized", data=False),
            ],
        )

    # ── Shared callback: threshold label ──────────────────────────────────
    @app.callback(
        Output("threshold-label", "children"),
        Input("threshold-slider", "value"),
    )
    def update_threshold_label(value):
        return f"≥ {value:.2f}"

    # ── Mode-specific callbacks ───────────────────────────────────────────
    if is_replay:
        reader = source
        max_tick = max(0, reader.tick_count - 1)

        # ── State manager: all inputs → replay-store ──────────────────────
        @app.callback(
            Output("replay-store", "data"),
            Input("update-interval", "n_intervals"),
            Input("replay-scrubber", "value"),
            Input("btn-play-pause", "n_clicks"),
            Input("btn-step-back", "n_clicks"),
            Input("btn-step-fwd", "n_clicks"),
            Input("btn-start", "n_clicks"),
            Input("btn-end", "n_clicks"),
            Input("speed-control", "value"),
            State("replay-store", "data"),
            prevent_initial_call=True,
        )
        def update_replay_state(
            n_intervals, scrubber_val, play_clicks, back_clicks,
            fwd_clicks, start_clicks, end_clicks, speed_val, store,
        ):
            store = store or {"tick_index": 0, "playing": False, "speed": 1.0}
            triggered = dash.callback_context.triggered_id

            if triggered == "replay-scrubber":
                return {**store, "tick_index": int(scrubber_val or 0), "playing": False}

            if triggered == "btn-play-pause":
                playing = not store["playing"]
                if store["tick_index"] >= max_tick:
                    playing = False  # at end: play is a no-op
                return {**store, "playing": playing}

            if triggered == "btn-step-back":
                return {**store,
                        "tick_index": max(0, store["tick_index"] - 1),
                        "playing": False}

            if triggered == "btn-step-fwd":
                return {**store,
                        "tick_index": min(max_tick, store["tick_index"] + 1),
                        "playing": False}

            if triggered == "btn-start":
                return {**store, "tick_index": 0, "playing": False}

            if triggered == "btn-end":
                return {**store, "tick_index": max_tick, "playing": False}

            if triggered == "speed-control":
                return {**store, "speed": float(speed_val or 1.0)}

            if triggered == "update-interval":
                if not store.get("playing", False):
                    return store
                dt = reader.meta.get("dt", 0.1)
                advance = max(1, round(0.25 * store.get("speed", 1.0) / dt))
                new_idx = min(max_tick, store["tick_index"] + advance)
                playing = new_idx < max_tick
                return {**store, "tick_index": new_idx, "playing": playing}

            return store

        # ── Graph update (replay) ─────────────────────────────────────────
        @app.callback(
            Output("awareness-graph", "elements"),
            Output("goal-badge", "children"),
            Output("queue-text", "children"),
            Output("elapsed-label", "children"),
            Output("replay-pos-label", "children"),
            Output("btn-play-pause", "children"),
            Output("layout-initialized", "data"),
            Input("replay-store", "data"),
            Input("threshold-slider", "value"),
            State("layout-initialized", "data"),
        )
        def update_graph_replay(store, threshold, layout_done):
            store = store or {"tick_index": 0}
            tick_index = store.get("tick_index", 0)
            playing = store.get("playing", False)
            snap = reader.snapshot_at(tick_index, threshold=threshold or 0.0)
            elements = snap["elements"] if not layout_done else _strip_positions(snap["elements"])
            cur_t = snap["elapsed"]
            total_t = reader.total_duration()
            return (
                elements,
                snap["goal_id"],
                f"replay: {snap['queue_text']}",
                f"t = {cur_t:.1f}s",
                f"tick {tick_index} / {max_tick}  ({cur_t:.1f}s / {total_t:.1f}s)",
                "⏸" if playing else "▶",
                True,
            )

        # ── Inspector (replay) ────────────────────────────────────────────
        @app.callback(
            Output("hover-tooltip", "children"),
            Output("sparkline-chart", "figure"),
            Input("awareness-graph", "mouseoverNodeData"),
            Input("replay-store", "data"),
        )
        def update_inspector_replay(node_data, store):
            store = store or {"tick_index": 0}
            tick_index = store.get("tick_index", 0)
            cursor_t = reader.elapsed_at(tick_index)

            if node_data is None:
                return _hover_card(None), _empty_sparkline()

            node_id = node_data.get("id", "")
            snap = reader.snapshot_at(tick_index, threshold=0.0)
            current_data = next(
                (e["data"] for e in snap["elements"] if e.get("data", {}).get("id") == node_id),
                node_data,
            )
            hist = reader.history_at(node_id, tick_index) if node_id else []
            return _hover_card(current_data), _make_sparkline(hist, node_id, cursor_t)

        # ── Events strip cursor update (Patch for efficiency) ─────────────
        @app.callback(
            Output("events-strip", "figure"),
            Input("replay-store", "data"),
        )
        def update_events_cursor(store):
            store = store or {"tick_index": 0}
            tick_index = store.get("tick_index", 0)
            t = reader.elapsed_at(tick_index)
            patched = Patch()
            patched["layout"]["shapes"] = [{
                "type": "line",
                "x0": t, "x1": t,
                "y0": 0, "y1": 1,
                "xref": "x", "yref": "paper",
                "line": {"color": "rgba(255,255,255,0.55)",
                         "width": 1.5, "dash": "dot"},
            }]
            return patched

    else:
        runner = source

        # ── Graph update (live) ───────────────────────────────────────────
        @app.callback(
            Output("awareness-graph", "elements"),
            Output("goal-badge", "children"),
            Output("queue-text", "children"),
            Output("budget-label", "children"),
            Output("budget-bar-slot", "children"),
            Output("elapsed-label", "children"),
            Output("layout-initialized", "data"),
            Output("controller-row", "style"),
            Output("ctrl-state-badge", "children"),
            Output("ctrl-state-badge", "style"),
            Output("ctrl-target-badge", "children"),
            Input("update-interval", "n_intervals"),
            Input("threshold-slider", "value"),
            State("layout-initialized", "data"),
        )
        def update_graph(n_intervals, threshold, layout_done):
            snap = runner.snapshot(threshold=threshold or 0.0)
            elements = snap["elements"] if not layout_done else _strip_positions(snap["elements"])
            n, b = snap["n_scheduled"], snap["budget"]
            unlimited = snap.get("budget_unlimited", False)
            if unlimited:
                sched_label = f"Scheduled: {n}  (Unlimited refresh)"
                bar = html.Span()
            else:
                sched_label = f"Scheduled: {n} / {b}"
                bar = _budget_bar_html(n, b)

            # Controller state panel (social_serving scenario with AMController hook)
            cs = snap.get("controller_state")
            if cs:
                state_str  = cs.get("state", "monitoring")
                target_str = cs.get("target")
                row_style  = {
                    "background": "#181828", "padding": "5px 16px",
                    "display": "flex", "align-items": "center", "gap": "10px",
                    "border-bottom": "1px solid #2a2a44", "flex-shrink": "0",
                }
                badge_color  = "#44cc88" if state_str == "monitoring" else "#f4944a"
                badge_style  = {"color": badge_color, "font-size": "10px",
                                "font-weight": "bold", "font-family": "monospace",
                                "white-space": "nowrap"}
                state_label  = state_str.upper()
                target_label = f"→ {target_str}" if target_str else ""
            else:
                row_style    = {"display": "none"}
                badge_style  = {}
                state_label  = ""
                target_label = ""

            return (
                elements,
                snap["goal_id"],
                f"queue: {snap['queue_text']}",
                sched_label,
                bar,
                f"t = {snap['elapsed']:.1f}s",
                True,
                row_style,
                state_label,
                badge_style,
                target_label,
            )

        # ── Inspector (live) ──────────────────────────────────────────────
        @app.callback(
            Output("hover-tooltip", "children"),
            Output("sparkline-chart", "figure"),
            Input("awareness-graph", "mouseoverNodeData"),
            Input("update-interval", "n_intervals"),
        )
        def update_inspector(node_data, _n):
            if node_data is None:
                return _hover_card(None), _empty_sparkline()
            node_id = node_data.get("id", "")
            snap = runner.snapshot(threshold=0.0)
            current_data = next(
                (e["data"] for e in snap["elements"] if e.get("data", {}).get("id") == node_id),
                node_data,
            )
            hist = runner.history(node_id) if node_id else []
            return _hover_card(current_data), _make_sparkline(hist, node_id)

        # ── Save trace button ─────────────────────────────────────────────
        @app.callback(
            Output("save-status", "children"),
            Input("save-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def save_trace(n_clicks):
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            out_dir = Path("traces") / f"flush_{timestamp}"
            try:
                runner.flush_trace(out_dir, scenario=scenario)
                return f"→ {out_dir}"
            except Exception as exc:
                return f"Error: {exc}"

    return app


def run(
    source: 'SimulationRunner | ReplayReader',
    source_b: 'ReplayReader | None' = None,
    scenario: str = "unknown",
    debug: bool = False,
    port: int = 8050,
) -> None:
    """Build the app, start the runner (live mode only), and block on the server."""
    from awareness_manager.visualization.replay_reader import ReplayReader

    app = build_app(source, source_b=source_b, scenario=scenario)

    if hasattr(source, 'start'):
        source.start()
        try:
            app.run(debug=debug, port=port, use_reloader=False)
        finally:
            source.stop()
    else:
        app.run(debug=debug, port=port, use_reloader=False)
