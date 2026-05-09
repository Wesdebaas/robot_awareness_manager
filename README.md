# Robot Awareness Management

Master's thesis project - TU Delft / CoreSense Horizon Europe.

**Top-down robot awareness management**: given a mission goal, the system pre-tunes which concepts the robot should observe *before* entering a situation. The Awareness Manager (AM) answers the question "which concepts should the robot refresh right now?" by combining spreading activation, anticipatory pre-tuning, and epistemic error prioritisation under a fixed observation budget.

## Overview

The system models a robot's knowledge state as a semantic graph where each concept carries an **epistemic error** E ∈ [0, 1] (0 = fresh, 1 = fully unknown). Concepts decay toward uncertainty over time; active observation reduces their error. Five grounding formulas determine which concepts to observe and by how much:

| Formula | Name | Role |
|---------|------|------|
| F1 | Spreading Activation | Relevance decays with graph distance from the goal |
| F2 | Anticipatory Horizon | Future goals contribute attention proportional to proximity in time |
| F3 | Utility Saturation | Refresh amount calibrated to drift accumulated since last observation |
| F4 | Quadratic Cost Constraint | Search depth derived from memory budget B |
| F5 | Epistemic Error / Entropy | Drift accumulates over time; priority = E x A |

Two baseline strategies are included for comparison: **AlwaysOnBaseline** (no budget, refreshes everything) and **ReactiveBaseline** (1-hop rule-based, round-robin, no anticipation).

## Installation

Requires **ROS2 Humble** and Python 3.10+. From the workspace root:

```bash
pip install dash plotly scipy
colcon build --packages-select awareness_manager
source install/setup.bash
```

For the dashboard only (no ROS2 required):

```bash
pip install dash plotly scipy
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/awareness_manager
```

## Quick Start

### Interactive Dashboard

Live simulation with the Awareness Manager on the PV inspection scenario:

```bash
python3 src/awareness_manager/demos/run_dashboard.py
```

Other modes:

```bash
# Run with a different strategy
python3 src/awareness_manager/demos/run_dashboard.py --strategy reactive
python3 src/awareness_manager/demos/run_dashboard.py --strategy always_on

# Log a trace to disk while running live
python3 src/awareness_manager/demos/run_dashboard.py --log
python3 src/awareness_manager/demos/run_dashboard.py --log --log-path traces/my_run

# Replay a saved trace
python3 src/awareness_manager/demos/run_dashboard.py --replay traces/my_run

# Side-by-side A/B comparison of two traces
python3 src/awareness_manager/demos/run_dashboard.py \
    --replay traces/am_run --compare traces/reactive_run
```

Open `http://localhost:8050` in a browser. The dashboard shows the semantic graph with
concept epistemic errors, per-concept attention sparklines, and an event timeline.

### Batch Evaluation

Run the full budget x observation-interval sweep (27 runs, ~3 min):

```bash
python3 -m awareness_manager.evaluation batch \
    --scenario pv_inspection \
    --strategies awareness_manager,reactive,always_on \
    --budget 1,2,4 \
    --obs-interval 1.0,5.0,10.0 \
    --output experiments/budget_obsint_sweep/
```

Generate the report from saved traces:

```bash
python3 -m awareness_manager.evaluation report \
    --experiment experiments/budget_obsint_sweep/ \
    --output reports/budget_obsint_sweep/
```

Outputs: `summary.csv`, `plots.html` (interactive Plotly figures), `summary.md`
(metric table, per-regime breakdown, Mann-Whitney U tests).

## Project Structure

```
src/awareness_manager/
├── awareness_manager/
│   ├── awareness_manager.py        # Core AM (all 5 formulas)
│   ├── knowledge_base.py           # Semantic graph with attention + epistemic error
│   ├── instance_knowledge_base.py  # Instance-level graph
│   ├── concept.py                  # Concept / InstanceConcept dataclasses
│   ├── feature_config.py           # Toggle formulas F1–F5 (ablation)
│   ├── baselines/
│   │   ├── strategy.py             # AttentionStrategy Protocol (PEP 544)
│   │   ├── always_on.py            # AlwaysOnBaseline
│   │   └── reactive.py             # ReactiveBaseline
│   ├── scenarios/
│   │   ├── pv_inspection.py        # Primary scenario: drone inspects solar plant (D7.1)
│   │   ├── manufacturing_d61.py    # Secondary scenario (D6.1)
│   │   └── birdhouse.py            # Toy scenario for unit tests
│   ├── visualization/
│   │   ├── dashboard.py            # Dash app (live / replay / compare)
│   │   ├── runner.py               # SimulationRunner (threaded real-time loop)
│   │   ├── trace_logger.py         # TraceLogger - streams ticks to JSONL
│   │   ├── replay_reader.py        # ReplayReader - offline playback
│   │   └── snapshot.py             # Layout computation, snapshot serialisation
│   └── evaluation/
│       ├── metrics.py              # M1–M6 metric library (pure functions over traces)
│       ├── batch.py                # Multi-run experiment driver
│       ├── report.py               # Report generator (CSV, HTML, Markdown)
│       └── __main__.py             # CLI entry point
├── demos/
│   └── run_dashboard.py            # Dashboard entry point
└── test/
    ├── test_awareness_manager.py
    ├── test_knowledge_base.py
    └── test_instance_knowledge_base.py
```

## Evaluation Metrics

Metrics are computed from saved traces - no re-running required.

| Metric | Description | Lower is better |
|--------|-------------|-----------------|
| M1 E_relevant | Mean epistemic error over goal-relevant concepts | Yes |
| M2 E_irrelevant | Mean E over irrelevant concepts | - |
| M3 budget_util | Fraction of budget used per tick | - |
| M4 lag (s) | Seconds to recover E < 0.1 after goal transition | Yes |
| M5 cache_hit | Was next goal's neighborhood pre-cached? | No (higher better) |
| M6 rel_fraction | Fraction of schedule slots on relevant concepts | No (higher better) |

**Goal neighborhood**: 1-hop semantic neighbors of the active goal with `decay_rate > 0`.
Strategy-agnostic - matches ReactiveBaseline's active set exactly.

### Key Result (budget x obs-interval sweep)

At observation interval T = 5 s and T = 10 s:
- AM achieves **zero cognitive lag** after goal transitions vs Reactive 6 s / 2.6 s
- AM M1 ≈ 0.027 (relevant concepts stay fresh) vs Reactive ≈ 0.099
- AlwaysOn M1 ≈ 0.000 (uncapped budget - the lower bound)
- Pooled M4 Mann-Whitney U: **p = 0.003** (AM vs Reactive, significant)

T = 1 s is a degenerate regime (refresh amplitude too small to overcome drift for all
strategies). See `reports/budget_obsint_sweep/summary.md` for the full breakdown.

## Scenario: PV Inspection (CoreSense D7.1)

A drone inspects a photovoltaic solar plant. Two task goals create the key behavioral contrast:

- **`inspect_pv_field`** (active at t = 0): attends to `solar_panel`, `drone_camera`, `light_conditions`, `wind_speed`, `drone_battery`, `drone_gps`
- **`emergency_landing`** (queued at ETA = 30 s): attends to `airspace`, `landing_zone`, `drone_battery`, `wind_speed`

The AM begins pre-tuning for `emergency_landing` as its ETA decreases (F2), so concepts
like `airspace` and `landing_zone` gain attention before the transition fires. Reactive
has no such mechanism and is always "surprised" by the goal switch.

## Running Tests

```bash
cd src/awareness_manager
python3 -m pytest test/ -v
```

## Context

- **CoreSense** (https://coresense.eu/) - Horizon Europe cognitive robotics project
- **Primary thesis claim**: Awareness is not a static database but a transient functional
  state that must be actively defended against entropy via parameterised update rates.
- **Closest related system**: CRESTA (Gasperini et al. 2026) - centralized manager between
  World Model and Task Awareness, similar architecture but focused on task execution
  rather than top-down attention allocation.
