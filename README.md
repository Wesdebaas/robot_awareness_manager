# Robot Awareness Management

Master's thesis project - TU Delft / CoreSense Horizon Europe.

**Top-down robot awareness management**: given a mission goal, the system pre-tunes which concepts the robot should observe *before* entering a situation. The Awareness Manager (AM) answers the question "which concepts should the robot refresh right now?" by combining spreading activation, anticipatory pre-tuning, and epistemic error prioritisation under a fixed observation budget.

## Overview

The system models a robot's knowledge state as a semantic graph where each concept carries an **epistemic error** E ∈ [0, 1] (0 = fresh, 1 = fully unknown). Concepts decay toward uncertainty over time; active observation reduces their error. Six grounding formulas determine which concepts to observe and by how much:

| Formula | Name | Role |
|---------|------|------|
| F1 | Spreading Activation | Relevance decays with graph distance from the goal |
| F2 | Anticipatory Horizon | Future goals contribute attention proportional to proximity in time |
| F3 | Utility Saturation | Refresh amount calibrated to drift accumulated since last observation |
| F4 | Quadratic Cost Constraint | Search depth derived from memory budget B |
| F5 | Epistemic Error / Entropy | Drift accumulates over time; priority = E x A |
| F6 | Spatial Opportunity Cost | Scheduling priority divided by travel cost to reach a concept |

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

Live simulation with the Awareness Manager on the social serving scenario:

```bash
python3 src/awareness_manager/demos/run_dashboard.py --scenario social_serving
```

Or on the PV inspection scenario:

```bash
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection
```

Other modes:

```bash
# Run with a different strategy
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection --strategy reactive
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection --strategy always_on

# Log a trace to disk while running live
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection --log
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection --log --log-path traces/my_run

# Replay a saved trace
python3 src/awareness_manager/demos/run_dashboard.py --replay traces/my_run

# Side-by-side A/B comparison of two traces
python3 src/awareness_manager/demos/run_dashboard.py \
    --replay traces/am_run --compare traces/reactive_run
```

Open `http://localhost:8050` in a browser. The dashboard shows the semantic graph with
concept epistemic errors, per-concept attention sparklines, and an event timeline.

### ROS2 Live Mode

Connect the dashboard to a running `awareness_node` instead of an internal simulation:

```bash
# Terminal 1 — start the full ROS2 stack (node + dashboard together)
ros2 launch awareness_manager awareness_demo.launch.py

# Or start them separately:
# Terminal 1
ros2 run awareness_manager awareness_node

# Terminal 2
python3 src/awareness_manager/demos/run_dashboard.py --ros --scenario social_serving
```

In `--ros` mode the dashboard subscribes to `awareness/state`, `awareness/schedule`,
`awareness/goal`, and `awareness/controller_state` topics and displays live data from
the running node. The controller state panel (MONITORING / SERVING badge + target person)
is driven by the node's internal state machine.

### Evaluation

Six focused experiments, each validating one contribution:

```bash
# All six in sequence (~5 min)
python3 -m awareness_manager.evaluation experiments

# Individual experiments
python3 -m awareness_manager.evaluation exp1   # Scaling: detection latency vs N and budget
python3 -m awareness_manager.evaluation exp2   # Goal conditioning: E×A vs pure epistemic
python3 -m awareness_manager.evaluation exp3   # Anticipatory horizon: F2 on vs off (ETA sweep)
python3 -m awareness_manager.evaluation exp4   # Instance KB vs class-only (multi-instance sweep)
python3 -m awareness_manager.evaluation exp5   # Formula ablation: F1 spreading activation
python3 -m awareness_manager.evaluation exp6   # F6 spatial opportunity cost

# Abstract scenario smoke test and learnable-δ verification
python3 -m awareness_manager.evaluation abstract --quick
python3 -m awareness_manager.evaluation learnable-decay
```

## Project Structure

```
src/awareness_manager/
├── awareness_manager/
│   ├── awareness_manager.py         # Core AM: all 6 formulas + modular priority
│   ├── knowledge_base.py            # Semantic graph with attention + epistemic error
│   ├── instance_knowledge_base.py   # Instance-level graph (individual objects)
│   ├── concept.py                   # Concept / InstanceConcept dataclasses
│   ├── feature_config.py            # FeatureConfig (F1-F6 flags) + PriorityWeights
│   ├── ros_node.py                  # ROS2 node: ticks AM, drives SS controller
│   ├── book_finding_node.py         # ROS2 node: Find My Book Gazebo integration
│   ├── scripted_navigator_node.py   # Demo helper: scripted room-tour for book_finding
│   ├── baselines/
│   │   ├── strategy.py              # AttentionStrategy Protocol (PEP 544)
│   │   ├── always_on.py             # AlwaysOnBaseline
│   │   └── reactive.py              # ReactiveBaseline
│   ├── scenarios/
│   │   ├── pv_inspection.py / .ttl  # PV inspection scenario (CoreSense D7.1)
│   │   ├── social_serving.py / .ttl # Social serving scenario (waiter robot, 10 persons)
│   │   ├── book_finding.py / .ttl   # Find My Book scenario (MIRTE in house world)
│   │   ├── abstract_n.py            # Abstract N-variable KB builder (Telogenesis eval)
│   │   └── loader.py                # Generic TTL→KnowledgeBase loader (pyoxigraph)
│   ├── visualization/
│   │   ├── dashboard.py             # Dash app (live / replay / compare / ROS)
│   │   ├── runner.py                # SimulationRunner (threaded real-time loop)
│   │   ├── ros_source.py            # RosStateSource — live data from ROS2 topics
│   │   ├── trace_logger.py          # TraceLogger — streams ticks to JSONL
│   │   ├── replay_reader.py         # ReplayReader — offline playback
│   │   └── snapshot.py              # Layout computation, snapshot serialisation
│   └── evaluation/
│       ├── metrics.py               # M1-M5 metric library (pure functions over traces)
│       ├── experiments.py           # Six focused experiments (exp1-exp6)
│       ├── abstract_runner.py       # Abstract N-variable simulation (5 strategies)
│       └── __main__.py              # CLI: exp1-exp6/experiments/abstract/learnable-decay
├── demos/
│   └── run_dashboard.py             # Dashboard entry point (live, replay, compare, ROS)
├── launch/
│   ├── awareness_demo.launch.py     # ROS2 launch: awareness_node + dashboard
│   ├── pv_inspection_demo.launch.py # ROS2 launch: PV inspection (awareness_node only)
│   └── book_finding.launch.py       # Full Gazebo: gzserver + MIRTE + Nav2 + BookFindingNode
└── test/
    ├── test_awareness_manager.py
    ├── test_knowledge_base.py
    ├── test_instance_knowledge_base.py
    └── test_prediction_error.py
```

## Priority Formula

The scheduling priority of concept c at each tick:

```
P(c) = w_ea    × (E(c) × A_mission(c))   — staleness × spreading activation
     + w_surp  × prediction_error(c)      — Telogenesis surprise term (S̃_i)
     + w_f2    × A_anticipatory(c)         — anticipatory horizon contribution
     + w_urg   × urgency(c)               — instance-level unmet-need accumulator

sort_key(c) = P(c) / travel_cost(c)^w_tc  — F6 spatial opportunity cost
```

All weights are continuous floats (default 1.0, except `w_surp` which defaults to 0.0
for backward compatibility). Setting any weight to 0.0 removes that component.
Weights are exposed as ROS2 parameters and as `PriorityWeights` in the API.
`FeatureConfig` controls the structural formulas (F1–F6, learnable decay).

## Evaluation Metrics

Trace-based metrics (M1–M5) are pure functions over saved JSONL traces — no re-running required. Used by the dashboard compare view.

| Key | Description | Better |
|-----|-------------|--------|
| M1 `m1_e_at_transition` | Max E in new-goal's 1-hop neighborhood at transition tick | ↓ |
| M2 `m2_pre_transition_attn` | Mean attention to incoming-goal hood before the switch | ↑ |
| M3 `m3_lag_seconds` | Seconds until E_max < 0.1 after goal transition | ↓ |
| M4 `m4_e_relevant` | Run-mean epistemic error over goal-relevant concepts | ↓ |
| M5 `m5_budget_util` | Schedule slots used / budget (sanity check) | — |

**Goal neighborhood**: 1-hop semantic neighbors of the active goal with `decay_rate > 0`.

### Experimental Results

**Exp 1 — Scaling** (abstract N-variable, 5 seeds, mean ± std):
- PRIORITY-AM and PRIORITY-EPISTEMIC: **0.00 ± 0.00 ticks** detection latency at all N (up to N=96)
- REACTIVE (goal-aware round-robin): **~2.4 ticks**, independent of N (cycles R=6 relevant only)
- ROTATION (blind round-robin): grows linearly with N (~N/2 ticks)
- Priority scheduling is O(1) in concept space size; goal-awareness alone bounds latency to R/budget

**Exp 2 — Goal conditioning** (K_overlap sweep 0→6, N=12, K=R=6, obs_interval=20 s, 5 seeds):
- PRIORITY-AM and REACTIVE nearly identical in long-run E_relevant — goal-awareness captures most advantage
- PRIORITY-EPISTEMIC flat at E_relevant ≈ 0.59 regardless of K_overlap (no goal structure)
- AM advantage over EPISTEMIC: **0.33 at K_overlap=0** → **0.23 at K_overlap=6** (monotone, std < 0.01)
- Epistemic priority (AM vs REACTIVE) shows as lower detection latency (exp1) not lower mean E

**Exp 3 — Anticipatory horizon** (PV inspection, budget sweep B=1,2,4):
- F2 on: E_mean = **0.13** at transition; F2 off: **0.19** (Δ = 0.05, consistent across all budgets)
- Pre-transition emergency schedule hits: **30–42%** of budget slots (F2 on) vs **20–30%** (F2 off)
- E_max is dominated by `airspace` (slow decay, far semantic path) and is identical in both conditions

**Exp 4 — Instance KB** (multi-instance sweep, all 7 PV instances):
- With instances: all 7 instances detected at **tick 0**, final E 0.73–0.97
- Without instances: **never scheduled** (only parent class concepts are addressable)
- Demonstrates generality: result holds for all instance types (panels, batteries, landing zones, camera)

**Exp 5 — Formula ablation: F1 spreading activation** (abstract N-variable, N=12, K=6, R=3, K_overlap=0, budget=1, 5 seeds):
- F1 on:  E_relevant = **0.21 ± 0.00** — spreading activation focuses budget on 1-hop relevant concepts
- F1 off: E_relevant = **0.84 ± 0.00** — uniform A=1.0, volatile (2-hop, spiked) monopolize budget
- Δ = **0.63** (positive = F1 reduces E_relevant; 3× improvement in goal-relevant freshness)

**Exp 6 — F6 spatial opportunity cost** (social serving, serve_person_01 goal, robot at table_area, 50 ticks):
- F6 on:  mean travel cost = **2.00 s**, same-zone fraction = **1.00** (always picks table_area persons)
- F6 off: mean travel cost = **2.65 s**, same-zone fraction = **0.74** (26% restock_zone trips for beers)
- Mann-Whitney U p = **0.00** — F6 significantly reduces mean travel cost

**Learnable decay verification** (8 volatile + 8 stable concepts, 500 ticks, obs_interval=20 s):
- Volatile group mean δ: **0.527**; stable group: **0.100** (ratio 5.27×)
- Mann-Whitney U = 64 (perfect separation), **p = 2×10⁻⁴**

## Scenarios

### PV Inspection (CoreSense D7.1)

A drone inspects a photovoltaic solar plant. Two task goals create the key behavioral contrast:

- **`inspect_pv_field`** (active at t = 0): attends to `solar_panel`, `drone_camera`, `light_conditions`, `wind_speed`, `drone_battery`, `drone_gps`
- **`emergency_landing`** (queued at ETA = 30 s): attends to `airspace`, `landing_zone`, `drone_battery`, `wind_speed`

The AM begins pre-tuning for `emergency_landing` as its ETA decreases (F2), so concepts
like `airspace` and `landing_zone` gain attention before the transition fires. Reactive
has no such mechanism and is always "surprised" by the goal switch.

### Social Serving (CoreSense D8.1)

A waiter robot serves drinks to 10 persons (child / adult / VIP) in a room divided into
zones. The robot cycles between **MONITORING** (scanning for thirsty persons) and
**SERVING** (delivering a drink to a selected target). Goal switches to
`serve_<person_id>` when the controller enters SERVING state, reshuffling attention via
F1 and F6 (spatial opportunity cost from the robot's current zone). This is the primary
showcase scenario for the full ROS2 integration.

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
