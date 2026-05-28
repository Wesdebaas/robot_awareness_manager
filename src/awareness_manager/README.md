# Awareness Manager

Top-down robot awareness management for cognitive robotics.  
Master's thesis — Wessel Remmelzwaan, TU Delft / CoreSense Horizon Europe.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble |
| Python | 3.10+ |

```bash
pip install dash plotly scipy pyoxigraph
```

Also requires [`triplestar_kb`](https://github.com/kas-lab/triplestar_kb) cloned into `src/`.

---

## Setup

### With ROS2 (required for the ROS node and Gazebo simulation)

```bash
cd ~/thesis_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select awareness_manager
source install/setup.bash
```

### Standalone Python (dashboard and evaluation only)

```bash
export PYTHONPATH=$PWD/src/awareness_manager:$PYTHONPATH
```

---

## Interactive Dashboard

```bash
# Live AM — social serving scenario (primary demo):
python3 src/awareness_manager/demos/run_dashboard.py --scenario social_serving

# Live AM — PV inspection scenario:
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection

# Baseline strategies:
python3 src/awareness_manager/demos/run_dashboard.py --strategy reactive
python3 src/awareness_manager/demos/run_dashboard.py --strategy always_on

# Log a trace to disk while running:
python3 src/awareness_manager/demos/run_dashboard.py --scenario pv_inspection --log

# Replay a saved trace:
python3 src/awareness_manager/demos/run_dashboard.py --replay traces/run_001

# Side-by-side A/B comparison:
python3 src/awareness_manager/demos/run_dashboard.py \
    --replay traces/am_run --compare traces/reactive_run
```

Open `http://localhost:8050` in a browser.

---

## Batch Evaluation

Run the standard budget × observation-interval sweep (27 runs):

```bash
python3 -m awareness_manager.evaluation batch \
    --scenario pv_inspection \
    --strategies awareness_manager,reactive,always_on \
    --budget 1,2,4 \
    --obs-interval 1.0,5.0,10.0 \
    --output experiments/budget_obsint_sweep/
```

Run the 5-step component ablation study:

```bash
python3 -m awareness_manager.evaluation ablation \
    --output experiments/ablation_study/
```

Generate a report from saved traces:

```bash
python3 -m awareness_manager.evaluation report \
    --experiment experiments/budget_obsint_sweep/ \
    --output reports/budget_obsint_sweep/
```

Outputs: `summary.csv`, `plots.html` (interactive Plotly figures), `summary.md`
(metric table, per-regime breakdown, Mann-Whitney U tests).

---

## ROS2 Node

```bash
# PV inspection / social serving (awareness_node):
ros2 launch awareness_manager awareness_demo.launch.py
ros2 launch awareness_manager pv_inspection_demo.launch.py

# Find My Book — full Gazebo simulation (MIRTE robot, Nav2, BookFindingNode):
ros2 launch awareness_manager book_finding.launch.py
```

---

## Tests

```bash
python3 -m pytest src/awareness_manager/test/ -v
```
