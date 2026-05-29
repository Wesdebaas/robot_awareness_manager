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

## Evaluation

Six focused experiments, each validating one grounding formula:

```bash
python3 -m awareness_manager.evaluation experiments      # all six
python3 -m awareness_manager.evaluation exp1             # scaling / detection latency (F1)
python3 -m awareness_manager.evaluation exp2             # goal conditioning (E×A)
python3 -m awareness_manager.evaluation exp3             # anticipatory horizon (F2)
python3 -m awareness_manager.evaluation exp4             # instance KB vs class-only
python3 -m awareness_manager.evaluation exp5             # formula ablation: F1
python3 -m awareness_manager.evaluation exp6             # F6 spatial opportunity cost
python3 -m awareness_manager.evaluation abstract --quick # abstract N-variable smoke test
python3 -m awareness_manager.evaluation learnable-decay  # learnable δ verification
```

---

## ROS2 Node

```bash
# Social serving / PV inspection (awareness_node + dashboard):
ros2 launch awareness_manager awareness_demo.launch.py

# Drink Serving — full Gazebo simulation (MIRTE robot, Nav2, AM-integrated patrol):
ros2 launch awareness_manager drink_serving.launch.py                     # AM condition
ros2 launch awareness_manager drink_serving.launch.py strategy:=reactive  # Reactive baseline
```

---

## Tests

```bash
python3 -m pytest src/awareness_manager/test/ -v
```
