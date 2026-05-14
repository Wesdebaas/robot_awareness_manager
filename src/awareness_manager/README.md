# Awareness Manager

Top-down robot awareness management for cognitive robotics.  
Master's thesis - Wessel Remmelzwaan, TU Delft / CoreSense Horizon Europe.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble |
| Python | 3.10+ |

Install Python dependencies:

```bash
pip install matplotlib networkx numpy pyoxigraph
```

---

## Setup

This package also requires [`triplestar_kb`](https://github.com/kas-lab/triplestar_kb) to be cloned into `src/` alongside this package — it provides the RDF/SPARQL backend used to load scenario ontologies.

### Option A - colcon (required for the ROS 2 node; also works for all demos)

```bash
cd ~/thesis_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select awareness_manager_msgs awareness_manager
source install/setup.bash
```

After sourcing, all scripts below can be run directly with `python3`.

### Option B - standalone Python (demos and evaluation only, no ROS 2 node)

No build step needed. Prefix every `python3` call with the package path:

```bash
cd ~/thesis_ws
PYTHONPATH=src/awareness_manager python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_anticipatory
```

Or set it once for the session:

```bash
cd ~/thesis_ws
export PYTHONPATH=$PWD/src/awareness_manager:$PYTHONPATH
```

---

## Live Comparison Demo - F2: Anticipatory Horizon

Shows the value of pre-tuning awareness for upcoming tasks.

**What you will see:**  
Both panels run the birdhouse scenario in lockstep at 2x real-time speed.
The robot starts with goal `build_birdhouse`. At **t = 25 simulated seconds**,
a disconnected `store_tools` task auto-promotes. 

- **Left (Full system):** the robot has been gradually pre-tuning attention
  on `store_tools` concepts as their ETA counted down (Anticipatory Horizon,
  Formula 2). At the switch their epistemic error is already low.
- **Right (F2 OFF):** `store_tools` concepts had attention = 0 the entire
  time - invisible until the switch fires. Their epistemic error has been
  drifting unchecked, causing a sharp spike in the bottom metric chart.

**With colcon build + source (Option A):**
```bash
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_anticipatory
```

**Standalone, no build (Option B):**
```bash
PYTHONPATH=src/awareness_manager python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_anticipatory
```

All five formula comparisons are available:

```bash
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_spreading    # F1
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_anticipatory # F2
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_saturation   # F3
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_budget       # F4
python3 src/awareness_manager/demos/run_birdhouse_compare.py --config no_drift        # F5 (default)
```

**Reading the visualizer:**

| Visual cue | Meaning |
|---|---|
| Node colour (green → red) | Epistemic error E (fresh → stale) |
| Node size (small → large) | Attention A (irrelevant → highly relevant) |
| Yellow ring | Concept just observed by the AM |
| Bottom chart | Attention-weighted epistemic error Σ(E·A)/Σ(A) over simulated time |

---

## Ablation Study (static figures)

Generates comparison figures for all five grounding formulas and saves them
to `evaluation/figures/`.

```bash
# All five formulas in one 2x3 figure:
python3 src/awareness_manager/evaluation/run_ablation.py --all

# One formula at a time (also prints a terminal summary table):
python3 src/awareness_manager/evaluation/run_ablation.py --config no_anticipatory
python3 src/awareness_manager/evaluation/run_ablation.py --config no_drift

# Custom combinations:
python3 src/awareness_manager/evaluation/run_ablation.py --disable F1 F3
```

---

## Other Demos

```bash
# Interactive birdhouse visualizer (single AM, click to observe):
python3 src/awareness_manager/demos/run_birdhouse_viz.py

# PV inspection scenario:
python3 src/awareness_manager/demos/run_pv_inspection_viz.py
```

---

## Evaluation (thesis figures)

Produces all figures used in the thesis (`evaluation/figures/fig1_*.png` … `fig9_*.png`):

```bash
python3 src/awareness_manager/evaluation/run_evaluation.py
```

---

## Tests

```bash
cd ~/thesis_ws
python3 -m pytest src/awareness_manager/test/ -v
```

---

## ROS 2 Node

After colcon build + source (Option A above):

```bash
ros2 run awareness_manager awareness_node
```

Or with a launch file:

```bash
ros2 launch awareness_manager awareness_demo.launch.py
```
