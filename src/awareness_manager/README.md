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

## Interactive Dashboard

The dashboard supports live simulation, offline replay, and A/B comparison.

```bash
# Live AM — PV inspection scenario (default):
python3 src/awareness_manager/demos/run_dashboard.py

# Live AM — social serving scenario (waiter robot, 10 persons, drink classes):
python3 src/awareness_manager/demos/run_dashboard.py --scenario social_serving

# Baseline strategies:
python3 src/awareness_manager/demos/run_dashboard.py --strategy reactive
python3 src/awareness_manager/demos/run_dashboard.py --strategy always_on

# Record a trace to disk while running:
python3 src/awareness_manager/demos/run_dashboard.py --log

# Replay a saved trace:
python3 src/awareness_manager/demos/run_dashboard.py --replay traces/run_001

# Side-by-side A/B comparison with M1–M6 overlay:
python3 src/awareness_manager/demos/run_dashboard.py \
    --replay traces/am_run --compare traces/reactive_run
```

The social serving dashboard includes a **controller state panel** (above the graph)
showing the current MONITORING / SERVING state and the person being served.

---

## Other Demos

```bash
# Interactive birdhouse visualizer (single AM, click to observe):
python3 src/awareness_manager/demos/run_birdhouse_viz.py

# PV inspection scenario (Matplotlib, no Dash):
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

## ROS 2 Node (Social Serving)

After colcon build + source (Option A above):

```bash
ros2 run awareness_manager awareness_node
```

Or with a launch file:

```bash
ros2 launch awareness_manager awareness_demo.launch.py
```

---

## Find My Book — Full Gazebo Simulation

A domestic robot navigates the KRR_Course_Small_house world and must locate a book.
The AM pre-tunes awareness for room-specific book locations as the robot moves,
demonstrating F1 (spreading activation), F2 (anticipatory pre-tuning on nav goals),
F5 (epistemic error drives room priority), and F6 (spatial opportunity cost).

### Additional prerequisites

| Requirement | Notes |
|---|---|
| Gazebo 11 | `sudo apt install gazebo` |
| `robocup_home_simulation` | cloned into `src/` — provides the house world + book props |
| `mirte-gazebo` | cloned into `src/` — MIRTE Master robot and spawn launch |
| `mirte_navigation` | cloned into `src/` — Nav2 params and map for the house world |
| `plasys_house_world` / `aws_robomaker_small_house_world` | ROS package dependencies of the world |

### Build

```bash
colcon build --packages-select \
    awareness_manager robocup_home_simulation mirte_navigation \
    --symlink-install
source install/setup.bash
```

### Launch

```bash
ros2 launch awareness_manager book_finding.launch.py
```

Optional arguments:

```bash
ros2 launch awareness_manager book_finding.launch.py \
    budget:=3 tick_rate:=10.0 observation_interval:=5.0 \
    alpha:=0.5 nav_eta:=15.0 f6:=true
```

### What to expect

After ~10 seconds the Gazebo window opens with the house world, and the MIRTE robot
appears inside. RViz shows the map and robot pose. Nav2 activates and AMCL localises
the robot using the LiDAR.

The robot does **not** move autonomously — use RViz's **"2D Nav Goal"** button to send
it to a room. When a goal is published:

- `book_finding_node` infers the target room from the goal coordinates
- Queues the corresponding book instance with an ETA for F2 anticipatory pre-tuning
- The AM raises priority for book-related concepts in that room before the robot arrives

When the robot enters a new room, it immediately observes the book there, reducing its
epistemic error.

### Topics published by BookFindingNode

| Topic | Type | Content |
|---|---|---|
| `awareness/state` | `std_msgs/String` (JSON) | Per-concept A, E, P values and channel breakdown |
| `awareness/schedule` | `std_msgs/String` (JSON) | Ordered list of concept IDs to observe this tick |
| `awareness/goal` | `std_msgs/String` | Active goal ID |
| `awareness/controller_state` | `std_msgs/String` (JSON) | Current zone, found_book, top-3 schedule |
