"""Social Serving Scenario — simulation and baseline comparison.

Robot serving drinks to 10 people. People periodically finish their drinks;
the robot must discover who needs a drink (by observing them) and fetch one.

Key design:
  - Drink state is epistemic: the robot only learns a person needs a drink by
    physically observing them (AM schedules person → robot goes there → discovers
    has_drink=False). No direct thirst events are injected.
  - Observation cost (F6): cross-zone travel (table_area ↔ restock_zone) costs 8s,
    making the AM prefer batching observations within a zone before crossing.
  - Dynamic goals: the controller pushes serve_<person> goals via am.set_goal()
    based on what it discovers. The AM autonomously shifts attention toward the
    relevant drink class and routes efficiently.

Two strategies compete on the same event stream:

  AM (Awareness Manager):
      Cost-aware epistemic scheduling (F1-F6). Discovers need-for-drink by
      proactively observing persons. Focuses on the preferred drink class and
      routes toward cheap nearby observations.

  Naive (greedy sequential loop):
      No epistemic model. Cycles through persons at a fixed interval. Discovers
      needs reactively, fetches drink without stock awareness, wastes trips when
      drinks are gone.

Metric: total person-seconds without a drink (lower is better).

Usage:
    python3 src/awareness_manager/demos/social_serving_sim.py
    python3 src/awareness_manager/demos/social_serving_sim.py --budget 3
    python3 src/awareness_manager/demos/social_serving_sim.py --compare
"""

import argparse
import random
import sys
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo))

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import PresenceState
from awareness_manager.scenarios.social_serving import (
    ZONE_TRAVEL_TIMES,
    build_social_serving_instance_kb,
    build_social_serving_kb,
    create_serve_goal,
    load_zone_assignment,
    preferred_drinks_for,
)

# Default simulation parameters
_FINISH_RATE = 1 / 40.0    # each person finishes their drink every ~40s (Poisson)
_DISAPPEAR_RATE = 0.008    # probability per drink per second of being taken


# ===========================================================================
# Shared event layer
# ===========================================================================

class EventStream:
    """Generates drink-finish events (person has_drink → False) and drink
    disappearance events, shared between both strategies so comparisons are fair."""

    def __init__(
        self,
        all_persons: list[str],
        all_drinks: list[str],
        finish_rate: float = _FINISH_RATE,
        disappear_rate: float = _DISAPPEAR_RATE,
        seed: int = 42,
    ) -> None:
        self._all_persons   = all_persons
        self._all_drinks    = all_drinks
        self._finish_rate   = finish_rate
        self._disappear_rate = disappear_rate
        self._rng           = random.Random(seed)

    def sample(self, dt: float) -> tuple[set[str], set[str]]:
        """Return (newly_finished, newly_disappeared) for this tick."""
        finished:    set[str] = set()
        disappeared: set[str] = set()
        for pid in self._all_persons:
            if self._rng.random() < self._finish_rate * dt:
                finished.add(pid)
        for did in self._all_drinks:
            if self._rng.random() < self._disappear_rate * dt:
                disappeared.add(did)
        return finished, disappeared


# ===========================================================================
# AM controller — MONITORING / SERVING state machine
# ===========================================================================

_MONITORING = 'monitoring'
_SERVING    = 'serving'


class AMController:
    """
    Minimal state machine that drives the AM via goal injection.

    The controller does NOT make awareness decisions — it only:
      1. Calls am.tick(dt, robot_pos) each step
      2. Executes the schedule: moves robot to each concept and calls am.observe()
      3. On observing a person: reads their has_drink state from ground truth,
         updates the instance KB, and — if they need a drink — calls set_goal()
      4. On observing a preferred drink while SERVING: attempts delivery

    This is intentionally thin so it can be replaced by a richer planner later.
    """

    def __init__(self, am: AwarenessManager, ikb, ground_truth_drinks: dict[str, bool]) -> None:
        self._am           = am
        self._ikb          = ikb
        self._gt_drinks    = ground_truth_drinks  # drink_id → present (bool)

        all_drink_classes = {'juice', 'cola', 'beer', 'wine', 'champagne'}
        self._all_drinks  = [
            iid for iid in ikb.instance_ids()
            if any(c in all_drink_classes for c in ikb.get_instance(iid).all_class_ids)
        ]
        self._all_persons = [iid for iid in ikb.instance_ids() if iid not in self._all_drinks]

        # Ground truth: each person starts with a drink
        self._gt_has_drink: dict[str, bool] = {p: True for p in self._all_persons}
        for pid in self._all_persons:
            ikb.get_instance(pid).properties['has_drink'] = True

        self._state          = _MONITORING
        self._serving_person = ''
        # Start at the table area so F6 immediately favours person observations
        # over the cheaper-to-drift drink instances in the restock zone.
        self._robot_pos: str | None = self._all_persons[0] if self._all_persons else None

        self._need_drink_seconds = 0.0  # accumulated person-seconds without a drink
        self._serves             = 0
        self._wasted_trips       = 0
        self._staleness_log: list[list] = []  # [drink_id, t_disappeared, t_detected | None]

    # ------------------------------------------------------------------
    # Event injection (called before step())
    # ------------------------------------------------------------------

    def apply_events(self, t: float, newly_finished: set[str], newly_disappeared: set[str]) -> None:
        """Inject ground-truth events. The robot learns of these only by observing."""
        for pid in newly_finished:
            if self._gt_has_drink.get(pid, True):
                self._gt_has_drink[pid] = False
                # do NOT update instance KB here — robot must observe to discover this
        for did in newly_disappeared:
            if self._gt_drinks.get(did, True):
                self._gt_drinks[did] = False
                self._staleness_log.append([did, t, None])

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float) -> None:
        schedule = self._am.tick(dt, robot_pos=self._robot_pos)

        for concept_id in schedule:
            self._robot_pos = concept_id  # robot moves to this concept

            if concept_id in self._all_persons:
                self._observe_person(t, concept_id)
            elif concept_id in self._all_drinks:
                self._observe_drink(t, concept_id)
            else:
                self._am.observe(concept_id)  # class or location concept

        # Accumulate metric
        for pid in self._all_persons:
            if not self._gt_has_drink.get(pid, True):
                self._need_drink_seconds += dt

    # ------------------------------------------------------------------
    # Observation handlers
    # ------------------------------------------------------------------

    def _observe_person(self, t: float, person_id: str) -> None:
        """Robot looks at person_id and discovers their has_drink state."""
        actual_has_drink = self._gt_has_drink[person_id]
        self._ikb.get_instance(person_id).properties['has_drink'] = actual_has_drink
        self._am.observe(person_id)

        if self._state == _MONITORING and not actual_has_drink:
            # Discover this person needs a drink → start serving them
            drink_class = self._pick_first_drink_class(person_id)
            if drink_class:
                goal_id = create_serve_goal(self._am.kb, person_id, drink_class)
                self._am.set_goal(goal_id)
                self._state          = _SERVING
                self._serving_person = person_id

    def _observe_drink(self, t: float, drink_id: str) -> None:
        """Robot looks at drink_id and checks if it's still present."""
        inst = self._ikb.get_instance(drink_id)

        if not self._gt_drinks.get(drink_id, True):
            # Drink is gone — confirm absence
            if inst.presence_state != PresenceState.CONFIRMED_ABSENT:
                self._ikb.mark_suspected_absent(drink_id)
                self._ikb.confirm_absent(drink_id)
            for entry in self._staleness_log:
                if entry[0] == drink_id and entry[2] is None:
                    entry[2] = t
                    break
            # If serving and this was the target drink class, try next class
            if self._state == _SERVING:
                self._try_next_drink_class(t)
        else:
            self._am.observe(drink_id)
            # If serving and this drink matches the target person's preference: deliver
            if self._state == _SERVING:
                person_inst = self._ikb.get_instance(self._serving_person)
                preferred = preferred_drinks_for(person_inst.all_class_ids)
                if any(c in inst.all_class_ids for c in preferred):
                    self._deliver(t, drink_id)

    # ------------------------------------------------------------------
    # Serving helpers
    # ------------------------------------------------------------------

    def _pick_first_drink_class(self, person_id: str) -> str:
        """Return the first preferred drink class that still has active instances."""
        person_inst = self._ikb.get_instance(person_id)
        for drink_class in preferred_drinks_for(person_inst.all_class_ids):
            active = [
                did for did in self._ikb.instances_of_class(drink_class)
                if self._ikb.get_instance(did).presence_state != PresenceState.CONFIRMED_ABSENT
            ]
            if active:
                return drink_class
        return ''

    def _try_next_drink_class(self, t: float) -> None:
        """After discovering a drink class is unavailable, try the next preference."""
        person_inst = self._ikb.get_instance(self._serving_person)
        next_class = self._pick_first_drink_class(self._serving_person)
        if next_class:
            goal_id = create_serve_goal(self._am.kb, self._serving_person, next_class)
            self._am.set_goal(goal_id)
        else:
            # No preferred drinks remain — give up and return to monitoring
            self._am.set_goal('serve_people_drinks')
            self._state = _MONITORING

    def _deliver(self, t: float, drink_id: str) -> None:
        """Deliver drink_id to serving_person. Robot then moves to person."""
        self._gt_has_drink[self._serving_person] = True
        self._ikb.get_instance(self._serving_person).properties['has_drink'] = True
        self._serves += 1
        self._robot_pos = self._serving_person  # robot is now at the person
        self._am.set_goal('serve_people_drinks')
        self._state = _MONITORING

    def metrics(self) -> dict:
        lags = [e[2] - e[1] for e in self._staleness_log if e[2] is not None]
        undetected = sum(1 for e in self._staleness_log if e[2] is None)
        return {
            'need_drink_person_seconds': round(self._need_drink_seconds, 1),
            'total_serves':              self._serves,
            'wasted_trips':              self._wasted_trips,
            'disappearances':            len(self._staleness_log),
            'detected':                  len(lags),
            'undetected':                undetected,
            'mean_detection_lag_s':      round(sum(lags) / len(lags), 2) if lags else None,
        }


# ===========================================================================
# Naive (greedy sequential) controller
# ===========================================================================

class NaiveController:
    """
    Baseline with no epistemic model — greedy sequential loop.

    Cycles through persons at a fixed interval. When it reaches a person without
    a drink it immediately tries to fetch one, discovering drink absences only
    by physically going to fetch and finding them gone.
    """

    def __init__(
        self,
        ikb,
        ground_truth_drinks: dict[str, bool],
        person_check_interval: float = 5.0,
    ) -> None:
        self._ikb            = ikb
        self._gt_drinks      = ground_truth_drinks
        self._check_interval = person_check_interval

        all_drink_classes = {'juice', 'cola', 'beer', 'wine', 'champagne'}
        self._all_drinks  = [
            iid for iid in ikb.instance_ids()
            if any(c in all_drink_classes for c in ikb.get_instance(iid).all_class_ids)
        ]
        self._all_persons = [iid for iid in ikb.instance_ids() if iid not in self._all_drinks]

        self._gt_has_drink: dict[str, bool] = {p: True for p in self._all_persons}

        self._person_idx       = 0
        self._time_since_check = 0.0
        self._known_gone: set[str] = set()

        self._need_drink_seconds = 0.0
        self._serves             = 0
        self._wasted_trips       = 0
        self._staleness_log: list[list] = []

    def apply_events(self, t: float, newly_finished: set[str], newly_disappeared: set[str]) -> None:
        for pid in newly_finished:
            self._gt_has_drink[pid] = False
        for did in newly_disappeared:
            if self._gt_drinks.get(did, True):
                self._gt_drinks[did] = False
                self._staleness_log.append([did, t, None])

    def step(self, t: float, dt: float) -> None:
        self._time_since_check += dt
        if self._time_since_check >= self._check_interval:
            self._time_since_check = 0.0
            self._visit_next_person(t)

        for pid in self._all_persons:
            if not self._gt_has_drink.get(pid, True):
                self._need_drink_seconds += dt

    def _visit_next_person(self, t: float) -> None:
        pid = self._all_persons[self._person_idx]
        self._person_idx = (self._person_idx + 1) % len(self._all_persons)

        if self._gt_has_drink.get(pid, True):
            return  # person has a drink, skip

        person_inst = self._ikb.get_instance(pid)
        preferred = preferred_drinks_for(person_inst.all_class_ids)
        for drink_class in preferred:
            for drink_id in self._ikb.instances_of_class(drink_class):
                if drink_id in self._known_gone:
                    continue
                if self._gt_drinks.get(drink_id, True):
                    self._gt_has_drink[pid] = True
                    self._serves += 1
                    return
                else:
                    self._wasted_trips += 1
                    self._known_gone.add(drink_id)
                    for entry in self._staleness_log:
                        if entry[0] == drink_id and entry[2] is None:
                            entry[2] = t
                            break

    def metrics(self) -> dict:
        lags = [e[2] - e[1] for e in self._staleness_log if e[2] is not None]
        undetected = sum(1 for e in self._staleness_log if e[2] is None)
        return {
            'need_drink_person_seconds': round(self._need_drink_seconds, 1),
            'total_serves':              self._serves,
            'wasted_trips':              self._wasted_trips,
            'disappearances':            len(self._staleness_log),
            'detected':                  len(lags),
            'undetected':                undetected,
            'mean_detection_lag_s':      round(sum(lags) / len(lags), 2) if lags else None,
        }


# ===========================================================================
# Run helpers
# ===========================================================================

def _categorise(ikb) -> tuple[list[str], list[str]]:
    all_drink_classes = {'juice', 'cola', 'beer', 'wine', 'champagne'}
    drinks  = [iid for iid in ikb.instance_ids()
               if any(c in all_drink_classes for c in ikb.get_instance(iid).all_class_ids)]
    persons = [iid for iid in ikb.instance_ids() if iid not in drinks]
    return persons, drinks


def run_am(
    duration: float = 180.0,
    dt: float = 0.5,
    budget: int = 2,
    obs_interval: float = 5.0,
    disappear_rate: float = _DISAPPEAR_RATE,
    finish_rate: float = _FINISH_RATE,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    kb           = build_social_serving_kb()
    ikb          = build_social_serving_instance_kb()
    zone_assign  = load_zone_assignment()
    am           = AwarenessManager(
        kb=kb, goal_id='serve_people_drinks', alpha=0.5,
        budget=budget, observation_interval=obs_interval,
        instance_kb=ikb,
        zone_assignment=zone_assign,
        zone_travel_times=ZONE_TRAVEL_TIMES,
    )
    persons, drinks = _categorise(ikb)
    gt_drinks    = {did: True for did in drinks}
    events       = EventStream(persons, drinks, finish_rate, disappear_rate, seed)
    ctrl         = AMController(am, ikb, gt_drinks)

    t = 0.0
    if verbose:
        _print_header('awareness_manager', budget, duration, dt, disappear_rate, finish_rate, seed)
    while t < duration:
        finished, disappeared = events.sample(dt)
        ctrl.apply_events(t, finished, disappeared)
        ctrl.step(t, dt)
        t += dt
        if verbose:
            _maybe_print_progress(t, dt, ctrl, persons)
    if verbose:
        _print_results(ctrl.metrics(), duration)
    return ctrl.metrics()


def run_naive(
    duration: float = 180.0,
    dt: float = 0.5,
    person_check_interval: float = 5.0,
    disappear_rate: float = _DISAPPEAR_RATE,
    finish_rate: float = _FINISH_RATE,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    ikb     = build_social_serving_instance_kb()
    persons, drinks = _categorise(ikb)
    gt_drinks = {did: True for did in drinks}
    events  = EventStream(persons, drinks, finish_rate, disappear_rate, seed)
    ctrl    = NaiveController(ikb, gt_drinks, person_check_interval)

    t = 0.0
    if verbose:
        _print_header(f'naive (every {person_check_interval}s)', '—', duration, dt,
                      disappear_rate, finish_rate, seed)
    while t < duration:
        finished, disappeared = events.sample(dt)
        ctrl.apply_events(t, finished, disappeared)
        ctrl.step(t, dt)
        t += dt
        if verbose:
            _maybe_print_progress(t, dt, ctrl, persons)
    if verbose:
        _print_results(ctrl.metrics(), duration)
    return ctrl.metrics()


def _print_header(strategy, budget, duration, dt, disappear_rate, finish_rate, seed):
    print(f"\n=== Social Serving Simulation ===")
    print(f"Strategy: {strategy}  |  Budget: {budget}  |  Duration: {duration}s  |  dt: {dt}s")
    print(f"Finish rate: {finish_rate:.4f}/s/person  |  Disappear rate: {disappear_rate}/s/drink")
    print(f"Seed: {seed}\n")


_last_print = [0.0]


def _maybe_print_progress(t: float, dt: float, ctrl, persons: list[str]) -> None:
    if t - _last_print[0] >= 30.0:
        _last_print[0] = t
        m = ctrl.metrics()
        if hasattr(ctrl, '_gt_has_drink'):
            needs = sum(1 for p in persons if not ctrl._gt_has_drink.get(p, True))
        else:
            needs = '?'
        print(
            f"t={t:6.1f}s  need_drink={needs}/10"
            f"  person_s={m['need_drink_person_seconds']:.0f}"
            f"  serves={m['total_serves']}  wasted={m['wasted_trips']}"
            f"  disappear={m['disappearances']}  detected={m['detected']}"
        )


def _print_results(metrics: dict, duration: float) -> None:
    print(f"\n=== Results after {duration}s simulated ===")
    print(f"  Person-seconds without drink:  {metrics['need_drink_person_seconds']:.1f}")
    print(f"  Total serves:                  {metrics['total_serves']}")
    print(f"  Wasted trips:                  {metrics['wasted_trips']}")
    print(f"  Drink disappearances:          {metrics['disappearances']}")
    print(f"  Detected:                      {metrics['detected']}")
    print(f"  Undetected:                    {metrics['undetected']}")
    if metrics['mean_detection_lag_s'] is not None:
        print(f"  Mean detection lag:            {metrics['mean_detection_lag_s']:.1f}s")
    else:
        print(f"  Mean detection lag:            — (no detections)")
    print()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description='Social serving scenario simulation.')
    parser.add_argument('--strategy', default='awareness_manager',
                        choices=['awareness_manager', 'naive'])
    parser.add_argument('--duration', type=float, default=180.0)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--obs-interval', type=float, default=5.0, dest='obs_interval')
    parser.add_argument('--disappear-rate', type=float, default=_DISAPPEAR_RATE,
                        dest='disappear_rate')
    parser.add_argument('--finish-rate', type=float, default=_FINISH_RATE, dest='finish_rate')
    parser.add_argument('--check-interval', type=float, default=5.0, dest='check_interval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    kwargs = dict(
        duration=args.duration, dt=args.dt,
        disappear_rate=args.disappear_rate,
        finish_rate=args.finish_rate,
        seed=args.seed,
        verbose=False,
    )

    if args.compare:
        print("Running comparison: awareness_manager vs naive\n")
        am_m = run_am(budget=args.budget, obs_interval=args.obs_interval, **kwargs)
        nv_m = run_naive(person_check_interval=args.check_interval, **kwargs)

        print(f"{'Metric':<35} {'AM':>12} {'Naive':>12}")
        print('-' * 61)
        for key in am_m:
            av = am_m[key]
            nv = nv_m[key]
            a_str = f"{av:.1f}" if isinstance(av, float) else str(av)
            n_str = f"{nv:.1f}" if isinstance(nv, float) else str(nv)
            print(f"{key:<35} {a_str:>12} {n_str:>12}")
        print()
        print("Interpretation:")
        print("  lower need_drink_person_seconds → robot keeps people served faster")
        print("  fewer wasted_trips              → better stock awareness (F6 advantage)")
        print("  higher detected                 → proactive vs reactive discovery")
    elif args.strategy == 'awareness_manager':
        _last_print[0] = 0.0
        kw = {k: v for k, v in kwargs.items() if k != 'verbose'}
        run_am(budget=args.budget, obs_interval=args.obs_interval, verbose=True, **kw)
    else:
        _last_print[0] = 0.0
        kw = {k: v for k, v in kwargs.items() if k != 'verbose'}
        run_naive(person_check_interval=args.check_interval, verbose=True, **kw)


if __name__ == '__main__':
    main()
