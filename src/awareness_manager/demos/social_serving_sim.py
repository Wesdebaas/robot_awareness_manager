"""Social Serving Scenario — simulation and baseline comparison.

Two strategies compete on the same event stream:

  AM (Awareness Manager):
      Maintains epistemic beliefs about all persons and drinks.
      Detects drink disappearances PROACTIVELY through scheduled observations.
      Serves a person only when it has fresh knowledge of both the person
      (person_E < threshold) and a preferred drink (drink_E < threshold).
      Rarely wastes a trip because it already knows the stock state.

  Naive (greedy sequential loop):
      No epistemic model. Cycles through people one by one.
      When it finds a thirsty person, immediately goes to fetch the preferred
      drink — without knowing whether it is still there.
      Discovers drink disappearances only when physically going to fetch a drink
      and finding it gone (reactive, costly).

Key metrics:
  total_thirst_seconds  — how long people wait overall (lower is better)
  wasted_trips          — serve attempts where the drink turned out to be gone
  detection_lag_s       — how quickly each disappearance was discovered

The AM's advantage is clearest at TIGHT budgets. At budget ≥ 5 both strategies
can refresh everything fast enough that wasted trips become rare for both.

Usage:
    python3 src/awareness_manager/demos/social_serving_sim.py
    python3 src/awareness_manager/demos/social_serving_sim.py --budget 3
    python3 src/awareness_manager/demos/social_serving_sim.py --compare
    python3 src/awareness_manager/demos/social_serving_sim.py --compare --budget 2
"""

import argparse
import random
import sys
import time
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo))

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import PresenceState
from awareness_manager.scenarios.social_serving import (
    build_social_serving_instance_kb,
    build_social_serving_kb,
    preferred_drinks_for,
)

# Epistemic thresholds for the AM serve condition.
# person_E < 0.7: robot has a reasonably fresh reading on the person (instance δ=0.04)
# drink_E  < 0.45: robot has a reasonably fresh reading on the drink (instance δ=0.05)
_PERSON_E_THRESHOLD = 0.7
_DRINK_E_THRESHOLD  = 0.45

# Default simulation parameters
_THIRST_RATE    = 1 / 30.0   # Poisson rate: each person requests a drink every ~30s
_DISAPPEAR_RATE = 0.01        # probability per drink per second of being taken by another agent


# ===========================================================================
# Shared event layer — generates the same Poisson events for both strategies
# ===========================================================================

class EventStream:
    """Generates person thirst events and drink disappearance events.

    Both strategies receive the same events so comparisons are fair.
    """

    def __init__(
        self,
        all_persons: list[str],
        all_drinks: list[str],
        thirst_rate: float = _THIRST_RATE,
        disappear_rate: float = _DISAPPEAR_RATE,
        seed: int = 42,
    ) -> None:
        self._all_persons    = all_persons
        self._all_drinks     = all_drinks
        self._thirst_rate    = thirst_rate
        self._disappear_rate = disappear_rate
        self._rng            = random.Random(seed)

    def sample(self, dt: float) -> tuple[set[str], set[str]]:
        """Return (newly_thirsty, newly_disappeared) for this tick."""
        thirsty:    set[str] = set()
        gone:       set[str] = set()
        for pid in self._all_persons:
            if self._rng.random() < self._thirst_rate * dt:
                thirsty.add(pid)
        for did in self._all_drinks:
            if self._rng.random() < self._disappear_rate * dt:
                gone.add(did)
        return thirsty, gone


# ===========================================================================
# AM controller
# ===========================================================================

class AMController:
    """
    Runs the Awareness Manager strategy on the social serving scenario.

    The AM observes whatever its priority scheduling puts in the schedule.
    Drink instances are checked against ground truth on observation.
    The robot serves a thirsty person whenever BOTH epistemic conditions hold:
      (a) person_E < PERSON_E_THRESHOLD
      (b) drink_E  < DRINK_E_THRESHOLD for at least one preferred drink

    When the serve condition is met, the robot goes to fetch the drink.
    If the drink turns out to be gone despite low E (stale belief), that is
    a wasted trip — the robot marks the drink suspected_absent and retries later.
    """

    def __init__(self, am: AwarenessManager, ikb, ground_truth: dict[str, bool]) -> None:
        self._am           = am
        self._ikb          = ikb
        self._ground_truth = ground_truth  # shared mutable dict: drink_id → present

        all_drink_classes = {'juice', 'cola', 'beer', 'wine', 'champagne'}
        self._all_drinks  = [
            iid for iid in ikb.instance_ids()
            if any(c in all_drink_classes for c in ikb.get_instance(iid).all_class_ids)
        ]
        self._all_persons = [iid for iid in ikb.instance_ids() if iid not in self._all_drinks]

        self._thirst_since: dict[str, float | None] = {p: None for p in self._all_persons}
        self._staleness_log: list[list] = []   # [drink_id, t_disappeared, t_detected_or_None]

        self._total_thirst_s = 0.0
        self._serves         = 0
        self._wasted_trips   = 0

    def apply_events(self, t: float, new_thirsty: set[str], new_gone: set[str]) -> None:
        """Inject ground-truth events — called before am.tick()."""
        for pid in new_thirsty:
            if self._thirst_since[pid] is None:
                self._thirst_since[pid] = t
        for did in new_gone:
            if self._ground_truth.get(did, True):
                self._ground_truth[did] = False
                self._staleness_log.append([did, t, None])

    def step(self, t: float, dt: float) -> None:
        """Run one AM tick, observe, and attempt to serve."""
        schedule = self._am.tick(dt)

        for concept_id in schedule:
            if concept_id in self._all_drinks:
                if not self._ground_truth.get(concept_id, True):
                    # Robot physically observes this drink and finds it gone.
                    # Confirm immediately so it leaves the priority queue
                    # (CONFIRMED_ABSENT freezes E drift and excludes from schedule).
                    inst = self._ikb.get_instance(concept_id)
                    if inst.presence_state != PresenceState.CONFIRMED_ABSENT:
                        self._ikb.mark_suspected_absent(concept_id)
                        self._ikb.confirm_absent(concept_id)
                    for entry in self._staleness_log:
                        if entry[0] == concept_id and entry[2] is None:
                            entry[2] = t
                            break
                    # Do not call observe — the drink is absent.
                else:
                    self._am.observe(concept_id)
            else:
                # Class concepts and person instances: observe normally.
                self._am.observe(concept_id)

        # Attempt to serve all thirsty persons whose epistemic conditions are met.
        for pid in self._all_persons:
            if self._thirst_since[pid] is None:
                continue
            drink_id = self._best_known_drink(pid)
            if drink_id is None:
                continue  # no drink with low enough E
            # Robot goes to fetch the drink — check ground truth.
            if self._ground_truth.get(drink_id, True):
                # Drink is actually there: serve succeeds.
                self._thirst_since[pid] = None
                self._serves += 1
            else:
                # Stale belief: drink is gone despite low E → wasted trip.
                self._wasted_trips += 1
                inst = self._ikb.get_instance(drink_id)
                if inst.presence_state != PresenceState.CONFIRMED_ABSENT:
                    self._ikb.mark_suspected_absent(drink_id)
                    self._ikb.confirm_absent(drink_id)
                for entry in self._staleness_log:
                    if entry[0] == drink_id and entry[2] is None:
                        entry[2] = t
                        break

        for pid in self._all_persons:
            if self._thirst_since[pid] is not None:
                self._total_thirst_s += dt

    def _best_known_drink(self, person_id: str) -> str | None:
        """Return the lowest-E preferred drink with E below threshold, or None."""
        person_inst = self._ikb.get_instance(person_id)
        if person_inst.epistemic_error >= _PERSON_E_THRESHOLD:
            return None

        best_id: str | None = None
        best_e = _DRINK_E_THRESHOLD  # must be strictly below this

        preferred = preferred_drinks_for(person_inst.all_class_ids)
        for drink_class in preferred:
            for drink_id in self._ikb.instances_of_class(drink_class):
                drink_inst = self._ikb.get_instance(drink_id)
                if drink_inst.presence_state == PresenceState.CONFIRMED_ABSENT:
                    continue
                if drink_inst.epistemic_error < best_e:
                    best_e = drink_inst.epistemic_error
                    best_id = drink_id
        return best_id

    def metrics(self) -> dict:
        lags = [e[2] - e[1] for e in self._staleness_log if e[2] is not None]
        undetected = sum(1 for e in self._staleness_log if e[2] is None)
        return {
            'total_thirst_seconds': round(self._total_thirst_s, 1),
            'total_serves':         self._serves,
            'wasted_trips':         self._wasted_trips,
            'disappearances':       len(self._staleness_log),
            'detected':             len(lags),
            'undetected':           undetected,
            'mean_detection_lag_s': round(sum(lags) / len(lags), 2) if lags else None,
            'max_detection_lag_s':  round(max(lags), 2) if lags else None,
        }


# ===========================================================================
# Naive (greedy sequential) controller
# ===========================================================================

class NaiveController:
    """
    Baseline with no epistemic model — greedy sequential loop.

    The robot cycles through all persons at a fixed rate. When it reaches a
    thirsty person it immediately goes to fetch the preferred drink without
    any awareness of whether the drink is still there.

    Drink disappearances are discovered only when the robot physically goes to
    fetch a drink and finds it gone (reactive). This is the key contrast with
    the AM: the naive robot can never anticipate stock problems in advance.
    """

    def __init__(
        self,
        ikb,
        ground_truth: dict[str, bool],
        person_check_interval: float = 3.0,
    ) -> None:
        self._ikb           = ikb
        self._ground_truth  = ground_truth  # shared mutable dict: drink_id → present
        self._check_interval = person_check_interval

        all_drink_classes = {'juice', 'cola', 'beer', 'wine', 'champagne'}
        self._all_drinks  = [
            iid for iid in ikb.instance_ids()
            if any(c in all_drink_classes for c in ikb.get_instance(iid).all_class_ids)
        ]
        self._all_persons = [iid for iid in ikb.instance_ids() if iid not in self._all_drinks]

        self._thirst_since: dict[str, float | None] = {p: None for p in self._all_persons}
        self._staleness_log: list[list] = []
        self._known_gone: set[str] = set()  # drinks discovered gone via physical fetch

        self._person_idx  = 0
        self._time_since_check = 0.0

        self._total_thirst_s = 0.0
        self._serves         = 0
        self._wasted_trips   = 0

    def apply_events(self, t: float, new_thirsty: set[str], new_gone: set[str]) -> None:
        for pid in new_thirsty:
            if self._thirst_since[pid] is None:
                self._thirst_since[pid] = t
        for did in new_gone:
            if self._ground_truth.get(did, True):
                self._ground_truth[did] = False
                self._staleness_log.append([did, t, None])

    def step(self, t: float, dt: float) -> None:
        self._time_since_check += dt
        if self._time_since_check >= self._check_interval:
            self._time_since_check = 0.0
            self._visit_next_person(t)

        for pid in self._all_persons:
            if self._thirst_since[pid] is not None:
                self._total_thirst_s += dt

    def _visit_next_person(self, t: float) -> None:
        """Cycle to the next person; if thirsty, immediately try to serve."""
        pid = self._all_persons[self._person_idx]
        self._person_idx = (self._person_idx + 1) % len(self._all_persons)

        if self._thirst_since[pid] is None:
            return

        # Naive: try preferred drinks in order, physically going to each one.
        # Skips drinks already discovered gone in a previous physical visit.
        preferred_classes = preferred_drinks_for(self._ikb.get_instance(pid).all_class_ids)
        for drink_class in preferred_classes:
            for drink_id in self._ikb.instances_of_class(drink_class):
                if drink_id in self._known_gone:
                    continue  # already discovered gone; skip the trip
                if self._ground_truth.get(drink_id, True):
                    # Drink is there — serve.
                    self._thirst_since[pid] = None
                    self._serves += 1
                    return
                else:
                    # Robot physically went to get the drink and found it gone.
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
            'total_thirst_seconds': round(self._total_thirst_s, 1),
            'total_serves':         self._serves,
            'wasted_trips':         self._wasted_trips,
            'disappearances':       len(self._staleness_log),
            'detected':             len(lags),
            'undetected':           undetected,
            'mean_detection_lag_s': round(sum(lags) / len(lags), 2) if lags else None,
            'max_detection_lag_s':  round(max(lags), 2) if lags else None,
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
    duration: float = 120.0,
    dt: float = 0.5,
    budget: int = 2,
    obs_interval: float = 5.0,
    disappear_rate: float = _DISAPPEAR_RATE,
    thirst_rate: float = _THIRST_RATE,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    kb  = build_social_serving_kb()
    ikb = build_social_serving_instance_kb()
    am  = AwarenessManager(
        kb=kb, goal_id='serve_drinks', alpha=0.5,
        budget=budget, observation_interval=obs_interval,
        instance_kb=ikb,
    )
    persons, drinks = _categorise(ikb)
    ground_truth = {did: True for did in drinks}
    events = EventStream(persons, drinks, thirst_rate, disappear_rate, seed)
    ctrl   = AMController(am, ikb, ground_truth)

    t = 0.0
    if verbose:
        _print_header('awareness_manager', budget, duration, dt, disappear_rate, thirst_rate, seed)
    while t < duration:
        new_thirsty, new_gone = events.sample(dt)
        ctrl.apply_events(t, new_thirsty, new_gone)
        ctrl.step(t, dt)
        t += dt
        if verbose:
            _maybe_print_progress(t, dt, ctrl)
    if verbose:
        _print_results(ctrl.metrics(), duration)
    return ctrl.metrics()


def run_naive(
    duration: float = 120.0,
    dt: float = 0.5,
    person_check_interval: float = 3.0,
    disappear_rate: float = _DISAPPEAR_RATE,
    thirst_rate: float = _THIRST_RATE,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    ikb     = build_social_serving_instance_kb()
    persons, drinks = _categorise(ikb)
    ground_truth = {did: True for did in drinks}
    events  = EventStream(persons, drinks, thirst_rate, disappear_rate, seed)
    ctrl    = NaiveController(ikb, ground_truth, person_check_interval)

    t = 0.0
    if verbose:
        _print_header('naive (greedy loop)', f'every {person_check_interval}s', duration, dt,
                      disappear_rate, thirst_rate, seed)
    while t < duration:
        new_thirsty, new_gone = events.sample(dt)
        ctrl.apply_events(t, new_thirsty, new_gone)
        ctrl.step(t, dt)
        t += dt
        if verbose:
            _maybe_print_progress(t, dt, ctrl)
    if verbose:
        _print_results(ctrl.metrics(), duration)
    return ctrl.metrics()


def _print_header(strategy, budget, duration, dt, disappear_rate, thirst_rate, seed):
    print(f"\n=== Social Serving Simulation ===")
    print(f"Strategy: {strategy}  |  Budget: {budget}  |  Duration: {duration}s  |  dt: {dt}s")
    print(f"Disappear rate: {disappear_rate}/s/drink  |  Thirst rate: {thirst_rate:.4f}/s/person")
    print(f"Seed: {seed}\n")


_last_print = [0.0]

def _maybe_print_progress(t: float, dt: float, ctrl) -> None:
    if t - _last_print[0] >= 20.0:
        _last_print[0] = t
        m = ctrl.metrics()
        thirsty = sum(1 for v in ctrl._thirst_since.values() if v is not None)
        print(
            f"t={t:6.1f}s  thirsty={thirsty}/10  thirst_s={m['total_thirst_seconds']:.0f}"
            f"  serves={m['total_serves']}  wasted={m['wasted_trips']}"
            f"  disappear={m['disappearances']}  detected={m['detected']}"
        )


def _print_results(metrics: dict, duration: float) -> None:
    print(f"\n=== Results after {duration}s simulated ===")
    print(f"  Total thirst time:        {metrics['total_thirst_seconds']:.1f} person-seconds")
    print(f"  Total serves:             {metrics['total_serves']}")
    print(f"  Wasted trips:             {metrics['wasted_trips']}")
    print(f"  Drink disappearances:     {metrics['disappearances']}")
    print(f"  Detected:                 {metrics['detected']}")
    print(f"  Undetected:               {metrics['undetected']}")
    if metrics['mean_detection_lag_s'] is not None:
        print(f"  Mean detection lag:       {metrics['mean_detection_lag_s']:.1f}s")
        print(f"  Max detection lag:        {metrics['max_detection_lag_s']:.1f}s")
    else:
        print(f"  Mean detection lag:       — (no detections)")
    print()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description='Social serving scenario simulation.')
    parser.add_argument('--strategy', default='awareness_manager',
                        choices=['awareness_manager', 'naive'],
                        help='Strategy to run (default: awareness_manager)')
    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--budget', type=int, default=2,
                        help='AM observation budget per tick (default: 2)')
    parser.add_argument('--obs-interval', type=float, default=5.0, dest='obs_interval')
    parser.add_argument('--disappear-rate', type=float, default=_DISAPPEAR_RATE,
                        dest='disappear_rate')
    parser.add_argument('--check-interval', type=float, default=3.0, dest='check_interval',
                        help='Naive: seconds between person visits (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare', action='store_true',
                        help='Run both strategies with the same events and compare')
    args = parser.parse_args()

    kwargs = dict(
        duration=args.duration, dt=args.dt,
        disappear_rate=args.disappear_rate, seed=args.seed,
        verbose=False,
    )

    if args.compare:
        print("Running comparison: awareness_manager vs naive (greedy loop)\n")
        am_m  = run_am(budget=args.budget, obs_interval=args.obs_interval, **kwargs)
        nv_m  = run_naive(person_check_interval=args.check_interval, **kwargs)

        print(f"{'Metric':<30} {'AM':>14} {'Naive':>14}")
        print('-' * 60)
        for key in am_m:
            av = am_m[key]
            nv = nv_m[key]
            a_str = f"{av:.1f}" if isinstance(av, float) else str(av)
            n_str = f"{nv:.1f}" if isinstance(nv, float) else str(nv)
            print(f"{key:<30} {a_str:>14} {n_str:>14}")
        print()
        print("Interpretation:")
        print("  fewer wasted_trips      → better stock awareness (AM advantage)")
        print("  lower total_thirst_seconds → faster service overall")
        print("  higher detected         → proactive vs reactive discovery")
    elif args.strategy == 'awareness_manager':
        _last_print[0] = 0.0
        kw = {k: v for k, v in kwargs.items() if k != 'verbose'}
        run_am(budget=args.budget, obs_interval=args.obs_interval,
               verbose=True, **kw)
    else:
        _last_print[0] = 0.0
        kw = {k: v for k, v in kwargs.items() if k != 'verbose'}
        run_naive(person_check_interval=args.check_interval,
                  verbose=True, **kw)


if __name__ == '__main__':
    main()
