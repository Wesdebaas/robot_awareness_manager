import math

from awareness_manager.knowledge_base import KnowledgeBase


class AwarenessManager:
    """
    Awareness Manager — sits on top of the KnowledgeBase and answers:
    "Which concepts should the robot observe right now?"

    Each tick the AM:
        1. Advances epistemic drift (kb.tick)                          — Formula 5
        2. Advances the mission queue, promoting goals whose ETA ≤ 0   — Formula 2
        3. Recomputes spreading activation from the current goal        — Formula 1
           blended with discounted future goals from the queue          — Formula 2
        4. Ranks every concept by priority = epistemic_error × attention
        5. Returns the top-N concept IDs as the refresh schedule

    Observations are executed via observe(), which applies Formula 3 to compute
    how much epistemic error to reduce:

        Formula 3 — Utility Saturation:  refresh(n) = 1 − e^(−δ(n) × T)

    where δ(n) is the concept's decay rate and T is the observation interval.
    This calibrates the refresh amount to the drift accumulated since the last
    observation: slow-decaying concepts get a small refresh, fast-decaying ones
    get a larger one — each observation exactly compensates for what was lost.

    Formula 2 — Anticipatory Horizon:
        A_combined(c) = A_current(c) + Σ_i [ e^{-λ × Δt_i} × A_i(c) ]
    Queued goals contribute attention proportional to their proximity in time.
    As ETA decreases, the discount e^{-λΔt} rises toward 1, causing the robot
    to gradually pre-tune its awareness for the upcoming goal before it activates.

    Formula 4 — Quadratic Cost Constraint:
        depth = √B − 1
    The number of graph nodes reachable within depth d grows as (1+d)². Given
    memory budget B, the maximum search depth is √B − 1, replacing the fixed
    max_distance with a resource-derived bound.

    Priority formula:
        priority(c) = E(c) × A(c)

    Task nodes have decay_rate=0 so E stays 0 and priority stays 0 — they are
    never scheduled.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        alpha: float = 0.5,
        max_distance: float = 4.0,
        budget: int = 3,
        observation_interval: float = 1.0,
        lambda_horizon: float = 0.5,
        memory_budget: int | None = None,
    ) -> None:
        """
        Args:
            kb:                   The semantic knowledge base to manage.
            goal_id:              The initial mission goal concept ID.
            alpha:                Spreading activation decay factor [0, 1].
            max_distance:         Maximum weighted graph distance for attention.
                                  Ignored when memory_budget is set.
            budget:               Maximum concepts to schedule per tick (top-N).
            observation_interval: Expected seconds between observations (T in
                                  Formula 3). Should match the caller's cadence
                                  so the refresh amount equals the accumulated
                                  drift.
            lambda_horizon:       λ in Formula 2. Controls how quickly the
                                  anticipatory discount decays with time-to-goal.
                                  Higher values mean only near-future goals
                                  influence current attention.
            memory_budget:        B in Formula 4. When set, max search depth is
                                  derived as √B − 1 instead of using max_distance.
                                  None disables Formula 4 (uses max_distance).
        """
        if goal_id not in kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")

        self._kb = kb
        self._goal_id = goal_id
        self._alpha = alpha
        self._max_distance = max_distance
        self._budget = budget
        self._observation_interval = observation_interval
        self._lambda_horizon = lambda_horizon
        self._memory_budget = memory_budget
        self._attention: dict[str, float] = {}

        # Mission queue: ordered list of (goal_id, time_remaining) tuples.
        # Maintained sorted by time_remaining ascending so the next goal to
        # promote is always at index 0.
        self._mission_queue: list[tuple[str, float]] = []

        self._recompute_attention()

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_goal(self, goal_id: str) -> None:
        """Switch mission goal and immediately recompute attention."""
        if goal_id not in self._kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        self._goal_id = goal_id
        self._recompute_attention()

    @property
    def goal_id(self) -> str:
        return self._goal_id

    # ------------------------------------------------------------------
    # Formula 4 — Quadratic Cost Constraint
    # ------------------------------------------------------------------

    @property
    def effective_max_distance(self) -> float:
        """
        Formula 4: depth = √B − 1, or the fixed max_distance if no budget set.

        The number of nodes within spreading-activation depth d scales as (1+d)².
        Given memory budget B, the maximum depth that fits is √B − 1.
        """
        if self._memory_budget is not None:
            return max(0.0, math.sqrt(self._memory_budget) - 1.0)
        return self._max_distance

    # ------------------------------------------------------------------
    # Formula 2 — Mission queue / Anticipatory Horizon
    # ------------------------------------------------------------------

    def queue_goal(self, goal_id: str, eta: float) -> None:
        """
        Queue a future goal with ETA in simulated seconds from now.

        The goal will auto-promote to the current goal when its ETA reaches 0
        during tick(). While queued, its attention values are blended into the
        current attention window, discounted by e^{-λ × ETA} (Formula 2).

        Args:
            goal_id: A concept ID that must exist in the knowledge base.
            eta:     Simulated seconds until this goal becomes active. Must be > 0.
        """
        if goal_id not in self._kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        if eta <= 0.0:
            raise ValueError(f"ETA must be > 0; got {eta}. Use set_goal() for immediate switches.")
        self._mission_queue.append((goal_id, eta))
        self._mission_queue.sort(key=lambda x: x[1])
        self._recompute_attention()

    @property
    def mission_queue(self) -> list[tuple[str, float]]:
        """Snapshot of [(goal_id, time_remaining), ...] sorted by ETA ascending."""
        return list(self._mission_queue)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """
        Advance simulation by dt seconds and return the refresh schedule.

        Steps:
            1. kb.tick(dt) — passive epistemic drift on all concepts        (Formula 5)
            2. _advance_mission_queue(dt) — decrement ETAs, promote arrived goals (Formula 2)
            3. Recompute attention (current goal + discounted future goals)  (Formulas 1+2)
            4. Rank all concepts by priority = E × A (descending)
            5. Return top-budget concept IDs

        Returns:
            List of up to `budget` concept IDs ordered by priority (highest first).
        """
        self._kb.tick(dt)
        self._advance_mission_queue(dt)
        self._recompute_attention()
        return self._top_n()

    # ------------------------------------------------------------------
    # Observation (Formula 3)
    # ------------------------------------------------------------------

    def observe(self, concept_id: str) -> float:
        """
        Formula 3 — Utility Saturation: execute one observation on concept_id.

        Computes the refresh amount as:
            refresh(n) = 1 − e^(−δ(n) × observation_interval)

        This equals the drift that accumulates over one observation interval,
        modelled with a saturating exponential so heavily-decaying concepts are
        refreshed proportionally more. Calls kb.refresh_concept with this value.

        Returns:
            The refresh amount applied (useful for logging).
        """
        decay_rate = self._kb.get_concept(concept_id).decay_rate
        refresh = 1.0 - math.exp(-decay_rate * self._observation_interval)
        self._kb.refresh_concept(concept_id, refresh=refresh)
        return refresh

    def observation_refresh_value(self, concept_id: str) -> float:
        """Return the Formula 3 refresh value for concept_id without applying it."""
        decay_rate = self._kb.get_concept(concept_id).decay_rate
        return 1.0 - math.exp(-decay_rate * self._observation_interval)

    # ------------------------------------------------------------------
    # Read-only snapshots
    # ------------------------------------------------------------------

    def priorities(self) -> dict[str, float]:
        """Current priority for every concept (E × A). Snapshot, not live."""
        return {
            cid: self._kb.get_concept(cid).epistemic_error * self._attention.get(cid, 0.0)
            for cid in self._kb.concept_ids()
        }

    def attention(self) -> dict[str, float]:
        """Current attention values (last computed). Snapshot, not live."""
        return dict(self._attention)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_mission_queue(self, dt: float) -> None:
        """
        Decrement all ETAs by dt and promote goals whose ETA has reached 0.

        Multiple goals may promote in one tick if dt is large. Each promotion
        calls set_goal() internally so attention is NOT recomputed here —
        _recompute_attention() is called once after this method returns.
        """
        updated = [(gid, eta - dt) for gid, eta in self._mission_queue]
        # Separate promoted (ETA ≤ 0) from still-pending, keep time-order
        promoted = [(gid, eta) for gid, eta in updated if eta <= 0.0]
        self._mission_queue = [(gid, eta) for gid, eta in updated if eta > 0.0]

        for gid, _ in promoted:
            print(f"[QUEUE ]  '{gid}' promoted → new active goal")
            self._goal_id = gid

    def _recompute_attention(self) -> None:
        """
        Formulas 1 + 2: spreading activation from current goal, blended with
        discounted attention from each queued future goal.

        combined[c] = A_current(c)
                    + Σ_i [ e^{-λ × Δt_i} × A_i(c) ]    (clamped to [0, 1])
        """
        combined = self._kb.compute_attention(
            self._goal_id,
            alpha=self._alpha,
            max_distance=self.effective_max_distance,
        )
        for future_goal, eta in self._mission_queue:
            discount = math.exp(-self._lambda_horizon * eta)
            future_attn = self._kb.compute_attention(
                future_goal,
                alpha=self._alpha,
                max_distance=self.effective_max_distance,
            )
            for cid, a in future_attn.items():
                combined[cid] = min(1.0, combined.get(cid, 0.0) + discount * a)
        self._attention = combined

    def _top_n(self) -> list[str]:
        p = self.priorities()
        return sorted(p, key=p.__getitem__, reverse=True)[: self._budget]
