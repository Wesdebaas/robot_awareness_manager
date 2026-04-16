import math

from awareness_manager.knowledge_base import KnowledgeBase


class AwarenessManager:
    """
    Awareness Manager — sits on top of the KnowledgeBase and answers:
    "Which concepts should the robot observe right now?"

    Each tick the AM:
        1. Advances epistemic drift (kb.tick)                          — Formula 5
        2. Recomputes spreading activation from the current goal       — Formula 1
        3. Ranks every concept by priority = epistemic_error × attention
        4. Returns the top-N concept IDs as the refresh schedule

    Observations are executed via observe(), which applies Formula 3 to compute
    how much epistemic error to reduce:

        Formula 3 — Utility Saturation:  refresh(n) = 1 − e^(−δ(n) × T)

    where δ(n) is the concept's decay rate and T is the observation interval.
    This calibrates the refresh amount to the drift accumulated since the last
    observation: slow-decaying concepts get a small refresh, fast-decaying ones
    get a larger one — each observation exactly compensates for what was lost.

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
    ) -> None:
        """
        Args:
            kb:                   The semantic knowledge base to manage.
            goal_id:              The initial mission goal concept ID.
            alpha:                Spreading activation decay factor [0, 1].
            max_distance:         Maximum weighted graph distance for attention.
            budget:               Maximum concepts to schedule per tick (top-N).
            observation_interval: Expected seconds between observations (T in
                                  Formula 3). Should match the caller's cadence
                                  so the refresh amount equals the accumulated
                                  drift.
        """
        if goal_id not in kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")

        self._kb = kb
        self._goal_id = goal_id
        self._alpha = alpha
        self._max_distance = max_distance
        self._budget = budget
        self._observation_interval = observation_interval
        self._attention: dict[str, float] = {}

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
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """
        Advance simulation by dt seconds and return the refresh schedule.

        Steps:
            1. kb.tick(dt) — passive epistemic drift on all concepts
            2. Recompute attention from current goal
            3. Rank all concepts by priority = E × A (descending)
            4. Return top-budget concept IDs

        Returns:
            List of up to `budget` concept IDs ordered by priority (highest first).
        """
        self._kb.tick(dt)
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

    def _recompute_attention(self) -> None:
        self._attention = self._kb.compute_attention(
            self._goal_id,
            alpha=self._alpha,
            max_distance=self._max_distance,
        )

    def _top_n(self) -> list[str]:
        p = self.priorities()
        return sorted(p, key=p.__getitem__, reverse=True)[: self._budget]
