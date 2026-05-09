"""
reactive.py - ReactiveBaseline attention strategy.

Semantics:  rule-based filtering. Only the active goal and its 1-hop semantic
            neighbors in the class graph receive attention (uniform 1.0). All
            other concepts are ignored. Budget-constrained scheduling via a
            round-robin cycle through the active set.
Purpose:    models the naive approach used in typical robot architectures without
            an awareness manager: "observe what is directly mentioned by the task,
            nothing more." No spreading activation, no anticipatory pre-tuning,
            no epistemic prioritisation, no relational reasoning.

Attention:  1.0 for concepts in the 1-hop goal neighborhood, 0.0 elsewhere.
Channels:   all attention attributed to 'mission'; other channels empty.
Schedule:   round-robin through the active set (up to budget per tick).
            Round-robin ensures all active concepts get refreshed over time
            without any epistemic reasoning - this is the key contrast with
            the AM's ExA priority scheduling.

Goal transitions: when the mission queue promotes a new goal, the active set
    immediately switches to the new goal's neighborhood with no anticipatory
    pre-loading. This is the behavioral contrast with F2 (Anticipatory Horizon):
    the reactive baseline is always surprised by goal transitions because it
    never pre-tunes.

Instance handling: instances of any class in the active set receive attention
    (same uniform 1.0 as their parent class). No relational boost - the reactive
    baseline has no instance-graph reasoning.

Architectural note (Phase 4):
    The 1-hop distance is configurable via hop_distance (default 1). A hop_distance
    of 0 would give "Goal-Only" - only the goal node itself is attended to. This
    degenerate case is left as an easy extension point: if Phase 5 evaluation shows
    the AM-vs-Reactive gap is too narrow to be compelling, a Goal-Only (0-hop)
    baseline can be added as a fourth strategy by constructing with hop_distance=0.
    The round-robin selection and all other logic remain unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase

from awareness_manager.knowledge_base import KnowledgeBase


class ReactiveBaseline:
    """
    Budget-constrained, rule-based baseline: observe the 1-hop goal neighborhood.

    Uses the same mission queue mechanics as AwarenessManager so goal transitions
    occur at the same simulated times. Selection within the active set is pure
    round-robin - no epistemic prioritisation.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        budget: int,
        observation_interval: float = 1.0,
        ikb: InstanceKnowledgeBase | None = None,
        hop_distance: int = 1,
    ) -> None:
        """
        Args:
            kb:                   Class-level semantic KB (shared with other strategies).
            goal_id:              Initial active goal concept ID.
            budget:               Max concepts to schedule per tick (same as AM budget).
            observation_interval: Seconds between observations, used in Formula 3
                                  refresh computation (same T as AM).
            ikb:                  Optional instance KB (shared with other strategies).
            hop_distance:         Neighborhood radius in the class graph.
                                  1 = goal + direct neighbors (default).
                                  0 = goal only (Goal-Only degenerate case).
        """
        if goal_id not in kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        self._kb = kb
        self._goal_id = goal_id
        self._budget = budget
        self._observation_interval = observation_interval
        self._instance_kb = ikb
        self._hop_distance = hop_distance
        self._mission_queue: list[tuple[str, float, str]] = []
        self._rr_index: int = 0

        # Computed from goal + KB structure, rebuilt on each goal transition
        self._attention: dict[str, float] = {}
        self._active_list: list[str] = []
        self._rebuild_active_set()

    # ------------------------------------------------------------------
    # Mission queue (mirrors AwarenessManager semantics exactly)
    # ------------------------------------------------------------------

    def queue_goal(self, goal_id: str, eta: float, level: str = 'task') -> None:
        """Queue a future goal with ETA in simulated seconds (same API as AM)."""
        if goal_id not in self._kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        if eta <= 0.0:
            raise ValueError("ETA must be > 0.")
        if level not in ('global', 'phase', 'task'):
            raise ValueError(f"Unknown level '{level}'.")
        self._mission_queue.append((goal_id, eta, level))
        self._mission_queue.sort(key=lambda x: x[1])

    def _advance_mission_queue(self, dt: float) -> None:
        updated = [(gid, eta - dt, lvl) for gid, eta, lvl in self._mission_queue]
        promoted = [(gid, eta, lvl) for gid, eta, lvl in updated if eta <= 0.0]
        self._mission_queue = [(gid, eta, lvl) for gid, eta, lvl in updated if eta > 0.0]
        for gid, _, _ in promoted:
            self._goal_id = gid

    # ------------------------------------------------------------------
    # Active set construction
    # ------------------------------------------------------------------

    def _active_class_ids(self) -> set[str]:
        """
        Return the set of class IDs in the hop-distance neighborhood of goal_id.

        Uses public semantic_edges() API - no access to private KB graph structure.
        For hop_distance=1: goal + all direct semantic neighbors.
        For hop_distance=0: goal only.
        """
        frontier = {self._goal_id}
        visited = {self._goal_id}
        for _ in range(self._hop_distance):
            next_frontier: set[str] = set()
            for id_a, id_b, _ in self._kb.semantic_edges():
                if id_a in frontier and id_b not in visited:
                    next_frontier.add(id_b)
                elif id_b in frontier and id_a not in visited:
                    next_frontier.add(id_a)
            frontier = next_frontier - visited
            visited |= frontier
        return visited

    def _rebuild_active_set(self) -> None:
        """Recompute attention dict and schedulable list from current goal."""
        active_classes = self._active_class_ids()

        self._attention = {}
        for cid in self._kb.concept_ids():
            self._attention[cid] = 1.0 if cid in active_classes else 0.0

        if self._instance_kb is not None:
            for iid in self._instance_kb.instance_ids():
                inst = self._instance_kb.get_instance(iid)
                self._attention[iid] = 1.0 if inst.class_id in active_classes else 0.0

        # Schedulable = active concepts with decay_rate > 0 (skip task nodes,
        # which have decay_rate=0 and are never worth refreshing).
        self._active_list = sorted([
            cid for cid, a in self._attention.items()
            if a > 0.0 and self._get_decay_rate(cid) > 0.0
        ])
        self._rr_index = 0

    def _get_decay_rate(self, concept_id: str) -> float:
        if concept_id in self._kb.concept_ids():
            return self._kb.get_concept(concept_id).decay_rate
        if self._instance_kb is not None and concept_id in self._instance_kb.instance_ids():
            return self._instance_kb.get_instance(concept_id).decay_rate
        return 0.0

    # ------------------------------------------------------------------
    # AttentionStrategy interface
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """Advance time, handle goal transitions, return round-robin schedule."""
        self._kb.tick(dt)
        if self._instance_kb is not None:
            self._instance_kb.tick(dt)

        prev_goal = self._goal_id
        self._advance_mission_queue(dt)

        # On goal transition: immediately switch to new neighborhood.
        # Note: no anticipatory pre-loading - the reactive baseline is always
        # surprised by goal transitions (this is the F2 contrast point).
        if self._goal_id != prev_goal:
            self._rebuild_active_set()

        # Round-robin: take next min(budget, |active|) concepts from the cycle
        n = len(self._active_list)
        if n == 0:
            return []
        take = min(self._budget, n)
        schedule = [
            self._active_list[(self._rr_index + i) % n]
            for i in range(take)
        ]
        self._rr_index = (self._rr_index + take) % n
        return schedule

    def observe(self, concept_id: str) -> float:
        """Formula 3 refresh (same computation as AwarenessManager.observe())."""
        if concept_id in self._kb.concept_ids():
            dr = self._kb.get_concept(concept_id).decay_rate
            refresh = 1.0 - math.exp(-dr * self._observation_interval) if dr > 0 else 0.0
            self._kb.refresh_concept(concept_id, refresh=refresh)
            return refresh
        if self._instance_kb is not None and concept_id in self._instance_kb.instance_ids():
            dr = self._instance_kb.get_instance(concept_id).decay_rate
            refresh = 1.0 - math.exp(-dr * self._observation_interval) if dr > 0 else 0.0
            self._instance_kb.refresh_instance(concept_id, refresh=refresh)
            return refresh
        return 0.0

    def attention(self) -> dict[str, float]:
        return dict(self._attention)

    def attention_channels(self) -> dict[str, dict[str, float]]:
        return {
            'mission':      {cid: a for cid, a in self._attention.items() if a > 0.0},
            'anticipatory': {},
            'relational':   {},
            'surprise':     {},
        }

    @property
    def goal_id(self) -> str:
        return self._goal_id

    @property
    def mission_queue(self) -> list[tuple[str, float, str]]:
        return list(self._mission_queue)

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def kb(self) -> KnowledgeBase:
        return self._kb

    @property
    def instance_kb(self) -> InstanceKnowledgeBase | None:
        return self._instance_kb

    def strategy_name(self) -> str:
        return "reactive"

    def strategy_params(self) -> dict:
        return {
            "hop_distance": self._hop_distance,
            "selection":    "round_robin",
        }
