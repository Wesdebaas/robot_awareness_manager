"""
always_on.py - AlwaysOnBaseline attention strategy.

Semantics:  no budget constraint, every concept refreshed every tick.
Purpose:    establishes the epistemic-error lower bound - what does knowledge
            state look like when the robot observes everything at full rate?
            The AM is evaluated by showing it approaches this lower bound for
            goal-relevant concepts while using only budget B refreshes per tick.

Attention:  uniform 1.0 for all concepts (no spreading activation).
Channels:   everything attributed to 'mission'; the other three channels are empty.
Budget:     -1 (unlimited sentinel) - displayed as "Unlimited refresh" in dashboard.
Refresh:    uses the same Formula 3 dynamics as AwarenessManager:
                refresh = 1 - exp(-decay_rate x observation_interval)
            This keeps the comparison fair: the same physical refresh law applies,
            the only difference is that here it runs for every concept, not just
            the budget-capped top-N.

Mission queue / goal promotion: identical to AwarenessManager so both strategies
    experience the same goal transitions at the same simulated times.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase

from awareness_manager.knowledge_base import KnowledgeBase


class AlwaysOnBaseline:
    """
    Unlimited-budget baseline: observes every concept every tick.

    The epistemic error dynamics (drift + Formula 3 refresh) still operate
    on the underlying KB - this is not a "freeze everything" baseline.
    The distinction: the AM must choose *which* concepts to refresh given
    budget B; AlwaysOnBaseline skips that choice and refreshes everything.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        goal_id: str,
        observation_interval: float = 1.0,
        ikb: InstanceKnowledgeBase | None = None,
    ) -> None:
        if goal_id not in kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        self._kb = kb
        self._goal_id = goal_id
        self._observation_interval = observation_interval
        self._instance_kb = ikb
        self._mission_queue: list[tuple[str, float, str]] = []

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
    # AttentionStrategy interface
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """Advance time, promote queued goals, then refresh every concept."""
        self._kb.tick(dt)
        if self._instance_kb is not None:
            self._instance_kb.tick(dt)
        self._advance_mission_queue(dt)

        all_ids = list(self._kb.concept_ids())
        if self._instance_kb is not None:
            all_ids += list(self._instance_kb.instance_ids())

        # Refresh everything - this is the defining characteristic of this baseline
        for cid in all_ids:
            self.observe(cid)

        return all_ids  # "schedule" = the full ontology every tick

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
        """All concepts at uniform 1.0 - no selective attention."""
        result = {cid: 1.0 for cid in self._kb.concept_ids()}
        if self._instance_kb is not None:
            for iid in self._instance_kb.instance_ids():
                result[iid] = 1.0
        return result

    def attention_channels(self) -> dict[str, dict[str, float]]:
        """Everything attributed to mission; other channels unused."""
        all_ids = list(self._kb.concept_ids())
        if self._instance_kb is not None:
            all_ids += list(self._instance_kb.instance_ids())
        return {
            'mission':      {cid: 1.0 for cid in all_ids},
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
        return -1  # unlimited sentinel; dashboard shows "Unlimited refresh"

    @property
    def kb(self) -> KnowledgeBase:
        return self._kb

    @property
    def instance_kb(self) -> InstanceKnowledgeBase | None:
        return self._instance_kb

    def strategy_name(self) -> str:
        return "always_on"

    def strategy_params(self) -> dict:
        return {}  # no hyperparameters; behaviour is fully determined by the KB
