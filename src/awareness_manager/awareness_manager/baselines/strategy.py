"""
strategy.py - AttentionStrategy Protocol.

Defines the structural interface that AwarenessManager and all baseline
strategies must satisfy so that SimulationRunner, TraceLogger, and
snapshot.py can drive any of them identically.

Design notes
------------
Using typing.Protocol (PEP 544) for structural subtyping means
AwarenessManager satisfies the interface without modification - any object
that has these methods and properties is automatically a valid AttentionStrategy.

budget convention
    Return a non-negative integer for strategies with a fixed scheduling budget.
    Return -1 to signal "unlimited" (AlwaysOnBaseline refreshes everything every
    tick, so no meaningful budget cap exists). The dashboard checks budget < 0
    and displays "Unlimited refresh" instead of a numeric bar.

strategy_params() conventions (per Phase 4 design decision):
    AlwaysOnBaseline  → {}
    ReactiveBaseline  → {"hop_distance": 1, "selection": "round_robin"}
    AwarenessManager  → full hyperparameter set (alpha, budget, lambdas, etc.)

Private coupling note
    The runner and logger access am._channel_surprise directly for surprise-event
    capture. That coupling is intentional and AM-specific; baselines without
    violation boosts simply don't have this attribute, and the logger guards
    with getattr(strategy, '_channel_surprise', {}).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
    from awareness_manager.knowledge_base import KnowledgeBase


@runtime_checkable
class AttentionStrategy(Protocol):
    """
    Structural interface for attention-management strategies.

    Any object that implements these methods and properties - AwarenessManager,
    AlwaysOnBaseline, ReactiveBaseline, or a future SOTA baseline - is a valid
    AttentionStrategy and can be passed to SimulationRunner and TraceLogger.
    """

    # ── Core tick loop ────────────────────────────────────────────────────

    def tick(self, dt: float) -> list[str]:
        """
        Advance by dt simulated seconds and return the refresh schedule.

        The schedule is a list of concept IDs the strategy wants observed
        this tick, in priority order. An empty list is valid (nothing scheduled).
        """
        ...

    def observe(self, concept_id: str) -> float:
        """Apply one observation refresh; return the refresh amount applied."""
        ...

    # ── State accessors ───────────────────────────────────────────────────

    def attention(self) -> dict[str, float]:
        """Current attention for every concept (class + instance) in [0, 1]."""
        ...

    def attention_channels(self) -> dict[str, dict[str, float]]:
        """
        Per-channel attention contributions from the last tick.

        Keys: 'mission', 'anticipatory', 'relational', 'surprise'.
        Baselines that don't implement a channel return an empty dict for it.
        """
        ...

    @property
    def goal_id(self) -> str:
        """Current active goal concept ID."""
        ...

    @property
    def mission_queue(self) -> list[tuple[str, float, str]]:
        """Queued future goals: [(goal_id, eta_seconds, level), ...]."""
        ...

    @property
    def budget(self) -> int:
        """
        Maximum concepts to schedule per tick, or -1 for unlimited.
        The dashboard renders -1 as "Unlimited refresh".
        """
        ...

    @property
    def kb(self) -> KnowledgeBase:
        """Class-level semantic knowledge base."""
        ...

    @property
    def instance_kb(self) -> InstanceKnowledgeBase | None:
        """Instance-level knowledge base, or None if not used."""
        ...

    # ── Trace metadata ────────────────────────────────────────────────────

    def strategy_name(self) -> str:
        """
        Short identifier written to meta.json 'strategy' field.
        Examples: "awareness_manager", "always_on", "reactive".
        """
        ...

    def strategy_params(self) -> dict:
        """
        Strategy-specific hyperparameters for meta.json 'strategy_params' field.
        Self-describing: must contain enough info to characterise this run.
        """
        ...
