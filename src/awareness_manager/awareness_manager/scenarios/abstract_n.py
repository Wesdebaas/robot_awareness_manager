"""
abstract_n.py — Abstract N-variable evaluation scenario builder.

Creates a minimal KB for controlled comparison of attention strategies,
mirroring Telogenesis (Deng et al. 2026) Experiment 1: N scalar concepts
with one goal node, R mission-relevant concepts (1-hop neighbours of goal),
and K volatile concepts whose epistemic error is spiked periodically.

Topology
--------
    goal (task, δ=0)
     ├── c0 (relevant)
     ├── c1 (relevant)
     │   ...
     └── c_{R-1} (relevant)

    c_R, c_{R+1}, ..., c_{N-1}  (irrelevant — isolated, no edge to goal)

Volatile concepts are a subset of size K drawn from the N schedulable
concepts.  K_overlap of them fall inside the relevant set (1-hop from
goal); the remaining K − K_overlap are irrelevant.

Returns
-------
    kb            : KnowledgeBase (goal + N concepts)
    volatile_ids  : list[str]  — IDs of the K volatile concepts
    relevant_ids  : list[str]  — IDs of the R relevant concepts (1-hop)
"""

from __future__ import annotations

import random

from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase

_DEFAULT_DECAY_RATE: float = 0.1  # δ for all schedulable concepts


def build_abstract_kb(
    N: int,
    K: int,
    R: int,
    K_overlap: int | None = None,
    seed: int = 42,
    decay_rate: float = _DEFAULT_DECAY_RATE,
) -> tuple[KnowledgeBase, list[str], list[str]]:
    """
    Build an abstract KB with N schedulable concepts and one goal node.

    Args:
        N:          Total number of schedulable concepts.
        K:          Number of volatile concepts (periodically spiked).
        R:          Number of mission-relevant concepts (1-hop from goal).
        K_overlap:  How many volatile concepts are also relevant.
                    Defaults to min(K, R) — maximum overlap.
        seed:       Random seed for deterministic volatile assignment.
        decay_rate: δ for all schedulable concepts (identical so that any
                    scheduling advantage comes from priority, not decay).

    Returns:
        (kb, volatile_ids, relevant_ids)
    """
    if R > N:
        raise ValueError(f"R={R} cannot exceed N={N}")
    if K > N:
        raise ValueError(f"K={K} cannot exceed N={N}")
    if K_overlap is None:
        K_overlap = min(K, R)
    if K_overlap > min(K, R):
        raise ValueError(f"K_overlap={K_overlap} > min(K={K}, R={R})")
    if K_overlap < max(0, K - (N - R)):
        raise ValueError(
            f"K_overlap={K_overlap} too small: {K - (N - R)} irrelevant slots needed"
        )

    rng = random.Random(seed)

    kb = KnowledgeBase()

    # Goal node — task type, δ=0, never scheduled
    goal = Concept(concept_id="goal", concept_type="task", decay_rate=0.0)
    kb.add_concept(goal)

    # N schedulable concepts
    concept_ids = [f"c{i}" for i in range(N)]
    relevant_ids = concept_ids[:R]
    irrelevant_ids = concept_ids[R:]

    for cid in concept_ids:
        kb.add_concept(Concept(
            concept_id=cid,
            concept_type="object",
            decay_rate=decay_rate,
            epistemic_error=0.5,   # start at mid-uncertainty
        ))

    # Connect relevant concepts to goal (weight=1.0 → 1-hop F1 attention)
    for cid in relevant_ids:
        kb.add_relation("goal", cid, weight=1.0)

    # Assign volatile concepts: K_overlap from relevant, rest from irrelevant
    volatile_relevant = rng.sample(relevant_ids, K_overlap)
    volatile_irrelevant = rng.sample(irrelevant_ids, K - K_overlap)
    volatile_ids = volatile_relevant + volatile_irrelevant

    return kb, volatile_ids, relevant_ids
