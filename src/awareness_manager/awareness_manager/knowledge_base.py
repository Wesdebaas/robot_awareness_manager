import time
from typing import TYPE_CHECKING

import networkx as nx

from awareness_manager.concept import Concept, PresenceState

if TYPE_CHECKING:
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase


class KnowledgeBase:

    """
    Semantic graph knowledge base for top-down robot awareness management.

    Concepts are nodes in a weighted undirected graph. Edge weights represent
    semantic distance: lower weight means tighter conceptual coupling. Initial
    weights are set per domain; future extensions may adapt them from task
    co-occurrence (see Cognitive Architecture Extensions: Experience-Driven
    Edge Weights).

    Implements two of the five grounding formulas directly:

        Formula 1 - Spreading Activation:    R_n = (1 - alpha)^d
        Formula 5 - Epistemic Error/Entropy: E_{t+1} = E_t + drift - refresh

    Formulas 2-4 (Anticipatory Horizon, Utility Saturation, Quadratic Cost)
    operate at the Awareness Manager level, which sits on top of this KB.

    Graph implementation rationale:
        The spreading activation formula is defined over a semantic network
        where d is the shortest weighted path distance from the goal node.
        NetworkX provides native Dijkstra shortest-path computation, making
        it the direct computational realisation of the model described in
        Collins & Loftus (1975) and used in ACT-R (Anderson et al. 2004).
        Weighted edges enable variable semantic proximity, grounding the
        quadratic cost constraint (Cormen et al. 2009) in traversal depth.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._causal_graph: nx.DiGraph = nx.DiGraph()
        self._concepts: dict[str, Concept] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_concept(self, concept: Concept) -> None:
        """Add a concept node to the knowledge graph."""
        self._concepts[concept.concept_id] = concept
        self._graph.add_node(concept.concept_id)

    def add_relation(self, id_a: str, id_b: str, weight: float = 1.0) -> None:
        """
        Add a weighted semantic relation between two concepts.

        Weight represents semantic distance: lower values indicate tighter
        conceptual coupling (e.g. hammer-nail: 1.0 vs. tool-location: 2.0).
        These initial values are domain-set; the architecture is designed so
        they can be updated from experience without structural changes.

        Raises:
            ValueError: if either concept has not been added to the KB.
        """
        if id_a not in self._concepts:
            raise ValueError(f"Concept '{id_a}' not in knowledge base.")
        if id_b not in self._concepts:
            raise ValueError(f"Concept '{id_b}' not in knowledge base.")
        self._graph.add_edge(id_a, id_b, weight=weight)

    def neighbors(self, concept_id: str) -> list[str]:
        """Direct semantic graph neighbors of a concept (1-hop, any edge weight)."""
        if concept_id not in self._graph:
            raise ValueError(f"Concept '{concept_id}' not in knowledge graph.")
        return list(self._graph.neighbors(concept_id))

    # ------------------------------------------------------------------
    # Causal graph — stateProject (D1.4 §3.3.3)
    # ------------------------------------------------------------------

    def add_causal_relation(self, source_id: str, target_id: str, propagation_weight: float) -> None:
        """
        Add a directed causal edge: observing source gives partial epistemic
        information about target.

        propagation_weight ∈ (0, 1]: fraction of the source observation's refresh
        amount that propagates to the target's epistemic error.  A value of 0.5
        means observing source delivers half the epistemic benefit of a direct
        observation of target.

        Causal edges are separate from semantic edges and do not affect spreading
        activation (A). They only affect E propagation on observation.

        Raises:
            ValueError: if either concept is not in the knowledge base.
        """
        if source_id not in self._concepts:
            raise ValueError(f"Concept '{source_id}' not in knowledge base.")
        if target_id not in self._concepts:
            raise ValueError(f"Concept '{target_id}' not in knowledge base.")
        self._causal_graph.add_node(source_id)
        self._causal_graph.add_node(target_id)
        self._causal_graph.add_edge(source_id, target_id, propagation_weight=propagation_weight)

    def causal_neighbors(self, concept_id: str) -> list[tuple[str, float]]:
        """
        Return [(target_id, propagation_weight), ...] for all direct causal
        successors of concept_id.  Empty list if concept has no outgoing causal edges.
        """
        if concept_id not in self._causal_graph:
            return []
        return [
            (tgt, data['propagation_weight'])
            for tgt, data in self._causal_graph[concept_id].items()
        ]

    def propagate_causal_refresh(self, concept_id: str, amount: float) -> None:
        """
        Apply a causal propagation refresh: reduce E by amount without adding drift.

        Unlike refresh_concept() (which applies Formula 5: E += decay_rate − refresh),
        this only subtracts — no drift is added — because no direct observation took
        place.  Clamped to [0, 1].  No-op for derived-mode concepts.
        """
        concept = self._concepts.get(concept_id)
        if concept is None or concept.class_e_mode == 'derived':
            return
        concept.epistemic_error = max(0.0, min(1.0, concept.epistemic_error - amount))

    # ------------------------------------------------------------------
    # Formula 1 - Spreading Activation
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        goal_id: str,
        alpha: float = 0.5,
        max_distance: float = 4.0,
        use_spreading_activation: bool = True,
    ) -> dict[str, float]:
        """
        Formula 1 - Spreading Activation: R_n = (1 - alpha)^d

        Computes attention values for all concepts reachable from the goal
        within max_distance weighted path length. Dijkstra finds the shortest
        weighted distance d from the goal to each concept; the formula then
        maps d to an attention value in (0, 1].

        The goal node always receives attention 1.0 (d = 0).
        Concepts beyond max_distance are not returned (implicitly zero).

        When use_spreading_activation is False (F1 OFF baseline), all reachable
        concepts receive uniform attention 1.0 - the reachability cutoff still
        applies, but distance decay is disabled. This wastes budget on semantically
        distant concepts, demonstrating the value of the formula.

        Args:
            goal_id:                  ID of the current mission goal concept.
            alpha:                    Decay factor per unit of semantic distance [0, 1].
            max_distance:             Maximum weighted graph distance to consider.
            use_spreading_activation: When False, returns uniform A=1.0 for all
                                      reachable concepts (F1 OFF baseline).

        Returns:
            Dict mapping concept_id -> attention value in (0, 1].
        """
        if goal_id not in self._graph:
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge graph.")

        distances = nx.single_source_dijkstra_path_length(
            self._graph, goal_id, cutoff=max_distance, weight='weight'
        )
        if use_spreading_activation:
            return {cid: (1.0 - alpha) ** d for cid, d in distances.items()}
        return {cid: 1.0 for cid in distances}

    # ------------------------------------------------------------------
    # Formula 5 - Epistemic Error / Entropy
    # ------------------------------------------------------------------

    def refresh_concept(self, concept_id: str, refresh: float) -> None:
        """
        Formula 5 - Epistemic Error: E_{t+1} = E_t + drift - refresh

        Apply an observation to reduce epistemic error for one concept.
        Drift (the concept's decay_rate) is always added, modelling the
        entropy increase from environmental dynamics. The refresh term
        represents the uncertainty reduction from an active observation.

        Epistemic error is clamped to [0, 1].

        No-op for derived-mode concepts — their E is managed entirely by
        update_derived_class_errors() and cannot be set through observation.
        """
        concept = self._concepts[concept_id]
        if concept.class_e_mode == 'derived':
            return
        concept.epistemic_error = max(0.0, min(1.0,
            concept.epistemic_error + concept.decay_rate - refresh
        ))
        concept.last_updated = time.time()

    def update_derived_class_errors(self, instance_kb: 'InstanceKnowledgeBase') -> None:
        """
        Recompute epistemic error for all derived-mode concepts from their instances.

        Contributing instances: PRESENT + SUSPECTED_ABSENT (CONFIRMED_ABSENT excluded).
        Aggregation is per-concept and configurable: 'mean', 'max', or 'min'.

        When no contributing instances exist, the concept's absent_fallback_e is used
        (default 1.0 — maximum uncertainty when the robot has lost all instances).

        Called by AwarenessManager.tick() after instance_kb.tick() so derived class E
        always reflects the freshest instance E values within a given tick.
        """
        for concept in self._concepts.values():
            if concept.class_e_mode != 'derived':
                continue
            contributing = [
                instance_kb.get_instance(iid)
                for iid in instance_kb.instances_of_class(concept.concept_id)
                if instance_kb.get_instance(iid).presence_state in (
                    PresenceState.PRESENT, PresenceState.SUSPECTED_ABSENT
                )
            ]
            if not contributing:
                concept.epistemic_error = concept.absent_fallback_e
            else:
                errors = [inst.epistemic_error for inst in contributing]
                agg = concept.derived_aggregation
                if agg == 'max':
                    concept.epistemic_error = max(errors)
                elif agg == 'min':
                    concept.epistemic_error = min(errors)
                else:  # 'mean' is the default
                    concept.epistemic_error = sum(errors) / len(errors)

    def tick(self, dt: float, apply_drift: bool = True) -> None:
        """
        Advance time by dt seconds, applying passive drift to all concepts.

        Without active refresh, epistemic error grows at each concept's
        decay_rate per second. This models the natural entropy increase
        described in formula 5 (drift term).

        When apply_drift is False (F5 OFF baseline), epistemic error is frozen -
        no drift is applied. Scheduling then falls back to attention-only priority
        (see AwarenessManager.priorities), removing entropy-driven urgency.
        """
        if not apply_drift:
            return
        for concept in self._concepts.values():
            if concept.class_e_mode == 'derived':
                continue  # E is managed by update_derived_class_errors; skip drift
            concept.epistemic_error = min(1.0,
                concept.epistemic_error + concept.decay_rate * dt
            )

    # ------------------------------------------------------------------
    # Learnable decay (FeatureConfig.use_learnable_decay)
    # ------------------------------------------------------------------

    def update_decay_rate(
        self,
        concept_id: str,
        surprise: float,
        tau: float = 0.05,
        delta_min: float = 0.01,
        delta_max: float = 1.0,
    ) -> float:
        """
        EMA update of a concept's decay rate from observed surprise.

            δ_i ← (1 − τ) · δ_i  +  τ · S_i(t)

        High surprise → larger δ → concept drifts faster when unobserved →
        concept is scheduled more frequently. Self-organising adaptation to
        volatility structure without labels (Telogenesis Experiment 3).

        Args:
            concept_id: Concept whose decay rate to update.
            surprise:   Normalised surprise signal S_i ∈ [0, 1] (typically
                        concept.prediction_error at observation time).
            tau:        EMA smoothing rate (default 0.05).
            delta_min:  Lower clamp for the learned decay rate.
            delta_max:  Upper clamp for the learned decay rate.

        Returns:
            Updated decay rate.
        """
        concept = self._concepts[concept_id]
        new_delta = (1.0 - tau) * concept.decay_rate + tau * surprise
        concept.decay_rate = max(delta_min, min(delta_max, new_delta))
        return concept.decay_rate

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_concept(self, concept_id: str) -> Concept:
        return self._concepts[concept_id]

    def concept_ids(self) -> list[str]:
        return list(self._concepts.keys())

    def semantic_edges(self) -> list[tuple[str, str, float]]:
        """Return all semantic edges as (id_a, id_b, weight) triples."""
        return [
            (u, v, self._graph[u][v]['weight'])
            for u, v in self._graph.edges()
        ]

    def __len__(self) -> int:
        return len(self._concepts)
