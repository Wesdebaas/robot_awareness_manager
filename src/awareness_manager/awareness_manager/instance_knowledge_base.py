import time

import networkx as nx

from awareness_manager.concept import InstanceConcept, PresenceState


class InstanceKnowledgeBase:
    """
    Instance-level knowledge graph for top-down robot awareness management.

    Stores specific physical individuals (instances) as nodes in a weighted
    undirected graph. Instances are connected by typed relational edges
    (e.g. locatedAt, heldBy, assignedTo) that reflect the actual causal and
    spatial structure of the environment - not taxonomic similarity.

    This graph sits alongside the class-level KnowledgeBase. Attention for
    instances is computed in two channels:

      1. Class gate (Formula 1 inherited):
         An instance inherits the attention of its parent class from the class
         graph. If a class has attention 0, all its instances get attention 0
         regardless of their relational links. This prevents irrelevant classes
         from polluting instance attention.

      2. Relational boost:
         Spreading activation within the instance graph, seeded from instances
         whose parent class has attention > 0. An instance that is relationally
         close to instances of relevant classes receives a scaled boost, even
         if its OWN class has low attention in the current mission frame.
         Scaled by instance_relational_weight (default 0.3).

    Combined formula:
        A_instance(i) = A_class[class_of(i)]
                      + relational_spread(i) * instance_relational_weight
        (clamped to [0, 1])

    Epistemic error for instances follows the same Formula 5 dynamics as
    class-level concepts:
        E_{t+1} = E_t + drift - refresh

    Grounding:
        Proposal section "Envisioned Approach": "instances inherit relevance
        from their classes and from their relational links to currently active
        individuals." The two-channel separation (relational vs. semantic
        propagation) also grounds the Perceptual Prediction Error mechanism
        described in the Active Inference section.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._instances: dict[str, InstanceConcept] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_instance(self, instance: InstanceConcept) -> None:
        """Add an instance node to the graph."""
        self._instances[instance.concept_id] = instance
        self._graph.add_node(instance.concept_id)

    def add_instance_relation(
        self,
        id_a: str,
        id_b: str,
        weight: float = 1.0,
        relation_type: str | None = None,
    ) -> None:
        """
        Add a typed relational edge between two instances.

        Weight represents relational distance (lower = tighter coupling).
        relation_type is stored as edge metadata but does not affect spreading
        activation weights - it is available for future relational propagation
        in the Perceptual Prediction Error mechanism.

        Args:
            id_a:          Source instance concept_id.
            id_b:          Target instance concept_id.
            weight:        Relational distance; lower means tighter coupling.
            relation_type: Optional semantic label (e.g. 'locatedAt', 'heldBy').

        Raises:
            ValueError: if either instance has not been added.
        """
        if id_a not in self._instances:
            raise ValueError(f"Instance '{id_a}' not in InstanceKnowledgeBase.")
        if id_b not in self._instances:
            raise ValueError(f"Instance '{id_b}' not in InstanceKnowledgeBase.")
        self._graph.add_edge(id_a, id_b, weight=weight, relation_type=relation_type)

    # ------------------------------------------------------------------
    # Presence state management
    # ------------------------------------------------------------------

    def mark_suspected_absent(self, instance_id: str) -> None:
        """Transition instance to SUSPECTED_ABSENT.

        The instance remains in the schedule at reduced priority so the robot
        can observe it to confirm or deny absence.
        """
        if instance_id not in self._instances:
            raise ValueError(f"Instance '{instance_id}' not in InstanceKnowledgeBase.")
        self._instances[instance_id].presence_state = PresenceState.SUSPECTED_ABSENT

    def confirm_present(self, instance_id: str) -> None:
        """Restore a suspected- or confirmed-absent instance to PRESENT.

        If the instance was CONFIRMED_ABSENT (removed from the graph), it is
        re-added as an isolated node; call add_instance_relation() to restore
        any edges that were lost.
        """
        if instance_id not in self._instances:
            raise ValueError(f"Instance '{instance_id}' not in InstanceKnowledgeBase.")
        self._instances[instance_id].presence_state = PresenceState.PRESENT
        if instance_id not in self._graph:
            self._graph.add_node(instance_id)

    def confirm_absent(self, instance_id: str) -> None:
        """Mark instance as CONFIRMED_ABSENT.

        Removes the node from the relational graph (no spreading activation),
        but keeps the InstanceConcept in the dict for history and re-discovery.
        """
        if instance_id not in self._instances:
            raise ValueError(f"Instance '{instance_id}' not in InstanceKnowledgeBase.")
        self._instances[instance_id].presence_state = PresenceState.CONFIRMED_ABSENT
        if instance_id in self._graph:
            self._graph.remove_node(instance_id)

    def active_instance_ids(self) -> list[str]:
        """Return IDs of PRESENT and SUSPECTED_ABSENT instances only."""
        return [
            iid for iid, inst in self._instances.items()
            if inst.presence_state != PresenceState.CONFIRMED_ABSENT
        ]

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def compute_instance_attention(
        self,
        class_attention: dict[str, float],
        alpha: float = 0.5,
        max_distance: float = 4.0,
        instance_relational_weight: float = 0.3,
        use_spreading_activation: bool = True,
    ) -> dict[str, float]:
        """
        Compute attention values for all instances.

        Channel 1 - Class gate:
            base(i) = class_attention.get(class_id_of(i), 0.0)
            An instance only inherits attention if its class is in the active
            attention window. Instances of irrelevant classes get 0 base attention.

        Channel 2 - Relational boost:
            Seeds = all instances whose class_attention > 0.
            For each seed s, run spreading activation through the instance graph:
                contribution to i = class_attention[class_of(s)] * (1-alpha)^dist(s, i)
            Take the maximum contribution across all seeds for each instance i.
            Only contributions from OTHER seeds (d > 0) are counted - the base
            attention already handles the self-seeding (d=0) case.
            The final relational boost is scaled by instance_relational_weight.

        Combined:
            A(i) = clamp(base(i) + relational_boost(i) * instance_relational_weight, 0, 1)

        Args:
            class_attention:           Dict from class concept_id to attention [0,1].
                                       Typically from KnowledgeBase.compute_attention().
            alpha:                     Spreading activation decay factor [0, 1].
            max_distance:              Max weighted path length in instance graph.
            instance_relational_weight: Scale factor for relational boost [0, 1].

        Returns:
            Dict mapping instance concept_id -> attention value in [0, 1].
        """
        # Only active instances (PRESENT + SUSPECTED_ABSENT) participate.
        # CONFIRMED_ABSENT instances are excluded from attention entirely.
        active = self.active_instance_ids()

        # Identify seed instances (active, whose ANY class has non-zero attention).
        # Multi-class instances take the max attention across all their memberships.
        seed_attention: dict[str, float] = {}
        for iid in active:
            inst = self._instances[iid]
            a = max((class_attention.get(c, 0.0) for c in inst.all_class_ids), default=0.0)
            if a > 0.0:
                seed_attention[iid] = a

        # Spreading activation through instance graph from each seed
        # relational_boost[i] = max contribution from any seed at d > 0
        relational_boost: dict[str, float] = {}

        for seed_id, seed_a in seed_attention.items():
            if seed_id not in self._graph or self._graph.degree(seed_id) == 0:
                continue
            distances = nx.single_source_dijkstra_path_length(
                self._graph, seed_id, cutoff=max_distance, weight='weight'
            )
            for iid, d in distances.items():
                if d == 0.0:
                    continue  # skip self - handled by base attention
                contrib = seed_a * ((1.0 - alpha) ** d if use_spreading_activation else 1.0)
                relational_boost[iid] = max(relational_boost.get(iid, 0.0), contrib)

        # Combine channels — active instances only.
        # Multi-class instances inherit the max attention across all their class memberships.
        result: dict[str, float] = {}
        for iid in active:
            inst = self._instances[iid]
            # Class-gate baseline + direct goal-to-instance attention (F6 parameterised goals:
            # a goal node in the class KB may have an edge directly to an instance ID, giving
            # that instance a targeted boost without routing through its class).
            base = max(
                max((class_attention.get(c, 0.0) for c in inst.all_class_ids), default=0.0),
                class_attention.get(iid, 0.0),
            )
            boost = relational_boost.get(iid, 0.0) * instance_relational_weight
            result[iid] = min(1.0, base + boost)

        return result

    def neighbors_of(self, instance_id: str) -> list[str]:
        """Return direct neighbors of an instance in the relational graph."""
        if instance_id not in self._graph:
            return []
        return list(self._graph.neighbors(instance_id))

    def get_relation_type(self, id_a: str, id_b: str) -> str | None:
        """Return the relation_type label for an edge, or None if no label."""
        if not self._graph.has_edge(id_a, id_b):
            return None
        return self._graph[id_a][id_b].get('relation_type')

    # ------------------------------------------------------------------
    # Formula 5 - Epistemic Error / Entropy
    # ------------------------------------------------------------------

    def refresh_instance(self, instance_id: str, refresh: float) -> None:
        """
        Apply an observation to reduce epistemic error for one instance.

        Mirrors KnowledgeBase.refresh_concept but operates on instance nodes.
        Epistemic error is clamped to [0, 1].
        """
        instance = self._instances[instance_id]
        instance.epistemic_error = max(0.0, min(1.0,
            instance.epistemic_error + instance.decay_rate - refresh
        ))
        instance.last_updated = time.time()

    def tick(self, dt: float, apply_drift: bool = True) -> None:
        """
        Advance time by dt seconds, applying passive drift to all instances.

        Mirrors KnowledgeBase.tick() but for instance nodes.
        When apply_drift is False (F5 OFF baseline), epistemic error is frozen.
        """
        if not apply_drift:
            return
        for instance in self._instances.values():
            if instance.presence_state == PresenceState.CONFIRMED_ABSENT:
                continue  # no longer in the world; don't accumulate drift
            instance.epistemic_error = min(1.0,
                instance.epistemic_error + instance.decay_rate * dt
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_instance(self, instance_id: str) -> InstanceConcept:
        return self._instances[instance_id]

    def instance_ids(self) -> list[str]:
        return list(self._instances.keys())

    def instances_of_class(self, class_id: str) -> list[str]:
        """Return all instance IDs that belong to the given class (any membership slot)."""
        return [iid for iid, inst in self._instances.items()
                if class_id in inst.all_class_ids]

    def relational_edges(self) -> list[tuple[str, str, float, str | None]]:
        """Return all relational edges as (id_a, id_b, weight, relation_type) tuples."""
        return [
            (u, v, self._graph[u][v]['weight'], self._graph[u][v].get('relation_type'))
            for u, v in self._graph.edges()
        ]

    def __len__(self) -> int:
        return len(self._instances)
