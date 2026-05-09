import math
from typing import TYPE_CHECKING

from awareness_manager.feature_config import FeatureConfig
from awareness_manager.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase


_LAMBDA_BY_LEVEL: dict[str, float] = {
    'global': 0.05,   # very slow decay - keeps long-horizon background awareness
    'phase':  0.20,   # medium decay  - phase-level planning window
    'task':   0.50,   # fast decay    - near-future task pre-tuning (legacy default)
}
"""Per-level anticipatory discount rates for Hierarchical Mission Horizons."""


class AwarenessManager:
    """
    Awareness Manager - sits on top of the KnowledgeBase and answers:
    "Which concepts should the robot observe right now?"

    Operates at two levels simultaneously:
        - Class level  (KnowledgeBase):      ontological categories (e.g. Hammer, Workbench)
        - Instance level (InstanceKnowledgeBase): specific individuals (e.g. hammer_rack_A)

    If an InstanceKnowledgeBase is provided, the budget constrains both levels.
    Instances of relevant classes inherit attention; instances relationally close
    to active instances receive a scaled relational boost.

    Each tick the AM:
        1. Advances epistemic drift (kb.tick + instance_kb.tick)        - Formula 5
        2. Advances the mission queue, promoting goals whose ETA ≤ 0    - Formula 2
        3. Recomputes class attention from current goal + queued goals   - Formulas 1+2
        4. Computes instance attention from class attention              - Formulas 1+2
        5. Ranks ALL concepts (class + instance) by priority = E x A
        6. Returns the top-N concept IDs as the refresh schedule

    Observations are executed via observe(), which applies Formula 3 to compute
    how much epistemic error to reduce:

        Formula 3 - Utility Saturation:  refresh(n) = 1 - e^(-δ(n) x T)

    where δ(n) is the concept's decay rate and T is the observation interval.
    This calibrates the refresh amount to the drift accumulated since the last
    observation: slow-decaying concepts get a small refresh, fast-decaying ones
    get a larger one - each observation exactly compensates for what was lost.

    Formula 2 - Anticipatory Horizon:
        A_combined(c) = A_current(c) + Σ_i [ e^{-λ x Δt_i} x A_i(c) ]
    Queued goals contribute attention proportional to their proximity in time.
    As ETA decreases, the discount e^{-λΔt} rises toward 1, causing the robot
    to gradually pre-tune its awareness for the upcoming goal before it activates.

    Formula 4 - Quadratic Cost Constraint:
        depth = √B - 1
    The number of graph nodes reachable within depth d grows as (1+d)². Given
    memory budget B, the maximum search depth is √B - 1, replacing the fixed
    max_distance with a resource-derived bound.

    Priority formula:
        priority(c) = E(c) x A(c)

    Task nodes have decay_rate=0 so E stays 0 and priority stays 0 - they are
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
        instance_kb: 'InstanceKnowledgeBase | None' = None,
        instance_relational_weight: float = 0.3,
        relational_spike_factor: float = 0.5,
        certainty_threshold: float = 0.0,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        """
        Args:
            kb:                        The class-level semantic knowledge base.
            goal_id:                   The initial mission goal concept ID.
            alpha:                     Spreading activation decay factor [0, 1].
            max_distance:              Maximum weighted graph distance for attention.
                                       Ignored when memory_budget is set.
            budget:                    Maximum concepts to schedule per tick (top-N).
                                       Constrains both class and instance concepts.
            observation_interval:      Expected seconds between observations (T in
                                       Formula 3). Should match the caller's cadence
                                       so the refresh amount equals the accumulated
                                       drift.
            lambda_horizon:            λ in Formula 2. Controls how quickly the
                                       anticipatory discount decays with time-to-goal.
                                       Higher values mean only near-future goals
                                       influence current attention.
            memory_budget:             B in Formula 4. When set, max search depth is
                                       derived as √B - 1 instead of using max_distance.
                                       None disables Formula 4 (uses max_distance).
            instance_kb:               Optional InstanceKnowledgeBase. When provided,
                                       instance-level concepts are included in attention
                                       computation and the schedule. When None, behaviour
                                       is identical to the class-only mode.
            instance_relational_weight: Scale factor [0, 1] for the relational boost
                                       applied to instances via the instance graph.
                                       Only used when instance_kb is provided.
            relational_spike_factor:   Scale factor [0, 1] for epistemic error spike
                                       propagated to instance-graph neighbors on a
                                       prediction-error violation. Only used when
                                       instance_kb is provided.
            certainty_threshold:       Probabilistic Forgetting gate (Phase 4).
                                       If a concept's epistemic error E ≤ this value
                                       its scheduling priority is forced to 0 -
                                       the concept is considered sufficiently known
                                       and budget is redirected to uncertain concepts.
                                       Default 0.0 disables the gate (matches
                                       existing behaviour: only E=0 concepts, i.e.
                                       task nodes, are skipped).
            feature_config:            Which of the five grounding formulas to
                                       activate. None (default) enables all formulas,
                                       preserving existing behaviour. Use
                                       FeatureConfig.with_disabled('f2') or
                                       FeatureConfig.all_off() for ablation studies.
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
        self._instance_kb = instance_kb
        self._instance_relational_weight = instance_relational_weight
        self._relational_spike_factor = relational_spike_factor
        self._certainty_threshold = certainty_threshold
        self._fc = feature_config if feature_config is not None else FeatureConfig()
        self._attention: dict[str, float] = {}

        # Per-channel attention contributions - stashed in _recompute_attention()
        # for Phase 2 introspection (channel breakdown tooltip, color-by-source).
        self._channel_mission: dict[str, float] = {}
        self._channel_anticipatory: dict[str, float] = {}
        self._channel_relational: dict[str, float] = {}
        self._channel_surprise: dict[str, float] = {}

        # Mission queue: ordered list of (goal_id, time_remaining, level) triples.
        # level is one of 'global' | 'phase' | 'task'.
        # Maintained sorted by time_remaining ascending so the next goal to
        # promote is always at index 0.
        self._mission_queue: list[tuple[str, float, str]] = []

        # One-tick attention boosts set by _handle_violation().
        # Applied and cleared in the next _recompute_attention() call.
        self._violation_boosts: dict[str, float] = {}

        # One-tick attention overrides set by override_attention() / set_attention service.
        # Replace (not add to) the computed attention for one recompute cycle.
        self._attention_overrides: dict[str, float] = {}

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
    # Formula 4 - Quadratic Cost Constraint
    # ------------------------------------------------------------------

    @property
    def effective_max_distance(self) -> float:
        """
        Formula 4: depth = √B - 1, or the fixed max_distance if no budget set.

        The number of nodes within spreading-activation depth d scales as (1+d)².
        Given memory budget B, the maximum depth that fits is √B - 1.
        """
        if self._fc.use_f4_memory_budget and self._memory_budget is not None:
            return max(0.0, math.sqrt(self._memory_budget) - 1.0)
        return self._max_distance

    # ------------------------------------------------------------------
    # Formula 2 - Mission queue / Anticipatory Horizon
    # ------------------------------------------------------------------

    def queue_goal(self, goal_id: str, eta: float, level: str = 'task') -> None:
        """
        Queue a future goal with ETA in simulated seconds from now.

        The goal will auto-promote to the current goal when its ETA reaches 0
        during tick(). While queued, its attention values are blended into the
        current attention window, discounted by e^{-λ_level x ETA} (Formula 2).

        Hierarchical Mission Horizons (Phase 5):
            level='global'  λ = 0.05  - strategic background awareness; a goal
                                         100 s away still contributes ~0.01 x A.
                                         Use for overarching mission objectives.
            level='phase'   λ = 0.20  - operational planning window; a goal 20 s
                                         away contributes ~0.02 x A, 5 s away ~0.37.
                                         Use for mission phases (e.g. "after inspection").
            level='task'    λ = 0.50  - near-future pre-tuning (legacy default);
                                         a goal 5 s away contributes ~0.08 x A,
                                         1 s away ~0.61 x A.
                                         Use for the next immediate sub-task.

        Blending formula across all queued goals:
            A_combined(c) = A_current(c)
                          + Σ_global  e^{-0.05 x Δt} x A_goal(c)
                          + Σ_phase   e^{-0.20 x Δt} x A_goal(c)
                          + Σ_task    e^{-0.50 x Δt} x A_goal(c)

        Args:
            goal_id: A concept ID that must exist in the knowledge base.
            eta:     Simulated seconds until this goal becomes active. Must be > 0.
            level:   Hierarchy level - 'global', 'phase', or 'task'. Controls the
                     anticipatory discount rate λ. Default 'task' matches the
                     legacy single-λ behaviour (lambda_horizon=0.5).
        """
        if goal_id not in self._kb.concept_ids():
            raise ValueError(f"Goal concept '{goal_id}' not in knowledge base.")
        if eta <= 0.0:
            raise ValueError(f"ETA must be > 0; got {eta}. Use set_goal() for immediate switches.")
        if level not in _LAMBDA_BY_LEVEL:
            raise ValueError(
                f"Unknown level '{level}'. Must be one of {list(_LAMBDA_BY_LEVEL)}."
            )
        self._mission_queue.append((goal_id, eta, level))
        self._mission_queue.sort(key=lambda x: x[1])
        self._recompute_attention()

    @property
    def mission_queue(self) -> list[tuple[str, float, str]]:
        """Snapshot of [(goal_id, time_remaining, level), ...] sorted by ETA ascending."""
        return list(self._mission_queue)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> list[str]:
        """
        Advance simulation by dt seconds and return the refresh schedule.

        Steps:
            1. kb.tick(dt) - passive epistemic drift on class concepts       (Formula 5)
               instance_kb.tick(dt) - drift on instance concepts (if present)(Formula 5)
            2. _advance_mission_queue(dt) - decrement ETAs, promote arrived goals (Formula 2)
            3. Recompute class attention (current goal + discounted future goals)  (Formulas 1+2)
               Compute instance attention from class attention                     (Formulas 1+2)
            4. Rank ALL concepts (class + instance) by priority = E x A (descending)
            5. Return top-budget concept IDs

        Returns:
            List of up to `budget` concept IDs ordered by priority (highest first).
        """
        self._kb.tick(dt)
        if self._instance_kb is not None:
            self._instance_kb.tick(dt)
        self._advance_mission_queue(dt)
        self._recompute_attention()
        return self._top_n()

    # ------------------------------------------------------------------
    # Observation (Formula 3)
    # ------------------------------------------------------------------

    def observe(self, concept_id: str) -> float:
        """
        Formula 3 - Utility Saturation: execute one observation on concept_id.

        Computes the refresh amount as:
            refresh(n) = 1 - e^(-δ(n) x observation_interval)

        This equals the drift that accumulates over one observation interval,
        modelled with a saturating exponential so heavily-decaying concepts are
        refreshed proportionally more.

        Works for both class-level concepts (in KnowledgeBase) and instance-level
        concepts (in InstanceKnowledgeBase). Class KB is checked first; if the
        concept_id is not found there, the instance KB is tried.

        Returns:
            The refresh amount applied (useful for logging).

        Raises:
            ValueError: if concept_id is not found in either KB.
        """
        if concept_id in self._kb.concept_ids():
            decay_rate = self._kb.get_concept(concept_id).decay_rate
            if self._fc.use_f3_utility_saturation:
                refresh = 1.0 - math.exp(-decay_rate * self._observation_interval)
            else:
                refresh = 1.0  # F3 OFF: full over-refresh regardless of drift
            self._kb.refresh_concept(concept_id, refresh=refresh)
            return refresh

        if self._instance_kb is not None and concept_id in self._instance_kb.instance_ids():
            decay_rate = self._instance_kb.get_instance(concept_id).decay_rate
            if self._fc.use_f3_utility_saturation:
                refresh = 1.0 - math.exp(-decay_rate * self._observation_interval)
            else:
                refresh = 1.0  # F3 OFF: full over-refresh regardless of drift
            self._instance_kb.refresh_instance(concept_id, refresh=refresh)
            return refresh

        raise ValueError(
            f"Concept '{concept_id}' not found in class KB or instance KB."
        )

    def observe_with_feedback(
        self,
        concept_id: str,
        observed_value: float,
        violation_threshold: float = 0.3,
    ) -> tuple[float, bool]:
        """
        Observation with Perceptual Prediction Error detection.

        Combines a normal refresh (Formula 3) with expectation-violation checking.
        On the first call for a concept, predicted_value is unset so no violation
        is possible - the observed value becomes the initial prediction.

        Violation handling (triggered when |observed − predicted| ≥ threshold):

          Relational channel (instance graph):
            Direct instance-graph neighbors of the violated concept have their
            epistemic error spiked by epsilon x relational_spike_factor.
            This bypasses the budget scheduler - the spike is immediate.

          Semantic channel (class graph):
            The class concept (for instances) or the concept itself (for classes)
            and its 1-hop class-graph neighbors receive a one-tick attention boost
            proportional to epsilon. The boost is cleared on the next tick.

        Args:
            concept_id:          Class or instance concept to observe.
            observed_value:      The newly observed quantity (a scalar in [0, 1]
                                 is conventional, e.g. an occupancy probability,
                                 normalised sensor reading, or confidence score).
            violation_threshold: Minimum |observed − predicted| to trigger
                                 a violation (default 0.3).

        Returns:
            (refresh_amount, was_violated)
        """
        # Locate concept
        if concept_id in self._kb.concept_ids():
            concept = self._kb.get_concept(concept_id)
            is_instance = False
        elif self._instance_kb is not None and concept_id in self._instance_kb.instance_ids():
            concept = self._instance_kb.get_instance(concept_id)
            is_instance = True
        else:
            raise ValueError(
                f"Concept '{concept_id}' not found in class KB or instance KB."
            )

        # Compute prediction error
        epsilon = 0.0
        was_violated = False
        if concept.predicted_value is not None:
            epsilon = abs(observed_value - concept.predicted_value)
            if epsilon >= violation_threshold:
                was_violated = True

        # Update prediction tracking before applying refresh
        concept.predicted_value = observed_value
        concept.prediction_error = epsilon

        # Apply normal refresh (Formula 3)
        refresh = self.observe(concept_id)

        # Handle violation
        if was_violated:
            self._handle_violation(concept_id, epsilon, is_instance=is_instance)

        return refresh, was_violated

    def _handle_violation(
        self,
        concept_id: str,
        epsilon: float,
        is_instance: bool = False,
    ) -> None:
        """
        Two-channel violation propagation triggered by observe_with_feedback().

        Relational channel:
            Immediate epistemic-error spike to all direct instance-graph neighbors
            of the violated node. Spike magnitude = epsilon x relational_spike_factor.

        Semantic channel:
            One-tick attention boost stored in _violation_boosts for the violated
            class (or the class of the violated instance) and its 1-hop class-graph
            neighbors. Boost magnitude = current_attention x epsilon.
            Applied and cleared in the next _recompute_attention() call.
        """
        # Determine the class-level node to propagate from semantically
        if is_instance and self._instance_kb is not None:
            class_id = self._instance_kb.get_instance(concept_id).class_id
        else:
            class_id = concept_id

        # --- Relational channel: spike neighbors in instance graph ---
        if self._instance_kb is not None and is_instance:
            for nb_id in self._instance_kb.neighbors_of(concept_id):
                nb = self._instance_kb.get_instance(nb_id)
                spike = epsilon * self._relational_spike_factor
                nb.epistemic_error = min(1.0, nb.epistemic_error + spike)

        # --- Semantic channel: one-tick boost on class-graph neighbors ---
        if class_id in self._kb.concept_ids():
            neighbors_1hop = list(self._kb._graph.neighbors(class_id))
            for nb_id in [class_id] + neighbors_1hop:
                current_a = self._attention.get(nb_id, 0.0)
                boost = current_a * epsilon
                self._violation_boosts[nb_id] = max(
                    self._violation_boosts.get(nb_id, 0.0), boost
                )

    def observation_refresh_value(self, concept_id: str) -> float:
        """
        Return the Formula 3 refresh value for concept_id without applying it.

        Works for both class and instance concepts.
        """
        if concept_id in self._kb.concept_ids():
            decay_rate = self._kb.get_concept(concept_id).decay_rate
            return 1.0 - math.exp(-decay_rate * self._observation_interval)
        if self._instance_kb is not None and concept_id in self._instance_kb.instance_ids():
            decay_rate = self._instance_kb.get_instance(concept_id).decay_rate
            return 1.0 - math.exp(-decay_rate * self._observation_interval)
        raise ValueError(
            f"Concept '{concept_id}' not found in class KB or instance KB."
        )

    # ------------------------------------------------------------------
    # Read-only snapshots
    # ------------------------------------------------------------------

    def priorities(self) -> dict[str, float]:
        """
        Current priority for every concept. Snapshot, not live.

        Priority formula (Probabilistic Forgetting gate applied):
            P(c) = E(c) x A(c)   if E(c) > certainty_threshold
            P(c) = 0              if E(c) ≤ certainty_threshold

        The certainty gate prevents wasting budget re-observing concepts that
        are already known with sufficient confidence. With the default threshold
        of 0.0 only task nodes (E=0) are excluded, preserving existing behaviour.

        Returns a merged dict covering both class-level and instance-level
        concepts when an InstanceKnowledgeBase is attached.
        """
        τ = self._certainty_threshold
        use_f5 = self._fc.use_f5_epistemic_drift

        def _priority(e: float, a: float) -> float:
            if use_f5:
                return e * a if e > τ else 0.0
            # F5 OFF: drift frozen → schedule by attention alone, no entropy weighting.
            return a

        result = {
            cid: _priority(
                self._kb.get_concept(cid).epistemic_error,
                self._attention.get(cid, 0.0),
            )
            for cid in self._kb.concept_ids()
        }
        if self._instance_kb is not None:
            for iid in self._instance_kb.instance_ids():
                result[iid] = _priority(
                    self._instance_kb.get_instance(iid).epistemic_error,
                    self._attention.get(iid, 0.0),
                )
        return result

    def attention(self) -> dict[str, float]:
        """
        Current attention values (last computed). Snapshot, not live.

        Contains both class-level and instance-level attention when an
        InstanceKnowledgeBase is attached.
        """
        return dict(self._attention)

    def override_attention(self, concept_id: str, value: float) -> None:
        """
        Set a one-tick attention override for a concept.

        The overridden value replaces the normally-computed attention on the
        next _recompute_attention() call, then the override is cleared.
        Useful for external control (e.g. the set_attention ROS2 service).

        Args:
            concept_id: Any concept in the class or instance KB.
            value:      Attention to force, clamped to [0, 1].

        Raises:
            ValueError: if concept_id is not found in either KB.
        """
        if concept_id not in self._kb.concept_ids():
            if self._instance_kb is None or concept_id not in self._instance_kb.instance_ids():
                raise ValueError(
                    f"Concept '{concept_id}' not found in class KB or instance KB."
                )
        self._attention_overrides[concept_id] = max(0.0, min(1.0, value))

    @property
    def instance_kb(self) -> 'InstanceKnowledgeBase | None':
        """The attached InstanceKnowledgeBase, or None if not set."""
        return self._instance_kb

    @property
    def kb(self) -> KnowledgeBase:
        """The class-level knowledge base."""
        return self._kb

    @property
    def budget(self) -> int:
        """Maximum concepts to schedule per tick (top-N)."""
        return self._budget

    @property
    def memory_budget(self) -> 'int | None':
        """Formula 4 memory budget B, or None if disabled."""
        return self._memory_budget

    def attention_channels(self) -> dict[str, dict[str, float]]:
        """
        Per-channel attention contributions from the last _recompute_attention().
        Snapshot, not live.

        Keys:
            'mission'       - F1 spreading activation from the current goal only.
            'anticipatory'  - F2 contributions summed across all queued goals.
            'relational'    - instance-graph relational boost above class-gate baseline.
            'surprise'      - one-tick violation boosts from observe_with_feedback().
        """
        return {
            'mission': dict(self._channel_mission),
            'anticipatory': dict(self._channel_anticipatory),
            'relational': dict(self._channel_relational),
            'surprise': dict(self._channel_surprise),
        }

    def strategy_name(self) -> str:
        """AttentionStrategy interface - identifies this strategy in trace metadata."""
        return "awareness_manager"

    def strategy_params(self) -> dict:
        """
        AttentionStrategy interface - full hyperparameter set for trace metadata.
        Replaces direct access to private attributes in TraceLogger._build_meta().
        """
        return {
            "alpha": self._alpha,
            "budget": self._budget,
            "observation_interval": self._observation_interval,
            "lambda_horizon": self._lambda_horizon,
            "memory_budget": self._memory_budget,
            "instance_relational_weight": self._instance_relational_weight,
            "feature_config": {
                "f1": self._fc.use_f1_spreading_activation,
                "f2": self._fc.use_f2_anticipatory_horizon,
                "f3": self._fc.use_f3_utility_saturation,
                "f4": self._fc.use_f4_memory_budget,
                "f5": self._fc.use_f5_epistemic_drift,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_mission_queue(self, dt: float) -> None:
        """
        Decrement all ETAs by dt and promote goals whose ETA has reached 0.

        Multiple goals may promote in one tick if dt is large. Each promotion
        calls set_goal() internally so attention is NOT recomputed here -
        _recompute_attention() is called once after this method returns.
        """
        updated = [(gid, eta - dt, lvl) for gid, eta, lvl in self._mission_queue]
        # Separate promoted (ETA ≤ 0) from still-pending, keep time-order
        promoted = [(gid, eta, lvl) for gid, eta, lvl in updated if eta <= 0.0]
        self._mission_queue = [(gid, eta, lvl) for gid, eta, lvl in updated if eta > 0.0]

        for gid, _, _lvl in promoted:
            self._goal_id = gid

    def _recompute_attention(self) -> None:
        """
        Formulas 1 + 2 + Hierarchical Mission Horizons (Phase 5):

        Spreading activation from the current goal, blended with discounted
        attention from each queued future goal. Each queued goal uses a
        level-specific λ from _LAMBDA_BY_LEVEL:

            A_combined(c) = A_current(c)
                          + Σ_global  e^{-0.05 x Δt} x A_goal(c)
                          + Σ_phase   e^{-0.20 x Δt} x A_goal(c)
                          + Σ_task    e^{-0.50 x Δt} x A_goal(c)
                          (clamped to [0, 1])

        Instance attention (if instance_kb is set):
            combined[i] = A_class[class_of(i)]
                        + relational_spread(i) * instance_relational_weight

        Per-channel stash: intermediate channel contributions are saved into
        _channel_mission, _channel_anticipatory, _channel_relational, and
        _channel_surprise for Phase 2 introspection (breakdown tooltip,
        color-by-source). The final _attention is unchanged from before.
        """
        # --- Channel 1: mission spreading activation from current goal (F1) ---
        mission_attn = self._kb.compute_attention(
            self._goal_id,
            alpha=self._alpha,
            max_distance=self.effective_max_distance,
            use_spreading_activation=self._fc.use_f1_spreading_activation,
        )
        combined = dict(mission_attn)

        # --- Channel 2: anticipatory contributions from queued goals (F2) ---
        anticipatory_attn: dict[str, float] = {}
        if self._fc.use_f2_anticipatory_horizon:
            for future_goal, eta, level in self._mission_queue:
                lam = _LAMBDA_BY_LEVEL.get(level, self._lambda_horizon)
                discount = math.exp(-lam * eta)
                future_attn = self._kb.compute_attention(
                    future_goal,
                    alpha=self._alpha,
                    max_distance=self.effective_max_distance,
                    use_spreading_activation=self._fc.use_f1_spreading_activation,
                )
                for cid, a in future_attn.items():
                    contrib = discount * a
                    anticipatory_attn[cid] = anticipatory_attn.get(cid, 0.0) + contrib
                    combined[cid] = min(1.0, combined.get(cid, 0.0) + contrib)

        # --- Instance-level attention + relational channel stash ---
        relational_attn: dict[str, float] = {}
        if self._instance_kb is not None:
            instance_attn = self._instance_kb.compute_instance_attention(
                combined,
                alpha=self._alpha,
                max_distance=self.effective_max_distance,
                instance_relational_weight=self._instance_relational_weight,
                use_spreading_activation=self._fc.use_f1_spreading_activation,
            )
            # Relational channel = boost above the class-gate baseline.
            # base = total class attention (F1 + F2) inherited by this instance.
            for iid in self._instance_kb.instance_ids():
                inst = self._instance_kb.get_instance(iid)
                base = combined.get(inst.class_id, 0.0)
                relational_attn[iid] = max(0.0, instance_attn[iid] - base)
            combined.update(instance_attn)

        # --- Channel 3: surprise - violation boosts (Perceptual Prediction Error) ---
        # Applied after normal attention is computed; cleared on each tick so
        # the boost is visible for exactly one recompute cycle.
        surprise_attn: dict[str, float] = dict(self._violation_boosts)
        for cid, boost in self._violation_boosts.items():
            combined[cid] = min(1.0, combined.get(cid, 0.0) + boost)
        self._violation_boosts.clear()

        # --- One-tick attention overrides (set_attention service) ---
        # Replaces (not adds to) the computed attention for one cycle.
        for cid, value in self._attention_overrides.items():
            combined[cid] = value
        self._attention_overrides.clear()

        # Stash channels for Phase 2 introspection
        self._channel_mission = mission_attn
        self._channel_anticipatory = anticipatory_attn
        self._channel_relational = relational_attn
        self._channel_surprise = surprise_attn

        self._attention = combined

    def _top_n(self) -> list[str]:
        p = self.priorities()
        # Only schedule concepts with strictly positive priority.
        # Zero-priority items (task nodes, or E ≤ certainty_threshold) are never
        # worth refreshing - task nodes have decay_rate=0 so E stays 0 and
        # querying them is wasteful; gated concepts are already sufficiently known.
        positive = {k: v for k, v in p.items() if v > 0.0}
        return sorted(positive, key=positive.__getitem__, reverse=True)[: self._budget]
