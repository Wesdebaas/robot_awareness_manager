from dataclasses import dataclass

_WEIGHT_MAP: dict[str, str] = {
    'w_ea':   'w_ea_product',
    'w_surp': 'w_surprise',
    'w_f2':   'w_f2_anticipatory',
    'w_urg':  'w_urgency',
    'w_tc':   'w_travel_cost',
}

_FLAG_MAP: dict[str, str] = {
    'f1': 'use_f1_spreading_activation',
    'f2': 'use_f2_anticipatory_horizon',
    'f3': 'use_f3_utility_saturation',
    'f4': 'use_f4_memory_budget',
    'f5': 'use_f5_epistemic_priority',
    'f6': 'use_f6_observation_cost',
    'ld': 'use_learnable_decay',
}


@dataclass(frozen=True)
class FeatureConfig:
    """
    Controls which of the six grounding formulas are active, plus learnable decay.

    All flags default to True (or False for use_learnable_decay) - existing
    behaviour is fully preserved when no FeatureConfig is passed to AwarenessManager.

    F1 - Spreading Activation:       KnowledgeBase.compute_attention
    F2 - Anticipatory Horizon:       _recompute_attention mission-queue blend
    F3 - Utility Saturation:         AwarenessManager.observe
    F4 - Quadratic Cost Constraint:  effective_max_distance property
    F5 - Epistemic Error / Entropy:  KnowledgeBase.tick + priorities
    F6 - Spatial Opportunity Cost:   sort_key divisor in priorities()
    LD - Learnable decay:            EMA δ update in AwarenessManager.observe
    """
    use_f1_spreading_activation: bool = True
    use_f2_anticipatory_horizon: bool = True
    use_f3_utility_saturation:   bool = True
    use_f4_memory_budget:        bool = True
    use_f5_epistemic_priority:   bool = True  # False → attention-only scheduling; drift still accumulates regardless
    use_f6_observation_cost:     bool = True
    use_learnable_decay:         bool = False

    @classmethod
    def all_on(cls) -> 'FeatureConfig':
        return cls()

    @classmethod
    def all_off(cls) -> 'FeatureConfig':
        return cls(False, False, False, False, False, False, False)

    @classmethod
    def with_disabled(cls, *flags: str) -> 'FeatureConfig':
        """
        Return a config with the named formulas disabled (all others on).

        Args:
            flags: Formula names to disable, e.g. 'f1', 'F2', 'f3'.

        Raises:
            ValueError: if an unrecognised flag name is given.
        """
        kwargs: dict[str, bool] = {v: True for v in _FLAG_MAP.values()}
        for flag in flags:
            key = flag.lower()
            if key not in _FLAG_MAP:
                raise ValueError(
                    f"Unknown flag '{flag}'. Valid names: {list(_FLAG_MAP)}."
                )
            kwargs[_FLAG_MAP[key]] = False
        return cls(**kwargs)


@dataclass(frozen=True)
class PriorityWeights:
    """
    Continuous weights for each additive component of the priority formula.

    The priority of concept c is computed as:

        P(c) = w_ea_product    × (E(c) × A_mission(c))   [stale AND relevant]
             + w_surprise       × prediction_error(c)     [Telogenesis S̃_i term]
             + w_f2_anticipatory × A_anticipatory(c)      [standalone F2 boost]
             + w_urgency         × urgency(c)             [unmet-need accumulator]

    Selection ordering applies the spatial opportunity cost (F6) as:

        sort_key(c) = P(c) / travel_cost(c)^w_travel_cost

    Setting a weight to 0.0 disables that component. w_surprise defaults to 0.0
    (backward compatible); set to 1.0 to activate the Telogenesis surprise term.
    Setting w_travel_cost to 0.0 disables the F6 spatial penalty.
    The w_travel_cost exponent must be ≥ 0.

    These weights compose with FeatureConfig: FeatureConfig controls whether
    drift accumulates (F5), spreading activation shape (F1), utility saturation
    (F3), budget depth (F4), and whether decay rates are learned (LD).
    PriorityWeights controls how the resulting signals are combined into a
    scheduling score.
    """
    w_ea_product:      float = 1.0  # E(c) × A_mission(c)
    w_surprise:        float = 0.0  # persistent prediction_error (Telogenesis S̃_i)
    w_f2_anticipatory: float = 1.0  # A_anticipatory(c) from queued goals
    w_urgency:         float = 1.0  # instance-level urgency accumulator
    w_travel_cost:     float = 1.0  # F6 divisor exponent (0 = no spatial penalty)

    @classmethod
    def all_default(cls) -> 'PriorityWeights':
        return cls()

    @classmethod
    def ea_only(cls) -> 'PriorityWeights':
        """Pure E×A scheduling — no F2 or urgency contributions."""
        return cls(w_f2_anticipatory=0.0, w_urgency=0.0)

    @classmethod
    def from_dict(cls, d: dict) -> 'PriorityWeights':
        """
        Construct from a dict using short or full key names.

        Short keys: 'w_ea', 'w_f2', 'w_urg', 'w_tc'.
        Full keys:  'w_ea_product', 'w_f2_anticipatory', 'w_urgency', 'w_travel_cost'.
        """
        kwargs: dict[str, float] = {}
        for k, v in d.items():
            attr = _WEIGHT_MAP.get(k.lower(), k)
            if attr not in ('w_ea_product', 'w_surprise', 'w_f2_anticipatory', 'w_urgency', 'w_travel_cost'):
                raise ValueError(
                    f"Unknown weight key '{k}'. "
                    f"Valid short keys: {list(_WEIGHT_MAP)}; "
                    "full keys: w_ea_product, w_surprise, w_f2_anticipatory, w_urgency, w_travel_cost."
                )
            kwargs[attr] = float(v)
        return cls(**kwargs)
