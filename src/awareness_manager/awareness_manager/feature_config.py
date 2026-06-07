from dataclasses import dataclass

_WEIGHT_MAP: dict[str, str] = {
    'w_ea':            'w_ea_product',
    'w_surp':          'w_surprise',
    'w_f2':            'w_f2_anticipatory',
    'w_urg':           'w_urgency',
    'w_tc':            'w_travel_cost',
    'w_causal':        'w_causal',
    'combination_rule':'combination_rule',
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
             + w_causal          × Σ_{Y: c→Y} prop(c,Y) × E(Y) × A(Y)   [R4 causal benefit]

    Selection ordering applies the spatial opportunity cost (F6) as:

        sort_key(c) = P(c) / travel_cost(c)^w_travel_cost

    Setting a weight to 0.0 disables that component. w_surprise and w_causal both
    default to 0.0 (backward compatible). Set w_causal > 0 to activate the causal
    benefit term so the scheduler prioritises observations that deliver indirect
    epistemic benefit to non-observable causally-implied concepts.
    Note: R4 is constitutive in KBs that contain non-observable concepts; in KBs
    without them the causal term is vacuously zero and w_causal=0.0 is the correct
    default (backward compatible with all existing scenarios).
    Setting w_travel_cost to 0.0 disables the F6 spatial penalty.
    The w_travel_cost exponent must be ≥ 0.

    combination_rule controls how E and A are combined in the E×A term:
      'multiplicative' (default): P_ea = w_ea × (E × A)  — joint filter
      'additive':                 P_ea = w_ea × (E + A)  — allows high-E to compensate low-A

    These weights compose with FeatureConfig: FeatureConfig controls whether
    drift accumulates (F5), spreading activation shape (F1), utility saturation
    (F3), budget depth (F4), and whether decay rates are learned (LD).
    PriorityWeights controls how the resulting signals are combined into a
    scheduling score.
    """
    w_ea_product:      float = 1.0  # E(c) × A_mission(c) (or E+A when combination_rule='additive')
    w_surprise:        float = 0.0  # persistent prediction_error (Telogenesis S̃_i)
    w_f2_anticipatory: float = 1.0  # A_anticipatory(c) from queued goals
    w_urgency:         float = 1.0  # instance-level urgency accumulator
    w_travel_cost:     float = 1.0  # F6 divisor exponent (0 = no spatial penalty)
    w_causal:          float = 0.0  # R4 causal benefit: Σ propagation_weight×E(Y)×A(Y) over causal successors Y
    combination_rule:  str   = 'multiplicative'  # 'multiplicative' (E×A) or 'additive' (E+A)

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
            _float_fields = ('w_ea_product', 'w_surprise', 'w_f2_anticipatory',
                             'w_urgency', 'w_travel_cost', 'w_causal')
            _str_fields   = ('combination_rule',)
            if attr not in _float_fields and attr not in _str_fields:
                raise ValueError(
                    f"Unknown weight key '{k}'. "
                    f"Valid short keys: {list(_WEIGHT_MAP)}; "
                    "full keys: w_ea_product, w_surprise, w_f2_anticipatory, "
                    "w_urgency, w_travel_cost, w_causal, combination_rule."
                )
            kwargs[attr] = str(v) if attr in _str_fields else float(v)
        return cls(**kwargs)
