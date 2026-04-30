from dataclasses import dataclass

_FLAG_MAP: dict[str, str] = {
    'f1': 'use_f1_spreading_activation',
    'f2': 'use_f2_anticipatory_horizon',
    'f3': 'use_f3_utility_saturation',
    'f4': 'use_f4_memory_budget',
    'f5': 'use_f5_epistemic_drift',
}


@dataclass(frozen=True)
class FeatureConfig:
    """
    Controls which of the five grounding formulas are active.

    All flags default to True — existing behaviour is fully preserved when
    no FeatureConfig is passed to AwarenessManager.

    F1 — Spreading Activation:       KnowledgeBase.compute_attention
    F2 — Anticipatory Horizon:       _recompute_attention mission-queue blend
    F3 — Utility Saturation:         AwarenessManager.observe
    F4 — Quadratic Cost Constraint:  effective_max_distance property
    F5 — Epistemic Error / Entropy:  KnowledgeBase.tick + priorities
    """
    use_f1_spreading_activation: bool = True
    use_f2_anticipatory_horizon: bool = True
    use_f3_utility_saturation:   bool = True
    use_f4_memory_budget:        bool = True
    use_f5_epistemic_drift:      bool = True

    @classmethod
    def all_on(cls) -> 'FeatureConfig':
        return cls()

    @classmethod
    def all_off(cls) -> 'FeatureConfig':
        return cls(False, False, False, False, False)

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
