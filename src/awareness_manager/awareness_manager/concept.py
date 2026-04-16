import time
from dataclasses import dataclass, field


@dataclass
class Concept:
    """
    A node in the semantic knowledge graph.

    Represents any entity the robot may need to be aware of: an object,
    tool, location, human state, or abstract task.

    The decay_rate (delta) is the core parameter that determines how quickly
    this concept's information becomes stale in the absence of active refresh.
    It drives formula 5 (Epistemic Error): E_{t+1} = E_t + drift - refresh.

    Grounding:
        Friston (2010): Free Energy Principle — agents act to minimise the
        entropy of their internal world model. decay_rate quantifies how fast
        that entropy grows for this specific concept.
    """

    concept_id: str
    concept_type: str       # 'object', 'action', 'location', 'state', 'task'
    decay_rate: float       # delta: epistemic error drift per second [0, inf)

    epistemic_error: float = 0.0                        # E_t in [0, 1]
    last_updated: float = field(default_factory=time.time)
