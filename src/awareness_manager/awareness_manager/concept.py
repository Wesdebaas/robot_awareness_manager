import time
from dataclasses import dataclass, field
from enum import Enum


class PresenceState(str, Enum):
    """Runtime presence state for instance-level concepts.

    PRESENT          — instance is actively tracked in the world.
    SUSPECTED_ABSENT — not seen recently; still scheduled at reduced priority
                       so the robot can look for it to confirm absence.
    CONFIRMED_ABSENT — definitively gone; excluded from attention, drift, and
                       relational spreading. Kept in the dict for re-discovery.
    """
    PRESENT          = 'present'
    SUSPECTED_ABSENT = 'suspected_absent'
    CONFIRMED_ABSENT = 'confirmed_absent'


@dataclass
class Concept:
    """
    A node in the class-level semantic knowledge graph.

    Represents a category of entities the robot may need to be aware of:
    an object class (Tool, Hammer), a location type, a state category, or an
    abstract task. Class-level concepts define the ontological structure and
    are connected by semantic relations in the KnowledgeBase graph.

    For specific physical individuals (e.g. hammer_rack_A), use InstanceConcept.

    The decay_rate (delta) is the core parameter that determines how quickly
    this concept's information becomes stale in the absence of active refresh.
    It drives formula 5 (Epistemic Error): E_{t+1} = E_t + drift - refresh.

    Grounding:
        Friston (2010): Free Energy Principle - agents act to minimise the
        entropy of their internal world model. decay_rate quantifies how fast
        that entropy grows for this specific concept.
    """

    concept_id: str
    concept_type: str       # 'object', 'action', 'location', 'state', 'task'
    decay_rate: float       # delta: epistemic error drift per second [0, inf)

    epistemic_error: float = 0.0                        # E_t in [0, 1]
    last_updated: float = field(default_factory=time.time)

    # F6 - Spatial Opportunity Cost: time (seconds) to examine this concept once
    # the robot is already at its location. Parsed from am:observationCost in TTL.
    observation_cost: float = 1.0

    # Perceptual Prediction Error (Formula PPE)
    predicted_value: float | None = None  # last cached expected observation
    prediction_error: float = 0.0         # |observed − predicted| at last update

    # Class E mode — controls how epistemic error is managed for this concept.
    # 'standalone': E drifts and is refreshed independently (default behaviour).
    # 'derived':    E = aggregation of instance E values; never drifted or refreshed
    #               directly. Requires an InstanceKnowledgeBase to be meaningful.
    class_e_mode: str = 'standalone'      # 'standalone' | 'derived'
    derived_aggregation: str = 'mean'     # 'mean' | 'max' | 'min'  (derived mode only)
    absent_fallback_e: float = 1.0        # E when zero active instances remain (derived only)


@dataclass
class InstanceConcept(Concept):
    """
    A node in the instance-level knowledge graph.

    Represents a specific physical individual in the world - a particular
    tool, person, location, or object that the robot may encounter. Instances
    belong to a class defined in the KnowledgeBase (class graph) and inherit
    relevance from it. They also gain relevance through relational links to
    other currently-active instances in the InstanceKnowledgeBase.

    Example:
        Class: 'hammer'  (in KnowledgeBase)
        Instance: 'hammer_rack_A'  (in InstanceKnowledgeBase, class_id='hammer')

    The class acts as a gate: an instance only receives attention if its
    parent class has non-zero attention from spreading activation. Additionally,
    instances relationally linked to instances of relevant classes receive a
    scaled boost via spreading activation in the instance graph.

    Grounding:
        Proposal sub-question: "how does the balance shift when budget constrains
        both ontological classes AND the instances populating them?"
    """

    class_id: str = ""      # concept_id of the primary parent class in KnowledgeBase
    extra_class_ids: list[str] = field(default_factory=list)  # additional class memberships
    properties: dict = field(default_factory=dict)  # arbitrary instance metadata
    presence_state: PresenceState = PresenceState.PRESENT

    @property
    def all_class_ids(self) -> list[str]:
        """All class memberships: primary class_id followed by any extra memberships."""
        ids = [self.class_id] if self.class_id else []
        return ids + self.extra_class_ids
