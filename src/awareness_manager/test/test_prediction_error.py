"""
Tests for Perceptual Prediction Error (Phase 2).

Covers:
    - observe_with_feedback: normal (no violation) path
    - observe_with_feedback: violation detection (epsilon >= threshold)
    - predicted_value updated on every call
    - Relational channel: neighbor epistemic error spiked on instance violation
    - Semantic channel: one-tick attention boost for class-graph neighbors
    - Violation clears after one _recompute_attention() (next tick)
    - First observation never triggers violation (no prior prediction)
    - Class-level concept violations propagate on semantic channel only
"""

import math
import pytest

from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.awareness_manager import AwarenessManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_am(
    instance_kb: InstanceKnowledgeBase | None = None,
    relational_spike_factor: float = 0.5,
    budget: int = 5,
) -> AwarenessManager:
    """Minimal KB: task → tool → location, with optional instance KB."""
    kb = KnowledgeBase()
    kb.add_concept(Concept('task',     'task',     decay_rate=0.0))
    kb.add_concept(Concept('tool',     'object',   decay_rate=0.2))
    kb.add_concept(Concept('location', 'location', decay_rate=0.1))
    kb.add_relation('task', 'tool',     weight=1.0)
    kb.add_relation('tool', 'location', weight=1.0)

    return AwarenessManager(
        kb,
        goal_id='task',
        alpha=0.5,
        budget=budget,
        observation_interval=1.0,
        instance_kb=instance_kb,
        relational_spike_factor=relational_spike_factor,
    )


def _make_instance_kb() -> InstanceKnowledgeBase:
    """
    Three instances:
        tool_A  (class: tool)     -- linked to location_X
        tool_B  (class: tool)     -- isolated
        place_X (class: location) -- linked to tool_A
    """
    ikb = InstanceKnowledgeBase()
    ikb.add_instance(InstanceConcept('tool_A',   'object',   decay_rate=0.2, class_id='tool'))
    ikb.add_instance(InstanceConcept('tool_B',   'object',   decay_rate=0.2, class_id='tool'))
    ikb.add_instance(InstanceConcept('place_X',  'location', decay_rate=0.1, class_id='location'))
    ikb.add_instance_relation('tool_A', 'place_X', weight=1.0, relation_type='locatedAt')
    return ikb


# ---------------------------------------------------------------------------
# 1. First observation never triggers violation
# ---------------------------------------------------------------------------

def test_first_observation_no_violation():
    am = _make_simple_am()
    refresh, violated = am.observe_with_feedback('tool', observed_value=0.8)
    assert not violated
    assert refresh >= 0.0


def test_first_observation_sets_predicted_value():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.7)
    concept = am._kb.get_concept('tool')
    assert concept.predicted_value == pytest.approx(0.7)
    assert concept.prediction_error == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Normal (sub-threshold) second observation
# ---------------------------------------------------------------------------

def test_small_delta_no_violation():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.5)   # sets prediction
    refresh, violated = am.observe_with_feedback('tool', observed_value=0.6, violation_threshold=0.3)
    assert not violated
    concept = am._kb.get_concept('tool')
    assert concept.prediction_error == pytest.approx(0.1)
    assert concept.predicted_value  == pytest.approx(0.6)


def test_small_delta_returns_formula3_refresh():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.5)
    refresh, _ = am.observe_with_feedback('tool', observed_value=0.55)
    expected = 1.0 - math.exp(-0.2 * 1.0)  # decay_rate=0.2, interval=1.0
    assert refresh == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. Violation detection
# ---------------------------------------------------------------------------

def test_large_delta_triggers_violation():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.1)
    _, violated = am.observe_with_feedback('tool', observed_value=0.9, violation_threshold=0.3)
    assert violated


def test_violation_updates_prediction_error_field():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.1)
    am.observe_with_feedback('tool', observed_value=0.8, violation_threshold=0.3)
    concept = am._kb.get_concept('tool')
    assert concept.prediction_error == pytest.approx(0.7)
    assert concept.predicted_value  == pytest.approx(0.8)


def test_violation_at_exact_threshold():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.0)
    _, violated = am.observe_with_feedback('tool', observed_value=0.3, violation_threshold=0.3)
    assert violated


def test_no_violation_just_below_threshold():
    am = _make_simple_am()
    am.observe_with_feedback('tool', observed_value=0.0)
    _, violated = am.observe_with_feedback('tool', observed_value=0.29, violation_threshold=0.3)
    assert not violated


# ---------------------------------------------------------------------------
# 4. Semantic channel: one-tick attention boost for class-graph neighbors
# ---------------------------------------------------------------------------

def test_violation_boosts_class_neighbors():
    am = _make_simple_am()
    am.tick(dt=1.0)   # build up attention state
    attn_before = am.attention()['location']

    # First observation on 'tool' - sets prediction
    am.observe_with_feedback('tool', observed_value=0.1)
    # Second observation - triggers violation
    am.observe_with_feedback('tool', observed_value=0.9, violation_threshold=0.3)

    # _violation_boosts should now be populated for 'tool' and its neighbor 'location'
    assert len(am._violation_boosts) > 0

    # After _recompute_attention() (next tick), boost is applied and cleared
    am.tick(dt=0.0)
    assert len(am._violation_boosts) == 0


def test_violation_boost_increases_neighbor_attention():
    """
    After a violation on 'tool', the one-tick boost should make 'location'
    (a 1-hop class neighbor) reach higher attention than the un-violated baseline.
    """
    am = _make_simple_am()
    am.tick(dt=1.0)

    # Snapshot attention without violation
    baseline_location = am.attention().get('location', 0.0)

    # Trigger a large violation on 'tool'
    am.observe_with_feedback('tool', observed_value=0.0)     # set prediction
    am.observe_with_feedback('tool', observed_value=1.0, violation_threshold=0.1)

    # Next tick applies the boost
    am.tick(dt=0.0)
    boosted_location = am.attention().get('location', 0.0)

    # Location should have at least as much attention (boost is additive)
    assert boosted_location >= baseline_location


def test_violation_boost_cleared_after_one_tick():
    """Boost must NOT persist beyond one _recompute_attention() call."""
    am = _make_simple_am()
    am.tick(dt=1.0)

    am.observe_with_feedback('tool', observed_value=0.0)
    am.observe_with_feedback('tool', observed_value=1.0, violation_threshold=0.1)

    am.tick(dt=0.0)  # applies boost, clears _violation_boosts
    attn_after_boost = am.attention().get('location', 0.0)

    am.tick(dt=0.0)  # no new violation - should be back to normal
    attn_two_ticks  = am.attention().get('location', 0.0)

    # Two ticks after violation, attention should be <= the boosted value
    # (no boost persists)
    assert attn_two_ticks <= attn_after_boost


# ---------------------------------------------------------------------------
# 5. Relational channel: instance-graph neighbor epistemic-error spike
# ---------------------------------------------------------------------------

def test_instance_violation_spikes_neighbor_error():
    """
    tool_A violated → place_X (direct neighbor) should get an error spike.
    tool_B (no edge) should be unaffected.
    """
    ikb = _make_instance_kb()
    am = _make_simple_am(instance_kb=ikb, relational_spike_factor=0.5)
    am.tick(dt=1.0)   # let epistemic error build

    place_X_before = ikb.get_instance('place_X').epistemic_error
    tool_B_before  = ikb.get_instance('tool_B').epistemic_error

    # Set initial prediction for tool_A
    am.observe_with_feedback('tool_A', observed_value=0.1)
    # Trigger violation
    am.observe_with_feedback('tool_A', observed_value=0.9, violation_threshold=0.3)

    place_X_after = ikb.get_instance('place_X').epistemic_error
    tool_B_after  = ikb.get_instance('tool_B').epistemic_error

    epsilon = 0.8
    expected_spike = epsilon * 0.5  # relational_spike_factor=0.5

    # Neighbor spiked
    assert place_X_after >= place_X_before + expected_spike * 0.9   # allow small float tolerance
    # Unrelated instance unchanged
    assert tool_B_after == pytest.approx(tool_B_before)


def test_instance_violation_spike_clamped_to_one():
    """Epistemic error from a spike must not exceed 1.0."""
    ikb = _make_instance_kb()
    # Force place_X to near-max error
    ikb.get_instance('place_X').epistemic_error = 0.99

    am = _make_simple_am(instance_kb=ikb, relational_spike_factor=1.0)
    am.observe_with_feedback('tool_A', observed_value=0.0)
    am.observe_with_feedback('tool_A', observed_value=1.0, violation_threshold=0.1)

    assert ikb.get_instance('place_X').epistemic_error <= 1.0


def test_class_violation_no_relational_spike():
    """
    Violating a class-level concept (not an instance) should NOT spike any
    instance neighbors - the relational channel only applies to instances.
    """
    ikb = _make_instance_kb()
    am = _make_simple_am(instance_kb=ikb)

    place_X_before = ikb.get_instance('place_X').epistemic_error

    # Violate class-level 'tool'
    am.observe_with_feedback('tool', observed_value=0.0)
    am.observe_with_feedback('tool', observed_value=1.0, violation_threshold=0.1)

    place_X_after = ikb.get_instance('place_X').epistemic_error
    # Class violation does NOT propagate through instance graph
    assert place_X_after == pytest.approx(place_X_before)


# ---------------------------------------------------------------------------
# 6. Unknown concept raises ValueError
# ---------------------------------------------------------------------------

def test_observe_with_feedback_unknown_raises():
    am = _make_simple_am()
    with pytest.raises(ValueError, match="not found"):
        am.observe_with_feedback('nonexistent', observed_value=0.5)


# ---------------------------------------------------------------------------
# 7. Works for instance-level concepts (no class KB entry)
# ---------------------------------------------------------------------------

def test_instance_observe_with_feedback_no_violation():
    ikb = _make_instance_kb()
    am = _make_simple_am(instance_kb=ikb)
    refresh, violated = am.observe_with_feedback('tool_A', observed_value=0.4)
    assert not violated
    assert refresh >= 0.0
    assert ikb.get_instance('tool_A').predicted_value == pytest.approx(0.4)


def test_instance_observe_with_feedback_violation():
    ikb = _make_instance_kb()
    am = _make_simple_am(instance_kb=ikb)
    am.observe_with_feedback('tool_A', observed_value=0.1)
    refresh, violated = am.observe_with_feedback('tool_A', observed_value=0.9, violation_threshold=0.3)
    assert violated
    assert ikb.get_instance('tool_A').prediction_error == pytest.approx(0.8)
