"""
Tests for InstanceKnowledgeBase and the class/instance two-layer architecture.

Covers:
  - InstanceKnowledgeBase construction and relational graph
  - Instance attention inheriting from class attention (class gate)
  - Relational boost from instance graph spreading activation
  - Epistemic error dynamics (tick, refresh)
  - AwarenessManager integration: budget spans class + instance concepts
"""

import math

import pytest

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.pv_inspection import (
    build_pv_inspection_instance_kb,
    build_pv_inspection_kb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_class_kb() -> KnowledgeBase:
    """Minimal class KB: task -> tool_a (w=1.0), task -> tool_b (w=3.0)."""
    kb = KnowledgeBase()
    kb.add_concept(Concept('task',   'task',   decay_rate=0.0))
    kb.add_concept(Concept('tool_a', 'object', decay_rate=0.1))
    kb.add_concept(Concept('tool_b', 'object', decay_rate=0.05))
    kb.add_relation('task', 'tool_a', weight=1.0)
    kb.add_relation('task', 'tool_b', weight=3.0)
    return kb


def _simple_instance_kb() -> InstanceKnowledgeBase:
    """
    Minimal instance KB for _simple_class_kb:
        tool_a_inst_1, tool_a_inst_2  → class: tool_a
        tool_b_inst_1                  → class: tool_b (not related to tool_a instances)
        isolated_inst                  → class: tool_a (no relational edges)
    Relations:
        tool_a_inst_1 --locatedAt--> tool_a_inst_2  (weight 1.0)
    """
    ikb = InstanceKnowledgeBase()
    ikb.add_instance(InstanceConcept('tool_a_inst_1', 'object', decay_rate=0.1,  class_id='tool_a'))
    ikb.add_instance(InstanceConcept('tool_a_inst_2', 'object', decay_rate=0.1,  class_id='tool_a'))
    ikb.add_instance(InstanceConcept('tool_b_inst_1', 'object', decay_rate=0.05, class_id='tool_b'))
    ikb.add_instance(InstanceConcept('isolated_inst', 'object', decay_rate=0.05, class_id='tool_a'))
    ikb.add_instance_relation('tool_a_inst_1', 'tool_a_inst_2', weight=1.0, relation_type='locatedAt')
    return ikb


# ---------------------------------------------------------------------------
# InstanceKnowledgeBase construction
# ---------------------------------------------------------------------------

class TestInstanceKBConstruction:

    def test_add_instance(self):
        ikb = InstanceKnowledgeBase()
        inst = InstanceConcept('hammer_1', 'object', decay_rate=0.01, class_id='hammer')
        ikb.add_instance(inst)
        assert 'hammer_1' in ikb.instance_ids()

    def test_add_instance_relation(self):
        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('a', 'object', decay_rate=0.01, class_id='x'))
        ikb.add_instance(InstanceConcept('b', 'object', decay_rate=0.01, class_id='x'))
        ikb.add_instance_relation('a', 'b', weight=1.0, relation_type='locatedAt')
        assert 'b' in ikb.neighbors_of('a')

    def test_add_relation_missing_instance_raises(self):
        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('a', 'object', decay_rate=0.01, class_id='x'))
        with pytest.raises(ValueError, match="b"):
            ikb.add_instance_relation('a', 'b')

    def test_relation_type_stored(self):
        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('a', 'object', decay_rate=0.01, class_id='x'))
        ikb.add_instance(InstanceConcept('b', 'object', decay_rate=0.01, class_id='x'))
        ikb.add_instance_relation('a', 'b', weight=1.0, relation_type='heldBy')
        assert ikb.get_relation_type('a', 'b') == 'heldBy'

    def test_instances_of_class(self):
        ikb = _simple_instance_kb()
        instances = ikb.instances_of_class('tool_a')
        assert set(instances) == {'tool_a_inst_1', 'tool_a_inst_2', 'isolated_inst'}

    def test_pv_inspection_instance_kb_loads(self):
        ikb = build_pv_inspection_instance_kb()
        assert len(ikb) == 7  # 3 panels + 1 battery + 2 landing zones + 1 camera


# ---------------------------------------------------------------------------
# Class gate: instance attention inherits from parent class
# ---------------------------------------------------------------------------

class TestClassGate:

    def test_zero_class_attention_gives_zero_instance_attention(self):
        # tool_b has low class attention when task only connects to tool_a
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        # Only provide attention for tool_a (gate open), not tool_b (gate closed)
        class_attn = {'task': 1.0, 'tool_a': 0.5}
        instance_attn = ikb.compute_instance_attention(class_attn, alpha=0.5)
        # tool_b_inst_1 should get 0 base (no class gate) and 0 relational boost
        assert instance_attn['tool_b_inst_1'] == pytest.approx(0.0)

    def test_instance_inherits_class_attention(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        class_attn = {'task': 1.0, 'tool_a': 0.5}
        instance_attn = ikb.compute_instance_attention(
            class_attn, alpha=0.5, instance_relational_weight=0.0
        )
        # With relational_weight=0, base attention = class attention
        assert instance_attn['tool_a_inst_1'] == pytest.approx(0.5)
        assert instance_attn['tool_a_inst_2'] == pytest.approx(0.5)
        assert instance_attn['isolated_inst']  == pytest.approx(0.5)

    def test_higher_class_attention_gives_higher_instance_attention(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        class_attn = {'task': 1.0, 'tool_a': 0.5, 'tool_b': 0.125}
        instance_attn = ikb.compute_instance_attention(
            class_attn, alpha=0.5, instance_relational_weight=0.0
        )
        assert instance_attn['tool_a_inst_1'] > instance_attn['tool_b_inst_1']

    def test_all_attention_values_in_range(self):
        ikb = _simple_instance_kb()
        class_attn = {'task': 1.0, 'tool_a': 0.8, 'tool_b': 0.3}
        instance_attn = ikb.compute_instance_attention(class_attn)
        for iid, val in instance_attn.items():
            assert 0.0 <= val <= 1.0, f"{iid} attention {val} outside [0, 1]"


# ---------------------------------------------------------------------------
# Relational boost via instance graph spreading
# ---------------------------------------------------------------------------

class TestRelationalBoost:

    def test_relational_boost_adds_to_base(self):
        ikb = _simple_instance_kb()
        class_attn = {'tool_a': 0.5, 'tool_b': 0.2}
        attn_with_boost = ikb.compute_instance_attention(
            class_attn, alpha=0.5, instance_relational_weight=0.3
        )
        # tool_b_inst_1 has no relational connections; stays at class gate
        assert attn_with_boost['tool_b_inst_1'] == pytest.approx(0.2)

    def test_instance_linked_to_relevant_class_gets_relational_boost(self):
        """
        An instance whose OWN class has 0 attention, but which is related to
        an instance of a relevant class, should receive a relational boost.
        """
        ikb = InstanceKnowledgeBase()
        # class 'a' is relevant, class 'b' is not
        ikb.add_instance(InstanceConcept('a_inst', 'object', decay_rate=0.1, class_id='a'))
        ikb.add_instance(InstanceConcept('b_inst', 'object', decay_rate=0.1, class_id='b'))
        ikb.add_instance_relation('a_inst', 'b_inst', weight=1.0, relation_type='locatedAt')

        class_attn = {'a': 0.5}  # 'b' has no attention (gate closed for b_inst base)

        attn_no_boost = ikb.compute_instance_attention(
            class_attn, alpha=0.5, instance_relational_weight=0.0
        )
        attn_with_boost = ikb.compute_instance_attention(
            class_attn, alpha=0.5, instance_relational_weight=0.3
        )

        # Without boost: b_inst base = 0 (class 'b' not attended)
        assert attn_no_boost['b_inst'] == pytest.approx(0.0)

        # With boost: b_inst is 1 hop from a_inst (seed), contribution = 0.5*(1-0.5)^1 * 0.3
        expected_boost = 0.5 * (1.0 - 0.5) ** 1.0 * 0.3
        assert attn_with_boost['b_inst'] == pytest.approx(expected_boost)

    def test_relational_boost_decays_with_distance(self):
        """Instances further from seeds receive smaller relational boost."""
        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('seed',   'object', decay_rate=0.1, class_id='relevant'))
        ikb.add_instance(InstanceConcept('close',  'object', decay_rate=0.1, class_id='other'))
        ikb.add_instance(InstanceConcept('far',    'object', decay_rate=0.1, class_id='other'))
        ikb.add_instance_relation('seed',  'close', weight=1.0)
        ikb.add_instance_relation('close', 'far',   weight=1.0)

        class_attn = {'relevant': 0.8}
        attn = ikb.compute_instance_attention(class_attn, alpha=0.5, instance_relational_weight=1.0)

        assert attn['close'] > attn['far']
        assert attn['close'] == pytest.approx(0.8 * 0.5 ** 1)
        assert attn['far']   == pytest.approx(0.8 * 0.5 ** 2)

    def test_relational_boost_respects_max_distance(self):
        """Instances beyond max_distance get no relational spread."""
        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('seed',    'object', decay_rate=0.1, class_id='relevant'))
        ikb.add_instance(InstanceConcept('too_far', 'object', decay_rate=0.1, class_id='other'))
        ikb.add_instance_relation('seed', 'too_far', weight=5.0)  # weight > max_distance

        class_attn = {'relevant': 1.0}
        attn = ikb.compute_instance_attention(
            class_attn, alpha=0.5, max_distance=4.0, instance_relational_weight=1.0
        )
        assert attn['too_far'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Epistemic error dynamics
# ---------------------------------------------------------------------------

class TestInstanceEpistemicError:

    def test_tick_increases_instance_error(self):
        ikb = _simple_instance_kb()
        before = ikb.get_instance('tool_a_inst_1').epistemic_error
        ikb.tick(dt=5.0)
        after = ikb.get_instance('tool_a_inst_1').epistemic_error
        assert after > before

    def test_tick_proportional_to_decay_rate(self):
        ikb = _simple_instance_kb()
        ikb.tick(dt=10.0)
        # tool_a_inst_1 (δ=0.1) decays faster than tool_b_inst_1 (δ=0.05)
        assert (ikb.get_instance('tool_a_inst_1').epistemic_error >
                ikb.get_instance('tool_b_inst_1').epistemic_error)

    def test_epistemic_error_clamped_at_one(self):
        ikb = _simple_instance_kb()
        ikb.tick(dt=100_000.0)
        assert ikb.get_instance('tool_a_inst_1').epistemic_error == pytest.approx(1.0)

    def test_refresh_reduces_instance_error(self):
        ikb = _simple_instance_kb()
        ikb.tick(dt=10.0)
        before = ikb.get_instance('tool_a_inst_1').epistemic_error
        ikb.refresh_instance('tool_a_inst_1', refresh=0.5)
        after = ikb.get_instance('tool_a_inst_1').epistemic_error
        assert after < before

    def test_epistemic_error_not_below_zero(self):
        ikb = _simple_instance_kb()
        # Large refresh on a fresh instance should not go negative
        ikb.refresh_instance('tool_a_inst_1', refresh=1.0)
        assert ikb.get_instance('tool_a_inst_1').epistemic_error >= 0.0


# ---------------------------------------------------------------------------
# AwarenessManager integration: budget spans class + instance
# ---------------------------------------------------------------------------

class TestAwarenessManagerWithInstances:

    def test_instance_in_schedule_when_kb_attached(self):
        # After ticking, instances of relevant classes should appear in the schedule
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', budget=10, instance_kb=ikb)
        am.tick(dt=100.0)  # drive epistemic errors up
        schedule = am.tick(dt=0.0)
        instance_ids = set(ikb.instance_ids())
        assert any(s in instance_ids for s in schedule), \
            f"Expected at least one instance in schedule, got: {schedule}"

    def test_instance_of_irrelevant_class_not_scheduled(self):
        # Create a KB where 'irrelevant_class' has no connection to the goal.
        kb = KnowledgeBase()
        kb.add_concept(Concept('goal',             'task',   decay_rate=0.0))
        kb.add_concept(Concept('relevant_class',   'object', decay_rate=0.1))
        kb.add_concept(Concept('irrelevant_class', 'object', decay_rate=0.1))
        kb.add_relation('goal', 'relevant_class', weight=1.0)

        ikb = InstanceKnowledgeBase()
        ikb.add_instance(InstanceConcept('relevant_inst',   'object', decay_rate=0.1,
                                         class_id='relevant_class'))
        ikb.add_instance(InstanceConcept('irrelevant_inst', 'object', decay_rate=0.1,
                                         class_id='irrelevant_class'))

        am = AwarenessManager(kb, 'goal', budget=2, instance_kb=ikb)
        am.tick(dt=100.0)
        schedule = am.tick(dt=0.0)

        assert 'irrelevant_inst' not in schedule
        assert 'relevant_inst' in schedule or 'relevant_class' in schedule

        p = am.priorities()
        assert p['irrelevant_inst'] == pytest.approx(0.0)

    def test_budget_respected_across_class_and_instance(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', budget=4, instance_kb=ikb)
        schedule = am.tick(dt=100.0)
        assert len(schedule) <= 4

    def test_observe_works_on_instance_id(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', observation_interval=2.0, instance_kb=ikb)
        am.tick(dt=10.0)
        before = ikb.get_instance('tool_a_inst_1').epistemic_error
        am.observe('tool_a_inst_1')
        after = ikb.get_instance('tool_a_inst_1').epistemic_error
        assert after < before

    def test_observe_instance_returns_refresh_amount(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', observation_interval=2.0, instance_kb=ikb)
        refresh = am.observe('tool_a_inst_1')
        # δ=0.1, T=2.0 → 1 - e^(-0.2)
        expected = 1.0 - math.exp(-0.1 * 2.0)
        assert refresh == pytest.approx(expected)

    def test_observe_unknown_id_raises(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', instance_kb=ikb)
        with pytest.raises(ValueError, match="nonexistent"):
            am.observe('nonexistent')

    def test_priorities_contains_class_and_instance_ids(self):
        kb = _simple_class_kb()
        ikb = _simple_instance_kb()
        am = AwarenessManager(kb, 'task', instance_kb=ikb)
        p = am.priorities()
        assert 'tool_a' in p
        assert 'task' in p
        assert 'tool_a_inst_1' in p
        assert 'tool_b_inst_1' in p

    def test_instance_attention_zero_without_instance_kb(self):
        kb = _simple_class_kb()
        am = AwarenessManager(kb, 'task')
        attn = am.attention()
        assert 'tool_a_inst_1' not in attn

    def test_goal_switch_reshuffles_instance_attention(self):
        # During inspect_pv_field: panel instances should have higher attention
        # During emergency_landing: battery and landing zone instances take over
        kb = build_pv_inspection_kb()
        ikb = build_pv_inspection_instance_kb()
        am = AwarenessManager(kb, 'inspect_pv_field', instance_kb=ikb)

        attn_inspect = am.attention()
        am.set_goal('emergency_landing')
        attn_emergency = am.attention()

        assert attn_inspect.get('panel_A1', 0.0) > attn_emergency.get('panel_A1', 0.0)
        assert (attn_emergency.get('battery_main', 0.0) >
                attn_inspect.get('battery_main', 0.0))
