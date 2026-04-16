import pytest

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_kb() -> KnowledgeBase:
    """Minimal 3-node KB: one task + two objects directly connected to it."""
    kb = KnowledgeBase()
    kb.add_concept(Concept('goal',    'task',   decay_rate=0.0))
    kb.add_concept(Concept('near',    'object', decay_rate=0.1))
    kb.add_concept(Concept('far',     'object', decay_rate=0.05))
    kb.add_relation('goal', 'near', weight=1.0)
    kb.add_relation('goal', 'far',  weight=3.0)
    return kb


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAwarenessManagerConstruction:

    def test_goal_node_never_scheduled(self):
        # task node has decay_rate=0 → E stays 0 → priority=0, never in top-N
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', budget=5)
        kb.tick(dt=10_000.0)  # drive all other E values up
        schedule = am.tick(dt=0.0)
        assert 'build_birdhouse' not in schedule

    def test_returns_at_most_budget_items(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', budget=3)
        schedule = am.tick(dt=100.0)
        assert len(schedule) <= 3

    def test_invalid_goal_raises(self):
        kb = build_birdhouse_kb()
        with pytest.raises(ValueError, match="nonexistent"):
            AwarenessManager(kb, goal_id='nonexistent')


# ---------------------------------------------------------------------------
# Priority ranking
# ---------------------------------------------------------------------------

class TestPriorityRanking:

    def test_high_attention_high_error_first(self):
        # 'near' is closer to goal (A higher) and has higher decay_rate
        # after ticking, near should beat far in priority
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2)
        am.tick(dt=10.0)
        p = am.priorities()
        assert p['near'] > p['far']

    def test_zero_attention_zero_priority(self):
        # concept outside max_distance has A=0 → priority=0
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', alpha=0.5, max_distance=1.5, budget=2)
        am.tick(dt=100.0)
        p = am.priorities()
        # 'far' has edge weight 3.0 > max_distance 1.5 → A=0
        assert p['far'] == pytest.approx(0.0)

    def test_fresh_concept_not_scheduled(self):
        # E=0 at start → priority=0 regardless of attention
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2)
        p = am.priorities()
        # All concepts start with E=0
        for val in p.values():
            assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Goal switching
# ---------------------------------------------------------------------------

class TestGoalSwitching:

    def test_set_goal_changes_attention(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', budget=3)
        am.tick(dt=100.0)
        # workbench is a location node; make it the goal
        am.set_goal('workbench')
        p_after = am.priorities()
        a_after = am.attention()
        # workbench is now goal → A=1.0
        assert a_after.get('workbench', 0.0) == pytest.approx(1.0)
        # build_birdhouse is now a regular node → A may differ
        assert am.goal_id == 'workbench'

    def test_set_goal_invalid_raises(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse')
        with pytest.raises(ValueError, match="bad_goal"):
            am.set_goal('bad_goal')


# ---------------------------------------------------------------------------
# Tick integration
# ---------------------------------------------------------------------------

class TestTickIntegration:

    def test_tick_returns_list(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', budget=3)
        result = am.tick(dt=1.0)
        assert isinstance(result, list)

    def test_tick_advances_epistemic_error(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse')
        before = kb.get_concept('human_hand').epistemic_error
        am.tick(dt=1.0)
        after = kb.get_concept('human_hand').epistemic_error
        assert after > before

    def test_schedule_reflects_highest_priority_node(self):
        # In _simple_kb: 'near' has A=0.5 and decay_rate=0.1, 'far' has A=0.125
        # and decay_rate=0.05. After ticking both reach E=1.0, so near wins on
        # attention and must appear first in the schedule.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=1)
        am.tick(dt=100.0)
        schedule = am.tick(dt=0.0)
        assert schedule[0] == 'near'


# ---------------------------------------------------------------------------
# Formula 3 — Utility Saturation
# ---------------------------------------------------------------------------

class TestObserve:

    def test_observe_reduces_epistemic_error(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', observation_interval=2.0)
        am.tick(dt=10.0)
        before = kb.get_concept('hammer').epistemic_error
        am.observe('hammer')
        after = kb.get_concept('hammer').epistemic_error
        assert after < before

    def test_observe_returns_refresh_amount(self):
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', observation_interval=2.0)
        refresh = am.observe('hammer')
        assert 0.0 < refresh <= 1.0

    def test_refresh_value_scales_with_decay_rate(self):
        # higher decay rate → larger refresh value
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', observation_interval=2.0)
        r_human_hand = am.observation_refresh_value('human_hand')   # δ=0.1
        r_hammer = am.observation_refresh_value('hammer')           # δ=0.01
        r_workbench = am.observation_refresh_value('workbench')     # δ=0.001
        assert r_human_hand > r_hammer > r_workbench

    def test_refresh_value_matches_formula(self):
        import math
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', observation_interval=2.0)
        # human_hand: δ=0.1, T=2.0 → 1 - e^(-0.2)
        expected = 1.0 - math.exp(-0.1 * 2.0)
        assert am.observation_refresh_value('human_hand') == pytest.approx(expected)

    def test_zero_decay_rate_gives_zero_refresh(self):
        # task node has δ=0 → refresh = 1 - e^0 = 0
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', observation_interval=2.0)
        assert am.observation_refresh_value('build_birdhouse') == pytest.approx(0.0)
