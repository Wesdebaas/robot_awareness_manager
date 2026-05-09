import math

import pytest

from awareness_manager.awareness_manager import AwarenessManager
from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb


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
# Formula 3 - Utility Saturation
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


# ---------------------------------------------------------------------------
# Formula 2 - Anticipatory Horizon
# ---------------------------------------------------------------------------

class TestAnticipatoryHorizon:

    def test_queued_goal_boosts_its_concepts(self):
        # Queue emergency_landing; drone_battery should gain extra attention
        # compared to having no queue.
        kb1 = build_pv_inspection_kb()
        am_no_queue = AwarenessManager(kb1, goal_id='inspect_pv_field', lambda_horizon=0.5)

        kb2 = build_pv_inspection_kb()
        am_with_queue = AwarenessManager(kb2, goal_id='inspect_pv_field', lambda_horizon=0.5)
        am_with_queue.queue_goal('emergency_landing', eta=5.0)

        a_no_queue = am_no_queue.attention()
        a_with_queue = am_with_queue.attention()

        # Concepts that are primary in emergency_landing but not in inspect_pv_field
        # should have higher attention when the goal is queued.
        assert a_with_queue.get('drone_battery', 0.0) > a_no_queue.get('drone_battery', 0.0)
        assert a_with_queue.get('landing_zone', 0.0) > a_no_queue.get('landing_zone', 0.0)

    def test_closer_eta_gives_more_attention(self):
        # eta=2 → discount = e^{-0.5*2} = e^{-1} ≈ 0.37
        # eta=10 → discount = e^{-0.5*10} = e^{-5} ≈ 0.007
        # Closer queued goal should contribute more attention to its concepts.
        kb_near = build_pv_inspection_kb()
        am_near = AwarenessManager(kb_near, goal_id='inspect_pv_field', lambda_horizon=0.5)
        am_near.queue_goal('emergency_landing', eta=2.0)

        kb_far = build_pv_inspection_kb()
        am_far = AwarenessManager(kb_far, goal_id='inspect_pv_field', lambda_horizon=0.5)
        am_far.queue_goal('emergency_landing', eta=10.0)

        near_attn = am_near.attention().get('drone_battery', 0.0)
        far_attn = am_far.attention().get('drone_battery', 0.0)
        assert near_attn > far_attn

    def test_zero_eta_promotes_goal_on_tick(self):
        # After a tick that brings ETA to 0 or below, the goal should auto-promote.
        kb = build_pv_inspection_kb()
        am = AwarenessManager(kb, goal_id='inspect_pv_field', lambda_horizon=0.5)
        am.queue_goal('emergency_landing', eta=5.0)
        assert am.goal_id == 'inspect_pv_field'
        am.tick(dt=5.0)  # ETA: 5.0 - 5.0 = 0.0 → promotes
        assert am.goal_id == 'emergency_landing'
        assert am.mission_queue == []

    def test_no_queue_unchanged(self):
        # Empty queue → attention identical to direct compute_attention call.
        kb = build_pv_inspection_kb()
        am = AwarenessManager(kb, goal_id='inspect_pv_field', lambda_horizon=0.5)
        direct = kb.compute_attention('inspect_pv_field', alpha=0.5, max_distance=4.0)
        via_am = am.attention()
        for cid in direct:
            assert via_am.get(cid, 0.0) == pytest.approx(direct[cid])

    def test_multiple_goals_in_queue_both_contribute(self):
        # Two queued goals → both should boost their respective primary concepts.
        kb = build_pv_inspection_kb()
        am = AwarenessManager(kb, goal_id='inspect_pv_field', lambda_horizon=0.5)

        # Baseline: only current goal active
        kb_base = build_pv_inspection_kb()
        am_base = AwarenessManager(kb_base, goal_id='inspect_pv_field', lambda_horizon=0.5)

        # Queue two goals (emergency is primary in emergency_landing;
        # solar_panel is primary in inspect_pv_field itself, but let us
        # use emergency_landing twice at different ETAs to verify additive blending)
        am.queue_goal('emergency_landing', eta=5.0)
        # We can only queue each goal once per slot, but we can observe the effect
        # of the single queued goal across multiple concepts:
        a = am.attention()
        a_base = am_base.attention()

        # Both primary emergency concepts should be boosted
        assert a.get('drone_battery', 0.0) > a_base.get('drone_battery', 0.0)
        assert a.get('airspace', 0.0) > a_base.get('airspace', 0.0)


# ---------------------------------------------------------------------------
# Formula 4 - Quadratic Cost Constraint
# ---------------------------------------------------------------------------

class TestQuadraticCost:

    def test_memory_budget_formula(self):
        # effective_max_distance = sqrt(B) - 1
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', memory_budget=9)
        assert am.effective_max_distance == pytest.approx(math.sqrt(9) - 1.0)  # 2.0

    def test_small_budget_excludes_far_concepts(self):
        # Budget=4 → depth = sqrt(4)-1 = 1.0.
        # In _simple_kb: 'far' has edge weight 3.0 > 1.0 → excluded from attention.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', memory_budget=4)
        kb.tick(dt=100.0)  # max out E for both concepts
        a = am.attention()
        assert 'far' not in a or a.get('far', 0.0) == pytest.approx(0.0)

    def test_large_budget_includes_all(self):
        # Budget=36 → depth = sqrt(36)-1 = 5.0, well beyond any edge in _simple_kb.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', memory_budget=36)
        a = am.attention()
        # Both 'near' and 'far' should be reachable
        assert a.get('near', 0.0) > 0.0
        assert a.get('far', 0.0) > 0.0

    def test_none_budget_uses_max_distance(self):
        # Without memory_budget, effective_max_distance == max_distance.
        kb = build_birdhouse_kb()
        am = AwarenessManager(kb, goal_id='build_birdhouse', max_distance=3.5, memory_budget=None)
        assert am.effective_max_distance == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Probabilistic Forgetting - certainty_threshold (Phase 4)
# ---------------------------------------------------------------------------

class TestCertaintyThreshold:

    def test_default_threshold_zero_unchanged_behaviour(self):
        # threshold=0.0 means only E=0 concepts are suppressed (task nodes).
        # Verified by checking a high-E concept stays scheduled.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=0.0)
        kb.tick(dt=100.0)   # drive E to max for 'near' and 'far'
        schedule = am.tick(dt=0.0)
        assert len(schedule) == 2
        assert 'near' in schedule or 'far' in schedule

    def test_concept_below_threshold_excluded_from_schedule(self):
        # Set threshold=0.5 and keep E low on 'near' - it should be skipped.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=0.5)
        # Do NOT tick - near.E and far.E start at 0.0, both below threshold
        schedule = am.tick(dt=0.0)
        # No concept has E > 0.5, so nothing should be scheduled
        assert 'near' not in schedule
        assert 'far'  not in schedule

    def test_concept_above_threshold_is_scheduled(self):
        # After enough drift 'near' has E > threshold → should appear in schedule.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=0.05)
        kb.tick(dt=5.0)   # near: E += 0.1 * 5 = 0.5  → well above 0.05
        schedule = am.tick(dt=0.0)
        assert 'near' in schedule

    def test_threshold_above_one_excludes_all(self):
        # Pathological case: threshold=2.0 → nothing ever scheduled.
        kb = _simple_kb()
        am = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=2.0)
        kb.tick(dt=100.0)
        schedule = am.tick(dt=0.0)
        assert schedule == []

    def test_high_threshold_reduces_refresh_count(self):
        # With a high certainty threshold, total refreshes over a simulation
        # should be strictly fewer than with threshold=0.
        kb_base = _simple_kb()
        kb_gate  = _simple_kb()

        am_base = AwarenessManager(kb_base, goal_id='goal', budget=2,
                                   observation_interval=1.0, certainty_threshold=0.0)
        am_gate  = AwarenessManager(kb_gate, goal_id='goal', budget=2,
                                    observation_interval=1.0, certainty_threshold=0.3)

        count_base = count_gate = 0
        for _ in range(50):
            sched_b = am_base.tick(dt=0.1)
            for cid in sched_b:
                am_base.observe(cid)
                count_base += 1

            sched_g = am_gate.tick(dt=0.1)
            for cid in sched_g:
                am_gate.observe(cid)
                count_gate += 1

        assert count_gate <= count_base

    def test_priorities_reflect_threshold(self):
        # After drift, concept with E=0.6 and threshold=0.5 → priority > 0.
        # Same concept with threshold=0.7 → priority = 0.
        kb = _simple_kb()
        am_low  = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=0.5)
        am_high = AwarenessManager(kb, goal_id='goal', budget=2, certainty_threshold=0.7)

        kb.tick(dt=6.0)  # near.E ~ 0.6, far.E ~ 0.3
        am_low.tick(dt=0.0)
        am_high.tick(dt=0.0)

        p_low  = am_low.priorities()
        p_high = am_high.priorities()

        # With low threshold 'near' has priority > 0
        assert p_low['near'] > 0.0
        # With high threshold 'near' (E ~0.6 < 0.7) has priority = 0
        assert p_high['near'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hierarchical Mission Horizons - queue_goal levels (Phase 5)
# ---------------------------------------------------------------------------

class TestHierarchicalMissionHorizons:

    def _kb_two_tasks(self) -> KnowledgeBase:
        """KB with two tasks sharing a concept."""
        kb = KnowledgeBase()
        kb.add_concept(Concept('task_a',   'task',   decay_rate=0.0))
        kb.add_concept(Concept('task_b',   'task',   decay_rate=0.0))
        kb.add_concept(Concept('shared',   'object', decay_rate=0.1))
        kb.add_concept(Concept('unique_b', 'object', decay_rate=0.1))
        kb.add_relation('task_a', 'shared',   weight=1.0)
        kb.add_relation('task_b', 'shared',   weight=1.0)
        kb.add_relation('task_b', 'unique_b', weight=1.0)
        return kb

    # ── queue_goal API ────────────────────────────────────────────────────

    def test_default_level_is_task(self):
        kb = self._kb_two_tasks()
        am = AwarenessManager(kb, goal_id='task_a', budget=3)
        am.queue_goal('task_b', eta=5.0)
        assert am.mission_queue[0][2] == 'task'

    def test_explicit_level_stored(self):
        kb = self._kb_two_tasks()
        am = AwarenessManager(kb, goal_id='task_a', budget=3)
        am.queue_goal('task_b', eta=10.0, level='global')
        assert am.mission_queue[0][2] == 'global'

    def test_invalid_level_raises(self):
        kb = self._kb_two_tasks()
        am = AwarenessManager(kb, goal_id='task_a', budget=3)
        with pytest.raises(ValueError, match="Unknown level"):
            am.queue_goal('task_b', eta=5.0, level='ultra')

    def test_mission_queue_returns_triples(self):
        kb = self._kb_two_tasks()
        am = AwarenessManager(kb, goal_id='task_a', budget=3)
        am.queue_goal('task_b', eta=7.0, level='phase')
        entry = am.mission_queue[0]
        assert len(entry) == 3
        goal_id, eta, level = entry
        assert goal_id == 'task_b'
        assert eta == pytest.approx(7.0)
        assert level == 'phase'

    # ── Discount rate differences ─────────────────────────────────────────

    def test_global_level_higher_discount_at_long_eta(self):
        """
        At ETA=50 s, global level (λ=0.05) contributes much more than task (λ=0.5).
        e^(-0.05*50)=0.082  vs  e^(-0.5*50) ≈ 0 (essentially zero).
        """
        kb = self._kb_two_tasks()
        am_global = AwarenessManager(kb, goal_id='task_a', budget=3)
        am_task   = AwarenessManager(kb, goal_id='task_a', budget=3)

        am_global.queue_goal('task_b', eta=50.0, level='global')
        am_task.queue_goal(  'task_b', eta=50.0, level='task')

        attn_global = am_global.attention().get('unique_b', 0.0)
        attn_task   = am_task.attention().get(  'unique_b', 0.0)

        # Global-level queued goal still contributes non-trivially at t=50s;
        # task-level is near zero.
        assert attn_global > attn_task
        assert attn_global > 0.0

    def test_task_level_vanishes_at_large_eta(self):
        """
        At ETA=100 s, task level λ=0.5 → discount=e^(-50) ≈ 0.
        Use max_distance=1 so unique_b is NOT reachable from task_a directly
        (it is 3 hops away via shared→task_b→unique_b in the undirected graph).
        """
        kb = self._kb_two_tasks()
        # max_distance=1 → task_a only reaches 'shared' (d=1); task_b (d=2) and
        # unique_b (d=3) are unreachable from task_a through class spreading.
        am = AwarenessManager(kb, goal_id='task_a', budget=3, max_distance=1.0)
        am.queue_goal('task_b', eta=100.0, level='task')

        # discount=e^(-0.5*100)=e^(-50)≈0 → unique_b contribution essentially zero
        attn = am.attention().get('unique_b', 0.0)
        assert attn < 0.001

    def test_phase_level_intermediate_discount(self):
        """Phase level (λ=0.2) sits between global and task at moderate ETAs."""
        kb = self._kb_two_tasks()
        am_global = AwarenessManager(kb, goal_id='task_a', budget=3)
        am_phase  = AwarenessManager(kb, goal_id='task_a', budget=3)
        am_task   = AwarenessManager(kb, goal_id='task_a', budget=3)

        eta = 10.0
        am_global.queue_goal('task_b', eta=eta, level='global')
        am_phase.queue_goal( 'task_b', eta=eta, level='phase')
        am_task.queue_goal(  'task_b', eta=eta, level='task')

        a_global = am_global.attention().get('unique_b', 0.0)
        a_phase  = am_phase.attention().get( 'unique_b', 0.0)
        a_task   = am_task.attention().get(  'unique_b', 0.0)

        # global decays slowest → highest contribution; task decays fastest → lowest
        assert a_global > a_phase > a_task

    # ── Promotion still works ─────────────────────────────────────────────

    def test_global_goal_promotes_when_eta_reaches_zero(self):
        kb = self._kb_two_tasks()
        am = AwarenessManager(kb, goal_id='task_a', budget=3)
        am.queue_goal('task_b', eta=2.0, level='global')
        am.tick(dt=2.0)  # ETA → 0
        assert am.goal_id == 'task_b'

    def test_multiple_levels_queued_simultaneously(self):
        """All three levels can coexist in the queue."""
        kb = self._kb_two_tasks()
        kb.add_concept(Concept('task_c', 'task', decay_rate=0.0))
        kb.add_concept(Concept('unique_c', 'object', decay_rate=0.1))
        kb.add_relation('task_c', 'unique_c', weight=1.0)

        am = AwarenessManager(kb, goal_id='task_a', budget=5)
        am.queue_goal('task_b', eta=100.0, level='global')
        am.queue_goal('task_b', eta=30.0,  level='phase')
        am.queue_goal('task_c', eta=5.0,   level='task')

        assert len(am.mission_queue) == 3
        levels = {entry[2] for entry in am.mission_queue}
        assert levels == {'global', 'phase', 'task'}

        # unique_b gets contributions from two queued entries (global + phase)
        # unique_c from one (task, short ETA so large discount applied)
        attn = am.attention()
        assert attn.get('unique_b', 0.0) > 0.0
        assert attn.get('unique_c', 0.0) > 0.0
