import pytest

from awareness_manager.concept import Concept
from awareness_manager.knowledge_base import KnowledgeBase
from awareness_manager.scenarios.birdhouse import build_birdhouse_kb
from awareness_manager.scenarios.pv_inspection import build_pv_inspection_kb


class TestKnowledgeBaseConstruction:

    def test_add_concept(self):
        kb = KnowledgeBase()
        kb.add_concept(Concept('hammer', 'object', decay_rate=0.01))
        assert 'hammer' in kb.concept_ids()

    def test_add_relation(self):
        kb = KnowledgeBase()
        kb.add_concept(Concept('hammer', 'object', decay_rate=0.01))
        kb.add_concept(Concept('nail', 'object', decay_rate=0.02))
        kb.add_relation('hammer', 'nail', weight=1.0)

    def test_add_relation_missing_concept_raises(self):
        kb = KnowledgeBase()
        kb.add_concept(Concept('hammer', 'object', decay_rate=0.01))
        with pytest.raises(ValueError, match="nail"):
            kb.add_relation('hammer', 'nail', weight=1.0)

    def test_birdhouse_scenario_loads(self):
        kb = build_birdhouse_kb()
        assert len(kb) == 10  # task + 3 tools + 2 fasteners + 1 material + 2 locations + 1 human
        assert 'build_birdhouse' in kb.concept_ids()

    def test_pv_inspection_scenario_loads(self):
        kb = build_pv_inspection_kb()
        assert len(kb) == 11  # 2 tasks + 3 plant + 3 drone + 3 environment
        assert 'inspect_pv_field' in kb.concept_ids()
        assert 'emergency_landing' in kb.concept_ids()

    def test_pv_inspection_goal_switch_reshuffles_attention(self):
        # The key thesis demo: switching from inspection to emergency changes
        # which concepts get the most attention.
        kb = build_pv_inspection_kb()
        a_inspect  = kb.compute_attention('inspect_pv_field',  alpha=0.5)
        a_emergency = kb.compute_attention('emergency_landing', alpha=0.5)
        # solar_panel is close to inspect goal, far/absent from emergency
        assert a_inspect.get('solar_panel', 0.0) > a_emergency.get('solar_panel', 0.0)
        # drone_battery is close to emergency goal, peripheral during inspection
        assert a_emergency.get('drone_battery', 0.0) > a_inspect.get('drone_battery', 0.0)


class TestSpreadingActivation:

    def test_goal_node_has_full_attention(self):
        kb = build_birdhouse_kb()
        attention = kb.compute_attention('build_birdhouse', alpha=0.5)
        assert attention['build_birdhouse'] == pytest.approx(1.0)

    def test_direct_neighbour_attention(self):
        # hammer is at distance 1.0 from goal → (1 - 0.5)^1.0 = 0.5
        kb = build_birdhouse_kb()
        attention = kb.compute_attention('build_birdhouse', alpha=0.5)
        assert attention['hammer'] == pytest.approx(0.5 ** 1.0)

    def test_attention_decreases_with_distance(self):
        kb = build_birdhouse_kb()
        attention = kb.compute_attention('build_birdhouse', alpha=0.5)
        # hammer (d=1.0) should have more attention than workbench (d=3.0)
        assert attention['hammer'] > attention['workbench']

    def test_tight_coupling_gives_more_attention_than_loose(self):
        kb = build_birdhouse_kb()
        attention = kb.compute_attention('build_birdhouse', alpha=0.5)
        # nail (d=1.0 via direct edge) vs workbench (d=3.0 via hammer→workbench)
        assert attention['nail'] > attention['workbench']

    def test_concepts_beyond_cutoff_absent(self):
        kb = build_birdhouse_kb()
        # human_hand is at distance 2.5 (goal→hammer:1.0, hammer→human_hand:1.5)
        attention = kb.compute_attention('build_birdhouse', alpha=0.5, max_distance=2.0)
        assert 'human_hand' not in attention

    def test_all_attention_values_in_range(self):
        kb = build_birdhouse_kb()
        attention = kb.compute_attention('build_birdhouse', alpha=0.5)
        for cid, val in attention.items():
            assert 0.0 < val <= 1.0, f"{cid} has attention {val} outside (0, 1]"

    def test_unknown_goal_raises(self):
        kb = build_birdhouse_kb()
        with pytest.raises(ValueError):
            kb.compute_attention('nonexistent_goal')


class TestEpistemicError:

    def test_tick_increases_error(self):
        kb = build_birdhouse_kb()
        before = kb.get_concept('hammer').epistemic_error
        kb.tick(dt=10.0)
        after = kb.get_concept('hammer').epistemic_error
        assert after > before

    def test_tick_proportional_to_decay_rate(self):
        kb = build_birdhouse_kb()
        kb.tick(dt=10.0)
        # human_hand (0.1/s) decays faster than hammer (0.01/s)
        assert (kb.get_concept('human_hand').epistemic_error >
                kb.get_concept('hammer').epistemic_error)

    def test_task_node_does_not_decay(self):
        kb = build_birdhouse_kb()
        kb.tick(dt=1000.0)
        assert kb.get_concept('build_birdhouse').epistemic_error == pytest.approx(0.0)

    def test_epistemic_error_clamped_at_one(self):
        kb = build_birdhouse_kb()
        kb.tick(dt=100_000.0)
        assert kb.get_concept('human_hand').epistemic_error == pytest.approx(1.0)

    def test_refresh_reduces_error(self):
        kb = build_birdhouse_kb()
        kb.tick(dt=10.0)
        before = kb.get_concept('hammer').epistemic_error
        kb.refresh_concept('hammer', refresh=0.5)
        after = kb.get_concept('hammer').epistemic_error
        assert after < before

    def test_epistemic_error_not_below_zero(self):
        kb = build_birdhouse_kb()
        # Large refresh on a fresh concept should not go below 0
        kb.refresh_concept('hammer', refresh=1.0)
        assert kb.get_concept('hammer').epistemic_error >= 0.0
