from pathlib import Path
from typing import TYPE_CHECKING

from pyoxigraph import RdfFormat, Store

from awareness_manager.concept import Concept, InstanceConcept, PresenceState
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from awareness_manager.awareness_manager import AwarenessManager

_AM_NS = 'http://coresense.eu/awareness/'

_CONCEPTS_QUERY = """
PREFIX am: <http://coresense.eu/awareness/>
SELECT ?conceptId ?conceptType ?decayRate ?classEMode ?derivedAggregation ?absentFallbackE WHERE {
    ?c a am:Concept ;
       am:conceptId ?conceptId ;
       am:conceptType ?conceptType ;
       am:decayRate ?decayRate .
    OPTIONAL { ?c am:classEMode ?classEMode . }
    OPTIONAL { ?c am:derivedAggregation ?derivedAggregation . }
    OPTIONAL { ?c am:absentFallbackE ?absentFallbackE . }
}
"""

_EDGES_QUERY = """
PREFIX am: <http://coresense.eu/awareness/>
SELECT ?sourceId ?targetId ?weight WHERE {
    ?e a am:SemanticEdge ;
       am:source ?src ;
       am:target ?tgt ;
       am:weight ?weight .
    ?src am:conceptId ?sourceId .
    ?tgt am:conceptId ?targetId .
}
"""

_INSTANCES_QUERY = """
PREFIX am: <http://coresense.eu/awareness/>
SELECT ?conceptId ?conceptType ?decayRate ?classId WHERE {
    ?i a am:InstanceConcept ;
       am:conceptId ?conceptId ;
       am:conceptType ?conceptType ;
       am:decayRate ?decayRate ;
       am:classId ?classId .
}
"""

_INSTANCE_EDGES_QUERY = """
PREFIX am: <http://coresense.eu/awareness/>
SELECT ?sourceId ?targetId ?weight ?relationType WHERE {
    ?e a am:InstanceEdge ;
       am:source ?src ;
       am:target ?tgt ;
       am:weight ?weight .
    OPTIONAL { ?e am:relationType ?relationType . }
    ?src am:conceptId ?sourceId .
    ?tgt am:conceptId ?targetId .
}
"""


def _open_store(ttl_path: Path) -> Store:
    store = Store()
    with open(ttl_path, 'r', encoding='utf-8') as f:
        store.load(f, format=RdfFormat.TURTLE, base_iri=_AM_NS)
    return store


def load_kb_from_ttl(ttl_path: Path) -> KnowledgeBase:
    """Load a KnowledgeBase from a Turtle ontology file via SPARQL queries."""
    store = _open_store(ttl_path)
    kb = KnowledgeBase()

    for sol in store.query(_CONCEPTS_QUERY):
        class_e_mode = sol['classEMode'].value if sol['classEMode'] is not None else 'standalone'
        derived_agg = sol['derivedAggregation'].value if sol['derivedAggregation'] is not None else 'mean'
        fallback_e = float(sol['absentFallbackE'].value) if sol['absentFallbackE'] is not None else 1.0
        kb.add_concept(Concept(
            concept_id=sol['conceptId'].value,
            concept_type=sol['conceptType'].value,
            decay_rate=float(sol['decayRate'].value),
            class_e_mode=class_e_mode,
            derived_aggregation=derived_agg,
            absent_fallback_e=fallback_e,
        ))

    for sol in store.query(_EDGES_QUERY):
        kb.add_relation(
            id_a=sol['sourceId'].value,
            id_b=sol['targetId'].value,
            weight=float(sol['weight'].value),
        )

    return kb


def load_instance_kb_from_ttl(ttl_path: Path) -> InstanceKnowledgeBase:
    """Load an InstanceKnowledgeBase from a Turtle ontology file via SPARQL queries.

    Supports multi-class membership: if an instance has multiple am:classId triples in
    the TTL, SPARQL returns one row per classId. Rows are grouped by conceptId so the
    first classId becomes InstanceConcept.class_id and any extras become extra_class_ids.
    """
    store = _open_store(ttl_path)
    ikb = InstanceKnowledgeBase()

    # Group by conceptId to collect multi-class memberships.
    raw: dict[str, dict] = {}
    for sol in store.query(_INSTANCES_QUERY):
        cid = sol['conceptId'].value
        if cid not in raw:
            raw[cid] = {
                'concept_type': sol['conceptType'].value,
                'decay_rate':   float(sol['decayRate'].value),
                'class_ids':    [],
            }
        raw[cid]['class_ids'].append(sol['classId'].value)

    for cid, d in raw.items():
        class_ids = d['class_ids']
        ikb.add_instance(InstanceConcept(
            concept_id=cid,
            concept_type=d['concept_type'],
            decay_rate=d['decay_rate'],
            class_id=class_ids[0],
            extra_class_ids=class_ids[1:],
        ))

    for sol in store.query(_INSTANCE_EDGES_QUERY):
        rel_type = sol['relationType'].value if sol['relationType'] is not None else None
        ikb.add_instance_relation(
            id_a=sol['sourceId'].value,
            id_b=sol['targetId'].value,
            weight=float(sol['weight'].value),
            relation_type=rel_type,
        )

    return ikb


def sync_instance_kb_from_store(
    am: 'AwarenessManager',
    store: Store,
) -> dict[str, list[str]]:
    """Sync the AM's InstanceKnowledgeBase against a live triplestar Store.

    Queries the provided pyoxigraph Store for current instances and reconciles
    against what the AM currently knows:

      - Instance in store, not in IKB (PRESENT/SUSPECTED)  → add_instance()
      - PRESENT instance in IKB, gone from store            → mark_suspected_absent()
      - SUSPECTED_ABSENT instance back in store             → confirm_present()
      - New instance edges in store, not yet in IKB         → add_instance_relation()

    confirm_absent() is never called automatically — the caller decides when to
    escalate from SUSPECTED_ABSENT to CONFIRMED_ABSENT.

    Args:
        am:    The AwarenessManager whose InstanceKnowledgeBase to sync.
        store: A live pyoxigraph Store (e.g. from TriplestarKnowledgeBase.store).

    Returns:
        {'added': [...], 'suspected_absent': [...], 'restored': [...]}
    """
    ikb = am._require_instance_kb()

    # Query current instances from store, grouping multi-class rows by conceptId.
    store_instances_raw: dict[str, dict] = {}
    for sol in store.query(_INSTANCES_QUERY):
        iid = sol['conceptId'].value
        if iid not in store_instances_raw:
            store_instances_raw[iid] = {
                'concept_type': sol['conceptType'].value,
                'decay_rate':   float(sol['decayRate'].value),
                'class_ids':    [],
            }
        store_instances_raw[iid]['class_ids'].append(sol['classId'].value)

    store_instances: dict[str, InstanceConcept] = {}
    for iid, d in store_instances_raw.items():
        class_ids = d['class_ids']
        store_instances[iid] = InstanceConcept(
            concept_id=iid,
            concept_type=d['concept_type'],
            decay_rate=d['decay_rate'],
            class_id=class_ids[0],
            extra_class_ids=class_ids[1:],
        )

    # Query current instance edges from store
    store_edges: list[tuple[str, str, float, str | None]] = []
    for sol in store.query(_INSTANCE_EDGES_QUERY):
        store_edges.append((
            sol['sourceId'].value,
            sol['targetId'].value,
            float(sol['weight'].value),
            sol['relationType'].value if sol['relationType'] is not None else None,
        ))

    current_ids = set(ikb.instance_ids())
    store_ids = set(store_instances)

    added: list[str] = []
    suspected: list[str] = []
    restored: list[str] = []

    # New instances discovered in store
    for iid in store_ids - current_ids:
        ikb.add_instance(store_instances[iid])
        added.append(iid)

    # Reconcile existing instances
    for iid in store_ids & current_ids:
        state = ikb.get_instance(iid).presence_state
        if state == PresenceState.SUSPECTED_ABSENT:
            ikb.confirm_present(iid)
            restored.append(iid)
        # CONFIRMED_ABSENT coming back: restore and re-add
        elif state == PresenceState.CONFIRMED_ABSENT:
            ikb.confirm_present(iid)
            restored.append(iid)

    # Instances no longer in store → mark suspected absent (if currently PRESENT)
    for iid in current_ids - store_ids:
        state = ikb.get_instance(iid).presence_state
        if state == PresenceState.PRESENT:
            ikb.mark_suspected_absent(iid)
            suspected.append(iid)

    # Add new edges (skip pairs already connected)
    for src, tgt, weight, rel_type in store_edges:
        if src in ikb.instance_ids() and tgt in ikb.instance_ids():
            if not ikb._graph.has_edge(src, tgt):
                ikb.add_instance_relation(src, tgt, weight=weight, relation_type=rel_type)

    return {'added': added, 'suspected_absent': suspected, 'restored': restored}
