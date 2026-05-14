from pathlib import Path

from pyoxigraph import RdfFormat, Store

from awareness_manager.concept import Concept, InstanceConcept
from awareness_manager.instance_knowledge_base import InstanceKnowledgeBase
from awareness_manager.knowledge_base import KnowledgeBase

_AM_NS = 'http://coresense.eu/awareness/'

_CONCEPTS_QUERY = """
PREFIX am: <http://coresense.eu/awareness/>
SELECT ?conceptId ?conceptType ?decayRate WHERE {
    ?c a am:Concept ;
       am:conceptId ?conceptId ;
       am:conceptType ?conceptType ;
       am:decayRate ?decayRate .
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
        kb.add_concept(Concept(
            concept_id=sol['conceptId'].value,
            concept_type=sol['conceptType'].value,
            decay_rate=float(sol['decayRate'].value),
        ))

    for sol in store.query(_EDGES_QUERY):
        kb.add_relation(
            id_a=sol['sourceId'].value,
            id_b=sol['targetId'].value,
            weight=float(sol['weight'].value),
        )

    return kb


def load_instance_kb_from_ttl(ttl_path: Path) -> InstanceKnowledgeBase:
    """Load an InstanceKnowledgeBase from a Turtle ontology file via SPARQL queries."""
    store = _open_store(ttl_path)
    ikb = InstanceKnowledgeBase()

    for sol in store.query(_INSTANCES_QUERY):
        ikb.add_instance(InstanceConcept(
            concept_id=sol['conceptId'].value,
            concept_type=sol['conceptType'].value,
            decay_rate=float(sol['decayRate'].value),
            class_id=sol['classId'].value,
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
