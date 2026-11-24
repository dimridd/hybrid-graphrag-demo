"""Fast checks that run without any API key (everything replays from the committed cache)."""
import json

import networkx as nx
import numpy as np

from hybridrag.config import BUILD, ONTOLOGY_DIR
from hybridrag.extract import validate
from hybridrag.graph import KnowledgeGraph
from hybridrag.ingest import load_corpus
from hybridrag.ontology import Ontology


def _kg():
    _, chunks, _ = load_corpus()
    onto = Ontology.load(ONTOLOGY_DIR / f"v{(ONTOLOGY_DIR / 'final.txt').read_text().strip()}.yaml")
    raw = json.loads((BUILD / "raw_final.json").read_text())
    return chunks, onto, raw, KnowledgeGraph(validate(raw, onto), {c.chunk_id: c.space for c in chunks})


def test_ppr_matches_networkx_pagerank():
    """Our power iteration equals networkx.pagerank with a personalization vector."""
    _, _, _, kg = _kg()
    seeds = {n: 1.0 for n in list(kg.entity_nodes)[:3] if kg.g.degree(n) > 0} or {kg.entity_nodes[0]: 1.0}
    r, _ = kg.ppr(seeds, damping=0.5, specificity=False)
    WT, dead, _ = kg._transition(None)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(kg.nodes)))
    W = WT.T.tocoo()
    for i, j, w in zip(W.row, W.col, W.data):
        G.add_edge(int(i), int(j), weight=float(w))
    p = kg.personalization(seeds, specificity=False)
    ref = nx.pagerank(G, alpha=0.5, personalization={i: float(p[i]) for i in range(len(p))},
                      dangling={i: float(p[i]) for i in range(len(p))}, weight="weight", tol=1e-12, max_iter=500)
    assert np.allclose(r, [ref[i] for i in range(len(p))], atol=1e-6)
    assert abs(r.sum() - 1) < 1e-9


def test_validator_is_pure_and_rules_apply():
    _, onto, raw, _ = _kg()
    a, b = validate(raw, onto), validate(raw, onto)
    assert a["stats"] == b["stats"]
    assert any(r["reason"] == "name_rule" for r in a["rejections"])        # learned junk rules fire


def test_permissions_hide_restricted_facts():
    chunks, _, _, kg = _kg()
    no_sec = frozenset({"ORBIT", "EVAL", "TEAM"})
    for u, v, d in kg.g.edges(data=True):
        if all(kg.chunk_space[c] == "SEC" for c in d["chunks"]):
            assert not kg.visible(d["chunks"], no_sec)
