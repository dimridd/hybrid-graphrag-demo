"""Hybrid retrieval (the SRAR pipeline in the demo):

  1. Permission scope      JWT groups -> allowed spaces -> allowed chunks (filter BEFORE ranking)
  2. Sparse + dense        BM25 (rank_bm25) and cosine ANN (ChromaDB, metadata filter on space)
  3. Entity linking        question -> seed entities (exact name/alias match + embedding similarity)
  4. Graph search          Personalized PageRank from the seeds over the permission-filtered graph
  5. Fusion                Reciprocal Rank Fusion of the three ranked lists
  6. Refine context        section expansion + token-budget trimmer that never cuts code blocks
"""

# python imports
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
import numpy as np
import chromadb
import time
import re

# local imports
from .graph import KnowledgeGraph
from .ontology import STRUCTURAL
from .extract import entity_key
from .ingest import Chunk
from .embed import embed

TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOP = set("a an the of to in on for and or is are was were be by with what which who how when why does do did from at as it this that these those i we you me my our".split())


def tokenize(text: str) -> list[str]:
    toks = TOKEN.findall(text.lower())
    out = []
    for t in toks:
        if t in STOP:
            continue
        out.append(t)
        if re.search(r"[-_.]", t):
            out += [p for p in re.split(r"[-_.]", t) if p and p not in STOP]
    return out


_COLLECTIONS: dict = {}


def chunk_collection(chunks: list[Chunk]):
    """One ChromaDB collection per corpus, shared by every retriever built on it (graph versions differ,
    the vector index does not). Space is stored as metadata so permission filtering happens inside the ANN query."""
    key = hash(tuple(c.chunk_id for c in chunks))
    if key not in _COLLECTIONS:
        client = chromadb.EphemeralClient()
        col = client.get_or_create_collection(f"chunks-{abs(key) % 10**9}", metadata={"hnsw:space": "cosine"})
        if col.count() == 0:
            vecs = embed([f"{c.title} — {c.breadcrumb}\n{c.text}" for c in chunks])
            col.add(ids=[c.chunk_id for c in chunks], embeddings=vecs.tolist(), documents=[c.text for c in chunks],
                    metadatas=[{"space": c.space, "doc_id": c.doc_id, "title": c.title} for c in chunks])
        _COLLECTIONS[key] = col
    return _COLLECTIONS[key]


@dataclass
class Result:
    question: str
    ranked: list[tuple[str, float]]                    # final fused (chunk_id, score)
    lists: dict[str, list[str]] = field(default_factory=dict)   # per-retriever ranked chunk ids
    seeds: dict[str, float] = field(default_factory=dict)
    top_nodes: list[tuple[str, float]] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    allowed: int = 0
    ppr_iters: int = 0


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], graph: KnowledgeGraph, aliases: dict[str, str] | None = None):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.order = [c.chunk_id for c in chunks]
        self.graph = graph
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.breadcrumb + " " + c.text) for c in chunks])
        self.col = chunk_collection(chunks)
        # entity index for linking (names of non-structural nodes)
        self.ent_keys = [n for n, d in graph.g.nodes(data=True) if d["type"] not in (STRUCTURAL - {"JiraIssue"}) | {"Passage"}]
        # ID-named entities (Jira keys) link by exact match only: IDs embed close to each other
        self.emb_keys = [n for n in self.ent_keys if graph.g.nodes[n]["type"] != "JiraIssue"]
        self.ent_vecs = embed([graph.g.nodes[n]["name"] for n in self.emb_keys]) \
            if self.ent_keys else np.zeros((0, 384))
        self.alias_to_key: dict[str, str] = {}
        for n in self.ent_keys:
            nm = graph.g.nodes[n]["name"].lower()
            if len(nm) >= 3:
                self.alias_to_key[nm] = n

    # ----- individual retrievers -----
    def allowed_chunks(self, spaces: frozenset | None) -> set[str]:
        return {c for c in self.order if spaces is None or self.chunks[c].space in spaces}

    def search_bm25(self, q: str, allowed: set[str], k: int = 30) -> list[str]:
        sc = self.bm25.get_scores(tokenize(q))
        ids = [self.order[i] for i in np.argsort(-sc) if sc[i] > 0 and self.order[i] in allowed]
        return ids[:k]

    def search_vector(self, q: str, spaces: frozenset | None, k: int = 30) -> list[str]:
        where = {"space": {"$in": sorted(spaces)}} if spaces is not None else None
        n = min(k, self.col.count())
        res = self.col.query(query_embeddings=embed([q]).tolist(), n_results=n, where=where)
        return res["ids"][0]

    def link_entities(self, q: str, min_sim: float = 0.72, top: int = 8) -> dict[str, float]:
        """Query-side entity linking: exact name matches (score 1.0) plus embedding matches between
        question n-grams (1-4 words) and entity names (score = cosine), like HippoRAG's query NER step."""
        ql = " " + re.sub(r"[^a-z0-9\-\.\s]", " ", q.lower()) + " "
        seeds: dict[str, float] = {}
        for nm, key in self.alias_to_key.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(nm) + r"(?![a-z0-9])", ql):
                seeds[key] = 1.0
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", q)
        grams = {" ".join(words[i:i + n]) for n in (1, 2, 3, 4) for i in range(len(words) - n + 1)}
        grams = [g for g in grams if not all(w.lower() in STOP for w in g.split()) and len(g) > 2]
        if len(self.emb_keys) and grams:
            sims = (embed(grams) @ self.ent_vecs.T).max(axis=0)
            for i in np.argsort(-sims)[:top * 2]:
                if sims[i] >= min_sim:
                    seeds[self.emb_keys[i]] = max(seeds.get(self.emb_keys[i], 0), float(sims[i]))
        # drop seeds that are substrings of longer matched seeds ("orbit" inside "orbit core team")
        names = {k: self.graph.g.nodes[k]["name"].lower() for k in seeds}
        for k in list(seeds):
            if seeds[k] == 1.0 and any(k != o and seeds[o] == 1.0 and names[k] in names[o] for o in seeds):
                del seeds[k]
        return dict(sorted(seeds.items(), key=lambda x: -x[1])[:top])

    def search_graph(self, q: str, spaces, allowed, method="ppr", damping=0.5, specificity=True, k=30):
        seeds = self.link_entities(q)
        if method == "ppr":
            node_scores, iters = self.graph.ppr(seeds, damping=damping, specificity=specificity, spaces=spaces)
        else:
            node_scores, iters = self.graph.one_hop(seeds, spaces=spaces), 0
        cs = self.graph.chunk_scores(node_scores, allowed, use_passages=(method == "ppr"))
        ranked = [c for c, _ in sorted(cs.items(), key=lambda x: -x[1])][:k]
        return ranked, seeds, node_scores, iters

    # ----- full pipeline -----
    def retrieve(self, q: str, spaces: frozenset | None = None, mode: str = "hybrid", k: int = 10,
                 damping: float = 0.5, specificity: bool = True, rrf_k: int = 60) -> Result:
        t = {}
        t0 = time.perf_counter()
        allowed = self.allowed_chunks(spaces)
        t["permission"] = time.perf_counter() - t0
        lists, seeds, node_scores, iters = {}, {}, None, 0
        if mode in ("bm25", "hybrid", "hybrid_text"):
            t0 = time.perf_counter(); lists["bm25"] = self.search_bm25(q, allowed); t["bm25"] = time.perf_counter() - t0
        if mode in ("vector", "hybrid", "hybrid_text"):
            t0 = time.perf_counter(); lists["vector"] = self.search_vector(q, spaces); t["vector"] = time.perf_counter() - t0
        if mode in ("graph", "graph_1hop", "hybrid"):
            t0 = time.perf_counter()
            method = "one_hop" if mode == "graph_1hop" else "ppr"
            lists["graph"], seeds, node_scores, iters = self.search_graph(q, spaces, allowed, method, damping, specificity)
            t["graph"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fused: dict[str, float] = {}
        for name, ids in lists.items():
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        t["fusion"] = time.perf_counter() - t0
        res = Result(question=q, ranked=ranked, lists=lists, seeds=seeds, timings=t, allowed=len(allowed), ppr_iters=iters)
        if node_scores is not None:
            idx = [i for i in np.argsort(-node_scores) if self.graph.is_entity(self.graph.nodes[i])][:12]
            res.top_nodes = [(self.graph.nodes[i], float(node_scores[i])) for i in idx if node_scores[i] > 0]
            res.facts = self.graph.top_facts(node_scores, seeds, spaces)
        return res


# ---------------- context refinement ----------------
def _split_blocks(text: str):
    """Split into prose sentences and atomic code blocks (a code block is kept or dropped whole)."""
    for p in re.split(r"(```.*?```)", text, flags=re.S):
        if not p.strip():
            continue
        if p.startswith("```"):
            yield ("code", p, True)
        else:
            for s in re.split(r"(?<=[.!?])\s+|\n(?=[-*\d#])", p):
                if s.strip():
                    yield ("prose", s.strip(), False)


def refine_context(q: str, ranked_ids: list[str], chunks: dict[str, Chunk], budget_words: int = 900):
    """Section expansion + budget trimmer. Returns [(chunk_id, text)] in rank order."""
    qt = set(tokenize(q))
    out, used = [], 0
    for cid in ranked_ids:
        c = chunks[cid]
        blocks = list(_split_blocks(c.text))
        if used + c.n_words <= budget_words:
            out.append((cid, f"[{c.title} › {c.breadcrumb}]\n{c.text}"))
            used += c.n_words
            continue
        # trim: keep code blocks whole, keep the most query-relevant prose sentences
        scored = sorted(range(len(blocks)), key=lambda i: -len(qt & set(tokenize(blocks[i][1]))))
        keep, w = set(), 0
        for i in scored:
            n = len(blocks[i][1].split())
            if used + w + n <= budget_words:
                keep.add(i); w += n
        if keep:
            out.append((cid, f"[{c.title} › {c.breadcrumb}] (trimmed)\n" + "\n".join(blocks[i][1] for i in sorted(keep))))
            used += w
        if used >= budget_words:
            break
    return out
