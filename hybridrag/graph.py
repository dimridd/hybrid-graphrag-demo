"""Knowledge graph on NetworkX + Personalized PageRank (HippoRAG-style) graph retrieval.

PPR:  r = (1 - d) * p + d * W^T r
  p  personalization (restart) vector: seed entities from the question, weighted by specificity
     (seed score / number of chunks the entity appears in), so hub entities do not dominate.
  W  row-normalised transition matrix; edge weight = link trust = 1 + log(#supporting chunks).
  r  converges in ~20-40 power iterations; dead ends restart at the seeds.
Graph layout follows HippoRAG 2: entity nodes, passage (chunk) nodes linked to the entities extracted from
them, and synonym edges between entities whose names embed close together. A chunk's graph score is the PPR
mass on its passage node, so multi-hop evidence is reached through shared entities.

Permission-aware: a node or edge is visible only if the user can open at least one of its source chunks.
"""

# python imports
from functools import lru_cache
import scipy.sparse as sp
import networkx as nx
import numpy as np
import math


PASSAGE = "Passage"
SYNONYM = "SYNONYM_OF"


class KnowledgeGraph:
    def __init__(self, validated: dict, chunk_space: dict[str, str], chunk_titles: dict[str, str] | None = None,
                 passages: bool = True, synonyms: bool = True, syn_threshold: float = 0.86):
        self.g = nx.Graph()
        self.chunk_space = chunk_space
        self.passages = passages
        for key, e in validated["entities"].items():
            self.g.add_node(key, name=e["name"], type=e["type"], chunks=set(e["chunks"]),
                            description=(e["descriptions"] or [""])[0])
        for r in validated["relationships"]:
            u, v = r["source"], r["target"]
            if self.g.has_edge(u, v):
                d = self.g[u][v]
                d["verbs"].add(r["verb"])
                d["chunks"] |= set(r["chunks"])
                d["facts"].append((u, r["verb"], v))
            else:
                self.g.add_edge(u, v, verbs={r["verb"]}, chunks=set(r["chunks"]), facts=[(u, r["verb"], v)])
        if synonyms:
            self._add_synonyms(syn_threshold)
        self.entity_nodes = [n for n in self.g.nodes]
        if passages:
            for key, e in validated["entities"].items():
                for c in e["chunks"]:
                    pn = "§" + c
                    if pn not in self.g:
                        self.g.add_node(pn, name=(chunk_titles or {}).get(c, c), type=PASSAGE, chunks={c}, description="")
                    self.g.add_edge(key, pn, verbs={"MENTIONED_IN"}, chunks={c}, facts=[])
        self.nodes = list(self.g.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.chunk_entities: dict[str, set[str]] = {}
        for n, d in self.g.nodes(data=True):
            if d["type"] == PASSAGE:
                continue
            for c in d["chunks"]:
                self.chunk_entities.setdefault(c, set()).add(n)
        self._W_cache: dict[frozenset, tuple] = {}

    def _add_synonyms(self, threshold: float):
        from .embed import embed
        ents = [n for n, d in self.g.nodes(data=True) if d["type"] != "JiraIssue"]
        if len(ents) < 2:
            return
        V = embed([self.g.nodes[n]["name"] for n in ents])
        S = V @ V.T
        np.fill_diagonal(S, 0)
        for i in range(len(ents)):
            for j in np.argsort(-S[i])[:3]:
                if S[i, j] >= threshold and not self.g.has_edge(ents[i], ents[j]):
                    ch = self.g.nodes[ents[i]]["chunks"] | self.g.nodes[ents[j]]["chunks"]
                    self.g.add_edge(ents[i], ents[j], verbs={SYNONYM}, chunks=ch, facts=[], sim=float(S[i, j]))

    def is_entity(self, n) -> bool:
        return self.g.nodes[n]["type"] != PASSAGE

    def entity_graph(self) -> nx.Graph:
        return self.g.subgraph([n for n in self.g if self.is_entity(n)])

    # ---------- stats ----------
    def stats(self) -> dict:
        eg = self.entity_graph()
        types = {}
        for _, d in eg.nodes(data=True):
            types[d["type"]] = types.get(d["type"], 0) + 1
        verbs = {}
        for _, _, d in eg.edges(data=True):
            for v in d["verbs"]:
                verbs[v] = verbs.get(v, 0) + 1
        iso = sum(1 for n in eg if eg.degree(n) == 0)
        return {"nodes": eg.number_of_nodes(), "edges": eg.number_of_edges(), "isolated": iso,
                "passages": self.g.number_of_nodes() - eg.number_of_nodes(),
                "types": dict(sorted(types.items(), key=lambda x: -x[1])),
                "verbs": dict(sorted(verbs.items(), key=lambda x: -x[1]))}

    # ---------- permissions ----------
    def visible(self, chunks: set[str], spaces: frozenset | None) -> bool:
        return spaces is None or any(self.chunk_space.get(c) in spaces for c in chunks)

    def _transition(self, spaces: frozenset | None):
        key = spaces or frozenset({"*"})
        if key in self._W_cache:
            return self._W_cache[key]
        n = len(self.nodes)
        node_ok = np.array([self.visible(self.g.nodes[x]["chunks"], spaces) for x in self.nodes], dtype=bool)
        rows, cols, vals = [], [], []
        for u, v, d in self.g.edges(data=True):
            if not self.visible(d["chunks"], spaces):
                continue
            i, j = self.index[u], self.index[v]
            if not (node_ok[i] and node_ok[j]):
                continue
            if SYNONYM in d["verbs"]:
                w = d.get("sim", 0.9)
            elif "MENTIONED_IN" in d["verbs"]:
                w = 1.0
            else:
                w = 1.0 + math.log(len(d["chunks"]))      # link trust: more supporting chunks, stronger link
            rows += [i, j]; cols += [j, i]; vals += [w, w]
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        out = np.asarray(A.sum(axis=1)).ravel()
        dead = out == 0
        inv = np.where(dead, 0.0, 1.0 / np.where(dead, 1, out))
        W = sp.diags(inv) @ A                              # row-normalised
        res = (W.T.tocsr(), dead, node_ok)
        self._W_cache[key] = res
        return res

    # ---------- retrieval ----------
    def personalization(self, seeds: dict[str, float], specificity: bool = True) -> np.ndarray:
        p = np.zeros(len(self.nodes))
        for n, w in seeds.items():
            if n in self.index:
                p[self.index[n]] += w / max(1, len(self.g.nodes[n]["chunks"])) if specificity else w
        return p / p.sum() if p.sum() > 0 else p

    def ppr(self, seeds: dict[str, float], damping: float = 0.5, specificity: bool = True,
            spaces: frozenset | None = None, tol: float = 1e-10, max_iter: int = 100):
        WT, dead, node_ok = self._transition(spaces)
        seeds = {s: w for s, w in seeds.items() if s in self.index and node_ok[self.index[s]]}
        p = self.personalization(seeds, specificity)
        if p.sum() == 0:
            return np.zeros(len(self.nodes)), 0
        r = p.copy()
        for it in range(1, max_iter + 1):
            nxt = (1 - damping) * p + damping * (WT @ r) + damping * r[dead].sum() * p
            if np.abs(nxt - r).sum() < tol:
                r = nxt
                break
            r = nxt
        return r, it

    def one_hop(self, seeds: dict[str, float], spaces: frozenset | None = None) -> np.ndarray:
        """Classic GraphRAG local-search style baseline: seeds + direct neighbours (half weight)."""
        _, _, node_ok = self._transition(spaces)
        s = np.zeros(len(self.nodes))
        for n, w in seeds.items():
            if n not in self.index or not node_ok[self.index[n]]:
                continue
            s[self.index[n]] += w
            for m in self.g.neighbors(n):
                if self.is_entity(m) and node_ok[self.index[m]] and self.visible(self.g[n][m]["chunks"], spaces):
                    s[self.index[m]] += 0.5 * w
        return s

    def chunk_scores(self, node_scores: np.ndarray, allowed_chunks: set[str] | None = None,
                     use_passages: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        if use_passages and self.passages:
            for c in self.chunk_entities:
                pn = "§" + c
                if pn in self.index and (allowed_chunks is None or c in allowed_chunks) and node_scores[self.index[pn]] > 0:
                    out[c] = float(node_scores[self.index[pn]])
            return out
        for c, ents in self.chunk_entities.items():
            if allowed_chunks is not None and c not in allowed_chunks:
                continue
            s = sum(node_scores[self.index[e]] for e in ents)
            if s > 0:
                out[c] = s
        return out

    def top_facts(self, node_scores: np.ndarray, seeds: dict[str, float], spaces=None, k: int = 10):
        ranked = []
        for u, v, d in self.g.edges(data=True):
            if not d["facts"] or not self.visible(d["chunks"], spaces):
                continue
            sc = node_scores[self.index[u]] + node_scores[self.index[v]]
            if sc > 0:
                for f in d["facts"]:
                    ranked.append((sc + (1 if (u in seeds or v in seeds) else 0), f, d["chunks"]))
        ranked.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for sc, f, ch in ranked:
            if f not in seen:
                seen.add(f)
                out.append({"source": self.g.nodes[f[0]]["name"], "verb": f[1], "target": self.g.nodes[f[2]]["name"],
                            "score": round(float(sc), 4), "chunks": sorted(ch)})
            if len(out) >= k:
                break
        return out
