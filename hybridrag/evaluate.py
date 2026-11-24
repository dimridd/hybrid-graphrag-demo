"""Golden-set evaluation: a chunk is relevant to an evidence quote if it contains the quote
(case, markdown and whitespace ignored). Metrics are computed per evidence item, so a multi-hop
question only gets full recall when every hop's evidence is retrieved."""

# python imports
from collections import defaultdict
import math
import yaml
import re

# local imports
from .config import GOLDEN


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_#>|]", "", s)).strip().lower()


def load_golden(chunks) -> list[dict]:
    qs = yaml.safe_load(GOLDEN.read_text())["questions"]
    normed = {c.chunk_id: _norm(c.text) for c in chunks}
    for q in qs:
        q["evidence_chunks"] = []
        for ev in q["evidence"]:
            hits = {cid for cid, t in normed.items() if _norm(ev) in t}
            if not hits:
                raise ValueError(f"{q['id']}: evidence not found in corpus: {ev!r}")
            q["evidence_chunks"].append(hits)
    return qs


def score(ranked: list[str], evidence: list[set[str]], k: int) -> dict:
    top = ranked[:k]
    found = [any(c in ev for c in top) for ev in evidence]
    recall = sum(found) / len(evidence)
    rel = set().union(*evidence)
    dcg = sum(1 / math.log2(i + 2) for i, c in enumerate(top) if c in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(evidence))))
    first = next((i for i, c in enumerate(ranked) if c in rel), None)
    return {"recall": recall, "ndcg": min(1.0, dcg / ideal) if ideal else 0.0,
            "mrr": 1 / (first + 1) if first is not None else 0.0,
            "hit1": float(first is not None and first < 1), "hit3": float(first is not None and first < 3),
            "full": float(all(found))}


def evaluate(retriever, questions, mode: str, k: int = 5, **kw) -> dict:
    per_q, by_type = [], defaultdict(list)
    for q in questions:
        r = retriever.retrieve(q["question"], spaces=None, mode=mode, k=max(k, 10), **kw)
        ids = [c for c, _ in r.ranked]
        s = score(ids, q["evidence_chunks"], k)
        per_q.append({"id": q["id"], "qtype": q["qtype"], **s, "top": ids[:k]})
        by_type[q["qtype"]].append(s)
    agg = {m: sum(x[m] for x in per_q) / len(per_q) for m in ("recall", "ndcg", "mrr", "hit1", "hit3", "full")}
    agg["by_type"] = {t: {m: sum(x[m] for x in v) / len(v) for m in ("recall", "ndcg", "mrr")} | {"n": len(v)}
                      for t, v in by_type.items()}
    agg["per_question"] = per_q
    return agg
