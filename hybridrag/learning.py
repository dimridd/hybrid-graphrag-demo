"""Ontology training loop.

  Step A  Build      extract under ontology vN, validate -> accepted graph + rejection log
  Step B  Propose    four independent evidence streams
            1. theme clustering      reads raw calibration chunks; LLM discovers missing types/links
            2. rejection judge       reads the rejection log; recurring patterns -> extend / wrong verb / junk
            3. junk check            reads accepted names; shape rules + two-model LLM audit
            4. golden misses         diagnoses failed golden questions; TAGS proposals, never votes
          -> consolidation (one LLM call merges synonyms) -> support gate -> plan + review sheet
  Step C  Split & gate  rules (patch) -> types (minor) -> links (minor); each group rebuilt and scored
          on its own (ablation). A group ships only if junk does not rise, schema-gap share rises <= 2 pts
          and graph Recall@5 / nDCG@5 drop <= 0.02 on calibration questions.
  Freeze  targeted re-extraction of chunks that match new types, then a held-out check.

All LLM calls are cached, so the full loop replays offline.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from . import llm
from .config import AUDIT_MODEL, EXTRACT_MODEL
from .embed import embed
from .evaluate import score
from .extract import entity_key, extract_all, validate
from .graph import KnowledgeGraph
from .ontology import STRUCTURAL, Ontology, norm_label, to_pascal, to_upper_snake
from .retrieve import HybridRetriever

MIN_CLUSTER_SUPPORT = 2       # production setting; the gate below scales it to the corpus
MIN_CHUNK_SUPPORT = 3
SMALL_CORPUS_PAGES = 40       # below this, one theme cluster is enough if chunk support holds (demo has 16 calibration pages)
MAX_NEW_TYPES_PER_CLUSTER = 2
MAX_NEW_LINKS_PER_CLUSTER = 3
GAP_TOLERANCE = 0.02          # schema-gap share may rise by at most 2 points
METRIC_TOLERANCE = 0.02       # Recall@5 / nDCG@5 may drop by at most 0.02
MAX_ROUNDS = 3
K = 5

# Fixed measurement catalog (independent of the ontology's own rules) used by the gate.
CODE_CATALOG = {
    "snake_case identifier": r"^_?[a-z][a-z0-9]*(_[a-z0-9]+)+$",
    "function call": r"\w\(\)?$|\w\(.*\)$",
    "camelCase identifier": r"^_?[a-z]+[A-Z][A-Za-z0-9]*$",
    "file name": r"\.(py|js|ts|json|yaml|md)$",
    "CONSTANT_NAME": r"^_?[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$",
    "dotted attribute": r"^[A-Za-z_]+\.[A-Za-z_]+",
}
ID_CATALOG = {
    "hex id": r"^[0-9a-f]{8,}$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-",
    "session/user id": r"^(sess|user|msg|conv)[-_][\w-]+$",
    "bare value": r"^\d[\d,\.]*\s*(s|ms|seconds|minutes|%|chars|characters|tokens|days|requests|calls)?$",
}


# ----------------------------------------------------------------------------------------------
# diversity sampling + splits
# ----------------------------------------------------------------------------------------------
def diversity_split(pages, frac: float = 0.75):
    """sqrt share per space, then farthest-first over page embeddings. Returns (calib_ids, holdout_ids)."""
    vecs = embed([f"{p.title}\n{p.text[:2000]}" for p in pages])
    by_space = defaultdict(list)
    for i, p in enumerate(pages):
        by_space[p.space].append(i)
    target = round(frac * len(pages))
    roots = {s: math.sqrt(len(v)) for s, v in by_space.items()}
    quota = {s: max(1, round(target * r / sum(roots.values()))) for s, r in roots.items()}
    quota = {s: min(q, len(by_space[s])) for s, q in quota.items()}
    centroid = vecs.mean(0)
    chosen = [int(np.argmax(vecs @ centroid))]
    taken = Counter({pages[chosen[0]].space: 1})
    while len(chosen) < min(target, sum(quota.values())):
        sims = vecs @ vecs[chosen].T
        mind = 1 - sims.max(1)
        best = max((i for i in range(len(pages)) if i not in chosen and taken[pages[i].space] < quota[pages[i].space]),
                   key=lambda i: mind[i], default=None)
        if best is None:
            break
        chosen.append(best)
        taken[pages[best].space] += 1
    calib = {pages[i].doc_id for i in chosen}
    return calib, {p.doc_id for p in pages} - calib, quota


# ----------------------------------------------------------------------------------------------
# metrics used by the gate
# ----------------------------------------------------------------------------------------------
def shape_counts(validated, audited_junk: set[str]):
    code = idl = aud = 0
    for e in validated["entities"].values():
        if e["type"] in {"JiraIssue", "APIEndpoint"}:
            continue
        n = e["name"]
        code += any(re.search(rx, n) for rx in CODE_CATALOG.values())
        idl += any(re.search(rx, n) for rx in ID_CATALOG.values())
        aud += entity_key(n) in audited_junk
    return code, idl, aud


def gap_share(validated) -> float:
    st = validated["stats"]
    gaps = sum(1 for r in validated["rejections"]
               if r["reason"] in ("unknown_type", "unknown_endpoint_type", "unknown_relationship", "bad_endpoints"))
    return gaps / max(1, st.get("entities_raw", 0) + st.get("relationships_raw", 0))


def graph_metrics(retriever, questions):
    rec = nd = 0.0
    per = {}
    for q in questions:
        r = retriever.retrieve(q["question"], mode="graph", k=10)
        s = score([c for c, _ in r.ranked], q["evidence_chunks"], K)
        rec += s["recall"]; nd += s["ndcg"]
        per[q["id"]] = s["recall"]
    n = max(1, len(questions))
    return rec / n, nd / n, per


def measure(onto, raw, chunks, chunk_space, questions, audited_junk):
    v = validate(raw, onto)
    kg = KnowledgeGraph(v, chunk_space)
    ret = HybridRetriever(chunks, kg)
    rec, nd, per = graph_metrics(ret, questions)
    code, idl, aud = shape_counts(v, audited_junk)
    st = kg.stats()
    return {"code_like": code, "id_like": idl, "audited_junk": aud, "gap_share": round(gap_share(v), 4),
            "recall": round(rec, 4), "ndcg": round(nd, 4), "nodes": st["nodes"], "edges": st["edges"],
            "isolated": st["isolated"], "rejections": len(v["rejections"]), "per_question": per}, v, kg, ret


def gate(before, after):
    checks = {
        "code-like names do not rise": after["code_like"] <= before["code_like"],
        "ID-like names do not rise": after["id_like"] <= before["id_like"],
        "audited junk does not rise": after["audited_junk"] <= before["audited_junk"],
        "schema-gap share ≤ old + 2 pts": after["gap_share"] <= before["gap_share"] + GAP_TOLERANCE,
        "graph Recall@5 drop ≤ 0.02": after["recall"] >= before["recall"] - METRIC_TOLERANCE,
        "graph nDCG@5 drop ≤ 0.02": after["ndcg"] >= before["ndcg"] - METRIC_TOLERANCE,
    }
    return all(checks.values()), checks


# ----------------------------------------------------------------------------------------------
# Step B.1  theme clustering
# ----------------------------------------------------------------------------------------------
DISCOVER_SYS = """You are an ontology engineer for an enterprise search knowledge graph.
Given the current ontology and a cluster of related document chunks, propose ONLY what is missing:
reusable entity TYPES (categories, never instances) and RELATIONSHIP verbs between types.
At most 2 new types and 3 new relationships. Propose aliases when a word in the text is a synonym of an
existing type. Cite the chunk ids that support each proposal.
You receive several clusters; answer for each one separately.
Return JSON: {"clusters":[{"cluster": <id>, "theme": "...", "new_types":[{"name":"PascalCase","description":"...","evidence_chunks":["id"]}],
"new_relationships":[{"verb":"UPPER_SNAKE","source_types":["T"],"target_types":["T"],"description":"...","evidence_chunks":["id"]}],
"new_aliases":[{"alias":"...","maps_to":"ExistingType"}]}]}"""


def theme_clustering(onto, calib_chunks):
    vecs = embed([f"{c.title} — {c.breadcrumb}\n{c.text}" for c in calib_chunks])
    k = max(3, round(math.sqrt(len(calib_chunks))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs)
    groups = {}
    for ci in range(k):
        members = [c for c, l in zip(calib_chunks, labels) if l == ci]
        if len(members) >= 2:
            groups[ci] = members
    body = "\n\n".join(
        f"##### CLUSTER {ci}\n" + "\n\n".join(f"[chunk {c.chunk_id}] ({c.title} › {c.breadcrumb})\n{c.text[:700]}"
                                               for c in members[:5])
        for ci, members in groups.items())
    resp = llm.chat_json(DISCOVER_SYS, f"CURRENT ONTOLOGY:\n{onto.prompt_block()}\n\n{body}", EXTRACT_MODEL)
    by_id = {int(x.get("cluster", -1)): x for x in resp.get("clusters", []) if str(x.get("cluster", "")).lstrip("-").isdigit()}
    clusters = []
    for ci, members in groups.items():
        out = by_id.get(ci, {})
        ids = {c.chunk_id for c in members}
        types = []
        for t in (out.get("new_types") or [])[:MAX_NEW_TYPES_PER_CLUSTER]:
            name = to_pascal(t.get("name", ""))
            if not name or onto.resolve_type(name):
                continue                                    # already a type or alias -> skip (not an alias candidate)
            types.append({"name": name, "description": t.get("description", ""),
                          "evidence": sorted(set(t.get("evidence_chunks") or []) & ids)})
        links = []
        for r in (out.get("new_relationships") or [])[:MAX_NEW_LINKS_PER_CLUSTER]:
            verb = to_upper_snake(r.get("verb", ""))
            if not verb:
                continue
            links.append({"verb": verb, "source": [to_pascal(x) for x in r.get("source_types") or []],
                          "target": [to_pascal(x) for x in r.get("target_types") or []],
                          "description": r.get("description", ""),
                          "evidence": sorted(set(r.get("evidence_chunks") or []) & ids)})
        aliases = []
        for a in out.get("new_aliases") or []:
            tgt = onto.resolve_type(a.get("maps_to", ""))
            if tgt and a.get("alias") and not onto.resolve_type(a["alias"]):
                aliases.append({"alias": to_pascal(a["alias"]), "maps_to": tgt})
        clusters.append({"cluster": ci, "theme": out.get("theme", f"cluster {ci}"), "size": len(members),
                         "chunks": sorted(ids), "types": types, "links": links, "aliases": aliases})
    return clusters


# ----------------------------------------------------------------------------------------------
# Step B.2  rejection judge
# ----------------------------------------------------------------------------------------------
JUDGE_SYS = """You review relationship patterns that an extractor produced but the ontology rejected.
For each pattern decide:
  "extend"              the relationship is real and useful for search; the ontology should allow it
  "wrong_relationship"  the meaning is already covered by an existing verb (give it in "use_verb")
  "junk"                not a meaningful fact
Return JSON: {"verdicts":[{"pattern_id":"...","verdict":"extend|wrong_relationship|junk","use_verb":null,"reason":"..."}]}"""


def rejection_judge(onto, validated, calib_ids: set[str]):
    rej = [r for r in validated["rejections"] if r["chunk"] in calib_ids]
    # unknown entity types (entities and relationship endpoints) -> candidates for consolidation
    type_chunks = defaultdict(set)
    type_examples = defaultdict(list)
    for r in rej:
        if r["reason"] == "unknown_type":
            type_chunks[r["type"]].add(r["chunk"])
            type_examples[r["type"]].append(r["name"])
        elif r["reason"] == "unknown_endpoint_type":
            for side in ("source", "target"):
                t = r[f"{side}_type"]
                if t and not onto.resolve_type(t):
                    type_chunks[t].add(r["chunk"])
                    type_examples[t].append(r[side])
    rj_types = [{"name": t, "chunks": sorted(c), "examples": sorted(set(type_examples[t]))[:6]}
                for t, c in type_chunks.items() if len(c) >= MIN_CHUNK_SUPPORT and t != "?"]

    # relationship patterns grouped by verb
    pats = defaultdict(lambda: {"chunks": set(), "pairs": Counter(), "examples": []})
    for r in rej:
        if r["reason"] in ("unknown_relationship", "bad_endpoints"):
            p = pats[(r["verb"], r["reason"])]
            p["chunks"].add(r["chunk"])
            p["pairs"][(r["source_type"], r["target_type"])] += 1
            if len(p["examples"]) < 4:
                p["examples"].append(f'{r["source"]} -{r["verb"]}-> {r["target"]}')
    candidates = []
    for (verb, reason), p in pats.items():
        if len(p["chunks"]) < MIN_CHUNK_SUPPORT:
            continue
        if not re.fullmatch(r"[A-Z][A-Z_]{1,30}", verb):           # rule check: verb shape
            continue
        candidates.append({"pattern_id": f"{verb}:{reason}", "verb": verb, "reason": reason,
                           "chunks": sorted(p["chunks"]), "pairs": [f"{a}->{b} ({n})" for (a, b), n in p["pairs"].most_common(4)],
                           "examples": p["examples"]})
    if candidates:
        body = "\n".join(f'- {c["pattern_id"]}: verb={c["verb"]} rejected_as={c["reason"]} endpoints={c["pairs"]} '
                         f'examples={c["examples"]} support={len(c["chunks"])} chunks' for c in candidates)
        out = llm.chat_json(JUDGE_SYS, f"ONTOLOGY:\n{onto.prompt_block()}\n\nPATTERNS:\n{body}", EXTRACT_MODEL)
        verdicts = {v["pattern_id"]: v for v in out.get("verdicts", [])}
        for c in candidates:
            v = verdicts.get(c["pattern_id"], {})
            c["verdict"] = v.get("verdict", "junk")
            c["use_verb"] = v.get("use_verb")
            c["judge_reason"] = v.get("reason", "")
    return rj_types, candidates


# ----------------------------------------------------------------------------------------------
# Step B.3  junk check
# ----------------------------------------------------------------------------------------------
AUDIT_SYS = """You audit entity names in an enterprise search knowledge graph. Flag names that are JUNK:
code identifiers, variable/function/parameter names, literal values, numbers, generic words, or sentence
fragments - things a user would never search for as a real-world entity. Also flag WRONG_TYPE when the
name is a real entity but its type is clearly wrong. Return JSON:
{"junk":[{"name":"...","reason":"..."}],"wrong_type":[{"name":"...","current":"...","suggested":"..."}]}"""
CONFIRM_SYS = """A first reviewer flagged these knowledge-graph entity names as junk. Confirm only the ones
that are truly junk for enterprise search (code identifiers, values, generic words). Dispute any that a user
could reasonably search for. Return JSON: {"confirm":["name",...],"dispute":["name",...]}"""


def junk_check(onto, validated, calib_ids: set[str]):
    ents = [e for e in validated["entities"].values()
            if e["chunks"] & calib_ids and e["type"] not in STRUCTURAL]
    exempt = set(onto.rules.get("shape_exempt", []))
    # (a) shape rules: nominate a catalog rule when >= 2 accepted names match it
    rules = []
    for kind, catalog, have in (("code_shapes", CODE_CATALOG, onto.rules.get("code_shapes", [])),
                                ("id_shapes", ID_CATALOG, onto.rules.get("id_shapes", []))):
        for label, rx in catalog.items():
            if rx in have:
                continue
            hits = sorted({e["name"] for e in ents if e["type"] not in exempt and re.search(rx, e["name"])})
            if len(hits) >= 2:
                rules.append({"kind": kind, "label": label, "regex": rx, "examples": hits[:8], "support": len(hits)})
    # (b) two-model audit
    listing = "\n".join(f"- {e['name']} ({e['type']})" for e in sorted(ents, key=lambda e: e["name"].lower()))
    a = llm.chat_json(AUDIT_SYS, listing, AUDIT_MODEL)          # model 1 (Qwen) flags
    flagged = [j for j in a.get("junk", []) if j.get("name")]
    confirmed, disputed = [], []
    if flagged:
        b = llm.chat_json(CONFIRM_SYS, "\n".join(f"- {j['name']}: {j.get('reason', '')}" for j in flagged),
                          EXTRACT_MODEL)                         # model 2 (Kimi) confirms
        conf = {norm_label(x) for x in b.get("confirm", [])}
        for j in flagged:
            (confirmed if norm_label(j["name"]) in conf else disputed).append(j)
    by_key = {entity_key(e["name"]): e for e in ents}
    degree = Counter()
    for r in validated["relationships"]:
        degree[r["source"]] += 1
        degree[r["target"]] += 1
    stop, single, protected = [], [], []
    for j in confirmed:
        key = entity_key(j["name"])
        e = by_key.get(key)
        n_chunks = len(e["chunks"]) if e else 0
        if degree[key] >= 2:                      # well-connected names need a human, not an automatic stop list
            protected.append({**j, "chunks": n_chunks, "degree": degree[key]})
        elif e and e["type"] in exempt:            # identifier-named types (DataStore, Metric, ...) are not junk by shape
            protected.append({**j, "chunks": n_chunks, "degree": degree[key]})
        else:
            (stop if n_chunks >= 2 else single).append({**j, "chunks": n_chunks})
    wrong_type = a.get("wrong_type", [])
    return {"rules": rules, "flagged": len(flagged), "confirmed": confirmed, "disputed": disputed,
            "stop_list": stop, "single_chunk_junk": single, "protected": protected, "wrong_type": wrong_type}


# ----------------------------------------------------------------------------------------------
# Step B.4  golden misses (diagnose, tag, never vote)
# ----------------------------------------------------------------------------------------------
def golden_misses(retriever, kg, validated, questions):
    rows = []
    for q in questions:
        r = retriever.retrieve(q["question"], mode="graph", k=10)
        top = [c for c, _ in r.ranked][:K]
        for ev in q["evidence_chunks"]:
            if any(c in ev for c in top):
                continue
            cid = sorted(ev)[0]
            ents = kg.chunk_entities.get(cid, set())
            seeds = set(r.seeds)
            if not ents:
                diag = "not_extracted"
            elif seeds & ents:
                diag = "ranked_low"
            elif seeds and any(nx.has_path(kg.g, s, e) for s in seeds for e in ents if s in kg.g and e in kg.g):
                diag = "ranked_low"
            else:
                diag = "not_linked"
            rejected = [x for x in validated["rejections"] if x["chunk"] == cid]
            rows.append({"question": q["id"], "text": q["question"], "chunk": cid, "diagnosis": diag,
                         "seeds": [kg.g.nodes[s]["name"] for s in r.seeds][:5],
                         "chunk_entities": [kg.g.nodes[e]["name"] for e in ents][:6],
                         "rejected_types": sorted({x.get("type") or "" for x in rejected if x["kind"] == "entity"} - {""}),
                         "rejected_verbs": sorted({x["verb"] for x in rejected if x["kind"] == "relationship"})})
    return rows


# ----------------------------------------------------------------------------------------------
# consolidation + support gate -> plan
# ----------------------------------------------------------------------------------------------
CONSOL_SYS = """You consolidate ontology proposals from several sources. Merge synonyms and near-duplicates
into one canonical proposal (PascalCase types, UPPER_SNAKE verbs). If a proposed type means the same as an
EXISTING type, set mapped_to_existing to that type (it becomes an alias). Keep endpoint types consistent with
the canonical type names. Return JSON:
{"types":[{"name":"...","description":"...","members":["proposal names merged"],"mapped_to_existing":null}],
 "relationships":[{"verb":"...","source":["T"],"target":["T"],"description":"...","members":["verbs merged"],"mapped_to_existing":null}]}"""


def min_clusters(n_pages: int) -> int:
    return 1 if n_pages < SMALL_CORPUS_PAGES else MIN_CLUSTER_SUPPORT


def consolidate(onto, clusters, rj_types, judge, validated, calib_ids, n_pages: int = 100):
    chunk_cluster = {cid: c["cluster"] for c in clusters for cid in c["chunks"]}
    type_props = defaultdict(lambda: {"clusters": set(), "chunks": set(), "sources": set(), "descriptions": []})
    for c in clusters:
        for t in c["types"]:
            p = type_props[t["name"]]
            p["clusters"].add(c["cluster"]); p["chunks"] |= set(t["evidence"]); p["sources"].add("themes")
            p["descriptions"].append(t["description"])
    for t in rj_types:
        p = type_props[t["name"]]
        p["chunks"] |= set(t["chunks"]); p["sources"].add("rejection_judge")
        p["descriptions"].append(f"rejected as unknown type, e.g. {', '.join(t['examples'][:3])}")
    link_props = defaultdict(lambda: {"clusters": set(), "chunks": set(), "sources": set(), "pairs": set(), "descriptions": []})
    for c in clusters:
        for l in c["links"]:
            p = link_props[l["verb"]]
            p["clusters"].add(c["cluster"]); p["chunks"] |= set(l["evidence"]); p["sources"].add("themes")
            p["pairs"] |= {(s, t) for s in l["source"] for t in l["target"]}; p["descriptions"].append(l["description"])
    for j in judge:
        if j.get("verdict") != "extend":
            continue
        p = link_props[j["verb"]]
        p["chunks"] |= set(j["chunks"]); p["sources"].add("rejection_judge")
        p["pairs"] |= {tuple(x.split(" ")[0].split("->")) for x in j["pairs"]}
        p["descriptions"].append(j.get("judge_reason", ""))
    if not type_props and not link_props:
        return [], []
    # raw usage support: chunks where the extractor itself used that type / verb
    for r in validated["rejections"]:
        if r["chunk"] not in calib_ids:
            continue
        if r["kind"] == "entity" and r["reason"] == "unknown_type":
            for name in type_props:
                if norm_label(name) == norm_label(r["type"]):
                    type_props[name]["chunks"].add(r["chunk"])
        if r["kind"] == "relationship":
            for name in link_props:
                if norm_label(name) == norm_label(r["verb"]):
                    link_props[name]["chunks"].add(r["chunk"])
    # cluster support = how many themes the supporting chunks span (proposing cluster or not)
    for props in (type_props, link_props):
        for p in props.values():
            p["clusters"] |= {chunk_cluster[c] for c in p["chunks"] if c in chunk_cluster}
    body = "TYPES:\n" + "\n".join(f"- {k}: {'; '.join(d for d in v['descriptions'] if d)[:300]}" for k, v in type_props.items())
    body += "\n\nRELATIONSHIPS:\n" + "\n".join(
        f"- {k}: {sorted(v['pairs'])[:6]} {'; '.join(d for d in v['descriptions'] if d)[:200]}" for k, v in link_props.items())
    out = llm.chat_json(CONSOL_SYS, f"EXISTING ONTOLOGY:\n{onto.prompt_block()}\n\nPROPOSALS:\n{body}", EXTRACT_MODEL)

    def merged(props, members):
        acc = {"clusters": set(), "chunks": set(), "sources": set()}
        for m in members:
            for key in (to_pascal(m), to_upper_snake(m), m):
                if key in props:
                    for f in acc:
                        acc[f] |= props[key][f]
                    break
        return acc

    types = []
    for t in out.get("types", []):
        name = to_pascal(t.get("name", ""))
        members = set(t.get("members") or []) | {name}
        acc = merged(type_props, members)
        existing = onto.resolve_type(t.get("mapped_to_existing") or "") if t.get("mapped_to_existing") else None
        passed = len(acc["chunks"]) >= MIN_CHUNK_SUPPORT and (len(acc["clusters"]) >= min_clusters(n_pages) or "rejection_judge" in acc["sources"])
        types.append({"name": name, "description": t.get("description", ""), "members": sorted(members - {name}),
                      "mapped_to_existing": existing, "cluster_support": len(acc["clusters"]),
                      "chunk_support": len(acc["chunks"]), "sources": sorted(acc["sources"]),
                      "evidence": sorted(acc["chunks"])[:8], "support_gate": passed})
    links = []
    for r in out.get("relationships", []):
        verb = to_upper_snake(r.get("verb", ""))
        members = set(r.get("members") or []) | {verb}
        acc = merged(link_props, members)
        existing = onto.resolve_verb(r.get("mapped_to_existing") or "") if r.get("mapped_to_existing") else None
        passed = len(acc["chunks"]) >= MIN_CHUNK_SUPPORT and (len(acc["clusters"]) >= min_clusters(n_pages) or "rejection_judge" in acc["sources"])
        links.append({"verb": verb, "source": [to_pascal(x) for x in r.get("source") or []],
                      "target": [to_pascal(x) for x in r.get("target") or []], "description": r.get("description", ""),
                      "members": sorted(members - {verb}), "mapped_to_existing": existing,
                      "cluster_support": len(acc["clusters"]), "chunk_support": len(acc["chunks"]),
                      "sources": sorted(acc["sources"]), "evidence": sorted(acc["chunks"])[:8], "support_gate": passed})
    return types, links


def make_plan(onto, types, links, junk, judge, misses):
    plan = {"rules": [], "types": [], "aliases": [], "links": [], "verb_aliases": []}
    review = []

    def tags_for(name: str):
        return sorted({m["question"] for m in misses if name in m["rejected_types"] or name in m["rejected_verbs"]})

    for r in junk["rules"]:
        plan["rules"].append({"kind": r["kind"], "regex": r["regex"], "label": r["label"]})
        review.append({"stream": "junk check", "proposal": f"name rule: {r['label']}", "group": "rules",
                       "support": f"{r['support']} names", "evidence": ", ".join(r["examples"][:5]), "golden_misses": "",
                       "decision": "planned"})
    if junk["stop_list"]:
        plan["rules"].append({"kind": "stop_list", "names": [j["name"] for j in junk["stop_list"]]})
    for j in junk["stop_list"]:
        review.append({"stream": "junk check", "proposal": f"stop list: {j['name']}", "group": "rules",
                       "support": f"{j['chunks']} chunks, 2 models agree", "evidence": j.get("reason", ""),
                       "golden_misses": "", "decision": "planned"})
    for j in junk["single_chunk_junk"]:
        review.append({"stream": "junk check", "proposal": f"stop list: {j['name']}", "group": "rules",
                       "support": "1 chunk", "evidence": j.get("reason", ""), "golden_misses": "",
                       "decision": "review only (needs ≥2 chunks)"})
    for j in junk.get("protected", []):
        review.append({"stream": "junk check", "proposal": f"stop list: {j['name']}", "group": "rules",
                       "support": f"{j['chunks']} chunks, graph degree {j['degree']}", "evidence": j.get("reason", ""),
                       "golden_misses": "", "decision": "human review (connected or identifier-named type)"})
    for j in junk["disputed"]:
        review.append({"stream": "junk check", "proposal": f"junk? {j['name']}", "group": "rules", "support": "models disagree",
                       "evidence": j.get("reason", ""), "golden_misses": "", "decision": "kept (disputed)"})
    for w in junk["wrong_type"][:12]:
        review.append({"stream": "junk check", "proposal": f"clarify type: {w.get('name')} {w.get('current')}→{w.get('suggested')}",
                       "group": "review", "support": "", "evidence": "", "golden_misses": "", "decision": "review only"})

    for t in types:
        row = {"stream": "+".join(t["sources"]) or "themes", "group": "types",
               "support": f"{t['cluster_support']} clusters, {t['chunk_support']} chunks",
               "evidence": ", ".join(t["evidence"][:5]), "golden_misses": ", ".join(tags_for(t["name"]))}
        if t["mapped_to_existing"]:
            ok = t["chunk_support"] >= 1
            if ok:
                plan["aliases"].append({"alias": t["name"], "maps_to": t["mapped_to_existing"]})
            review.append({**row, "proposal": f"alias {t['name']} → {t['mapped_to_existing']}", "decision": "planned" if ok else "dropped"})
            continue
        if t["support_gate"]:
            plan["types"].append({"name": t["name"], "description": t["description"], "aliases": t["members"][:4]})
        review.append({**row, "proposal": f"new type {t['name']}", "decision": "planned" if t["support_gate"] else "dropped (support gate)"})

    for l in links:
        row = {"stream": "+".join(l["sources"]) or "themes", "group": "links",
               "support": f"{l['cluster_support']} clusters, {l['chunk_support']} chunks",
               "evidence": ", ".join(l["evidence"][:5]), "golden_misses": ", ".join(tags_for(l["verb"]))}
        existing_same = onto.resolve_verb(l["verb"])
        if existing_same and l["source"] and l["target"] and l["support_gate"]:
            plan["links"].append({"verb": existing_same, "source": l["source"], "target": l["target"],
                                  "description": l["description"]})
            review.append({**row, "proposal": f"widen {existing_same}: +{'|'.join(l['source'])} → {'|'.join(l['target'])}",
                           "decision": "planned"})
            continue
        if l["mapped_to_existing"]:
            if existing_same:
                continue                                         # already covered
            plan["verb_aliases"].append({"alias": l["verb"], "maps_to": l["mapped_to_existing"]})
            review.append({**row, "proposal": f"verb alias {l['verb']} → {l['mapped_to_existing']}", "decision": "planned"})
            continue
        if l["support_gate"] and l["source"] and l["target"]:
            plan["links"].append({"verb": l["verb"], "source": l["source"], "target": l["target"], "description": l["description"]})
        review.append({**row, "proposal": f"link {l['verb']}: {'|'.join(l['source'])} → {'|'.join(l['target'])}",
                       "decision": "planned" if l["support_gate"] else "dropped (support gate)"})
    for j in judge:
        if j.get("verdict") == "extend":
            continue
        review.append({"stream": "rejection judge", "proposal": f"{j['verb']} ({j['reason']})", "group": "—",
                       "support": f"{len(j['chunks'])} chunks", "evidence": "; ".join(j["examples"][:2]),
                       "golden_misses": "", "decision": f"{j.get('verdict')}" + (f" → use {j['use_verb']}" if j.get("use_verb") else "")})
    return plan, review


# ----------------------------------------------------------------------------------------------
# Step C  apply groups
# ----------------------------------------------------------------------------------------------
def bump(version: str, part: str) -> str:
    a, b, c = (int(x) for x in version.split("."))
    return f"{a}.{b}.{c + 1}" if part == "patch" else f"{a}.{b + 1}.0"


def apply_group(onto: Ontology, plan: dict, group: str) -> tuple[Ontology, list[str]]:
    o = onto.copy()
    notes = []
    if group == "rules":
        for r in plan["rules"]:
            if r["kind"] == "stop_list":
                o.rules["stop_list"] = sorted(set(o.rules.get("stop_list", [])) | set(r["names"]))
                notes.append(f"stop list +{len(r['names'])}")
            else:
                o.rules.setdefault(r["kind"], [])
                if r["regex"] not in o.rules[r["kind"]]:
                    o.rules[r["kind"]].append(r["regex"])
                    notes.append(f"{r['kind']}: {r['label']}")
        o.version = bump(o.version, "patch")
    elif group == "types":
        for t in plan["types"]:
            if not o.resolve_type(t["name"]):
                o.types[t["name"]] = {"description": t["description"], "strategy": "llm", "aliases": t.get("aliases", [])}
                notes.append(f"type {t['name']}")
        for a in plan["aliases"]:
            o.types[a["maps_to"]].setdefault("aliases", [])
            if a["alias"] not in o.types[a["maps_to"]]["aliases"]:
                o.types[a["maps_to"]]["aliases"].append(a["alias"])
                notes.append(f"alias {a['alias']}→{a['maps_to']}")
        o.version = bump(o.version, "minor")
    elif group == "links":
        for l in plan["links"]:
            src = [o.resolve_type(s) for s in l["source"]]
            tgt = [o.resolve_type(t) for t in l["target"]]
            src, tgt = [s for s in src if s], [t for t in tgt if t]
            if not src or not tgt:                       # subplan: endpoint types missing -> drop link
                notes.append(f"dropped {l['verb']} (endpoint type missing)")
                continue
            existing = o.resolve_verb(l["verb"])
            if existing:
                spec = o.relationships[existing]
                spec["source"] = sorted(set(spec["source"]) | set(src))
                spec["target"] = sorted(set(spec["target"]) | set(tgt))
                notes.append(f"widen {existing}")
            else:
                o.relationships[l["verb"]] = {"source": sorted(set(src)), "target": sorted(set(tgt)), "description": l["description"]}
                notes.append(f"link {l['verb']}")
        for a in plan["verb_aliases"]:
            o.relationships[a["maps_to"]].setdefault("aliases", [])
            if a["alias"] not in o.relationships[a["maps_to"]]["aliases"]:
                o.relationships[a["maps_to"]]["aliases"].append(a["alias"])
                notes.append(f"verb alias {a['alias']}→{a['maps_to']}")
        o.version = bump(o.version, "minor")
    return o, notes


def plan_has(plan, group):
    return {"rules": bool(plan["rules"]), "types": bool(plan["types"] or plan["aliases"]),
            "links": bool(plan["links"] or plan["verb_aliases"])}[group]


# ----------------------------------------------------------------------------------------------
# freeze: targeted re-extraction
# ----------------------------------------------------------------------------------------------
def targeted_reextract(onto_new, onto_old, raw, chunks, validated_old, threshold: float = 0.42):
    new_types = [t for t in onto_new.types if t not in onto_old.types]
    if not new_types:
        return raw, []
    descs = [f"{t}: {onto_new.types[t].get('description', '')}" for t in new_types]
    tv = embed(descs)
    cv = embed([c.text for c in chunks])
    sims = cv @ tv.T
    rejected_chunks = {r["chunk"] for r in validated_old["rejections"]
                       if r["reason"] in ("unknown_type", "unknown_endpoint_type", "unknown_relationship")}
    targets = [c for i, c in enumerate(chunks) if sims[i].max() >= threshold or c.chunk_id in rejected_chunks]
    fresh = extract_all(targets, onto_new)
    out = dict(raw)
    out.update(fresh)
    return out, [c.chunk_id for c in targets]
