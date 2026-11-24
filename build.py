"""Offline build: extraction -> ontology training rounds -> final graph -> evaluation.

    python build.py            # replays from the committed LLM cache (no API key needed)
    OPENAI_API_KEY=... python build.py --fresh   # ignore nothing; new prompts call the API and are cached

Writes data/build/*.json and data/ontologies/v*.yaml, which the Streamlit app reads.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

from hybridrag import embed as E
from hybridrag import learning as L
from hybridrag import llm
from hybridrag.config import BUILD, ONTOLOGY_DIR
from hybridrag.evaluate import evaluate, load_golden
from hybridrag.extract import entity_key, extract_all, validate
from hybridrag.graph import KnowledgeGraph
from hybridrag.ingest import load_corpus
from hybridrag.ontology import Ontology
from hybridrag.retrieve import HybridRetriever


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def strip_per_q(m):
    return {k: v for k, v in m.items() if k != "per_question"}


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    pages, chunks, dropped = load_corpus()
    chunk_space = {c.chunk_id: c.space for c in chunks}
    golden = load_golden(chunks)
    log(f"{len(pages)} pages, {len(chunks)} chunks, {len(dropped)} near-duplicates dropped, {len(golden)} golden questions")

    calib_docs, hold_docs, quota = L.diversity_split(pages)
    calib_chunks = [c for c in chunks if c.doc_id in calib_docs]
    calib_ids = {c.chunk_id for c in calib_chunks}
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}
    q_hold = [q for q in golden if all(chunk_doc[c] in hold_docs for ev in q["evidence_chunks"] for c in ev)]
    q_cal = [q for q in golden if q not in q_hold]
    log(f"split: {len(calib_docs)} calibration pages / {len(hold_docs)} held-out; questions {len(q_cal)} / {len(q_hold)}")

    seed = Ontology.load(ONTOLOGY_DIR / "seed_v1.0.0.yaml")
    seed.save(ONTOLOGY_DIR / f"v{seed.version}.yaml")
    log("extracting with seed ontology ...")
    raw = extract_all(chunks, seed)
    raw_seed = raw
    (BUILD / "raw_seed.json").write_text(json.dumps(raw_seed, ensure_ascii=False))

    accepted = seed
    raw_by_version = {seed.version: raw_seed}
    audited_junk: set[str] = set()
    versions = [{"version": seed.version, "notes": ["seed (hand-written)"], "round": 0}]
    rounds = []
    for rnd in range(1, L.MAX_ROUNDS + 1):
        log(f"--- round {rnd} (ontology v{accepted.version}) ---")
        base_m, v_base, kg_base, ret_base = L.measure(accepted, raw, chunks, chunk_space, q_cal, audited_junk)
        clusters = L.theme_clustering(accepted, calib_chunks)
        rj_types, judge = L.rejection_judge(accepted, v_base, calib_ids)
        junk = L.junk_check(accepted, v_base, calib_ids)
        misses = L.golden_misses(ret_base, kg_base, v_base, q_cal)
        types, links = L.consolidate(accepted, clusters, rj_types, judge, v_base, calib_ids, n_pages=len(calib_docs))
        plan, review = L.make_plan(accepted, types, links, junk, judge, misses)
        audited_junk |= {entity_key(j["name"]) for j in junk["confirmed"]}
        base_m, *_ = L.measure(accepted, raw, chunks, chunk_space, q_cal, audited_junk)
        log(f"proposals: {len(plan['rules'])} rule groups, {len(plan['types'])} types, {len(plan['aliases'])} aliases, "
            f"{len(plan['links'])} links, {len(plan['verb_aliases'])} verb aliases; misses {len(misses)}")

        groups = []
        reextracted = []
        start = accepted
        current_m = base_m
        for g in ("rules", "types", "links"):
            if not L.plan_has(plan, g):
                continue
            cand, notes = L.apply_group(accepted, plan, g)
            raw_c, re_ids = raw, []
            if g == "types":   # new types need new extractions: re-extract only the chunks they target
                raw_c, re_ids = L.targeted_reextract(cand, accepted, raw, chunks, v_base)
                notes = notes + [f"re-extracted {len(re_ids)} targeted chunks"]
                reextracted += re_ids
            m, *_ = L.measure(cand, raw_c, chunks, chunk_space, q_cal, audited_junk)
            ok, checks = L.gate(current_m, m)
            groups.append({"group": g, "version": cand.version, "notes": notes, "passed": ok, "checks": checks,
                           "before": strip_per_q(current_m), "after": strip_per_q(m)})
            log(f"  {g:5s} -> v{cand.version}: {'PASS' if ok else 'ROLL BACK'}  recall {current_m['recall']:.3f}->{m['recall']:.3f} "
                f"ndcg {current_m['ndcg']:.3f}->{m['ndcg']:.3f} gap {current_m['gap_share']:.3f}->{m['gap_share']:.3f} "
                f"code {current_m['code_like']}->{m['code_like']} id {current_m['id_like']}->{m['id_like']}")
            if ok:
                raw = raw_c
                cand.changelog = accepted.changelog + [f"{cand.version} round {rnd} {g}: " + "; ".join(notes)]
                accepted = cand
                current_m = m
                accepted.save(ONTOLOGY_DIR / f"v{accepted.version}.yaml")
                raw_by_version[accepted.version] = raw
                versions.append({"version": accepted.version, "notes": notes, "round": rnd, "group": g})

        after_m, *_ = L.measure(accepted, raw, chunks, chunk_space, q_cal, audited_junk)
        rounds.append({"round": rnd, "base_version": start.version, "end_version": accepted.version,
                       "metrics_start": strip_per_q(base_m), "metrics_end": strip_per_q(after_m),
                       "themes": clusters, "rejection_types": rj_types, "judge": judge, "junk": junk,
                       "golden_misses": misses, "consolidated_types": types, "consolidated_links": links,
                       "plan": plan, "review": review, "groups": groups, "reextracted": reextracted})
        if accepted.version == start.version:
            log("  no group passed -> converged")
            break

    # ---------------- held-out check: freeze the newest version that passes ----------------
    seed_hold, *_ = L.measure(seed, raw_seed, chunks, chunk_space, q_hold, audited_junk)
    holdout_trials = []
    final, final_hold, hold_ok = seed, seed_hold, True
    for ver in reversed([v["version"] for v in versions]):
        o = Ontology.load(ONTOLOGY_DIR / f"v{ver}.yaml")
        m, *_ = L.measure(o, raw_by_version[ver], chunks, chunk_space, q_hold, audited_junk)
        # held-out set is small (one rank swap moves nDCG by ~0.05), so the check is per question:
        # no held-out question may lose evidence it found under the seed ontology
        lost = [q for q, rcl in seed_hold["per_question"].items() if m["per_question"][q] < rcl - 1e-9]
        ok = not lost
        holdout_trials.append({"version": ver, "recall": m["recall"], "ndcg": m["ndcg"], "passed": ok, "lost": lost})
        log(f"held-out ({len(q_hold)} q) v{ver}: recall {seed_hold['recall']:.3f}->{m['recall']:.3f} "
            f"ndcg {seed_hold['ndcg']:.3f}->{m['ndcg']:.3f} lost={lost}  {'FREEZE' if ok else 'fail -> try previous version'}")
        if ok:
            final, final_hold = o, m
            break
    raw = raw_by_version[final.version]
    (ONTOLOGY_DIR / "final.txt").write_text(final.version)

    # ---------------- final evaluation ----------------
    log("evaluating retrievers on all golden questions ...")
    v_final = validate(raw, final)
    kg = KnowledgeGraph(v_final, chunk_space)
    ret = HybridRetriever(chunks, kg)
    v_seed = validate(raw_seed, seed)
    ret_seed = HybridRetriever(chunks, KnowledgeGraph(v_seed, chunk_space))
    modes = {"BM25": "bm25", "Vector (Chroma)": "vector", "Graph 1-hop": "graph_1hop", "Graph PPR": "graph",
             "Hybrid BM25+Vector": "hybrid_text", "Hybrid + Graph PPR": "hybrid"}
    results = {name: evaluate(ret, golden, mode) for name, mode in modes.items()}
    results["Graph PPR (seed ontology)"] = evaluate(ret_seed, golden, "graph")
    results["Hybrid + Graph PPR (seed ontology)"] = evaluate(ret_seed, golden, "hybrid")
    results["Graph PPR, no specificity"] = evaluate(ret, golden, "graph", specificity=False)
    ret_ent = HybridRetriever(chunks, KnowledgeGraph(v_final, chunk_space, passages=False, synonyms=False))
    results["Graph PPR, entity nodes only"] = evaluate(ret_ent, golden, "graph")
    sweep = {d: strip_per_q(evaluate(ret, golden, "graph", damping=d)) for d in (0.3, 0.5, 0.7, 0.85)}
    t0 = time.perf_counter()
    for q in golden:
        ret.retrieve(q["question"], mode="graph")
    graph_ms = (time.perf_counter() - t0) * 1000 / len(golden)
    t0 = time.perf_counter()
    for _ in range(20):
        kg.ppr(ret.link_entities(golden[0]["question"]))
    ppr_ms = (time.perf_counter() - t0) * 1000 / 20
    for name, r in results.items():
        log(f"  {name:36s} R@5 {r['recall']:.3f}  nDCG@5 {r['ndcg']:.3f}  MRR {r['mrr']:.3f}  Hit@1 {r['hit1']:.3f}  Hit@3 {r['hit3']:.3f}")

    BUILD.joinpath("raw_final.json").write_text(json.dumps(raw, ensure_ascii=False))
    BUILD.joinpath("learning_log.json").write_text(json.dumps({
        "split": {"calibration_docs": sorted(calib_docs), "holdout_docs": sorted(hold_docs), "quota": quota,
                  "calibration_questions": [q["id"] for q in q_cal], "holdout_questions": [q["id"] for q in q_hold]},
        "versions": versions, "final_version": final.version, "rounds": rounds,
        "holdout": {"seed": strip_per_q(seed_hold), "final": strip_per_q(final_hold), "passed": hold_ok,
                    "trials": holdout_trials},
        "settings": {"MIN_CLUSTER_SUPPORT": L.MIN_CLUSTER_SUPPORT, "MIN_CHUNK_SUPPORT": L.MIN_CHUNK_SUPPORT,
                     "GAP_TOLERANCE": L.GAP_TOLERANCE, "METRIC_TOLERANCE": L.METRIC_TOLERANCE, "MAX_ROUNDS": L.MAX_ROUNDS,
                     "K": L.K},
    }, ensure_ascii=False, default=list, indent=1))
    BUILD.joinpath("eval.json").write_text(json.dumps({
        "k": 5, "n_questions": len(golden), "results": results, "damping_sweep": sweep,
        "latency_ms": {"graph_retrieval_per_question": round(graph_ms, 2), "ppr_per_question": round(ppr_ms, 3)},
        "graph": {"seed": KnowledgeGraph(v_seed, chunk_space).stats(), "final": kg.stats()},
        "corpus": {"pages": len(pages), "chunks": len(chunks), "near_duplicates": len(dropped)},
    }, ensure_ascii=False, default=list, indent=1))
    # cached answers for the starter questions: ONE batched call, same retrieval as the app's default settings
    from hybridrag.answer import SYSTEM as ANSWER_SYSTEM, build_prompt
    from hybridrag.config import DEFAULT_K, EXAMPLES, EXTRACT_MODEL
    from hybridrag.retrieve import refine_context
    cmap = {c.chunk_id: c for c in chunks}
    blocks = []
    for i, q in enumerate(EXAMPLES):
        r = ret.retrieve(q, spaces=None, mode="hybrid", k=DEFAULT_K)
        blocks.append(f"##### QUESTION {i}\n{build_prompt(q, refine_context(q, [c for c, _ in r.ranked], cmap))}")
    sys_prompt = ANSWER_SYSTEM + '\nYou receive several questions, each with its own context. Return JSON {"answers":[{"id":<n>,"answer":"markdown"}]}.'
    out = llm.chat_json(sys_prompt, "\n\n".join(blocks), EXTRACT_MODEL)
    ans = {int(a["id"]): a["answer"] for a in out.get("answers", []) if str(a.get("id", "")).isdigit()}
    BUILD.joinpath("example_answers.json").write_text(json.dumps(
        {q: {"answer": ans.get(i, ""), "model": "Kimi K2.5"} for i, q in enumerate(EXAMPLES) if ans.get(i)}, ensure_ascii=False, indent=1))
    E.save()
    log(f"LLM calls: {llm.STATS}; pruned {llm.prune_cache()} stale cache entries")


if __name__ == "__main__":
    main()
