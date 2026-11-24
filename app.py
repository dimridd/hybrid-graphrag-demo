"""Hybrid GraphRAG demo — Streamlit UI.

Run:  streamlit run app.py
Everything is precomputed by build.py; search runs live (BM25 + ChromaDB + Personalized PageRank + RRF).
"""

# python imports
import plotly.graph_objects as go
import streamlit as st
import networkx as nx
import pandas as pd
import json
import time
import re

# local imports
from hybridrag.config import BUILD, DEFAULT_K, EXAMPLES, ONTOLOGY_DIR, ROOT, USERS, allowed_spaces
from hybridrag.answer import MODE_GUIDE, answer_backend, extractive_answer, stream_llm_answer
from hybridrag.retrieve import HybridRetriever, refine_context
from hybridrag.ontology import STRUCTURAL, Ontology
from hybridrag.evaluate import load_golden
from hybridrag.graph import KnowledgeGraph
from hybridrag.ingest import load_corpus
from hybridrag.extract import validate

st.set_page_config(page_title="Hybrid GraphRAG — Enterprise Search Demo", page_icon="🕸️", layout="wide")

TYPE_COLORS = ["#4a7fd4", "#58a347", "#e0892c", "#8a5bd6", "#df5a4c", "#2f9c8f", "#c9a227", "#d65b9a",
               "#6b7a8f", "#3fa7d6", "#8c6d3f", "#7cb342", "#b55d4c", "#5c6bc0"]


# ------------------------------------------------------------------------------------------------
# load (cached)
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building indexes (BM25, ChromaDB, knowledge graph)…")
def load_system():
    pages, chunks, _ = load_corpus()
    chunk_space = {c.chunk_id: c.space for c in chunks}
    final_version = (ONTOLOGY_DIR / "final.txt").read_text().strip()
    onto = Ontology.load(ONTOLOGY_DIR / f"v{final_version}.yaml")
    raw = json.loads((BUILD / "raw_final.json").read_text())
    validated = validate(raw, onto)
    kg = KnowledgeGraph(validated, chunk_space)
    retriever = HybridRetriever(chunks, kg)
    golden = load_golden(chunks)
    evald = json.loads((BUILD / "eval.json").read_text())
    learn = json.loads((BUILD / "learning_log.json").read_text())
    ex_path = BUILD / "example_answers.json"
    examples = json.loads(ex_path.read_text()) if ex_path.exists() else {}
    return {"pages": pages, "chunks": chunks, "chunk_map": {c.chunk_id: c for c in chunks}, "onto": onto,
            "raw": raw, "validated": validated, "kg": kg, "ret": retriever, "golden": golden, "eval": evald,
            "learn": learn, "chunk_space": chunk_space, "examples": examples}


@st.cache_resource(show_spinner="Re-validating cached extractions under that ontology version…")
def version_graph(version: str):
    S = load_system()
    onto = Ontology.load(ONTOLOGY_DIR / f"v{version}.yaml")
    v = validate(S["raw"], onto)
    return onto, v, KnowledgeGraph(v, S["chunk_space"])


S = load_system()
kg: KnowledgeGraph = S["kg"]
ret: HybridRetriever = S["ret"]


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def type_color_map(types):
    return {t: TYPE_COLORS[i % len(TYPE_COLORS)] for i, t in enumerate(sorted(types))}


def graph_figure(g: nx.Graph, highlight: dict[str, float] | None = None, seeds: set[str] | None = None, height=430):
    if g.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(g, seed=7, k=0.9 / max(1, g.number_of_nodes()) ** 0.35)
    cmap = type_color_map({d["type"] for _, d in g.nodes(data=True)})
    ex, ey, mx, my, mt = [], [], [], [], []
    for u, v, d in g.edges(data=True):
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        mx.append((pos[u][0] + pos[v][0]) / 2); my.append((pos[u][1] + pos[v][1]) / 2)
        mt.append(", ".join(sorted(d.get("verbs", []))))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#b8c2d1", width=1.2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=4, color="#b8c2d1"), text=mt,
                             hoverinfo="text"))
    for t, col in cmap.items():
        ns = [n for n, d in g.nodes(data=True) if d["type"] == t]
        size = [14 + 60 * (highlight or {}).get(n, 0) ** 0.5 for n in ns]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=t,
            marker=dict(size=size, color=col, line=dict(width=[3 if seeds and n in seeds else 1 for n in ns],
                                                        color=["#111" if seeds and n in seeds else "#fff" for n in ns])),
            text=[g.nodes[n]["name"][:28] for n in ns], textposition="top center", textfont=dict(size=10),
            hovertext=[f"{g.nodes[n]['name']} ({t})<br>score {(highlight or {}).get(n, 0):.3f}" for n in ns],
            hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
                      legend=dict(orientation="h", y=-0.05), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def metric_table(results: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        r = results[n]
        rows.append({"Retriever": n, "Recall@5": r["recall"], "nDCG@5": r["ndcg"], "MRR": r["mrr"],
                     "Hit@1": r["hit1"], "Hit@3": r["hit3"], "All hops found": r["full"]})
    return pd.DataFrame(rows).set_index("Retriever")


# ------------------------------------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔐 Signed in as")
    user = st.selectbox("User (JWT groups)", list(USERS), index=0, label_visibility="collapsed")
    spaces = frozenset(allowed_spaces(USERS[user]))
    st.caption("Groups: " + ", ".join(sorted(USERS[user])))
    st.caption("Can read spaces: " + " · ".join(f"`{s}`" for s in sorted(spaces)))
    st.markdown("### ⚙️ Retrieval")
    mode_label = st.radio("Mode", ["Hybrid + Graph PPR", "Hybrid BM25+Vector", "Graph PPR", "Graph 1-hop", "BM25", "Vector (Chroma)"])
    mode = {"Hybrid + Graph PPR": "hybrid", "Hybrid BM25+Vector": "hybrid_text", "Graph PPR": "graph",
            "Graph 1-hop": "graph_1hop", "BM25": "bm25", "Vector (Chroma)": "vector"}[mode_label]
    damping = st.slider("PPR damping d", 0.1, 0.9, 0.5, 0.05, help="r = (1−d)·p + d·Wᵀr")
    specificity = st.toggle("Seed specificity (down-weight hub entities)", value=True)
    k = st.slider("Chunks to retrieve", 3, 10, DEFAULT_K)
    st.markdown("### 🤖 Answer model")
    api_key = st.text_input("OpenAI API key (optional)", type="password",
                            help="Used only in your session for gpt-4o-mini answers. Without it you get an extractive, cited answer.")
    backend = answer_backend(api_key)
    st.caption(f"Live answers: **{backend['label']}**" if backend else "Live answers: off (cached / extractive answers)")
    st.divider()
    st.caption("Demo build of the ChunkMind architecture on one connector (Confluence-style pages). "
               "NetworkX stands in for FalkorDB, ChromaDB for AstraDB, a batch build for Kafka.")

st.title("🕸️ Hybrid GraphRAG — Enterprise Search Demo")
st.caption("Ontology-governed knowledge graph + BM25 + vector search, fused with Reciprocal Rank Fusion. "
           "Graph retrieval uses Personalized PageRank (HippoRAG). Corpus: 21 fictional Confluence pages in 6 spaces.")

with st.expander("ℹ️ How this demo runs — with or without an API key", expanded=False):
    st.markdown(MODE_GUIDE)
    st.markdown(f"**Current answer mode:** {backend['label'] if backend else 'no LLM — cached starter answers + extractive answers'}")

tab_ask, tab_graph, tab_onto, tab_eval, tab_arch = st.tabs(
    ["🔎 Ask", "🕸 Knowledge graph", "🧬 Ontology training", "📊 Evaluation", "🏗 Architecture"])

# ------------------------------------------------------------------------------------------------
# ASK
# ------------------------------------------------------------------------------------------------
with tab_ask:
    if "q" not in st.session_state:
        st.session_state.q = EXAMPLES[0]
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex[:48] + "…", key=f"ex{i}", width="stretch", help=ex):
            st.session_state.q = ex
    q = st.text_input("Ask a question about the engineering docs", key="q")
    if q:
        t0 = time.perf_counter()
        res = ret.retrieve(q, spaces=spaces, mode=mode, k=k, damping=damping, specificity=specificity)
        contexts = refine_context(q, [c for c, _ in res.ranked], S["chunk_map"])
        t_total = (time.perf_counter() - t0) * 1000
        hidden = len(S["chunks"]) - res.allowed

        m = st.columns(5)
        m[0].metric("Retrieval latency", f"{t_total:.0f} ms")
        m[1].metric("Chunks you can read", f"{res.allowed} / {len(S['chunks'])}", delta=f"-{hidden} filtered" if hidden else None,
                    delta_color="off")
        m[2].metric("Seed entities", len(res.seeds))
        m[3].metric("PPR iterations", res.ppr_iters or "—")
        m[4].metric("Context words", sum(len(t.split()) for _, t in contexts))

        st.markdown("#### Answer")
        box = st.container(border=True)
        if not contexts:
            box.warning("NOT FOUND in the documents you can access.")
        elif backend:
            try:
                box.write_stream(stream_llm_answer(q, contexts, backend))
                box.caption(f"Generated live by {backend['label']} from the context below.")
            except Exception as e:  # noqa: BLE001
                cached = S["examples"].get(q)
                box.markdown(cached["answer"] if cached else extractive_answer(q, contexts, res.facts))
                box.caption(f"Live LLM unavailable ({type(e).__name__}: check the key in .env or the sidebar) — "
                            + ("showing the answer cached at build time." if cached else "showing the extractive answer."))
        elif (q in S["examples"] and mode == "hybrid" and len(spaces) == 6 and k == DEFAULT_K
              and damping == 0.5 and specificity):
            box.markdown(S["examples"][q]["answer"])
            box.caption(f"Answer generated at build time by {S['examples'][q]['model']} from exactly this context "
                        "(cached, so the public demo needs no key). Change any setting or ask your own question to go live.")
        else:
            box.markdown(extractive_answer(q, contexts, res.facts))
            box.caption("Extractive answer (no API key). Add an OpenAI key in the sidebar for a generated answer.")

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### Evidence (fused ranking)")
            rank_of = {name: {c: i + 1 for i, c in enumerate(ids)} for name, ids in res.lists.items()}
            for i, (cid, sc) in enumerate(res.ranked):
                c = S["chunk_map"][cid]
                badges = " · ".join(f"{n} #{rank_of[n][cid]}" if cid in rank_of[n] else f"{n} –" for n in res.lists)
                with st.expander(f"[C{i + 1}] {c.title} › {c.breadcrumb}", expanded=i < 2):
                    st.caption(f"space `{c.space}` · RRF {sc:.4f} · {badges}")
                    st.markdown(re.sub(r"^#{1,6}\s*(.+)$", r"**\1**", c.text, flags=re.M))
        with right:
            st.markdown("#### Graph reasoning")
            if res.seeds:
                seeds_df = pd.DataFrame([{"Seed entity": kg.g.nodes[n]["name"], "Type": kg.g.nodes[n]["type"],
                                          "Link score": round(w, 3), "Chunks": len(kg.g.nodes[n]["chunks"])}
                                         for n, w in res.seeds.items()])
                st.dataframe(seeds_df, hide_index=True, width="stretch")
                top = dict(res.top_nodes)
                sub_nodes = set(top) | set(res.seeds)
                for n in list(sub_nodes):
                    if n in kg.g:
                        sub_nodes |= {m for m in kg.g.neighbors(n) if m in top}
                sub = kg.g.subgraph([n for n in sub_nodes if n in kg.g and
                                     kg.visible(kg.g.nodes[n]["chunks"], spaces)]).copy()
                fig = graph_figure(sub, highlight=top, seeds=set(res.seeds))
                if fig:
                    st.plotly_chart(fig, width="stretch")
                st.caption("Node size = PPR score r; black ring = seed (restart) entity.")
                if res.facts:
                    st.markdown("**Top graph facts** (permission-filtered)")
                    for f in res.facts[:8]:
                        st.markdown(f"- {f['source']} **{f['verb']}** {f['target']}")
            else:
                st.info("No entities linked from this question in the current mode.")
            with st.expander("Stage timings"):
                st.json({k_: f"{v * 1000:.1f} ms" for k_, v in res.timings.items()})

# ------------------------------------------------------------------------------------------------
# KNOWLEDGE GRAPH
# ------------------------------------------------------------------------------------------------
with tab_graph:
    ev = S["eval"]
    seed_s, fin_s = ev["graph"]["seed"], ev["graph"]["final"]
    c = st.columns(4)
    c[0].metric("Entities", fin_s["nodes"], delta=f"{fin_s['nodes'] - seed_s['nodes']} vs seed", delta_color="off")
    c[1].metric("Relationships", fin_s["edges"], delta=f"{fin_s['edges'] - seed_s['edges']} vs seed", delta_color="off")
    c[2].metric("Isolated entities", fin_s["isolated"], delta=fin_s["isolated"] - seed_s["isolated"], delta_color="inverse")
    c[3].metric("Ontology", f"v{S['onto'].version}", delta=f"from v1.0.0", delta_color="off")
    st.caption("Deltas compare the final graph with the graph built under the hand-written seed ontology: the learned "
               "name rules removed code identifiers (e.g. `_shell`, `build_contexts`) that were polluting the graph, and "
               f"{fin_s['passages']} passage nodes link entities to the chunks they came from (HippoRAG 2 layout).")
    a, b = st.columns(2)
    with a:
        st.markdown("**Entities by type**")
        st.bar_chart(pd.Series(fin_s["types"], name="entities"), horizontal=True)
    with b:
        st.markdown("**Relationships by verb**")
        st.bar_chart(pd.Series(fin_s["verbs"], name="edges"), horizontal=True)
    st.markdown("#### Entity explorer")
    EG = kg.entity_graph()
    names = sorted(((d["name"], n) for n, d in EG.nodes(data=True) if EG.degree(n) > 0), key=lambda x: x[0].lower())
    default = next((i for i, (nm, _) in enumerate(names) if nm.lower() == "orbit core team"), 0)
    pick = st.selectbox("Entity", names, index=default, format_func=lambda x: f"{x[0]} ({kg.g.nodes[x[1]]['type']})")
    if pick:
        node = pick[1]
        d = kg.g.nodes[node]
        e1, e2 = st.columns([1, 1.3])
        with e1:
            st.markdown(f"**{d['name']}** · `{d['type']}`")
            st.write(d.get("description") or "")
            docs = sorted({S["chunk_map"][c].title for c in d["chunks"] if c in S["chunk_map"]})
            st.markdown("**Mentioned in:** " + "; ".join(docs))
            rows = []
            for m in EG.neighbors(node):
                for (u, v_, w) in kg.g[node][m]["facts"]:
                    rows.append({"Fact": f"{kg.g.nodes[u]['name']} —{v_}→ {kg.g.nodes[w]['name']}",
                                 "Sources": len(kg.g[node][m]["chunks"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with e2:
            ego = nx.ego_graph(EG, node, radius=2)
            if ego.number_of_nodes() > 40:
                ego = nx.ego_graph(EG, node, radius=1)
            st.plotly_chart(graph_figure(ego, seeds={node}, height=460), width="stretch")

# ------------------------------------------------------------------------------------------------
# ONTOLOGY TRAINING
# ------------------------------------------------------------------------------------------------
with tab_onto:
    L = S["learn"]
    st.markdown("""
The ontology is treated as a **trainable artefact**. Each round: **A. Build** (extract + validate → graph + rejection log)
→ **B. Propose** from four independent evidence streams → **C. Split & gate** (rules → types → links, each rebuilt and
scored alone) → **Freeze** (targeted re-extraction + held-out check). Golden misses only *tag* proposals — they never vote,
so the test set does not leak into training.""")
    sp = L["split"]
    c = st.columns(4)
    c[0].metric("Calibration pages", len(sp["calibration_docs"]))
    c[1].metric("Held-out pages", len(sp["holdout_docs"]))
    c[2].metric("Calibration questions", len(sp["calibration_questions"]))
    c[3].metric("Held-out questions", len(sp["holdout_questions"]))
    with st.expander("Diversity sample: √-share per space, then farthest-first over page embeddings"):
        st.write({"quota per space": sp["quota"]})
        st.write({"held-out pages": sp["holdout_docs"]})

    st.markdown("#### Version history")
    st.dataframe(pd.DataFrame([{"Version": "v" + v["version"], "Round": v["round"], "Group": v.get("group", "seed"),
                                "Changes": "; ".join(v["notes"])} for v in L["versions"]]),
                 hide_index=True, width="stretch")

    for rd in L["rounds"]:
        st.markdown(f"### Round {rd['round']}: v{rd['base_version']} → v{rd['end_version']}")
        s1, s2, s3, s4, s5, s6 = st.tabs(["1 · Theme clustering", "2 · Rejection judge", "3 · Junk check",
                                          "4 · Golden misses", "Plan & review sheet", "Split & gate"])
        with s1:
            for cl in rd["themes"]:
                props = [f"type **{t['name']}**" for t in cl["types"]] + \
                        [f"link **{l['verb']}** {'|'.join(l['source'])}→{'|'.join(l['target'])}" for l in cl["links"]] + \
                        [f"alias {a['alias']}→{a['maps_to']}" for a in cl["aliases"]]
                st.markdown(f"- **Cluster {cl['cluster']}** — _{cl['theme']}_ ({cl['size']} chunks): " + (", ".join(props) or "nothing missing"))
            st.caption("Discovery is capped at 2 new types and 3 new links per cluster; a type must then be supported by "
                       "≥2 clusters (or the rejection log) and ≥3 chunks.")
        with s2:
            if rd["rejection_types"]:
                st.markdown("**Unknown entity types in ≥3 chunks** → sent to consolidation")
                st.dataframe(pd.DataFrame([{"Type": t["name"], "Chunks": len(t["chunks"]), "Examples": ", ".join(t["examples"])}
                                           for t in rd["rejection_types"]]), hide_index=True, width="stretch")
            if rd["judge"]:
                st.markdown("**Rejected relationship patterns in ≥3 chunks** → LLM verdict (only *extend* becomes a proposal)")
                st.dataframe(pd.DataFrame([{"Verb": j["verb"], "Rejected as": j["reason"], "Chunks": len(j["chunks"]),
                                            "Endpoints": ", ".join(j["pairs"]), "Verdict": j.get("verdict"),
                                            "Use verb": j.get("use_verb") or "", "Why": j.get("judge_reason", "")}
                                           for j in rd["judge"]]), hide_index=True, width="stretch")
            if not rd["rejection_types"] and not rd["judge"]:
                st.info("No rejection pattern reached the 3-chunk support threshold.")
        with s3:
            jk = rd["junk"]
            if jk["rules"]:
                st.markdown("**Shape rules nominated** (need ≥2 accepted names)")
                st.dataframe(pd.DataFrame([{"Rule": r["label"], "Kind": r["kind"], "Regex": r["regex"], "Names": r["support"],
                                            "Examples": ", ".join(r["examples"][:6])} for r in jk["rules"]]),
                             hide_index=True, width="stretch")
            st.markdown(f"**Two-model audit**: Qwen flagged {jk['flagged']} names, Kimi confirmed {len(jk['confirmed'])}, "
                        f"disputed {len(jk['disputed'])} (disputed names are kept). Auto stop list: {len(jk['stop_list'])} "
                        f"(needs ≥2 chunks); sent to human review: {len(jk.get('protected', []))} "
                        "(well-connected entities like a real service, or identifier-named types).")
            if jk["confirmed"]:
                st.write(", ".join(f"`{j['name']}`" for j in jk["confirmed"][:40]))
        with s4:
            if rd["golden_misses"]:
                st.dataframe(pd.DataFrame([{"Question": m["question"], "Diagnosis": m["diagnosis"],
                                            "Seeds": ", ".join(m["seeds"]), "Entities in evidence chunk": ", ".join(m["chunk_entities"]),
                                            "Rejected there": ", ".join(m["rejected_types"] + m["rejected_verbs"])}
                                           for m in rd["golden_misses"]]), hide_index=True, width="stretch")
                st.caption("not_extracted: evidence chunk has no accepted entities · not_linked: no path from the question's "
                           "seeds · ranked_low: reachable but outside top 5. These only tag proposals for reviewers.")
            else:
                st.success("Graph retrieval found every calibration evidence chunk in the top 5.")
        with s5:
            st.dataframe(pd.DataFrame(rd["review"]), hide_index=True, width="stretch")
            st.caption(f"{len(rd['reextracted'])} chunks re-extracted after the round (targeted: chunks that match a new type "
                       "or had a rejection the new version can now accept).")
        with s6:
            for g in rd["groups"]:
                icon = "✅" if g["passed"] else "↩️ rolled back"
                st.markdown(f"**{g['group']} → v{g['version']}** {icon} — {'; '.join(g['notes'])}")
                b_, a_ = g["before"], g["after"]
                st.dataframe(pd.DataFrame([
                    {"Check": name, "Before": "", "After": "", "Pass": "✅" if ok else "❌"} for name, ok in g["checks"].items()
                ]).assign(Before=[b_["code_like"], b_["id_like"], b_["audited_junk"], b_["gap_share"], b_["recall"], b_["ndcg"]],
                          After=[a_["code_like"], a_["id_like"], a_["audited_junk"], a_["gap_share"], a_["recall"], a_["ndcg"]]),
                    hide_index=True, width="stretch")
    h = L["holdout"]
    st.markdown("### Held-out check before freezing")
    st.markdown(f"Graph Recall@5 **{h['seed']['recall']:.3f} → {h['final']['recall']:.3f}**, "
                f"nDCG@5 **{h['seed']['ndcg']:.3f} → {h['final']['ndcg']:.3f}** on held-out questions — "
                + ("✅ frozen as v" + L["final_version"] if h["passed"] else "❌ not frozen"))

    st.markdown("#### Try it: rebuild the graph under any version (no LLM calls)")
    vers = [v["version"] for v in L["versions"]]
    vsel = st.select_slider("Ontology version", options=vers, value=vers[-1])
    o_v, val_v, kg_v = version_graph(vsel)
    st_v = kg_v.stats()
    x = st.columns(4)
    x[0].metric("Entities", st_v["nodes"]); x[1].metric("Relationships", st_v["edges"])
    x[2].metric("Rejected extractions", len(val_v["rejections"])); x[3].metric("Entity types", len(o_v.types))
    st.caption("Raw LLM output is cached per chunk, so a new ontology version is applied by re-validation alone.")
    with st.expander(f"Ontology v{vsel} (YAML)"):
        st.code((ONTOLOGY_DIR / f"v{vsel}.yaml").read_text(), language="yaml")

# ------------------------------------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------------------------------------
with tab_eval:
    ev = S["eval"]
    R = ev["results"]
    st.markdown(f"Golden set: **{ev['n_questions']} questions** (fact, process, multi-hop, aggregation) with evidence quotes. "
                "A chunk is relevant if it contains a quote; recall is per evidence item, so multi-hop questions need every hop.")
    main = ["BM25", "Vector (Chroma)", "Graph 1-hop", "Graph PPR", "Hybrid BM25+Vector", "Hybrid + Graph PPR"]
    df = metric_table(R, main)
    st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0, color="#d5efcb"), width="stretch")
    st.bar_chart(df[["Recall@5", "nDCG@5", "Hit@1"]], stack=False, height=320)
    a, b = st.columns(2)
    with a:
        st.markdown("**By question type (Recall@5)**")
        types = ["fact", "process", "multi-hop", "aggregation"]
        bt = pd.DataFrame({n: {t: R[n]["by_type"].get(t, {}).get("recall", float("nan")) for t in types} for n in main}).T
        bt.columns = [f"{t} (n={R['BM25']['by_type'].get(t, {}).get('n', 0)})" for t in types]
        st.dataframe(bt.style.format("{:.3f}"), width="stretch")
    with b:
        st.markdown("**Ontology training effect**")
        st.dataframe(metric_table(R, ["Graph PPR (seed ontology)", "Graph PPR", "Hybrid + Graph PPR (seed ontology)",
                                      "Hybrid + Graph PPR"]).style.format("{:.3f}"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PPR damping sweep (graph only)**")
        st.dataframe(pd.DataFrame({f"d={d}": {"Recall@5": v["recall"], "nDCG@5": v["ndcg"], "MRR": v["mrr"]}
                                   for d, v in ev["damping_sweep"].items()}).T.style.format("{:.3f}"), width="stretch")
    with c2:
        st.markdown("**Ablation: seed specificity**")
        st.dataframe(metric_table(R, ["Graph PPR", "Graph PPR, no specificity"]).style.format("{:.3f}"), width="stretch")
        st.caption(f"Graph retrieval ≈ {ev['latency_ms']['graph_retrieval_per_question']} ms per question "
                   f"(PPR itself ≈ {ev['latency_ms']['ppr_per_question']} ms).")
    with st.expander("Per-question results (Hybrid + Graph PPR)"):
        st.dataframe(pd.DataFrame(R["Hybrid + Graph PPR"]["per_question"]).drop(columns=["top"]), hide_index=True,
                     width="stretch")
    st.caption("Small golden set: treat differences below ~0.04 (one question) as noise. The release gate uses a 0.02 tolerance.")

# ------------------------------------------------------------------------------------------------
# ARCHITECTURE
# ------------------------------------------------------------------------------------------------
with tab_arch:
    img = ROOT / "docs" / "architecture.png"
    if img.exists():
        st.image(str(img), caption="Production architecture (ChunkMind + Odyn). This demo implements the Confluence path.")
    st.markdown("""
| Production (ChunkMind) | This demo | Module |
|---|---|---|
| Kafka change events → incremental indexing | batch build, extraction cache keyed by chunk + prompt | `build.py`, `hybridrag/llm.py` |
| Structure-aware chunking, near-duplicate removal | heading-aware chunks, code blocks kept whole, shingle Jaccard | `hybridrag/ingest.py` |
| Tiered extraction (metadata/regex → LLM) | Tier 0 Jira keys + metadata, Tier 1 LLM guided by the ontology | `hybridrag/extract.py` |
| Ontology validator + rejection log | same | `hybridrag/extract.py`, `hybridrag/ontology.py` |
| Ontology training loop + release gate | same: 4 evidence streams, ablation per change group, held-out check | `hybridrag/learning.py` |
| FalkorDB graph | NetworkX graph | `hybridrag/graph.py` |
| AstraDB (BM25 + cosine ANN) | rank_bm25 + ChromaDB (local MiniLM ONNX embeddings) | `hybridrag/retrieve.py` |
| Graph expansion | Personalized PageRank with seed specificity + link trust | `hybridrag/graph.py` |
| RRF fusion + permission filter (JWT groups) | same; graph facts also filtered by source-chunk permission | `hybridrag/retrieve.py` |
| Code Expander + AST-aware token trimmer | section breadcrumb + budget trimmer that never cuts code blocks | `hybridrag/retrieve.py` |
| GPT-4o answer over SSE | gpt-4o-mini streaming (your key) or extractive cited answer | `hybridrag/answer.py` |
| Odyn evaluation (P@k, R@k, nDCG, MRR, LLM judge) | golden set, per-type metrics, ablations | `hybridrag/evaluate.py` |
""")
