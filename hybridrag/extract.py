"""Tiered extraction + ontology validation.

Tier 0 (no LLM): structural facts from connector metadata, Jira keys via regex.
Tier 1 (LLM):    entities and relationships, guided by the current ontology. The model may use a NEW
                 type or verb when nothing fits - those are rejected by the validator and become
                 evidence for the ontology training loop (rejection log).
Raw LLM output is cached per (chunk text, prompt) - re-validating under a new ontology costs no LLM calls.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .config import EXTRACT_BATCH, EXTRACT_MODEL
from .ingest import Chunk
from .ontology import JIRA_KEY, Ontology, norm_label, to_pascal, to_upper_snake

SYSTEM = """You extract a knowledge graph from enterprise documentation for search.
Extract EVERY named thing a user could search for: systems, components, teams, people, data stores,
endpoints, models, metrics, tickets, incidents, pipeline steps, events, policies, documents - typically
5-15 entities per chunk - and the relationships that the text states between them (connect every entity
to at least one other entity where the text supports it).
Use the ontology below when a type or verb fits. When NO listed type fits, use a new concise type
(PascalCase) instead of forcing a wrong one; likewise use a new verb (UPPER_SNAKE) when no listed verb fits.
New types and verbs are reviewed by the ontology team, so be honest rather than conservative.
Use the most specific name as written in the text.
You receive several chunks; extract from each chunk separately. Return JSON:
{"chunks":[{"chunk_id":"..","entities":[{"name":..,"type":..,"description":"short"}],
"relationships":[{"source":..,"source_type":..,"verb":..,"target":..,"target_type":..}]}]}"""


def tier0(chunk: Chunk) -> dict:
    ents = [{"name": k, "type": "JiraIssue", "description": "Jira issue referenced in the page"}
            for k in sorted(set(JIRA_KEY.findall(chunk.text)))]
    return {"entities": ents, "relationships": []}


def tier1_batch(batch: list[Chunk], onto: Ontology) -> dict[str, dict]:
    body = "\n\n".join(f"=== chunk_id: {c.chunk_id}\nDOCUMENT: {c.title}\nSECTION: {c.breadcrumb}\n{c.text}" for c in batch)
    out = llm.chat_json(SYSTEM, f"{onto.prompt_block()}\n\nCHUNKS:\n{body}", EXTRACT_MODEL)
    res = {c.chunk_id: {"entities": [], "relationships": []} for c in batch}
    for item in out.get("chunks", []) or []:
        cid = str(item.get("chunk_id", ""))
        if cid in res:
            res[cid] = {"entities": item.get("entities") or [], "relationships": item.get("relationships") or []}
    return res


def extract_all(chunks: list[Chunk], onto: Ontology, workers: int = 8) -> dict[str, dict]:
    """Return raw extraction per chunk_id (tier0 + tier1 merged). Chunks are batched per LLM call."""
    batches = [chunks[i:i + EXTRACT_BATCH] for i in range(0, len(chunks), EXTRACT_BATCH)]
    with ThreadPoolExecutor(workers) as ex:
        t1 = {}
        for part in ex.map(lambda b: tier1_batch(b, onto), batches):
            t1.update(part)
    out = {}
    for c in chunks:
        t0 = tier0(c)
        out[c.chunk_id] = {"entities": t0["entities"] + t1[c.chunk_id]["entities"],
                           "relationships": t1[c.chunk_id]["relationships"], "prompt_version": onto.version}
    return out


def entity_key(name: str) -> str:
    n = norm_label(name)
    return n[3:] if n.startswith("the") and len(n) > 6 else n


def _coerce_type(name: str, raw_type: str) -> str:
    if JIRA_KEY.fullmatch(name.strip()):
        return "JiraIssue"            # ID-named type: the shape decides, not the LLM
    return raw_type


def validate(raw: dict[str, dict], onto: Ontology) -> dict:
    """Validate raw extractions under an ontology. Pure function of (raw, ontology) - no LLM."""
    entities: dict[str, dict] = {}
    rels: dict[tuple, dict] = {}
    rejections: list[dict] = []
    stats = defaultdict(int)

    for cid, ex in raw.items():
        local: dict[str, str] = {}   # entity key -> canonical type (accepted in this chunk)

        def accept_entity(name, etype, desc):
            key = entity_key(name)
            if not key:
                return None
            e = entities.setdefault(key, {"key": key, "name": name.strip(), "type": etype, "chunks": set(),
                                          "descriptions": []})
            e["chunks"].add(cid)
            if desc and len(e["descriptions"]) < 4:
                e["descriptions"].append(desc)
            local[key] = e["type"]
            return key

        def check_entity(name, raw_type, desc):
            stats["entities_raw"] += 1
            name = str(name or "").strip()
            raw_type = _coerce_type(name, str(raw_type or ""))
            t = onto.resolve_type(raw_type)
            if t is None:
                rejections.append({"chunk": cid, "kind": "entity", "reason": "unknown_type",
                                   "type": to_pascal(raw_type) or "?", "name": name, "description": desc or ""})
                return None
            v = onto.name_violation(name, t)
            if v:
                rejections.append({"chunk": cid, "kind": "entity", "reason": "name_rule", "rule": v,
                                   "type": t, "name": name, "description": desc or ""})
                return None
            return accept_entity(name, t, desc)

        for e in ex.get("entities", []):
            check_entity(e.get("name"), e.get("type"), e.get("description"))

        for r in ex.get("relationships", []):
            stats["relationships_raw"] += 1
            src, tgt = str(r.get("source") or "").strip(), str(r.get("target") or "").strip()
            st, tt = _coerce_type(src, str(r.get("source_type") or "")), _coerce_type(tgt, str(r.get("target_type") or ""))
            verb_raw = to_upper_snake(str(r.get("verb") or ""))
            base = {"chunk": cid, "kind": "relationship", "verb": verb_raw, "source": src, "target": tgt,
                    "source_type": to_pascal(st), "target_type": to_pascal(tt), "description": r.get("description", "")}
            st_c, tt_c = onto.resolve_type(st), onto.resolve_type(tt)
            if st_c is None or tt_c is None:
                rejections.append({**base, "reason": "unknown_endpoint_type"})
                continue
            verb = onto.resolve_verb(verb_raw)
            if verb is None:
                rejections.append({**base, "source_type": st_c, "target_type": tt_c, "reason": "unknown_relationship"})
                continue
            spec = onto.relationships[verb]
            if st_c not in spec["source"] or tt_c not in spec["target"]:
                rejections.append({**base, "verb": verb, "source_type": st_c, "target_type": tt_c, "reason": "bad_endpoints"})
                continue
            sk = local.get(entity_key(src)) and entity_key(src) or check_entity(src, st_c, "")
            tk = local.get(entity_key(tgt)) and entity_key(tgt) or check_entity(tgt, tt_c, "")
            if not sk or not tk or sk == tk:
                rejections.append({**base, "verb": verb, "reason": "dangling_endpoint"})
                continue
            key = (sk, verb, tk)
            rel = rels.setdefault(key, {"source": sk, "verb": verb, "target": tk, "chunks": set(), "description": r.get("description", "")})
            rel["chunks"].add(cid)

    # an entity keeps the type it was first accepted with; count type conflicts for the report
    stats["entities_accepted"] = len(entities)
    stats["relationships_accepted"] = len(rels)
    stats["rejections"] = len(rejections)
    return {"entities": entities, "relationships": list(rels.values()), "rejections": rejections, "stats": dict(stats)}
