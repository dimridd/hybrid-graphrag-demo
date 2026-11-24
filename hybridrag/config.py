"""Paths, permission model and demo users."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
GOLDEN = DATA / "golden_set.yaml"
ONTOLOGY_DIR = DATA / "ontologies"
BUILD = DATA / "build"
LLM_CACHE = BUILD / "llm_cache"

# Offline build models (any OpenAI-compatible endpoint: LLM_BASE_URL + LLM_API_KEY).
# The committed cache was built with Kimi K2.5 and Qwen3-235B; replaying it needs no key.
EXTRACT_MODEL = "moonshotai.kimi-k2.5"        # extraction, theme discovery, rejection judge, consolidation, junk confirm
AUDIT_MODEL = "qwen.qwen3-235b-a22b-2507"     # first junk auditor (a different model family than the confirmer)
ANSWER_MODEL = "gpt-4o-mini"                  # live answers in the app, with the visitor's own OpenAI key
EXTRACT_BATCH = 6                             # chunks per extraction call (61 chunks -> 11 extraction calls)

# Space-level permissions, as a Confluence connector would sync them.
SPACE_GROUPS = {
    "ORBIT": {"engineering", "product", "admins"},
    "PLATFORM": {"engineering", "admins"},
    "EVAL": {"engineering", "product", "admins"},
    "OPS": {"engineering", "security", "admins"},
    "SEC": {"security", "admins"},
    "TEAM": {"engineering", "product", "security", "admins"},
}

USERS = {
    "Admin (all spaces)": {"admins"},
    "Ana — Engineering": {"engineering"},
    "Ravi — Product": {"product"},
    "Hannah — Security": {"security"},
}


def allowed_spaces(groups: set[str]) -> set[str]:
    return {s for s, g in SPACE_GROUPS.items() if g & groups}


# Starter questions: multi-hop / process questions where graph fusion measurably helps (see Evaluation tab).
EXAMPLES = [
    "Which team owns the service that sends SSE heartbeats, and who leads it?",
    "Who was the incident commander for ORB-2291 and which rotation does that person belong to?",
    "Which model produces embeddings for the Astra DB collections, and which team owns that service?",
    "What are the steps of Phase I in the Deal Filter Pipeline?",
    "What was the root cause of incident ORB-2291?",
]
DEFAULT_K = 6
