"""Thin OpenAI wrapper with a disk cache.

Every call is cached by (model, prompt). The cache is committed to the repo, so the whole build
(extraction + ontology training) replays offline without an API key and gives identical results.
"""

# python imports
import hashlib
import json
import os

# local imports
from .config import LLM_CACHE

_client = None
STATS = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
USED: set[str] = set()


def _key(model: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()[:24]


def has_key() -> bool:
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def client():
    """OpenAI-compatible client. LLM_BASE_URL/LLM_API_KEY select any provider (OpenAI, Bedrock Mantle, Groq, ...)."""
    global _client
    if _client is None:
        from openai import OpenAI
        base = os.environ.get("LLM_BASE_URL")
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        _client = OpenAI(base_url=base, api_key=key, timeout=900) if base else OpenAI(api_key=key, timeout=900)
    return _client


def chat_json(system: str, user: str, model: str, offline_ok: bool = True) -> dict:
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path = LLM_CACHE / f"{_key(model, system, user)}.json"
    USED.add(path.name)
    if path.exists():
        STATS["cache_hits"] += 1
        return json.loads(path.read_text())["response"]
    if not has_key():
        raise RuntimeError("No cached LLM response and OPENAI_API_KEY is not set")
    kw = {"max_completion_tokens": 16000} if os.environ.get("LLM_BASE_URL") else {}
    r = client().chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    STATS["calls"] += 1
    STATS["prompt_tokens"] += r.usage.prompt_tokens
    STATS["completion_tokens"] += r.usage.completion_tokens
    txt = r.choices[0].message.content or "{}"
    txt = txt[txt.find("{"): txt.rfind("}") + 1] or "{}"
    out = json.loads(txt)
    path.write_text(json.dumps({"model": model, "response": out}, ensure_ascii=False))
    return out


def prune_cache() -> int:
    """Delete cache entries the last build did not use (stale prompts)."""
    n = 0
    for f in LLM_CACHE.glob("*.json"):
        if f.name not in USED:
            f.unlink()
            n += 1
    return n
