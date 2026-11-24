"""Answer generation.

* With an OpenAI key: gpt-4o-mini writes a grounded answer that cites [C1], [C2] ... (streamed).
* Without a key: an extractive answer - the most question-relevant sentences from the top chunks,
  each with its citation. A grounding check returns NOT FOUND when nothing relevant is retrieved.
"""
from __future__ import annotations

import os
import re

from .config import ANSWER_MODEL
from .retrieve import tokenize

SYSTEM = """Answer the question using ONLY the numbered context blocks. Cite every claim with its block
number in square brackets, e.g. [C2]. If the context does not contain the answer, reply exactly:
NOT FOUND in the documents you can access. Be concise: 2-6 sentences or a short list."""


def build_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[C{i + 1}] {text}" for i, (_, text) in enumerate(contexts))
    return f"CONTEXT:\n{blocks}\n\nQUESTION: {question}"


def answer_backend(sidebar_key: str = ""):
    """Pick the answer LLM: key typed in the sidebar > OPENAI_API_KEY > LLM_BASE_URL + LLM_API_KEY (from .env)."""
    if sidebar_key:
        return {"api_key": sidebar_key, "base_url": None, "model": ANSWER_MODEL, "label": f"{ANSWER_MODEL} (your key)"}
    if os.environ.get("OPENAI_API_KEY"):
        return {"api_key": os.environ["OPENAI_API_KEY"], "base_url": None, "model": ANSWER_MODEL, "label": f"{ANSWER_MODEL} (.env)"}
    if os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
        model = os.environ.get("ANSWER_MODEL", "moonshotai.kimi-k2.5")
        return {"api_key": os.environ["LLM_API_KEY"], "base_url": os.environ["LLM_BASE_URL"], "model": model, "label": f"{model} (.env)"}
    return None


def stream_llm_answer(question: str, contexts, backend: dict):
    from openai import OpenAI
    client = OpenAI(api_key=backend["api_key"], base_url=backend["base_url"]) if backend["base_url"] else OpenAI(api_key=backend["api_key"])
    stream = client.chat.completions.create(model=backend["model"], temperature=0, stream=True,
                                            messages=[{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": build_prompt(question, contexts)}])
    for ev in stream:
        delta = ev.choices[0].delta.content if ev.choices else None
        if delta:
            yield delta


def _stem(tokens):
    return {t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens}


def extractive_answer(question: str, contexts, facts=None, max_sentences: int = 4, min_overlap: int = 2) -> str:
    qt = _stem(tokenize(question))
    cands = []
    for ci, (_, text) in enumerate(contexts[:5]):
        body = text.split("\n", 1)[-1]
        body = re.sub(r"```.*?```", " ", body, flags=re.S)
        for s in re.split(r"(?<=[.!?])\s+|\n+", body):
            s = s.replace("**", "").strip(" -*#")
            if len(s.split()) < 3:
                continue
            ov = len(qt & _stem(tokenize(s)))
            cands.append((ov - 0.15 * ci, ov, ci, s))
    cands.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for _, ov, ci, s in cands:
        if ov < min_overlap or s in seen:
            continue
        seen.add(s)
        picked.append(f"- {s} [C{ci + 1}]")
        if len(picked) >= max_sentences:
            break
    if not picked:
        return "NOT FOUND in the documents you can access."
    out = "\n".join(picked)
    rel = [f for f in (facts or []) if len(qt & _stem(tokenize(f"{f['source']} {f['verb']} {f['target']}")) ) >= 1][:3]
    if rel:
        out += "\n\n**Graph facts used:** " + " · ".join(f"{f['source']} —{f['verb']}→ {f['target']}" for f in rel)
    return out
