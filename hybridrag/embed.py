"""Local ONNX MiniLM embeddings (Chroma's default model) with a small persistent cache."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import BUILD

_ef = None
_cache: dict[str, np.ndarray] | None = None
_dirty = False


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        k, v = BUILD / "emb_keys.json", BUILD / "emb_vecs.npy"
        if k.exists() and v.exists():
            keys, vecs = json.loads(k.read_text()), np.load(v)
            _cache = {kk: vecs[i] for i, kk in enumerate(keys)}
    return _cache


def ef():
    global _ef
    if _ef is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _ef = DefaultEmbeddingFunction()
    return _ef


def embed(texts: list[str]) -> np.ndarray:
    global _dirty
    cache = _load()
    keys = [hashlib.sha1(t.encode()).hexdigest()[:16] for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if missing:
        uniq = dict(missing)
        vecs = ef()(list(uniq.values()))
        for k, v in zip(uniq, vecs):
            cache[k] = np.asarray(v, dtype=np.float32)
        _dirty = True
    out = np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 384), dtype=np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def save():
    if _dirty and _cache:
        BUILD.mkdir(parents=True, exist_ok=True)
        keys = list(_cache)
        (BUILD / "emb_keys.json").write_text(json.dumps(keys))
        np.save(BUILD / "emb_vecs.npy", np.stack([_cache[k] for k in keys]))
