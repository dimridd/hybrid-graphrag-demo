"""Load Confluence-style pages and split them into structure-aware chunks.

* Front matter carries the connector metadata (space, labels, author) -> structural graph layer.
* Chunks follow headings; code blocks are never split; each chunk keeps its heading path (breadcrumb).
* Near-duplicate chunks are dropped with MinHash-style shingle Jaccard.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import CORPUS

TARGET_WORDS = 45
MAX_WORDS = 120


@dataclass
class Page:
    doc_id: str
    title: str
    space: str
    labels: list[str]
    author: str
    updated: str
    text: str
    path: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    space: str
    index: int
    breadcrumb: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def load_pages(root: Path = CORPUS) -> list[Page]:
    pages = []
    for p in sorted(root.glob("*/*.md")):
        raw = p.read_text(encoding="utf-8")
        _, fm, body = raw.split("---", 2)
        m = yaml.safe_load(fm)
        pages.append(Page(doc_id=f"{m['space']}/{p.stem}", title=m["title"], space=m["space"],
                          labels=m.get("labels", []), author=m.get("author", ""), updated=str(m.get("updated", "")),
                          text=body.strip(), path=str(p.relative_to(root))))
    return pages


def _blocks(text: str):
    """Yield (heading_level, heading, block_text). Code fences stay whole."""
    lines, buf, in_code = text.splitlines(), [], False
    level, heading = 0, ""
    for ln in lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#{1,6})\s+(.*)", ln) if not in_code else None
        if m:
            if buf:
                yield level, heading, "\n".join(buf).strip()
                buf = []
            level, heading = len(m.group(1)), m.group(2).strip()
            buf.append(ln)
        else:
            buf.append(ln)
    if buf:
        yield level, heading, "\n".join(buf).strip()


def chunk_page(page: Page) -> list[Chunk]:
    path: dict[int, str] = {}
    pieces: list[tuple[str, str]] = []  # (breadcrumb, text)
    for level, heading, block in _blocks(page.text):
        if level:
            path = {k: v for k, v in path.items() if k < level}
            path[level] = heading
        crumb = " › ".join(path[k] for k in sorted(path) if k > 1) or page.title
        pieces.append((crumb, block))
    # merge small consecutive sections, split very large ones on paragraphs (never inside code)
    chunks, cur_txt, cur_crumb = [], [], None
    def flush():
        if cur_txt:
            chunks.append((cur_crumb, "\n\n".join(cur_txt)))
    for crumb, block in pieces:
        paras = re.split(r"\n\s*\n(?=(?:[^`]*```[^`]*```)*[^`]*$)", block)
        for para in paras:
            words = len(" ".join(cur_txt).split())
            if cur_txt and words + len(para.split()) > MAX_WORDS or (cur_txt and words >= TARGET_WORDS and para.lstrip().startswith("#")):
                flush()
                cur_txt, cur_crumb = [], None
            if cur_crumb is None:
                cur_crumb = crumb
            cur_txt.append(para)
    flush()
    out = []
    for i, (crumb, txt) in enumerate(chunks):
        cid = hashlib.sha1(f"{page.doc_id}|{i}|{txt}".encode()).hexdigest()[:12]
        out.append(Chunk(chunk_id=cid, doc_id=page.doc_id, title=page.title, space=page.space, index=i,
                         breadcrumb=crumb, text=txt, meta={"labels": page.labels, "author": page.author}))
    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def dedupe(chunks: list[Chunk], threshold: float = 0.9) -> tuple[list[Chunk], list[tuple[str, str]]]:
    """Drop chunks whose 5-word-shingle Jaccard with an earlier chunk is >= threshold."""
    kept, dropped, sigs = [], [], []
    for c in chunks:
        s = _shingles(c.text)
        dup = next((k.chunk_id for k, ks in sigs if len(s & ks) / max(1, len(s | ks)) >= threshold), None)
        if dup:
            dropped.append((c.chunk_id, dup))
        else:
            kept.append(c)
            sigs.append((c, s))
    return kept, dropped


def load_corpus():
    pages = load_pages()
    chunks = [c for p in pages for c in chunk_page(p)]
    chunks, dropped = dedupe(chunks)
    return pages, chunks, dropped
