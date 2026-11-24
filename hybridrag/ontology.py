"""Ontology: allowed entity types, relationships (verb + endpoint types), aliases and name rules.

Three layers (see README):
  * structural types from connector metadata / regex  (Space, Document, Label, JiraIssue)  - no LLM
  * shared core types                                   (Person, Team, Service, ...)
  * learned types/links added by the ontology training loop (versioned, gated)
"""

# python imports
from dataclasses import dataclass, field
from pathlib import Path
import copy
import yaml
import re

STRUCTURAL = {"Space", "Document", "Label", "JiraIssue"}
JIRA_KEY = re.compile(r"\b([A-Z][A-Z0-9]{1,9}-\d{1,6})\b")


def to_pascal(s: str) -> str:
    parts = re.split(r"[\s_\-/]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s.strip()))
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


def to_upper_snake(s: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s.strip())
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()


def norm_label(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s).casefold()


def norm_name(s: str) -> str:
    s = re.sub(r"[`*\"']", "", s).strip()
    return re.sub(r"\s+", " ", s).casefold()


@dataclass
class Ontology:
    name: str
    version: str
    types: dict[str, dict]                      # name -> {description, aliases, strategy}
    relationships: dict[str, dict]              # VERB -> {source: [...], target: [...], description}
    rules: dict = field(default_factory=dict)   # code_shapes, id_shapes, stop_list, shape_exempt
    changelog: list[str] = field(default_factory=list)

    # ---------- io ----------
    @classmethod
    def load(cls, path: Path) -> "Ontology":
        d = yaml.safe_load(Path(path).read_text())
        return cls(name=d["metadata"]["name"], version=str(d["metadata"]["version"]),
                   types={t["name"]: {k: v for k, v in t.items() if k != "name"} for t in d["entity_types"]},
                   relationships={r["verb"]: {k: v for k, v in r.items() if k != "verb"} for r in d["relationships"]},
                   rules=d.get("name_rules", {}), changelog=d.get("changelog", []))

    def to_dict(self) -> dict:
        return {"metadata": {"name": self.name, "version": self.version},
                "changelog": self.changelog,
                "entity_types": [{"name": k, **v} for k, v in self.types.items()],
                "relationships": [{"verb": k, **v} for k, v in self.relationships.items()],
                "name_rules": self.rules}

    def save(self, path: Path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=120))

    def copy(self) -> "Ontology":
        return copy.deepcopy(self)

    # ---------- lookups ----------
    def resolve_type(self, t: str) -> str | None:
        """Map a raw type (any casing / alias) to a canonical type, or None if unknown."""
        n = norm_label(t)
        for name, spec in self.types.items():
            if norm_label(name) == n or n in {norm_label(a) for a in spec.get("aliases", [])}:
                return name
        return None

    def resolve_verb(self, v: str) -> str | None:
        n = norm_label(v)
        for verb, spec in self.relationships.items():
            if norm_label(verb) == n or n in {norm_label(a) for a in spec.get("aliases", [])}:
                return verb
        return None

    def llm_types(self) -> list[str]:
        return [t for t, s in self.types.items() if s.get("strategy", "llm") in ("llm", "hybrid")]

    def prompt_block(self) -> str:
        lines = ["ENTITY TYPES:"]
        for t, s in self.types.items():
            if t in STRUCTURAL - {"JiraIssue"}:
                continue
            al = f" (aliases: {', '.join(s['aliases'])})" if s.get("aliases") else ""
            lines.append(f"- {t}: {s.get('description', '')}{al}")
        lines.append("RELATIONSHIPS (VERB: source types -> target types):")
        for v, s in self.relationships.items():
            lines.append(f"- {v}: {'|'.join(s['source'])} -> {'|'.join(s['target'])}  # {s.get('description', '')}")
        return "\n".join(lines)

    # ---------- name rules ----------
    def name_violation(self, name: str, etype: str) -> str | None:
        r = self.rules
        n = name.strip()
        if len(n) < 2 or n.isdigit():
            return "too_short_or_numeric"
        if norm_name(n) in {norm_name(s) for s in r.get("stop_list", [])}:
            return "stop_list"
        if etype in set(r.get("shape_exempt", [])):
            return None
        for rx in r.get("code_shapes", []):
            if re.search(rx, n):
                return f"code_shape:{rx}"
        for rx in r.get("id_shapes", []):
            if re.search(rx, n):
                return f"id_shape:{rx}"
        return None
