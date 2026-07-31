from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ExperienceHit:
    principle_id: str
    score: float
    description: str
    category: str
    triples: list[dict[str, str]]
    usage_score: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_triples(raw_triples: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if head or relation or tail:
                normalized.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                    }
                )
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 3 and not any(isinstance(x, (list, tuple, dict)) for x in item):
                head = str(item[0]).strip()
                relation = str(item[1]).strip()
                tail = str(item[2]).strip()
                if head or relation or tail:
                    normalized.append(
                        {
                            "head": head,
                            "relation": relation,
                            "tail": tail,
                        }
                    )
                return
            for child in item:
                visit(child)
            return
        text = str(item).strip()
        if text:
            normalized.append({"head": text, "relation": "", "tail": ""})

    visit(raw_triples)
    return normalized


def render_principle_text(description: str, triples: Any) -> str:
    triples = normalize_triples(triples)
    triple_lines = []
    for triple in triples:
        head = triple.get("head", "")
        relation = triple.get("relation", "")
        tail = triple.get("tail", "")
        if head or relation or tail:
            triple_lines.append(f"{head} {relation} {tail}".strip())
    if triple_lines:
        return f"{description}\n" + "\n".join(triple_lines)
    return description


class ExperienceStore:
    def __init__(
        self,
        path: str | Path,
        similarity_threshold: float = 0.72,
        prune_threshold: float = 0.20,
        min_usage_before_prune: int = 4,
    ):
        self.path = Path(path)
        self.similarity_threshold = similarity_threshold
        self.prune_threshold = prune_threshold
        self.min_usage_before_prune = min_usage_before_prune
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"principles": []}
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return {"principles": []}
        loaded.setdefault("principles", [])
        return loaded

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)

    @property
    def principles(self) -> list[dict[str, Any]]:
        return self.data.setdefault("principles", [])

    def search(
        self,
        query: str,
        top_k: int = 3,
        owner: str | None = None,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        if not self.principles:
            return []
        filtered = self._filter_principles(owner=owner, include_global=include_global)
        if not filtered:
            return []
        corpus = [
            render_principle_text(
                str(principle.get("description", "")),
                list(principle.get("triples", [])),
            )
            for principle in filtered
        ]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[dict[str, Any]] = []
        for idx in top_indices:
            principle = dict(filtered[int(idx)])
            principle["match_score"] = float(scores[int(idx)])
            principle["usage_score"] = float(principle.get("score", 0.5))
            hits.append(principle)
        return hits

    def _filter_principles(
        self,
        owner: str | None = None,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for principle in self.principles:
            principle_owner = str(principle.get("owner", "global"))
            if owner is None:
                filtered.append(principle)
                continue
            if principle_owner == owner:
                filtered.append(principle)
                continue
            if include_global and principle_owner == "global":
                filtered.append(principle)
        return filtered

    def _best_match_index(
        self,
        candidate_text: str,
        owner: str = "global",
        include_global: bool = True,
    ) -> tuple[int | None, float]:
        scoped = self._filter_principles(owner=owner, include_global=include_global)
        if not scoped:
            return None, 0.0
        corpus = [
            render_principle_text(
                str(principle.get("description", "")),
                list(principle.get("triples", [])),
            )
            for principle in scoped
        ]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform(corpus + [candidate_text])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        if scores.size == 0:
            return None, 0.0
        best_idx = int(np.argmax(scores))
        best_id = str(scoped[best_idx]["id"])
        real_idx = next((idx for idx, item in enumerate(self.principles) if str(item.get("id")) == best_id), None)
        return real_idx, float(scores[best_idx]) if real_idx is not None else (None, 0.0)

    def mark_usage(self, retrieved_ids: list[str], success: bool) -> None:
        if not retrieved_ids:
            return
        id_set = set(retrieved_ids)
        for principle in self.principles:
            if principle.get("id") not in id_set:
                continue
            principle["use_count"] = int(principle.get("use_count", 0)) + 1
            if success:
                principle["success_count"] = int(principle.get("success_count", 0)) + 1
            principle["score"] = self._score(principle)
            principle["updated_at"] = utc_now_iso()

    @staticmethod
    def _score(principle: dict[str, Any]) -> float:
        success_count = int(principle.get("success_count", 0))
        use_count = int(principle.get("use_count", 0))
        return float((success_count + 1) / (use_count + 2))

    def integrate(
        self,
        principles: list[dict[str, Any]],
        case_id: str,
        trajectory_summary: str,
        success: bool,
        owner: str = "global",
    ) -> list[str]:
        added_or_merged_ids: list[str] = []
        for principle in principles:
            description = " ".join(str(principle.get("description", "")).split()).strip()
            triples = normalize_triples(principle.get("triples", []))
            if not description:
                continue
            candidate_text = render_principle_text(description, triples)
            match_idx, match_score = self._best_match_index(
                candidate_text,
                owner=owner,
                include_global=(owner != "global"),
            )
            if match_idx is not None and match_score >= self.similarity_threshold:
                existing = self.principles[match_idx]
                existing_sources = list(existing.get("source_cases", []))
                if case_id not in existing_sources:
                    existing_sources.append(case_id)
                existing["source_cases"] = existing_sources[-12:]
                history = list(existing.get("trajectory_summaries", []))
                history.append(trajectory_summary[:1200])
                existing["trajectory_summaries"] = history[-6:]
                existing["updated_at"] = utc_now_iso()
                added_or_merged_ids.append(str(existing["id"]))
                continue

            principle_id = f"principle_{len(self.principles) + 1:05d}"
            record = {
                "id": principle_id,
                "description": description,
                "category": str(principle.get("category", "guiding")).strip() or "guiding",
                "owner": owner,
                "triples": triples if isinstance(triples, list) else [],
                "source_cases": [case_id],
                "trajectory_summaries": [trajectory_summary[:1200]],
                "use_count": 0,
                "success_count": 0,
                "score": 0.5,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
            self.principles.append(record)
            added_or_merged_ids.append(principle_id)

        self._prune()
        return added_or_merged_ids

    def _prune(self) -> None:
        kept: list[dict[str, Any]] = []
        for principle in self.principles:
            use_count = int(principle.get("use_count", 0))
            score = float(principle.get("score", 0.5))
            if use_count >= self.min_usage_before_prune and score < self.prune_threshold:
                continue
            kept.append(principle)
        self.data["principles"] = kept
