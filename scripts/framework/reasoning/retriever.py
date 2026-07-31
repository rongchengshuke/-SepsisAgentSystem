from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeChunk:
    doc_id: str
    title: str
    source: str
    text: str


class TfidfKnowledgeBase:
    def __init__(self, docs_dir: str | Path | Iterable[str | Path], min_chunk_chars: int = 120):
        if isinstance(docs_dir, (str, Path)):
            self.docs_dirs = [Path(docs_dir)]
        else:
            self.docs_dirs = [Path(item) for item in docs_dir]
        self.min_chunk_chars = min_chunk_chars
        self.chunks = self._load_chunks()
        corpus = [chunk.text for chunk in self.chunks] if self.chunks else [""]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _load_chunks(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for docs_dir in self.docs_dirs:
            if not docs_dir.exists():
                continue
            for path in sorted(docs_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                title = path.stem.replace("_", " ")
                source = ""
                body_lines: list[str] = []
                for line in text.splitlines():
                    lower = line.lower()
                    if lower.startswith("source:"):
                        source = line.split(":", 1)[1].strip()
                    elif lower.startswith("title:"):
                        title = line.split(":", 1)[1].strip() or title
                    elif lower.startswith("chunk:"):
                        continue
                    elif line.strip():
                        body_lines.append(line.strip())
                parts = [part.strip() for part in "\n".join(body_lines).split("\n\n") if part.strip()]
                if not parts:
                    parts = ["\n".join(body_lines)]
                for idx, part in enumerate(parts):
                    if len(part) < self.min_chunk_chars and idx > 0:
                        continue
                    chunks.append(
                        KnowledgeChunk(
                            doc_id=f"{path.stem}#{idx}",
                            title=title,
                            source=source,
                            text=part,
                        )
                    )
        return chunks

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str | float]]:
        if not self.chunks:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": float(scores[int(idx)]),
                }
            )
        return results
