"""Search FAISS indexes and aggregate results at the paper level."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

import config
from index.embedder import embed_texts, get_client
from retrieval.query_parser import ParsedQuery, QueryCategory

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk: dict
    score: float


@dataclass
class PaperResult:
    paper_id: str
    title: str | None
    doi: str | None
    best_score: float
    evidence_chunks: list[SearchResult]


class Retriever:
    """Loads FAISS indexes from disk and provides search methods."""

    def __init__(self) -> None:
        self.main_index: faiss.Index | None = None
        self.eq_index: faiss.Index | None = None
        self.main_chunks: list[dict] = []
        self.eq_chunks: list[dict] = []
        self._client = get_client()
        self._load()

    def _load(self) -> None:
        if config.MAIN_INDEX_PATH.exists():
            self.main_index = faiss.read_index(str(config.MAIN_INDEX_PATH))
            self.main_chunks = json.loads(config.MAIN_CHUNKS_PATH.read_text("utf-8"))
            logger.info("Loaded main index with %d vectors", self.main_index.ntotal)
        else:
            logger.warning("Main index not found at %s", config.MAIN_INDEX_PATH)

        if config.EQUATION_INDEX_PATH.exists():
            self.eq_index = faiss.read_index(str(config.EQUATION_INDEX_PATH))
            self.eq_chunks = json.loads(config.EQUATION_CHUNKS_PATH.read_text("utf-8"))
            logger.info("Loaded equation index with %d vectors", self.eq_index.ntotal)

    def search(self, parsed: ParsedQuery, top_k: int = config.RETRIEVAL_TOP_K) -> list[PaperResult]:
        """Run retrieval based on query category and return paper-level results."""
        results: list[SearchResult] = []

        if parsed.category == QueryCategory.EQUATION_MODEL:
            results += self._search_index(self.main_index, self.main_chunks, parsed.search_query, top_k)
            results += self._search_index(self.eq_index, self.eq_chunks, parsed.search_query, top_k)
        elif parsed.category == QueryCategory.FIGURE_TABLE:
            raw = self._search_index(self.main_index, self.main_chunks, parsed.search_query, top_k * 2)
            for r in raw:
                ct = r.chunk.get("chunk_type", "")
                if ct in ("figure_caption", "table_caption"):
                    r.score *= 1.3
            results = raw
        else:
            results = self._search_index(self.main_index, self.main_chunks, parsed.search_query, top_k)

        return self._aggregate(results, top_k)

    def _search_index(
        self, index: faiss.Index | None, chunks: list[dict], query: str, top_k: int
    ) -> list[SearchResult]:
        if index is None or not chunks:
            return []
        q_vec = embed_texts([query], client=self._client)
        scores, ids = index.search(q_vec, min(top_k, index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=chunks[idx], score=float(score)))
        return results

    @staticmethod
    def _aggregate(results: list[SearchResult], top_k: int) -> list[PaperResult]:
        """Group results by paper and pick the best chunk per paper."""
        paper_map: dict[str, PaperResult] = {}
        for r in results:
            pid = r.chunk.get("paper_id", "unknown")
            if pid not in paper_map:
                paper_map[pid] = PaperResult(
                    paper_id=pid,
                    title=r.chunk.get("title"),
                    doi=r.chunk.get("doi"),
                    best_score=r.score,
                    evidence_chunks=[r],
                )
            else:
                pr = paper_map[pid]
                pr.evidence_chunks.append(r)
                if r.score > pr.best_score:
                    pr.best_score = r.score

        ranked = sorted(paper_map.values(), key=lambda p: p.best_score, reverse=True)
        for pr in ranked:
            pr.evidence_chunks.sort(key=lambda r: r.score, reverse=True)
            pr.evidence_chunks = pr.evidence_chunks[:3]

        return ranked[:top_k]
