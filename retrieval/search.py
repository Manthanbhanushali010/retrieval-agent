"""
retrieval/search.py

Hybrid BM25 + semantic search with cross-encoder reranking.
Composable, tunable weights. Returns score breakdown per result.
"""
import pickle
import time
from dataclasses import dataclass

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder

from shared.config import settings
from shared.embedder import embedder
import structlog

log = structlog.get_logger()


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    title: str
    date: str
    bm25_score: float
    semantic_score: float
    rerank_score: float
    final_score: float


class HybridSearcher:
    def __init__(self):
        # Load BM25 index
        with open(settings.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self.chunk_id_to_idx = {
            c["chunk_id"]: i for i, c in enumerate(self.chunks)
        }

        # ChromaDB
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = client.get_collection("documents")

        # Cross-encoder reranker — lazy loaded on first rerank call
        self._reranker = None

        log.info("searcher_ready",
                 chunks=len(self.chunks),
                 chroma_count=self.collection.count())

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            log.info("loading_reranker", model=settings.rerank_model)
            self._reranker = CrossEncoder(settings.rerank_model)
        return self._reranker

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, normalised_score)]"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0
        return [
            (self.chunks[i]["chunk_id"], float(scores[i]) / max_score)
            for i in top_indices
            if scores[i] > 0
        ]

    def _semantic_search(self, query: str, top_k: int,
                         filters: dict = None) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine_score)]"""
        vec = embedder.embed_one(query).tolist()
        where = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append({k: {"$eq": v}})
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        kwargs = dict(
            query_embeddings=[vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        ids = results["ids"][0]
        distances = results["distances"][0]
        # ChromaDB cosine distance → similarity: score = 1 - distance
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def _rrf_fusion(self,
                    bm25_results: list[tuple[str, float]],
                    semantic_results: list[tuple[str, float]],
                    k: int = 60) -> list[tuple[str, float, float, float]]:
        """
        Reciprocal Rank Fusion.
        Returns [(chunk_id, bm25_score, semantic_score, rrf_score)]

        Why RRF over linear combination:
        RRF is rank-based — no score normalisation needed.
        BM25 scores are unbounded counts; cosine is 0-1.
        RRF handles this mismatch by using rank position, not raw score.
        """
        bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_results)}
        sem_ranks  = {cid: rank + 1 for rank, (cid, _) in enumerate(semantic_results)}
        bm25_dict  = dict(bm25_results)
        sem_dict   = dict(semantic_results)

        all_ids = set(bm25_ranks) | set(sem_ranks)
        fused = []
        for cid in all_ids:
            rrf = 0.0
            if cid in bm25_ranks:
                rrf += settings.bm25_weight / (k + bm25_ranks[cid])
            if cid in sem_ranks:
                rrf += settings.semantic_weight / (k + sem_ranks[cid])
            fused.append((
                cid,
                bm25_dict.get(cid, 0.0),
                sem_dict.get(cid, 0.0),
                rrf,
            ))

        fused.sort(key=lambda x: x[3], reverse=True)
        return fused

    def search(self,
               query: str,
               top_k: int = None,
               use_rerank: bool = True,
               filters: dict = None,
               bm25_weight: float = None,
               semantic_weight: float = None) -> tuple[list[SearchResult], dict]:
        """
        Main search entry point.

        Args:
            query: natural language query
            top_k: number of results (default from settings)
            use_rerank: whether to apply cross-encoder reranking
            filters: metadata filters e.g. {"source": "scifact"}
            bm25_weight: override default BM25 weight
            semantic_weight: override default semantic weight

        Returns:
            (results, latency_breakdown)
        """
        top_k = top_k or settings.top_k_final
        retrieval_k = settings.top_k_retrieval

        if bm25_weight is not None:
            settings.bm25_weight = bm25_weight
        if semantic_weight is not None:
            settings.semantic_weight = semantic_weight

        timings = {}
        t0 = time.perf_counter()

        # BM25
        t1 = time.perf_counter()
        bm25_res = self._bm25_search(query, retrieval_k)
        timings["bm25_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        # Semantic
        t2 = time.perf_counter()
        sem_res = self._semantic_search(query, retrieval_k, filters)
        timings["semantic_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        # RRF fusion
        fused = self._rrf_fusion(bm25_res, sem_res)
        candidates = fused[:retrieval_k]

        # Build result objects
        chunk_map = {c["chunk_id"]: c for c in self.chunks}
        results = []
        for cid, bm25_s, sem_s, rrf_s in candidates:
            chunk = chunk_map.get(cid)
            if not chunk:
                continue
            results.append(SearchResult(
                chunk_id=cid,
                doc_id=chunk["doc_id"],
                text=chunk["text"],
                source=chunk.get("source", ""),
                title=chunk.get("title", ""),
                date=chunk.get("date", ""),
                bm25_score=round(bm25_s, 4),
                semantic_score=round(sem_s, 4),
                rerank_score=0.0,
                final_score=round(rrf_s, 6),
            ))

        # Rerank
        if use_rerank and results:
            t3 = time.perf_counter()
            reranker = self._get_reranker()
            pairs = [(query, r.text) for r in results]
            rerank_scores = reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r.rerank_score = round(float(s), 4)
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            timings["rerank_ms"] = round((time.perf_counter() - t3) * 1000, 2)

        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return results[:top_k], timings


# Singleton
searcher = HybridSearcher()