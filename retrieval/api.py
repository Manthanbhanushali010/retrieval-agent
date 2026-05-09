"""
retrieval/api.py
POST /search — returns top-5 with full score breakdown
"""
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from retrieval.search import searcher

app = FastAPI(title="Retrieval API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True
    filters: Optional[dict] = None
    bm25_weight: Optional[float] = None
    semantic_weight: Optional[float] = None


class ScoreBreakdown(BaseModel):
    bm25: float
    semantic: float
    reranker: float
    final: float


class SearchResultItem(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    source: str
    title: str
    date: str
    text: str
    scores: ScoreBreakdown


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    latency: dict
    config: dict


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results, latency = searcher.search(
        query=req.query,
        top_k=req.top_k,
        use_rerank=req.use_rerank,
        filters=req.filters,
        bm25_weight=req.bm25_weight,
        semantic_weight=req.semantic_weight,
    )

    items = []
    for rank, r in enumerate(results, 1):
        items.append(SearchResultItem(
            rank=rank,
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            source=r.source,
            title=r.title,
            date=r.date,
            text=r.text[:500],  # truncate for API response
            scores=ScoreBreakdown(
                bm25=r.bm25_score,
                semantic=r.semantic_score,
                reranker=r.rerank_score,
                final=r.final_score,
            )
        ))

    return SearchResponse(
        query=req.query,
        results=items,
        latency=latency,
        config={
            "bm25_weight": req.bm25_weight or 0.3,
            "semantic_weight": req.semantic_weight or 0.7,
            "rerank": req.use_rerank,
            "top_k": req.top_k,
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": searcher.collection.count()}