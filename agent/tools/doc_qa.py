import httpx
import structlog

log = structlog.get_logger()

SEARCH_URL = "http://localhost:8000/search"

def doc_qa(query: str, top_k: int = 5, filters: dict = None) -> dict:
    """
    Query the retrieval system for document-grounded answers.
    Calls the /search API endpoint — keeps agent decoupled from ChromaDB.
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_rerank": True,
            "filters": filters,
        }
        resp = httpx.post(SEARCH_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        passages = [
            {
                "rank":   r["rank"],
                "source": r["source"],
                "title":  r["title"],
                "text":   r["text"],
                "scores": r["scores"],
            }
            for r in data["results"]
        ]
        log.info("doc_qa_done", query=query, passages=len(passages))
        return {
            "success":  True,
            "query":    query,
            "passages": passages,
            "latency":  data["latency"],
        }
    except Exception as e:
        log.error("doc_qa_failed", error=str(e))
        return {"success": False, "error": str(e), "passages": []}