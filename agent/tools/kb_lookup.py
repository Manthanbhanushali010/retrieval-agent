import chromadb
from shared.config import settings
import structlog

log = structlog.get_logger()

def kb_lookup(source: str = None, tags: str = None, limit: int = 10) -> dict:
    """
    Structured metadata lookup directly against ChromaDB.
    Use when you need filtered access by source/tag, not semantic search.
    Complements doc_qa (semantic) with structured filtering.
    """
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_collection("documents")

        where = None
        if source:
            where = {"source": {"$eq": source}}
        elif tags:
            where = {"tags": {"$eq": tags}}

        kwargs = dict(
            query_embeddings=None,
            n_results=limit,
            include=["documents", "metadatas"],
        )

        # ChromaDB requires a query — use a dummy get() for pure metadata lookup
        results = collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

        docs = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            docs.append({
                "id":     doc_id,
                "source": meta.get("source", ""),
                "title":  meta.get("title", ""),
                "text":   results["documents"][i][:300] if results["documents"] else "",
            })

        log.info("kb_lookup_done", source=source, tags=tags, found=len(docs))
        return {"success": True, "documents": docs, "count": len(docs)}

    except Exception as e:
        log.error("kb_lookup_failed", error=str(e))
        return {"success": False, "error": str(e), "documents": []}