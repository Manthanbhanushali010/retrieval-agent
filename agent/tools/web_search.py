from duckduckgo_search import DDGS
import structlog

log = structlog.get_logger()

def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo.
    No API key required. Returns title, url, snippet per result.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        log.info("web_search_done", query=query, results=len(results))
        return {
            "success": True,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url":   r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        }
    except Exception as e:
        log.error("web_search_failed", error=str(e))
        return {"success": False, "error": str(e), "results": []}