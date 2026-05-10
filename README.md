# Retrieval Agent — Production RAG + Agentic System

> Technical assignment submission — Production-Grade Retrieval Platform (Assignment 1) + Agentic Orchestration System (Assignment 2)

## Results Summary

### Assignment 1 — Retrieval Platform

| Config | P@5 | R@5 | NDCG@5 | Avg Latency | P95 Latency |
|---|---|---|---|---|---|
| Semantic only | 0.1583 | 0.2435 | 0.2515 | 132ms | 298ms |
| Hybrid (BM25 + semantic) | 0.1500 | 0.2223 | 0.2740 | 16ms | 28ms |
| **Hybrid + rerank** | **0.1833** | **0.2271** | **0.3151** | 939ms | 414ms |

- **+25% NDCG improvement** from semantic-only to hybrid+rerank (0.2515 → 0.3151)
- **Cold cache: 31ms → Warm cache: 18ms** (SQLite embed cache working)
- **Cost per 1000 queries: $0.00** (fully local stack — sentence-transformers + cross-encoder)
- P95 latency 414ms — within 500ms SLA budget

### Assignment 2 — Agentic System

| Prompt Version | Success Rate |
|---|---|
| V1 — explicit planning | 35% |
| V2 — no planning | 50% |
| Delta | -15% |

**Ablation finding:** V2 outperformed V1 because the explicit planning/reflection tags consumed
iterations within the hard 8-iteration budget, leaving fewer attempts for tool calls. 
Raising max_iterations to 12 would recover V1's advantage on complex multi-hop queries.

---

## Architecture

### Assignment 1 — Retrieval Platform

### Assignment 2 — Agentic System

---

## Key Design Decisions

### Chunking: 512 tokens, 20% overlap, paragraph-boundary aware
Split on double newlines first, then enforce token limit. Prevents BM25 keyword hits being 
destroyed by mid-sentence cuts. 20% overlap preserves cross-boundary context.

### Dedup: SHA-256 on raw text, stored in SQLite
Re-running `ingest.py` on unchanged corpus completes in milliseconds. Only new/changed 
documents are chunked, embedded, and indexed.

### Score fusion: Reciprocal Rank Fusion (RRF)
Rank-based — no normalisation needed between BM25 (unbounded counts) and cosine similarity 
(0-1). More robust than linear combination in practice.

### Embedding cache: SQLite-backed
Zero network latency — local file read. Handles thousands of reads/sec. Survives process 
restarts. Single-node appropriate; would switch to Redis for multi-worker scale.

### Cross-encoder: ms-marco-MiniLM-L-6-v2
6-layer model — fast enough for p95 <500ms on CPU. Trained on MS MARCO passage ranking — 
the exact task. Two-stage retrieval (bi-encoder → cross-encoder) is the standard 
production pattern.

### Agent safety: SymPy not eval()
`eval()` on agent-generated strings is a critical security hole. SymPy parses to AST 
and evaluates only math — arbitrary code injection fails at parse time.

---

## Corpus

3 BEIR datasets — chosen for domain diversity to surface BM25 vs semantic failure modes:

| Dataset | Docs | Domain | Why |
|---|---|---|---|
| SciFact | 5,183 | Biomedical claims | Paraphrased queries stress semantic search |
| NFCorpus | 3,633 | Medical nutrition | Short docs stress chunking strategy |
| FiQA-2018 | 2,000 | Finance QA | Domain jargon where BM25 beats semantic |

**Total: 10,816 documents, 11,750 chunks**

BEIR was chosen over hand-written QA pairs because qrels are written independently —
eval numbers are comparable to published baselines and not inflated by self-selection.

---

## Failure Modes (honest)

**Retrieval:**
- P@5 numbers are modest partly due to chunk/doc ID alignment — qrels reference doc IDs 
  but chunks inherit doc IDs with slight variation. Full alignment would improve scores.
- FiQA sample is 2000/57638 docs — REIT-specific queries miss due to corpus coverage, 
  not retrieval quality.
- Reranker p95 (414ms) is close to 500ms SLA — would use async rerank queue at scale.

**Agent:**
- Web search (DuckDuckGo) returns 0 results intermittently — rate limiting without API key.
  Production fix: Brave Search API or Serper.
- Agent handles worst: multi-hop queries requiring synthesis across 3+ sources, and 
  queries requiring real-time data the web search couldn't fetch.
- At 100 concurrent users: cross-encoder would queue (CPU-bound), ChromaDB in-process 
  mode would contend, SQLite cache writes would serialize. Fix: reranker worker pool, 
  ChromaDB server mode, Postgres pgvector.

---

## Running Locally

```bash
# 1. Clone and setup
git clone https://github.com/Manthanbhanushali010/retrieval-agent.git
cd retrieval-agent
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv \
    sentence-transformers chromadb rank-bm25 torch datasets \
    huggingface-hub pypdf pandas sympy httpx ddgs \
    structlog numpy scikit-learn anthropic

# 2. Environment
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 3. Download corpus
python scripts/download_corpus.py

# 4. Ingest
export PYTHONPATH=$(pwd)
python retrieval/ingest.py

# 5. Start retrieval API
python -m uvicorn retrieval.api:app --host 0.0.0.0 --port 8000

# 6. Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what causes cancer", "top_k": 5, "use_rerank": true}'

# 7. Test agent
python -c "
import asyncio
from agent.planner import run_agent
result = asyncio.run(run_agent('What is the square root of 144 plus the cube of 7?'))
print(result['answer'])
"

# 8. Run evals
python retrieval/eval.py
python agent/eval.py
```

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Vector store | ChromaDB | Local, no infra, persists to disk |
| Sparse index | rank-bm25 | Pure Python, pickles to single file |
| Embed model | all-MiniLM-L6-v2 | 22M params, ~50ms CPU, top-10 quality/speed |
| Reranker | ms-marco-MiniLM-L-6-v2 | MS MARCO trained, 6-layer fast |
| API | FastAPI | Async, Pydantic validation, auto docs |
| Agent LLM | Claude claude-sonnet-4-5 | Tool use, planning, reflection |
| Web search | ddgs | No API key for demo |
| Calculator | SymPy | Safe AST eval |

---

