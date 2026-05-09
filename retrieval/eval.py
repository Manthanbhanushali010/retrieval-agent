import json, time, csv, math
from pathlib import Path
from dataclasses import dataclass, field
from retrieval.search import HybridSearcher

EVAL_DIR = Path("data/eval")

def load_eval_data(dataset, max_queries=10):
    qrels_path = EVAL_DIR / f"{dataset}_qrels.tsv"
    queries_path = EVAL_DIR / f"{dataset}_queries.jsonl"
    relevant_map = {}
    with open(qrels_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid, cid, score = row["query-id"], row["corpus-id"], int(row["score"])
            if score > 0:
                relevant_map.setdefault(qid, set()).add(cid)
    queries = []
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            if q["id"] in relevant_map:
                queries.append({"id": q["id"], "text": q["text"], "relevant": relevant_map[q["id"]]})
            if len(queries) >= max_queries:
                break
    return queries

def precision_at_k(retrieved, relevant, k=5):
    return sum(1 for r in retrieved[:k] if r in relevant) / k

def recall_at_k(retrieved, relevant, k=5):
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(retrieved, relevant, k=5):
    dcg = sum(1.0 / math.log2(i+2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    idcg = sum(1.0 / math.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def run_config(searcher, queries, name, semantic_only=False, use_rerank=False, bm25_w=0.3, sem_w=0.7):
    ps, rs, ns, lats = [], [], [], []
    for q in queries:
        t0 = time.perf_counter()
        if semantic_only:
            results, _ = searcher.search(q["text"], top_k=5, use_rerank=False, bm25_weight=0.0, semantic_weight=1.0)
        else:
            results, _ = searcher.search(q["text"], top_k=5, use_rerank=use_rerank, bm25_weight=bm25_w, semantic_weight=sem_w)
        lat = (time.perf_counter() - t0) * 1000
        ids = [r.doc_id for r in results]
        ps.append(precision_at_k(ids, q["relevant"]))
        rs.append(recall_at_k(ids, q["relevant"]))
        ns.append(ndcg_at_k(ids, q["relevant"]))
        lats.append(lat)
    lats_s = sorted(lats)
    p95 = lats_s[int(len(lats_s)*0.95)]
    return {"name": name, "p5": round(sum(ps)/len(ps),4), "r5": round(sum(rs)/len(rs),4), "ndcg": round(sum(ns)/len(ns),4), "avg_ms": round(sum(lats)/len(lats),1), "p95_ms": round(p95,1)}

def main():
    searcher = HybridSearcher()
    all_queries = []
    for ds in ["scifact", "nfcorpus", "fiqa"]:
        all_queries.extend(load_eval_data(ds, max_queries=8))
    print(f"Total eval queries: {len(all_queries)}")

    configs = [
        dict(name="1_semantic_only", semantic_only=True),
        dict(name="2_hybrid", semantic_only=False, use_rerank=False),
        dict(name="3_hybrid_rerank", semantic_only=False, use_rerank=True),
    ]

    results = []
    for cfg in configs:
        print(f"Running {cfg['name']}...")
        r = run_config(searcher, all_queries, **cfg)
        results.append(r)
        print(f"  P@5={r['p5']} R@5={r['r5']} NDCG={r['ndcg']} avg={r['avg_ms']}ms p95={r['p95_ms']}ms")

    print("\n" + "="*65)
    print(f"{'Config':<25} {'P@5':>6} {'R@5':>6} {'NDCG@5':>8} {'Avg ms':>8} {'P95 ms':>8}")
    print("-"*65)
    for r in results:
        print(f"{r['name']:<25} {r['p5']:>6.4f} {r['r5']:>6.4f} {r['ndcg']:>8.4f} {r['avg_ms']:>8.1f} {r['p95_ms']:>8.1f}")

    print("\n--- Cold vs Warm Cache ---")
    q = "novel therapeutic targets for alzheimer disease"
    t0 = time.perf_counter(); searcher.search(q, use_rerank=False); print(f"Cold: {(time.perf_counter()-t0)*1000:.1f}ms")
    t0 = time.perf_counter(); searcher.search(q, use_rerank=False); print(f"Warm: {(time.perf_counter()-t0)*1000:.1f}ms")

    print("\n--- Cost Projection ---")
    print("Embedding: local sentence-transformers = $0.00/query")
    print("Reranker:  local cross-encoder = $0.00/query")
    print("Cost per 1000 queries: $0.00 (fully local stack)")
    print("Equivalent OpenAI ada-002: ~$0.10 per 1000 queries")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    open(EVAL_DIR / "retrieval_results.json", "w").write(json.dumps(results, indent=2))
    print("\nSaved to data/eval/retrieval_results.json")

main()
