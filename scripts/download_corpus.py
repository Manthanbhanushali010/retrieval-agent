from datasets import load_dataset
import json, pathlib, random

ROOT = pathlib.Path("data")
CORPUS_OUT = ROOT / "corpus"
EVAL_OUT   = ROOT / "eval"
CORPUS_OUT.mkdir(parents=True, exist_ok=True)
EVAL_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("scifact",  "test", 5183),
    ("nfcorpus", "test", 3633),
    ("fiqa",     "test", 2000),
]

all_docs = []

for name, qrels_split, limit in DATASETS:
    print(f"\n--- {name} ---")
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    sample = corpus_ds.select(range(min(limit, len(corpus_ds))))
    for row in sample:
        all_docs.append({
            "id":     row["_id"],
            "title":  row.get("title", ""),
            "text":   row["text"],
            "source": name,
            "date":   "2024-01-01",
            "tags":   [name],
        })
    print(f"  corpus: {len(sample)} docs")
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    with open(EVAL_OUT / f"{name}_queries.jsonl", "w") as f:
        for q in queries_ds:
            f.write(json.dumps({"id": q["_id"], "text": q["text"]}) + "\n")
    print(f"  queries: {len(queries_ds)}")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=qrels_split)
    with open(EVAL_OUT / f"{name}_qrels.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in qrels_ds:
            f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")
    print(f"  qrels: {len(qrels_ds)}")

random.shuffle(all_docs)
with open(CORPUS_OUT / "corpus.jsonl", "w") as f:
    for doc in all_docs:
        f.write(json.dumps(doc) + "\n")

print(f"\nTotal: {len(all_docs)} documents")
print("Done.")
