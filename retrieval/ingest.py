import json
import hashlib
import pickle
import sqlite3
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from shared.config import settings
from shared.embedder import embedder
import structlog

log = structlog.get_logger()

def _get_dedup_conn():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.data_dir / "dedup.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested (
            doc_hash TEXT PRIMARY KEY,
            doc_id   TEXT,
            chunks   INTEGER
        )
    """)
    conn.commit()
    return conn

def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def _count_tokens(text: str) -> int:
    return len(text) // 4

def chunk_document(doc_id: str, title: str, text: str) -> list[dict]:
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current_paras: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if para_tokens > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = _count_tokens(sent)
                if current_tokens + sent_tokens > chunk_size and current_paras:
                    chunk_text = " ".join(current_paras)
                    chunks.append(_make_chunk(doc_id, title, chunk_text, chunk_idx))
                    chunk_idx += 1
                    overlap_text = chunk_text[-(overlap * 4):]
                    current_paras = [overlap_text]
                    current_tokens = _count_tokens(overlap_text)
                current_paras.append(sent)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > chunk_size and current_paras:
                chunk_text = " ".join(current_paras)
                chunks.append(_make_chunk(doc_id, title, chunk_text, chunk_idx))
                chunk_idx += 1
                overlap_text = chunk_text[-(overlap * 4):]
                current_paras = [overlap_text]
                current_tokens = _count_tokens(overlap_text)
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunks.append(_make_chunk(doc_id, title, " ".join(current_paras), chunk_idx))

    return chunks

def _make_chunk(doc_id, title, text, idx):
    return {
        "chunk_id": f"{doc_id}_chunk_{idx}",
        "doc_id":   doc_id,
        "title":    title,
        "text":     text,
        "tokens":   _count_tokens(text),
    }

def _get_collection():
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )

def ingest(corpus_path: Path = None, force: bool = False):
    corpus_path = corpus_path or settings.corpus_path
    dedup = _get_dedup_conn()
    collection = _get_collection()

    all_chunks = []
    new_chunk_ids = []
    new_chunk_texts = []
    new_chunk_metas = []
    skipped = 0
    processed = 0

    with open(corpus_path) as f:
        docs = [json.loads(line) for line in f if line.strip()]

    log.info("ingest_start", total_docs=len(docs))

    for doc in docs:
        doc_id = doc["id"]
        title  = doc.get("title", "")
        text   = doc.get("text", "")
        source = doc.get("source", "unknown")
        date   = doc.get("date", "")
        tags   = doc.get("tags", [])

        if not text.strip():
            continue

        h = _doc_hash(text)
        chunks = chunk_document(doc_id, title, text)
        all_chunks.extend(chunks)

        exists = dedup.execute(
            "SELECT 1 FROM ingested WHERE doc_hash=?", (h,)
        ).fetchone()

        if exists and not force:
            skipped += 1
            continue

        processed += 1
        meta = {
            "doc_id": doc_id,
            "source": source,
            "date":   date,
            "tags":   ",".join(tags),
            "title":  title[:200],
        }
        for chunk in chunks:
            new_chunk_ids.append(chunk["chunk_id"])
            new_chunk_texts.append(chunk["text"])
            new_chunk_metas.append({**meta, "chunk_idx": chunk["chunk_id"]})

        dedup.execute(
            "INSERT OR REPLACE INTO ingested VALUES (?,?,?)",
            (h, doc_id, len(chunks))
        )

    dedup.commit()

    BATCH = 256
    if new_chunk_texts:
        log.info("embedding_chunks", count=len(new_chunk_texts))
        for i in range(0, len(new_chunk_texts), BATCH):
            bt = new_chunk_texts[i:i+BATCH]
            bi = new_chunk_ids[i:i+BATCH]
            bm = new_chunk_metas[i:i+BATCH]
            vecs = embedder.embed(bt)
            collection.upsert(
                ids=bi,
                embeddings=vecs.tolist(),
                documents=bt,
                metadatas=bm,
            )
            print(f"  chroma batch {i//BATCH + 1} done ({len(bt)} chunks)")

    log.info("building_bm25", total_chunks=len(all_chunks))
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    settings.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

    print(f"\n✓ Processed {processed} new docs, skipped {skipped} unchanged")
    print(f"✓ Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    ingest()