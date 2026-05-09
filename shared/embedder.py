import sqlite3
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from shared.config import settings
import structlog

log = structlog.get_logger()

class CachedEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._init_cache()
        log.info("embedder_ready", model=settings.embed_model, dim=self.dim)

    def _init_cache(self):
        settings.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(settings.cache_db), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embed_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        results = {}
        to_embed = []
        for text in texts:
            h = self._hash(text)
            row = self.conn.execute(
                "SELECT embedding FROM embed_cache WHERE text_hash=?", (h,)
            ).fetchone()
            if row:
                results[h] = np.frombuffer(row[0], dtype=np.float32)
            else:
                to_embed.append((h, text))

        if to_embed:
            log.info("embedding_new", count=len(to_embed))
            hashes, raw_texts = zip(*to_embed)
            vecs = self.model.encode(
                list(raw_texts),
                batch_size=settings.embed_batch_size,
                show_progress_bar=len(raw_texts) > 100,
                normalize_embeddings=True,
            )
            for h, vec in zip(hashes, vecs):
                results[h] = vec
                self.conn.execute(
                    "INSERT OR IGNORE INTO embed_cache VALUES (?, ?)",
                    (h, vec.astype(np.float32).tobytes())
                )
            self.conn.commit()

        return np.array([results[self._hash(t)] for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

embedder = CachedEmbedder()