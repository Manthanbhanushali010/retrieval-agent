from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    data_dir: Path = Path("data")
    corpus_path: Path = Path("data/corpus/corpus.jsonl")
    chroma_dir: Path = Path("data/chroma")
    bm25_path: Path = Path("data/bm25_index.pkl")
    cache_db: Path = Path("data/embed_cache.db")
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 64
    chunk_size: int = 512
    chunk_overlap: int = 102
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    top_k_retrieval: int = 20
    top_k_final: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    agent_model: str = "claude-sonnet-4-5"
    max_iterations: int = 8
    token_budget: int = 4000
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()