from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    huggingfacehub_api_token:str
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Retrieval
    vector_k: int = 6
    bm25_k: int = 6
    final_top_k: int = 4
    rrf_k_vector: int = 60
    rrf_k_bm25: int = 30

    # Logging
    log_file: str = "logs/rag_logs.jsonl"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()