from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    # Paths
    RAW_PDF_DIR: str = "data/raw_pdfs"
    PROCESSED_DIR: str = "data/processed"

    DOCS_JSONL: str = "data/processed/docs.jsonl"
    CHUNKS_PARQUET: str = "data/processed/chunks.parquet"

    FAISS_INDEX_PATH: str = "data/processed/index.faiss"
    FAISS_META_PATH: str = "data/processed/index_meta.parquet"

    # Chunking
    CHUNK_MAX_CHARS: int = 3500      # v0 heuristic (approx tokens)
    CHUNK_OVERLAP_CHARS: int = 400

    # Retrieval
    TOP_K: int = 6

    # LLM (optional; wire later)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # openai/anthropic/local
    LLM_MODEL_CHEAP: str = os.getenv("LLM_MODEL_CHEAP", "gpt-4o-mini")
    LLM_MODEL_STRONG: str = os.getenv("LLM_MODEL_STRONG", "gpt-4o")

    # Embeddings
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

settings = Settings()
