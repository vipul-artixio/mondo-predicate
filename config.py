import os
from pathlib import Path
from typing import Any, Dict


class Config:
    DB_HOST: str = os.getenv("PG_HOST", "localhost")
    DB_PORT: int = int(os.getenv("PG_PORT", "5432"))
    DB_NAME: str = os.getenv("PG_DATABASE", "postgres")
    DB_USER: str = os.getenv("PG_USER", "postgres")
    DB_PASSWORD: str = os.getenv("PG_PASSWORD", "postgres")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MONDO_BASE_URL: str = "https://github.com/monarch-initiative/mondo/releases/download/v2025-11-04/mondo_nodes.tsv"
    LLM_CACHE_DIR: str = os.getenv(
        "LLM_CACHE_DIR", str((Path(__file__).resolve().parent / "llm_join" / "cache").absolute())
    )
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "5"))
    LLM_THRESHOLD: float = float(os.getenv("LLM_THRESHOLD", "0.75"))
    LLM_MIN_CONFIDENCE: float = float(os.getenv("LLM_MIN_CONFIDENCE", "0.6"))
    OPENAI_THROTTLE_EVERY: int = int(os.getenv("OPENAI_THROTTLE_EVERY", "10"))
    OPENAI_THROTTLE_SLEEP: float = float(os.getenv("OPENAI_THROTTLE_SLEEP", "2.0"))
    OPEN_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-proj-1234567890")
    OPEN_AI_MODEL: str = os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL_NAME: str = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")


    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        return {
            "DB_HOST": cls.DB_HOST,
            "DB_PORT": cls.DB_PORT,
            "DB_NAME": cls.DB_NAME,
            "DB_USER": cls.DB_USER,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "OPEN_AI_MODEL": cls.OPEN_AI_MODEL,
            "OPENAI_EMBEDDING_MODEL_NAME": cls.OPENAI_EMBEDDING_MODEL_NAME,
        }

