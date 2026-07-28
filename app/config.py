"""Runtime configuration loaded from the environment."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _default_db_path() -> str:
    # Shared sqlite file in the project root: <root>/sashar.db.
    # This repo lives at <root>/sashar-dev-api-ml, so go up one level.
    return str(Path(__file__).resolve().parent.parent.parent / "sashar.db")


class Settings:
    """Process configuration. All values overridable via environment variables."""

    def __init__(self) -> None:
        self.port: int = int(os.getenv("PORT", "8002"))
        self.db_path: str = os.getenv("SASHAR_DB_PATH", _default_db_path())
        # ML endpoints must only be reachable by the frontend (CLAUDE.md).
        self.frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
        self.allowed_origins: list[str] = [
            o.strip()
            for o in os.getenv(
                "ALLOWED_ORIGINS",
                "https://sashar.dev,http://localhost:5173,http://localhost:4173",
            ).split(",")
            if o.strip()
        ]
        # LLM pipeline: instruction-tuned Qwen2.5-0.5B in 4-bit GGUF, run on CPU
        # via llama.cpp. Downloaded from the HF hub on first use if not present.
        self.model_name: str = os.getenv("MODEL_NAME", "qwen2.5-0.5b-instruct")
        self.spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
        self.gguf_repo_id: str = os.getenv("GGUF_REPO_ID", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        self.gguf_filename: str = os.getenv("GGUF_FILENAME", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        self.gguf_model_path: str = os.getenv(
            "GGUF_MODEL_PATH",
            str(Path(__file__).resolve().parent.parent / "models" / self.gguf_filename),
        )
        self.llm_n_ctx: int = int(os.getenv("LLM_N_CTX", "2048"))
        self.llm_n_threads: int = int(os.getenv("LLM_N_THREADS", "4"))
        # Hard cap on any torch training loop (CLAUDE.md: no GPU, <=50 epochs).
        self.max_epochs: int = int(os.getenv("MAX_EPOCHS", "50"))
        # Optional shared secret; when set, ML routes require this header value.
        self.internal_secret: str = os.getenv("INTERNAL_API_SECRET", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
