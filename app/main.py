"""FastAPI application for the sashar.dev ML + LLM backend (:8002).

Serves the Time Series & ML playground (/ml/v1) and the LLM pipeline (/llm/v1).
CORS is restricted to the frontend origins (CLAUDE.md: ML services are only
reachable by the frontend)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import llm, ml

settings = get_settings()

app = FastAPI(
    title="sashar.dev ML & LLM API",
    version="0.1.0",
    description="ML / time-series playground and LLM pipeline for sashar.dev.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml.router)
app.include_router(llm.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "model": settings.model_name}
