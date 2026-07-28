"""LLM playground pipeline routes under /llm/v1."""
from __future__ import annotations

from fastapi import APIRouter

from app.llm import pipeline
from app.schemas import (
    ClusterRequest,
    ClusterResult,
    EmbedRequest,
    EmbedResult,
    EncodeRequest,
    EncodeResult,
    GenerateRequest,
    GenerateResult,
    TokenizeRequest,
    TokenizeResult,
)

router = APIRouter(prefix="/llm/v1", tags=["llm"])


@router.post("/tokenize", response_model=TokenizeResult)
def tokenize(req: TokenizeRequest) -> TokenizeResult:
    return TokenizeResult(tokens=pipeline.tokenize(req.prompt))


@router.post("/encode", response_model=EncodeResult)
def encode(req: EncodeRequest) -> EncodeResult:
    return EncodeResult(ids=pipeline.encode(req.tokens))


@router.post("/embed", response_model=EmbedResult)
def embed(req: EmbedRequest) -> EmbedResult:
    points = pipeline.embed(req.tokens, req.ids)
    return EmbedResult(points=points)


@router.post("/generate", response_model=GenerateResult)
def generate(req: GenerateRequest) -> GenerateResult:
    out = pipeline.generate(req.prompt)
    return GenerateResult(
        output_tokens=out["tokens"],
        output_ids=out["ids"],
        output_points=out["points"],
    )


@router.post("/cluster", response_model=ClusterResult)
def cluster(req: ClusterRequest) -> ClusterResult:
    pts = [{"x": p.x, "y": p.y, "label": p.label} for p in req.points]
    out = pipeline.cluster(pts, req.n_clusters)
    return ClusterResult(n_groups=out["n_groups"], groups=out["groups"], assignments=out["assignments"])
