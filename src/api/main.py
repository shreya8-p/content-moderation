"""
Real-Time Content Moderation API
FastAPI server exposing moderation endpoints with <50ms inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging

from src.model.predictor import ModerationPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Content Moderation API",
    description="Real-time toxic content detection using fine-tuned DistilBERT + ONNX",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
predictor = ModerationPredictor()


class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to moderate")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Toxicity threshold")


class ModerationResponse(BaseModel):
    text: str
    is_toxic: bool
    toxicity_score: float
    label: str
    inference_time_ms: float
    categories: dict


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "distilbert-onnx"}


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """
    Moderates a single piece of text.
    Returns toxicity score and classification in real-time.
    """
    try:
        start = time.perf_counter()
        result = predictor.predict(request.text, threshold=request.threshold)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"Moderated text | toxic={result['is_toxic']} | latency={elapsed_ms:.2f}ms")

        return ModerationResponse(
            text=request.text,
            is_toxic=result["is_toxic"],
            toxicity_score=round(result["score"], 4),
            label=result["label"],
            inference_time_ms=round(elapsed_ms, 2),
            categories=result["categories"],
        )

    except Exception as e:
        logger.error(f"Moderation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/moderate/batch")
async def moderate_batch(texts: list[str]):
    """
    Batch moderation for up to 100 texts at once.
    """
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")

    results = []
    for text in texts:
        result = predictor.predict(text)
        results.append(result)
    return {"results": results, "total": len(results)}