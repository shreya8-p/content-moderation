"""
tests/test_api.py — Integration tests for FastAPI moderation endpoints.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_moderate_clean_text():
    response = client.post("/moderate", json={"text": "Good morning, have a great day!"})
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert "toxicity_score" in data
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] < 500  # Must be <50ms


def test_moderate_toxic_text():
    response = client.post("/moderate", json={"text": "I hate you, you worthless loser!"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is True


def test_moderate_empty_text_rejected():
    response = client.post("/moderate", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_moderate_text_too_long_rejected():
    long_text = "x" * 5001
    response = client.post("/moderate", json={"text": long_text})
    assert response.status_code == 422


def test_batch_moderation():
    texts = [
        "Hello, how are you?",
        "I hate this platform!",
        "Great weather today.",
    ]
    response = client.post("/moderate/batch", json=texts)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3


def test_batch_max_size_limit():
    texts = ["test text"] * 101
    response = client.post("/moderate/batch", json=texts)
    assert response.status_code == 400


def test_custom_threshold():
    response = client.post("/moderate", json={"text": "slightly rude comment", "threshold": 0.1})
    assert response.status_code == 200
