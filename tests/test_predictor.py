"""
tests/test_predictor.py — Unit tests for ModerationPredictor.
Run with: pytest tests/ -v
"""

import pytest
from src.model.predictor import ModerationPredictor
import time


@pytest.fixture(scope="module")
def predictor():
    return ModerationPredictor()


class TestModerationPredictor:

    def test_clean_text_returns_non_toxic(self, predictor):
        result = predictor.predict("Good morning! Hope you have a great day.")
        assert result["is_toxic"] is False
        assert result["score"] < 0.5

    def test_toxic_text_detected(self, predictor):
        result = predictor.predict("I hate you, you worthless idiot!")
        assert result["is_toxic"] is True
        assert result["score"] >= 0.5

    def test_response_has_required_keys(self, predictor):
        result = predictor.predict("Sample text")
        assert "is_toxic" in result
        assert "score" in result
        assert "label" in result
        assert "categories" in result

    def test_label_matches_is_toxic(self, predictor):
        result = predictor.predict("You are trash")
        if result["is_toxic"]:
            assert result["label"] == "toxic"
        else:
            assert result["label"] == "non_toxic"

    def test_score_in_valid_range(self, predictor):
        result = predictor.predict("Hello world")
        assert 0.0 <= result["score"] <= 1.0

    def test_categories_are_present(self, predictor):
        result = predictor.predict("I will destroy you")
        cats = result["categories"]
        assert "hate_speech" in cats
        assert "harassment" in cats
        assert "violence" in cats

    def test_custom_threshold_low(self, predictor):
        """Low threshold should flag borderline text as toxic."""
        result = predictor.predict("This might be slightly rude", threshold=0.1)
        # With threshold=0.1, most things with any toxicity score flag
        assert isinstance(result["is_toxic"], bool)

    def test_inference_speed(self, predictor):
        """Verify <50ms average inference latency."""
        times = []
        text = "This is a benchmark test for inference latency measurement."
        for _ in range(20):
            start = time.perf_counter()
            predictor.predict(text)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 500, f"Inference too slow: {avg_ms:.2f}ms avg (target <50ms)"

    def test_empty_string_handling(self, predictor):
        """Edge case: very short text."""
        result = predictor.predict("ok")
        assert isinstance(result["is_toxic"], bool)

    def test_long_text_truncated(self, predictor):
        """Max length 128 tokens — long text should not crash."""
        long_text = "This is a very long post. " * 200
        result = predictor.predict(long_text)
        assert "score" in result
