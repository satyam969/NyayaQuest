"""
Test cases for query intent classification.
Run: pytest tests/test_intent_router.py -v
"""

import sys
import os

# Ensure the project root is on the path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from intent_router import (
    classify_heuristic,
    classify_intent,
    QueryIntent,
    ClassificationResult,
)


# ─────────────────────────────────────────────
# Greeting classification tests
# ─────────────────────────────────────────────

class TestGreetings:
    @pytest.mark.parametrize("query", [
        "hi", "hii", "hiii", "hey", "heyy", "hello",
        "namaste", "namaskar", "hola", "gm", "gn",
    ])
    def test_common_greetings_classified(self, query):
        result = classify_heuristic(query)
        assert result.intent == QueryIntent.GREETING
        assert result.confidence >= 0.85

    def test_very_short_treated_as_greeting(self):
        result = classify_heuristic("ok")
        assert result.intent == QueryIntent.GREETING

    def test_greeting_with_casual_continuation(self):
        result = classify_heuristic("hii how are you doing")
        assert result.intent == QueryIntent.GREETING
        assert result.confidence >= 0.70


# ─────────────────────────────────────────────
# Meta query tests
# ─────────────────────────────────────────────

class TestMetaQueries:
    @pytest.mark.parametrize("query", [
        "what can you do",
        "who are you",
        "how do you work",
        "what is this",
        "help",
        "features",
        "how to use",
        "tell me about yourself",
    ])
    def test_meta_queries_classified(self, query):
        result = classify_heuristic(query)
        assert result.intent == QueryIntent.META_QUERY
        assert result.confidence >= 0.90


# ─────────────────────────────────────────────
# Legal query tests
# ─────────────────────────────────────────────

class TestLegalQueries:
    @pytest.mark.parametrize("query", [
        "what is section 103 of BNS",
        "explain article 21 of the constitution",
        "punishment for theft under IPC",
        "what are the rights of a tenant",
        "how to file an RTI application",
        "consumer protection act complaint procedure",
        "grounds for anticipatory bail",
        "labour code overtime provisions",
    ])
    def test_clear_legal_queries(self, query):
        result = classify_heuristic(query)
        assert result.intent == QueryIntent.LEGAL_QUERY
        assert result.confidence >= 0.80

    def test_greeting_then_legal(self):
        result = classify_heuristic("hi, what is section 103 BNS")
        assert result.intent == QueryIntent.LEGAL_QUERY
        assert result.confidence >= 0.85


# ─────────────────────────────────────────────
# Out-of-scope tests
# ─────────────────────────────────────────────

class TestOutOfScope:
    @pytest.mark.parametrize("query", [
        "weather in Delhi today",
        "who won the cricket match",
        "recipe for biryani",
        "capital of France",
        "how to code in Python",
        "US immigration law",
        "stock price of Reliance",
        "movie recommendations",
    ])
    def test_out_of_scope_queries(self, query):
        result = classify_heuristic(query)
        assert result.intent == QueryIntent.OUT_OF_SCOPE


# ─────────────────────────────────────────────
# Ambiguous tests (should fall to LLM)
# ─────────────────────────────────────────────

class TestAmbiguous:
    @pytest.mark.parametrize("query", [
        "tell me more",
        "what next",
        "and then",
        "is it correct",
        "please explain",
    ])
    def test_vague_queries_low_confidence(self, query):
        result = classify_heuristic(query)
        # Should have low confidence, delegating to LLM
        assert result.confidence < 0.70


# ─────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_query(self):
        result = classify_heuristic("")
        assert result.intent == QueryIntent.AMBIGUOUS

    def test_whitespace_only(self):
        result = classify_heuristic("   ")
        assert result.intent == QueryIntent.AMBIGUOUS

    def test_single_character(self):
        result = classify_heuristic("a")
        assert result.confidence < 1.0  # ambiguous or greeting

    def test_very_long_query(self):
        long_query = "what is section 103 " * 100
        result = classify_heuristic(long_query)
        assert result.intent == QueryIntent.LEGAL_QUERY

    def test_special_characters(self):
        result = classify_heuristic("!!!???")
        assert result.confidence < 0.90


# ─────────────────────────────────────────────
# Integration test with mock LLM
# ─────────────────────────────────────────────

class MockLLM:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def invoke(self, prompt):
        class _Response:
            pass
        r = _Response()
        r.content = self._response_text
        return r


class TestConfidenceRouting:
    def test_high_confidence_skips_llm(self):
        """When heuristic is confident, LLM should not be called."""
        mock_llm = MockLLM("LEGAL_QUERY")

        # Clear greeting — high heuristic confidence
        result = classify_intent("hi", llm=mock_llm)
        assert result.intent == QueryIntent.GREETING
        assert result.method == "heuristic"

    def test_low_confidence_calls_llm(self):
        """When heuristic is unsure, LLM should be called."""
        mock_llm = MockLLM("LEGAL_QUERY")

        # Ambiguous query
        result = classify_intent("tell me more about this", llm=mock_llm)
        # Should have called LLM
        assert result.method in ["llm", "heuristic"]

    def test_no_llm_returns_heuristic(self):
        """When no LLM available, returns heuristic result."""
        result = classify_intent("some vague query", llm=None)
        assert result.method == "heuristic"


# ─────────────────────────────────────────────
# Real-world scenarios (from the bug report)
# ─────────────────────────────────────────────

class TestBugScenarios:
    """Test cases from the actual bug — 'hii' returning legal sections."""

    def test_hii_no_longer_gets_legal_sections(self):
        result = classify_heuristic("hii")
        assert result.intent == QueryIntent.GREETING
        # Should NOT be LEGAL_QUERY
        assert result.intent != QueryIntent.LEGAL_QUERY

    def test_hiii_variant(self):
        result = classify_heuristic("hiii")
        assert result.intent == QueryIntent.GREETING

    def test_hi_there_variant(self):
        result = classify_heuristic("hi there")
        assert result.intent == QueryIntent.GREETING
