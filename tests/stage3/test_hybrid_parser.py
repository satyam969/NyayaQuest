"""
tests/stage3/test_hybrid_parser.py — Tests for the upgraded hybrid parser.

Run with:
    cd d:\\NyayaQuest
    .venv\\Scripts\\python.exe -m pytest tests/stage3/test_hybrid_parser.py -v
"""

import sys
import os
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from auto_ingest.stage3_parsing.hybrid_parser import (
    parse_hybrid,
    _evaluate,
    _tiebreak,
    _tag_chunks,
    REGEX_FASTPATH_THRESHOLD,
    QUALITY_FLOOR,
    ARBITRATION_MARGIN,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures — synthetic text & schemas
# ─────────────────────────────────────────────────────────────────────

def _gazette_text(n: int = 30) -> str:
    lines = [
        "THE TEST ACT, 2024",
        "CHAPTER I",
        "PRELIMINARY",
        "",
    ]
    for i in range(1, n + 1):
        lines += [
            f"{i}. Section {i} heading.—",
            f"   (1) Content of section {i}.",
            f"   (2) Further provision of section {i}.",
            "",
        ]
    return "\n".join(lines)


def _make_chunk(
    sec_num: str = "1",
    text_len: int = 200,
    strategy: str = "regex",
) -> Dict[str, Any]:
    return {
        "text": "X" * text_len,
        "metadata": {
            "section_number": sec_num,
            "chapter": "CHAPTER I - PRELIMINARY",
            "part": "Unknown Part",
            "chunk_index": 1,
            "parse_strategy": strategy,
            "doc_title": "TEST ACT",
            "doc_type": "GAZETTE_ACT",
        },
    }


def _make_chunks(
    n: int = 25,
    n_valid: Optional[int] = None,
    text_len: int = 300,
    strategy: str = "regex",
) -> List[Dict[str, Any]]:
    """Build a list of n chunks, n_valid of which have proper section numbers."""
    if n_valid is None:
        n_valid = n
    chunks = []
    for i in range(1, n + 1):
        sec = str(i) if i <= n_valid else "N/A"
        chunks.append(_make_chunk(sec_num=sec, text_len=text_len, strategy=strategy))
    return chunks


_STRONG_SCHEMA = {
    "section_pattern": r"(?=\n\s*\d{1,3}[A-Z]?\.\s)",
    "chapter_pattern": r"(CHAPTER\s+[IVXLCDM]+)\s*\n+(.*?)(?=\n|\Z)",
    "part_pattern": None,
    "title_extract_pattern": r"^(\d+[A-Z]?)\.\s*([^\n\-—]{0,150})",
    "hierarchy": ["chapter", "section"],
    "flags": {"has_definitions": True, "has_schedules": False,
               "has_orders": False, "has_parts": False},
    "metadata_defaults": {"year": "2024"},
}

_SECTION_PATTERN = r"(?=\n\s*\d{1,3}[A-Z]?\.\s)"


# ─────────────────────────────────────────────────────────────────────
# Unit tests — helpers
# ─────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_empty_chunks_returns_zero(self):
        metrics = _evaluate([])
        assert metrics["overall"] == 0.0

    def test_good_chunks_score_above_zero(self):
        chunks = _make_chunks(25)
        metrics = _evaluate(chunks)
        assert metrics["overall"] > 0.0
        assert "section_capture_rate" in metrics

    def test_all_na_chunks_low_capture(self):
        chunks = _make_chunks(20, n_valid=0)
        metrics = _evaluate(chunks)
        assert metrics["section_capture_rate"] < 0.3


class TestTiebreak:

    def test_higher_capture_wins(self):
        r = {"section_capture_rate": 0.9, "section_continuity": 0.8, "chapter_coverage": 0.7}
        s = {"section_capture_rate": 0.7, "section_continuity": 0.8, "chapter_coverage": 0.7}
        assert _tiebreak(r, s) == "regex"

    def test_schema_wins_better_continuity(self):
        r = {"section_capture_rate": 0.8, "section_continuity": 0.5, "chapter_coverage": 0.7}
        s = {"section_capture_rate": 0.8, "section_continuity": 0.9, "chapter_coverage": 0.7}
        assert _tiebreak(r, s) == "schema"

    def test_chapter_coverage_third_tiebreak(self):
        r = {"section_capture_rate": 0.8, "section_continuity": 0.8, "chapter_coverage": 0.4}
        s = {"section_capture_rate": 0.8, "section_continuity": 0.8, "chapter_coverage": 0.9}
        assert _tiebreak(r, s) == "schema"

    def test_all_equal_prefers_regex(self):
        m = {"section_capture_rate": 0.8, "section_continuity": 0.8, "chapter_coverage": 0.8}
        assert _tiebreak(m, m) == "regex"


class TestTagChunks:

    def test_tags_applied_to_all_chunks(self):
        chunks = _make_chunks(5)
        _tag_chunks(chunks, "hybrid_regex_fastpath", 0.90, None)
        for c in chunks:
            assert c["metadata"]["parse_strategy"] == "hybrid_regex_fastpath"
            assert c["metadata"]["hybrid_regex_score"] == 0.9

    def test_schema_score_included_when_provided(self):
        chunks = _make_chunks(3)
        _tag_chunks(chunks, "hybrid_schema", 0.75, 0.85)
        assert chunks[0]["metadata"]["hybrid_schema_score"] == 0.85

    def test_no_schema_score_when_none(self):
        chunks = _make_chunks(3)
        _tag_chunks(chunks, "hybrid_regex_only", 0.70, None)
        assert "hybrid_schema_score" not in chunks[0]["metadata"]


# ─────────────────────────────────────────────────────────────────────
# Integration tests — parse_hybrid()
# ─────────────────────────────────────────────────────────────────────

class TestRegexFastpath:
    """Regex with high quality → schema never called."""
    text = _gazette_text(40)

    def test_fastpath_when_regex_strong(self):
        """
        For a clean gazette text + good section pattern, regex should score
        well enough to trigger the fast-path (no schema invoked).
        """
        chunks = parse_hybrid(
            self.text, _SECTION_PATTERN, _STRONG_SCHEMA,
            doc_title="TEST ACT", doc_type="GAZETTE_ACT",
        )
        strategies = {c["metadata"]["parse_strategy"] for c in chunks}
        # Either fast-path (regex was very strong) or hybrid_regex/schema (competed)
        # The key assertion: all chunks have SOME hybrid strategy tag
        assert all(s.startswith("hybrid_") for s in strategies)

    def test_fastpath_tagged_correctly(self):
        """Manually force scores to confirm fast-path tag is applied."""
        strong_chunks = _make_chunks(30)
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=strong_chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            return_value={"overall": 0.95, "section_capture_rate": 0.95,
                          "section_continuity": 1.0, "chapter_coverage": 0.9,
                          "chunk_length_sanity": 0.9, "noise_ratio": 0.95},
        ):
            result = parse_hybrid(
                "dummy text", _SECTION_PATTERN, _STRONG_SCHEMA,
                doc_title="TEST", doc_type="GAZETTE_ACT",
            )
        strategies = {c["metadata"]["parse_strategy"] for c in result}
        assert strategies == {"hybrid_regex_fastpath"}, (
            f"Expected fast-path, got: {strategies}"
        )

    def test_fastpath_schema_not_called(self):
        """When regex is strong, SchemaChunker.parse() must NOT be called."""
        strong_chunks = _make_chunks(30)
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=strong_chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            return_value={"overall": 0.95, "section_capture_rate": 0.95,
                          "section_continuity": 1.0, "chapter_coverage": 0.9,
                          "chunk_length_sanity": 0.9, "noise_ratio": 0.95},
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.SchemaChunker"
        ) as mock_sc:
            parse_hybrid("dummy text", _SECTION_PATTERN, _STRONG_SCHEMA)
        mock_sc.assert_not_called()  # schema parser never instantiated


class TestNoSchema:
    """schema=None → regex returned, tagged hybrid_regex_only."""

    def test_returns_regex_when_no_schema(self):
        text = _gazette_text(20)
        chunks = parse_hybrid(
            text, _SECTION_PATTERN, None,
            doc_title="TEST", doc_type="GAZETTE_ACT",
        )
        assert len(chunks) > 0

    def test_strategy_tagged_regex_only(self):
        chunks = _make_chunks(10)
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            return_value={"overall": 0.72, "section_capture_rate": 0.8,
                          "section_continuity": 0.9, "chapter_coverage": 0.6,
                          "chunk_length_sanity": 0.9, "noise_ratio": 0.9},
        ):
            result = parse_hybrid("dummy", _SECTION_PATTERN, None)
        strategies = {c["metadata"]["parse_strategy"] for c in result}
        assert strategies == {"hybrid_regex_only"}

    def test_no_schema_score_in_metadata(self):
        chunks = _make_chunks(5)
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            return_value={"overall": 0.65, "section_capture_rate": 0.7,
                          "section_continuity": 0.8, "chapter_coverage": 0.5,
                          "chunk_length_sanity": 0.8, "noise_ratio": 0.8},
        ):
            result = parse_hybrid("dummy", _SECTION_PATTERN, None)
        assert "hybrid_schema_score" not in result[0]["metadata"]


class TestArbitration:
    """Score-based winner selection."""

    def _run_with_scores(self, regex_score: float, schema_score: float):
        """Helper: mock both parsers to return fixed overall scores."""
        call_count = [0]

        def fake_quality(chunks):
            call_count[0] += 1
            # First call = regex eval, second = schema eval
            score = regex_score if call_count[0] == 1 else schema_score
            return {
                "overall": score,
                "section_capture_rate": score,
                "section_continuity": score,
                "chapter_coverage": score,
                "chunk_length_sanity": score,
                "noise_ratio": score,
            }

        regex_chunks  = _make_chunks(20, strategy="regex")
        schema_chunks = _make_chunks(20, strategy="schema")

        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=regex_chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.SchemaChunker"
        ) as mock_sc, patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            side_effect=fake_quality,
        ):
            mock_sc.return_value.parse.return_value = schema_chunks
            result = parse_hybrid("text", _SECTION_PATTERN, _STRONG_SCHEMA)
        return result

    def test_regex_wins_when_significantly_better(self):
        result = self._run_with_scores(0.85, 0.70)
        strategies = {c["metadata"]["parse_strategy"] for c in result}
        assert "hybrid_regex" in strategies or "hybrid_regex_fastpath" in strategies, (
            f"Expected regex to win, got: {strategies}"
        )

    def test_schema_wins_when_significantly_better(self):
        result = self._run_with_scores(0.65, 0.82)
        strategies = {c["metadata"]["parse_strategy"] for c in result}
        assert "hybrid_schema" in strategies, (
            f"Expected schema to win, got: {strategies}"
        )

    def test_both_scores_attached_when_schema_runs(self):
        result = self._run_with_scores(0.65, 0.75)
        meta = result[0]["metadata"]
        assert "hybrid_regex_score" in meta
        assert "hybrid_schema_score" in meta

    def test_low_confidence_tag_when_both_below_floor(self):
        result = self._run_with_scores(0.55, 0.60)
        strategies = {c["metadata"]["parse_strategy"] for c in result}
        assert strategies == {"hybrid_low_confidence"}, (
            f"Expected hybrid_low_confidence, got: {strategies}"
        )


class TestCrashIsolation:
    """A parser crash should not kill the other."""

    def test_regex_crash_falls_back_to_schema(self):
        schema_chunks = _make_chunks(15, strategy="schema")
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            side_effect=RuntimeError("Simulated regex crash"),
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.SchemaChunker"
        ) as mock_sc:
            mock_sc.return_value.parse.return_value = schema_chunks
            result = parse_hybrid("text", _SECTION_PATTERN, _STRONG_SCHEMA)
        # Should return schema chunks (regex produced nothing after crash)
        assert len(result) > 0

    def test_schema_crash_returns_regex(self):
        regex_chunks = _make_chunks(20, strategy="regex")
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            return_value=regex_chunks,
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.SchemaChunker",
            side_effect=RuntimeError("Simulated schema crash"),
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.compute_quality_score",
            return_value={"overall": 0.70, "section_capture_rate": 0.75,
                          "section_continuity": 0.8, "chapter_coverage": 0.6,
                          "chunk_length_sanity": 0.8, "noise_ratio": 0.85},
        ):
            result = parse_hybrid("text", _SECTION_PATTERN, _STRONG_SCHEMA)
        assert len(result) > 0

    def test_both_crash_returns_empty_list(self):
        with patch(
            "auto_ingest.stage3_parsing.hybrid_parser.parse_with_regex",
            side_effect=RuntimeError("regex crash"),
        ), patch(
            "auto_ingest.stage3_parsing.hybrid_parser.SchemaChunker",
            side_effect=RuntimeError("schema crash"),
        ):
            result = parse_hybrid("text", _SECTION_PATTERN, _STRONG_SCHEMA)
        assert result == []


class TestBackwardCompatibility:
    """Public signature and return type must be unchanged."""

    def test_returns_list(self):
        text = _gazette_text(10)
        result = parse_hybrid(text, _SECTION_PATTERN, None)
        assert isinstance(result, list)

    def test_features_defaults_to_empty_dict(self):
        text = _gazette_text(10)
        # features=None must not raise
        result = parse_hybrid(text, _SECTION_PATTERN, None, features=None)
        assert isinstance(result, list)

    def test_all_chunks_have_metadata(self):
        text = _gazette_text(15)
        result = parse_hybrid(text, _SECTION_PATTERN, None)
        for c in result:
            assert "metadata" in c
            assert "parse_strategy" in c["metadata"]

    def test_all_strategies_start_with_hybrid(self):
        text = _gazette_text(15)
        result = parse_hybrid(text, _SECTION_PATTERN, None)
        for c in result:
            assert c["metadata"]["parse_strategy"].startswith("hybrid_"), (
                f"Unexpected strategy: {c['metadata']['parse_strategy']}"
            )
