"""
stage3_parsing/hybrid_parser.py — Regex-first adaptive hybrid parser.

Upgrades over v1
----------------
v1 always ran both parsers on the full PDF and picked the winner by raw
section-ID count — wrong metric, wasted CPU on every HYBRID document.

v2 design
---------
1. Run regex parser first (cheap, deterministic).
2. Evaluate with Stage 4 quality metrics (not just ID count).
3. If regex is strong (overall >= REGEX_FASTPATH_THRESHOLD):
       Return immediately — skip schema entirely.
       Tag: parse_strategy = "hybrid_regex_fastpath"
4. If no schema is available:
       Return regex output.
       Tag: parse_strategy = "hybrid_regex_only"
5. Run schema parser (only when regex is medium/weak).
6. Evaluate schema with the same Stage 4 metrics.
7. Score-based arbitration with configurable margin:
       regex > schema + MARGIN → regex wins
       schema > regex + MARGIN → schema wins
       tied → tiebreakers: capture_rate → continuity → chapter_coverage
8. If BOTH parsers score below QUALITY_FLOOR:
       Return higher-scoring result.
       Tag: parse_strategy = "hybrid_low_confidence"
9. All winning chunks carry: parse_strategy, hybrid_regex_score,
   hybrid_schema_score for observability.
10. Crash isolation: a crash in one parser does not kill the other.

Public API is unchanged:
    parse_hybrid(text, section_pattern, schema, doc_title, doc_type, features)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .regex_parser   import parse_with_regex
from .schema_chunker import SchemaChunker
from ..utils.scoring import compute_quality_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Thresholds (all tunable without changing logic)
# ─────────────────────────────────────────────────────────────────────

# If regex overall >= this, skip schema entirely (fast-path)
REGEX_FASTPATH_THRESHOLD: float = 0.82

# Minimum score considered "acceptable quality"
QUALITY_FLOOR: float = 0.70

# Score margin required to prefer one parser over the other
ARBITRATION_MARGIN: float = 0.05


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _evaluate(chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Run Stage 4 quality metrics on a chunk list.
    Returns an empty dict (overall=0) when chunks is empty.
    """
    if not chunks:
        return {"overall": 0.0}
    return compute_quality_score(chunks)


def _tiebreak(
    regex_metrics:  Dict[str, float],
    schema_metrics: Dict[str, float],
) -> str:
    """
    Deterministic tiebreaker when scores are within ARBITRATION_MARGIN.

    Ordered criteria:
      1. section_capture_rate  — prefer more sections identified
      2. section_continuity    — prefer monotone numbering
      3. chapter_coverage      — prefer richer chapter metadata

    Returns "regex" or "schema".
    """
    for key in ("section_capture_rate", "section_continuity", "chapter_coverage"):
        r = regex_metrics.get(key, 0.0)
        s = schema_metrics.get(key, 0.0)
        if r > s:
            return "regex"
        if s > r:
            return "schema"
    # All tiebreakers equal — prefer regex (deterministic, cheaper to debug)
    return "regex"


def _tag_chunks(
    chunks: List[Dict[str, Any]],
    strategy: str,
    regex_score: float,
    schema_score: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Attach observability metadata to every chunk in the winning set.
    Mutates in place and returns the list for convenience.
    """
    for c in chunks:
        meta = c.setdefault("metadata", {})
        meta["parse_strategy"]     = strategy
        meta["hybrid_regex_score"] = round(regex_score, 4)
        if schema_score is not None:
            meta["hybrid_schema_score"] = round(schema_score, 4)
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def parse_hybrid(
    text:             str,
    section_pattern:  str,
    schema:           Optional[Dict[str, Any]],
    doc_title:        str = "Unknown",
    doc_type:         str = "GENERIC",
    features:         Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Adaptive hybrid parser: regex-first with quality-gated schema competition.

    Parameters
    ----------
    text             : full cleaned document text
    section_pattern  : section-split regex from Stage 2
    schema           : LLM-generated schema dict (may be None)
    doc_title        : document label for chunk text prefixes
    doc_type         : doc-type string from Stage 1 (informational only)
    features         : structural feature flags from Stage 1

    Returns
    -------
    List of chunk dicts, each with parse_strategy and score metadata.
    """
    if features is None:
        features = {}

    # ── Step 1: Run regex parser ─────────────────────────────────────
    regex_chunks: List[Dict[str, Any]] = []
    try:
        regex_chunks = parse_with_regex(
            text, section_pattern, doc_title, doc_type, features
        )
    except Exception as exc:
        logger.error(f"[hybrid] regex parser crashed: {exc}")

    regex_metrics = _evaluate(regex_chunks)
    regex_score   = regex_metrics.get("overall", 0.0)

    logger.info(
        f"[hybrid] regex: chunks={len(regex_chunks)}  "
        f"overall={regex_score:.3f}  "
        f"capture={regex_metrics.get('section_capture_rate', 0):.2f}  "
        f"continuity={regex_metrics.get('section_continuity', 0):.2f}"
    )

    # ── Step 2: Fast-path — regex already strong ─────────────────────
    if regex_score >= REGEX_FASTPATH_THRESHOLD:
        logger.info(
            f"[hybrid] regex fast-path accepted "
            f"(score={regex_score:.3f} >= {REGEX_FASTPATH_THRESHOLD})"
        )
        return _tag_chunks(regex_chunks, "hybrid_regex_fastpath", regex_score, None)

    # ── Step 3: No schema available ──────────────────────────────────
    if not schema:
        logger.info(
            f"[hybrid] no schema — returning regex output "
            f"(score={regex_score:.3f})"
        )
        return _tag_chunks(regex_chunks, "hybrid_regex_only", regex_score, None)

    # ── Step 4: Run schema parser (regex was medium/weak) ────────────
    schema_chunks: List[Dict[str, Any]] = []
    try:
        schema_chunks = SchemaChunker(schema, doc_title).parse(text)
    except Exception as exc:
        logger.error(f"[hybrid] schema parser crashed: {exc}")

    schema_metrics = _evaluate(schema_chunks)
    schema_score   = schema_metrics.get("overall", 0.0)

    logger.info(
        f"[hybrid] schema: chunks={len(schema_chunks)}  "
        f"overall={schema_score:.3f}  "
        f"capture={schema_metrics.get('section_capture_rate', 0):.2f}  "
        f"continuity={schema_metrics.get('section_continuity', 0):.2f}"
    )

    # ── Step 5: Arbitration ──────────────────────────────────────────
    diff = regex_score - schema_score

    if diff > ARBITRATION_MARGIN:
        winner = "regex"
    elif -diff > ARBITRATION_MARGIN:
        winner = "schema"
    else:
        # Scores within margin — use tiebreakers
        winner = _tiebreak(regex_metrics, schema_metrics)
        logger.debug(
            f"[hybrid] tiebreak: margin={diff:+.3f}  winner={winner}"
        )

    # ── Step 6: Low-confidence both ──────────────────────────────────
    both_low   = regex_score < QUALITY_FLOOR and schema_score < QUALITY_FLOOR
    best_score = max(regex_score, schema_score)

    if both_low:
        strategy = "hybrid_low_confidence"
        logger.warning(
            f"[hybrid] both parsers below quality floor "
            f"(regex={regex_score:.3f}  schema={schema_score:.3f})  "
            f"→ returning higher-scoring result ({winner})"
        )
    else:
        strategy = f"hybrid_{winner}"

    logger.info(
        f"[hybrid] selected={winner}  strategy={strategy}  "
        f"regex={regex_score:.3f}  schema={schema_score:.3f}  "
        f"margin={diff:+.3f}"
    )

    winning_chunks = regex_chunks if winner == "regex" else schema_chunks
    return _tag_chunks(winning_chunks, strategy, regex_score, schema_score)
