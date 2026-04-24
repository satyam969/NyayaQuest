"""
stage5_refinement/retry_controller.py — Schema refinement final safety net.

Role in the new architecture
-----------------------------
The primary intelligent retry loop now lives in Stage 2:
    stage2_strategy/schema_strategy.generate_validated_schema()

That loop runs up to 3 attempts on sample windows BEFORE the full parse.
If a good schema is found, Stage 4 passes and this module is never called.

This module is the FINAL SAFETY NET — called only when:
  1. Stage 2 generated a schema (which passed sample validation)
  2. The full PDF parse completed
  3. Stage 4 quality evaluation still failed

Decision flow
-------------
  A. Parse full document with initial schema → evaluate
  B. If passed → return immediately (converged=True)
  C. Check severity of failure:
       - Near-threshold (overall >= MINOR_FAIL_FLOOR) with no severe structural
         issues → skip refinement, return best available (LLM cost not worth it)
       - Severe failure → run ONE final evidence-driven patch attempt
  D. After patch attempt → evaluate refined result
       - If improved → accept refined
       - If not improved → keep original best
  E. Return (best_chunks, best_metrics, converged)

MAX_RETRIES = 1  (one final attempt only — primary retries done in Stage 2)
"""

import logging
from typing import Any, Dict, List, Tuple

from ..stage3_parsing.schema_chunker       import SchemaChunker
from ..stage4_evaluation.quality_evaluator import evaluate_chunks
from ..stage4_evaluation.critic            import analyze_failures
from .schema_refiner                       import refine_schema

logger = logging.getLogger(__name__)

MAX_RETRIES = 1   # safety net — primary 3-attempt loop is in generate_validated_schema()

# ── Thresholds ────────────────────────────────────────────────────────────────

# If overall >= this AND no severe structural issues → skip refinement
MINOR_FAIL_FLOOR: float = 0.66

# Metric thresholds that indicate a SEVERE failure worth one retry
_SEVERE_CAPTURE_BELOW:    float = 0.50
_SEVERE_CONTINUITY_BELOW: float = 0.45
_SEVERE_LENGTH_BELOW:     float = 0.45
_SEVERE_NOISE_BELOW:      float = 0.50


# ─────────────────────────────────────────────────────────────────────
# Severity detection
# ─────────────────────────────────────────────────────────────────────

def _is_severe_failure(metrics: Dict[str, float]) -> bool:
    """
    Return True when the failure is structural enough to justify one
    LLM refinement call.

    Minor failures (score just below threshold, decent structure) are
    not worth the token cost — route straight to Stage 6 fallback.
    """
    if metrics.get("section_capture_rate", 1.0) < _SEVERE_CAPTURE_BELOW:
        return True
    if metrics.get("section_continuity",   1.0) < _SEVERE_CONTINUITY_BELOW:
        return True
    if metrics.get("chunk_length_sanity",  1.0) < _SEVERE_LENGTH_BELOW:
        return True
    if metrics.get("noise_ratio",          1.0) < _SEVERE_NOISE_BELOW:
        return True
    return False


def _build_richer_text_sample(text: str) -> str:
    """
    Build a text sample that combines the document head and tail for the
    refinement prompt — giving the LLM visibility of both the start
    (which sample validation saw) and the end (where full-doc failures
    typically originate).
    """
    head = text[:2_500]
    tail = text[-2_500:] if len(text) > 5_000 else ""
    if tail:
        return head + "\n\n[... middle of document ...]\n\n" + tail
    return head


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def schema_refinement_loop(
    llm,
    initial_schema: Dict[str, Any],
    text:           str,
    doc_title:      str,
    text_sample:    str,
    threshold:      float = 0.70,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], bool]:
    """
    Final safety-net refinement: one evidence-driven patch attempt after
    full-document Stage 4 failure.

    Parameters
    ----------
    llm            : LangChain LLM instance
    initial_schema : schema from Stage 2 (already passed sample validation)
    text           : full document text (used for full parse + richer sample)
    doc_title      : document label for SchemaChunker prefixes
    text_sample    : text sample passed from orchestrator (used as fallback)
    threshold      : quality-pass threshold (default 0.70)

    Returns
    -------
    best_chunks  : List[dict]  — highest-scoring chunk list found
    best_metrics : Dict        — metrics for the best chunk list
    converged    : bool        — True only if threshold was passed
    """
    # ── A. Full parse with initial schema ────────────────────────────
    logger.info("[stage5] Running full parse with initial schema")
    try:
        initial_chunks = SchemaChunker(initial_schema, doc_title).parse(text)
    except Exception as exc:
        logger.error(f"[stage5] Full parse crashed: {exc}")
        return [], {"overall": 0.0}, False

    passed, metrics = evaluate_chunks(initial_chunks, threshold)
    logger.info(
        f"[stage5] Full parse: chunks={len(initial_chunks)}  "
        f"overall={metrics['overall']:.3f}  passed={passed}"
    )

    best_chunks:  List[Dict[str, Any]] = initial_chunks
    best_metrics: Dict[str, float]     = metrics

    # ── B. Already passed → return immediately ───────────────────────
    if passed:
        logger.info("[stage5] Full parse passed — no refinement needed")
        return best_chunks, best_metrics, True

    overall = metrics.get("overall", 0.0)
    severe  = _is_severe_failure(metrics)

    # ── C. Near-threshold acceptance ──────────────────────────────────
    if overall >= MINOR_FAIL_FLOOR and not severe:
        logger.info(
            f"[stage5] Near-threshold acceptance (overall={overall:.3f} >= "
            f"{MINOR_FAIL_FLOOR}, no severe structural issues) — "
            f"accepting parse, skipping Stage 6"
        )
        for c in best_chunks:
            c.setdefault("metadata", {})["parse_strategy"] = "accepted_near_threshold"
        return best_chunks, best_metrics, False  # converged=False (below threshold)

    # ── D. Severe failure → one final patch attempt ──────────────────
    logger.info(
        f"[stage5] Severe failure detected (overall={overall:.3f}  "
        f"severe={severe}) — running final patch attempt"
    )

    failure_report = analyze_failures(initial_chunks, metrics)

    # Build a richer text sample (head + tail) so the refiner can see
    # both where sample validation succeeded and where full-doc failed
    rich_sample = _build_richer_text_sample(text)

    refined_schema = refine_schema(llm, initial_schema, failure_report, rich_sample)

    if refined_schema is None:
        logger.warning("[stage5] Refiner returned None — keeping original result")
        return best_chunks, best_metrics, False

    # ── E. Evaluate refined schema ────────────────────────────────────
    try:
        refined_chunks = SchemaChunker(refined_schema, doc_title).parse(text)
    except Exception as exc:
        logger.error(f"[stage5] Refined parse crashed: {exc} — keeping original")
        return best_chunks, best_metrics, False

    ref_passed, ref_metrics = evaluate_chunks(refined_chunks, threshold)
    ref_overall = ref_metrics.get("overall", 0.0)

    logger.info(
        f"[stage5] Refined parse: chunks={len(refined_chunks)}  "
        f"overall={ref_overall:.3f}  passed={ref_passed}"
    )

    # Accept refined result only if it genuinely improves on original
    if ref_overall > overall:
        logger.info(
            f"[stage5] Improvement: {overall:.3f} → {ref_overall:.3f} "
            f"— accepting refined schema"
        )
        best_chunks  = refined_chunks
        best_metrics = ref_metrics
        if ref_passed:
            return best_chunks, best_metrics, True
    else:
        logger.info(
            f"[stage5] No improvement ({ref_overall:.3f} <= {overall:.3f}) "
            f"— keeping original result"
        )

    return best_chunks, best_metrics, False
