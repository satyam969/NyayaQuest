"""
stage4_evaluation/quality_evaluator.py — Chunk quality gate.

Computes a weighted quality score and decides pass/fail against
QUALITY_THRESHOLD (default 0.70).
"""

import logging
from typing import List, Dict, Any, Tuple

from ..utils.scoring import compute_quality_score

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.70


def evaluate_chunks(
    chunks: List[Dict[str, Any]],
    threshold: float = QUALITY_THRESHOLD,
) -> Tuple[bool, Dict[str, float]]:
    """
    Evaluate chunk quality.

    Returns
    -------
    passed  : bool              — True when overall score ≥ threshold
    metrics : Dict[str, float]  — per-metric scores + "overall"
    """
    if not chunks:
        logger.warning("evaluate_chunks: received empty list → auto-fail")
        return False, {"overall": 0.0}

    metrics = compute_quality_score(chunks)
    passed  = metrics["overall"] >= threshold

    logger.info(
        f"Quality gate: overall={metrics['overall']:.3f} "
        f"(threshold={threshold}) → {'PASS' if passed else 'FAIL'}"
    )
    return passed, metrics
