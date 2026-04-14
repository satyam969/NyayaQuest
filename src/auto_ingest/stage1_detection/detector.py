"""
stage1_detection/detector.py — Heuristic document-type detector.

Scores the first 5 000 characters of the cleaned text against signature
patterns for every known document type.  Returns:

    (doc_type: str, confidence: float, features: dict)

confidence is the fraction of pattern hits for the winning type.
"""

import re
from typing import Tuple, Dict, Any

from ..utils.patterns import DOC_TYPE_SIGNATURES


def detect_document_type(
    text: str,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Score text against all known document-type signatures.

    Returns
    -------
    doc_type   : str   — best-matching type key (e.g. "CPC", "RTI")
    confidence : float — fraction of that type's patterns that fired  [0, 1]
    features   : dict  — structural feature flags + raw score breakdown
    """
    sample = text[:5000]
    scores: Dict[str, float] = {}

    for doc_type, patterns in DOC_TYPE_SIGNATURES.items():
        if doc_type == "GENERIC" or not patterns:
            scores[doc_type] = 0.0
            continue
        hits = sum(
            1 for p in patterns if re.search(p, sample, re.IGNORECASE)
        )
        scores[doc_type] = hits / len(patterns)

    best_type = max(scores, key=scores.__getitem__)
    confidence = min(scores[best_type], 1.0)

    # If nothing scored > 0 → fall back to GENERIC
    if confidence == 0.0:
        best_type = "GENERIC"

    features = _extract_structural_features(text)
    features["score_breakdown"] = scores

    return best_type, confidence, features


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_structural_features(text: str) -> Dict[str, Any]:
    """Detect high-level structural markers present in the document."""
    return {
        "has_chapters": bool(
            re.search(r"CHAPTER\s+[IVXLCDM]+", text, re.IGNORECASE)
        ),
        "has_parts": bool(
            re.search(r"PART\s+[IVXLCDM]+", text, re.IGNORECASE)
        ),
        "has_orders": bool(
            re.search(r"\bORDER\s+[IVXLCDM]+\b", text)
        ),
        "has_schedules": bool(
            re.search(
                r"(THE FIRST SCHEDULE|THE SECOND SCHEDULE|"
                r"SCHEDULE [IVXLCDM]|THE SCHEDULE)",
                text,
                re.IGNORECASE,
            )
        ),
        "approx_section_count": len(
            re.findall(r"\n\s*\d{1,3}[A-Z]?\.\s", text)
        ),
        "has_definitions": bool(
            re.search(r'\bmeans\b.{0,200}?\binclude', text, re.IGNORECASE | re.DOTALL)
        ),
        "approx_char_count": len(text),
    }
