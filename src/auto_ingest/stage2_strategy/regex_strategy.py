"""
stage2_strategy/regex_strategy.py — Regex pattern selector.

Scores every SECTION_PATTERN against the document text and returns
the one that produces the most plausible section count.

Heuristic: a good pattern yields 5–400 sections.
Too few (< 5) means it missed most sections.
Too many (> 500) means it over-split.
"""

import re
from typing import Dict, Any, Optional

from ..utils.patterns import SECTION_PATTERNS, CHAPTER_PATTERNS


def select_section_pattern(
    text: str,
    doc_type: str,
    features: Dict[str, Any],
) -> str:
    """
    Return the SECTION_PATTERN that best matches this document.
    Falls back to the last (most permissive) pattern if scoring fails.
    """
    sample = text[:15_000]
    scored = []

    for pattern in SECTION_PATTERNS:
        try:
            count = len(re.findall(pattern, sample))
            scored.append((pattern, count))
        except re.error:
            continue

    if not scored:
        return SECTION_PATTERNS[-1]

    def _score(item):
        _, cnt = item
        if cnt < 5:
            return 0
        if cnt > 500:
            return max(0, 500 - cnt)
        return cnt

    best_pattern, best_count = max(scored, key=_score)
    return best_pattern


def select_chapter_pattern(
    text: str,
    doc_type: str,
) -> Optional[str]:
    """
    Return the first CHAPTER_PATTERN that fires in the document sample.
    Returns None if the document appears to have no chapter structure.
    """
    sample = text[:8_000]
    for pattern in CHAPTER_PATTERNS:
        if re.search(pattern, sample, re.IGNORECASE):
            return pattern
    return None
