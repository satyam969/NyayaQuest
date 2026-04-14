"""
stage2_strategy/hybrid_selector.py — Strategy selector.

Decides which parsing path to take based on detection confidence:

  confidence ≥ 0.60 → REGEX   (deterministic, fast)
  0.30 ≤ conf < 0.60 → HYBRID  (regex primary + optional LLM schema assist)
  confidence < 0.30  → SCHEMA  (full LLM-schema driven parse)
"""

from enum import Enum
from typing import Dict, Any


class Strategy(str, Enum):
    REGEX  = "regex"
    HYBRID = "hybrid"
    SCHEMA = "schema"


_REGEX_THRESHOLD  = 0.60
_HYBRID_THRESHOLD = 0.30


def select_strategy(confidence: float, features: Dict[str, Any]) -> Strategy:
    """
    Return the appropriate parsing strategy.

    Parameters
    ----------
    confidence : float  — doc-type detection confidence from Stage 1
    features   : dict   — structural features (used for future overrides)
    """
    if confidence >= _REGEX_THRESHOLD:
        return Strategy.REGEX
    if confidence >= _HYBRID_THRESHOLD:
        return Strategy.HYBRID
    return Strategy.SCHEMA
