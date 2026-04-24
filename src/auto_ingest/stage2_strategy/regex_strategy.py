"""
stage2_strategy/regex_strategy.py — Doc-type-aware regex pattern selector.

Selects the best section-split pattern for a given document.

Three improvements over the original scorer
-------------------------------------------
1. Doc-type priority pool
   Each doc_type gets a ranked candidate list.  Patterns that are a
   natural fit for the type are tried first and receive a small priority
   bonus (+10) to break ties against equally-matched generic patterns.

2. Feature-guided candidates
   features["has_orders"] = True  →  add RULE_PATTERNS to the pool so
       numbered rules inside CPC-style schedules can be split correctly.
   features["approx_section_count"] (large values) → allow looser patterns.

3. Continuity bonus
   A pattern that extracts section numbers [1, 2, 3, 4 …] is objectively
   better than one that extracts [1, 5, 12, 88].  After counting matches
   the raw count is scaled by a continuity multiplier in [0.60, 1.00]:

       final_score = raw_count × (0.60 + 0.40 × continuity_ratio)

Public API (unchanged):
    select_section_pattern(text, doc_type, features) → str
    select_chapter_pattern(text, doc_type)           → Optional[str]
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple

from ..utils.patterns import SECTION_PATTERNS, CHAPTER_PATTERNS, RULE_PATTERNS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Extra candidate pattern for Rules documents whose rules are written
# as "Rule 1.  Short title." (with the word "Rule" before the number).
# This is common in subordinate legislation (Motor Vehicles Rules etc.)
# ─────────────────────────────────────────────────────────────────────
_RULE_PREFIX_PATTERN = r"(?=\n\s*Rule\s+(\d{1,3}[A-Z]?)\.?\s)"

# Plausible section count window (same as before)
_MIN_SECTIONS = 5
_MAX_SECTIONS = 500

# Sampling size for pattern counting
_SAMPLE_SIZE = 15_000


# ─────────────────────────────────────────────────────────────────────
# Pattern pools per doc_type
# Each list is ordered priority-first (earlier = larger priority bonus).
# ─────────────────────────────────────────────────────────────────────
_POOLS: Dict[str, List[str]] = {
    # Codified acts (CPC, IPC) have [1. / *[1. / 1A. bracket prefixes.
    # The strict pattern (index 0) handles those; try it first.
    "CODIFIED_ACT": [
        SECTION_PATTERNS[0],   # strict — bracket-prefix & capital/paren
        SECTION_PATTERNS[1],   # moderate — any numbered + whitespace
        SECTION_PATTERNS[2],   # loose — [N. bracket-wrapped only
    ],

    # Gazette / Bare Acts have clean "1. Short title." numbering.
    # The moderate pattern (index 1) is the best fit.
    "GAZETTE_ACT": [
        SECTION_PATTERNS[1],
        SECTION_PATTERNS[0],
        SECTION_PATTERNS[2],
    ],
    "BARE_ACT": [
        SECTION_PATTERNS[1],
        SECTION_PATTERNS[0],
        SECTION_PATTERNS[2],
    ],

    # Schedule Rules may use "Rule 1." prefixed format; try that first,
    # then fall back to general section patterns.
    "SCHEDULE_RULES": [
        _RULE_PREFIX_PATTERN,  # "Rule N." style (Motor Vehicles, etc.)
        RULE_PATTERNS[0],      # bare numbered lookahead (same as SP[1] but tighter)
        SECTION_PATTERNS[1],
        SECTION_PATTERNS[0],
    ],

    # Compendium / Generic: use all section patterns in default order.
    "COMPENDIUM": SECTION_PATTERNS[:],
    "GENERIC":    SECTION_PATTERNS[:],
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _build_candidate_pool(
    doc_type: str,
    features: Dict[str, Any],
) -> List[Tuple[str, int]]:
    """
    Return the ordered list of (pattern, priority_bonus) tuples for this
    document.

    Priority bonus is awarded by position in the doc_type pool:
      position 0 → +20, position 1 → +10, everything else → 0.
    This means the preferred pattern must be significantly worse in raw
    count before it loses out to a lower-priority candidate.

    Feature overrides applied after pool construction:
      has_orders=True  → prepend RULE_PATTERNS[0] if not already present.
    """
    pool_patterns = _POOLS.get(doc_type, SECTION_PATTERNS[:])

    # Build (pattern, priority_bonus) pairs
    candidates: List[Tuple[str, int]] = []
    bonuses = [20, 10]
    for i, pat in enumerate(pool_patterns):
        bonus = bonuses[i] if i < len(bonuses) else 0
        candidates.append((pat, bonus))

    # Feature override: has_orders → Rule-pattern gets top priority
    if features.get("has_orders") and RULE_PATTERNS[0] not in pool_patterns:
        candidates.insert(0, (RULE_PATTERNS[0], 20))

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[Tuple[str, int]] = []
    for pat, bonus in candidates:
        if pat not in seen:
            seen.add(pat)
            unique.append((pat, bonus))

    return unique


def _continuity_ratio(matches: List[str]) -> float:
    """
    Given a list of extracted section-number strings (e.g. ['1','2','3A','4']),
    compute what fraction of consecutive pairs are non-decreasing by their
    leading integer.

    Returns 0.5 when there is insufficient data (< 3 matches) — neutral.
    Returns 1.0 when perfectly monotone, 0.0 when perfectly reversed.
    """
    nums: List[int] = []
    for m in matches:
        lead = re.match(r"\d+", m or "")
        if lead:
            nums.append(int(lead.group()))

    if len(nums) < 3:
        return 0.5   # not enough data — neutral

    increases = sum(1 for i in range(1, len(nums)) if nums[i] >= nums[i - 1])
    return increases / (len(nums) - 1)


def _score_pattern(
    pattern: str,
    sample: str,
    priority_bonus: int,
) -> float:
    """
    Compute the final score for a single pattern against the sample text.

    Formula:
        raw_count  = len(findall(pattern, sample))
        continuity = _continuity_ratio(findall result groups)
        base_score = raw_count  if  _MIN_SECTIONS ≤ raw_count ≤ _MAX_SECTIONS
                   = 0          if  raw_count < _MIN_SECTIONS
                   = max(0, _MAX_SECTIONS - raw_count)  if oversplit
        final = base_score × (0.60 + 0.40 × continuity) + priority_bonus
    """
    try:
        matches = re.findall(pattern, sample)
    except re.error as e:
        logger.debug(f"Pattern regex error ({pattern!r}): {e}")
        return 0.0

    count = len(matches)

    if count < _MIN_SECTIONS:
        base = 0.0
    elif count > _MAX_SECTIONS:
        base = float(max(0, _MAX_SECTIONS - count))
    else:
        base = float(count)

    # Continuity bonus — extract the capture group (section numbers)
    continuity = _continuity_ratio(
        [m if isinstance(m, str) else (m[0] if m else "") for m in matches]
    )
    multiplier = 0.60 + 0.40 * continuity

    final = base * multiplier + priority_bonus
    logger.debug(
        f"  pattern={'...' + pattern[-30:]!r:>34s}  "
        f"count={count:4d}  cont={continuity:.2f}  "
        f"pri={priority_bonus:2d}  →  score={final:.1f}"
    )
    return final


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def select_section_pattern(
    text: str,
    doc_type: str,
    features: Dict[str, Any],
) -> str:
    """
    Return the section-split pattern that best matches this document.

    Parameters
    ----------
    text     : str           — cleaned full document text
    doc_type : str           — classification from Stage 1 (e.g. "GAZETTE_ACT")
    features : Dict[str,Any] — structural flags from Stage 1

    Returns
    -------
    str — the winning regex pattern (a lookahead suitable for re.split)
    """
    sample = text[:_SAMPLE_SIZE]
    candidates = _build_candidate_pool(doc_type, features)

    logger.debug(
        f"[regex_strategy] doc_type={doc_type}  "
        f"pool_size={len(candidates)}  "
        f"has_orders={features.get('has_orders')}  "
        f"has_schedules={features.get('has_schedules')}"
    )

    best_pattern = SECTION_PATTERNS[-1]   # safest fallback
    best_score   = -1.0

    for pattern, priority_bonus in candidates:
        score = _score_pattern(pattern, sample, priority_bonus)
        if score > best_score:
            best_score   = score
            best_pattern = pattern

    logger.info(
        f"[regex_strategy] selected pattern for {doc_type}: "
        f"score={best_score:.1f}  "
        f"pattern={'...' + best_pattern[-40:]!r}"
    )
    return best_pattern


def select_chapter_pattern(
    text: str,
    doc_type: str,
) -> Optional[str]:
    """
    Return the first CHAPTER_PATTERN that fires in the document sample.
    Returns None if the document has no chapter structure.

    Uses the first 8 000 chars — after TOC removal this reliably contains
    the first chapter heading for documents that have one.
    """
    sample = text[:8_000]
    for pattern in CHAPTER_PATTERNS:
        if re.search(pattern, sample, re.IGNORECASE):
            return pattern
    return None
