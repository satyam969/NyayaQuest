"""
utils/scoring.py — Quality scoring utilities for chunk evaluation.

Five metrics are computed and combined with configurable weights.
Overall score >= 0.70 means the parse quality is acceptable.

v2 additions (targeted, backward-compatible):
  - Giant chunk penalty applied inside score_chunk_length_sanity()
  - Duplicate chunk prefix detection penalises score
  - score_section_capture_rate() validates realistic legal IDs
  - score_section_continuity() resets sequence on chapter/part/order transitions
  - compute_quality_score() adds metrics["quality_band"]
  - All existing function signatures and metrics["overall"] unchanged
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────
# Legal ID validator
# ─────────────────────────────────────────────────────────────────────

# Pattern for realistic legal section / rule identifiers
# Accepts: "1", "12", "100", "1A", "12B", "100C", "Rule 3", "ORDER I RULE 2"
_LEGAL_ID_RE = re.compile(
    r"^(\d{1,3}[A-Z]?|Rule\s+\d+[A-Z]?|ORDER\s+[IVXLCDM]+\s+RULE\s+\d+)$",
    re.IGNORECASE,
)

# Junk values that are not real section IDs
_JUNK_IDS = frozenset({"N/A", "", "None", "Unknown", "0", "na", "n/a"})


def _is_valid_legal_id(value: Optional[str]) -> bool:
    """Return True if value looks like a real legal section/rule identifier."""
    if not value:
        return False
    v = str(value).strip()
    if v in _JUNK_IDS:
        return False
    return bool(_LEGAL_ID_RE.match(v))


# ─────────────────────────────────────────────────────────────────────
# Individual metric functions
# Each accepts the full list of chunk dicts and returns a float in [0, 1].
# ─────────────────────────────────────────────────────────────────────

def score_section_capture_rate(chunks: List[Dict[str, Any]], expected_min: int = 5) -> float:
    """
    Fraction of chunks that carry a valid, realistic legal section/rule number.

    v2: validates against _LEGAL_ID_RE instead of just checking != "N/A".
    """
    if not chunks:
        return 0.0
    valid = sum(
        1 for c in chunks
        if _is_valid_legal_id(
            c.get("metadata", {}).get(
                "section_number",
                c.get("metadata", {}).get("rule", "N/A")
            )
        )
    )
    rate = valid / len(chunks)
    # Penalise when absolute count is suspiciously low
    if valid < expected_min:
        rate *= 0.5
    return min(rate, 1.0)


def score_chunk_length_sanity(
    chunks:  List[Dict[str, Any]],
    min_len: int = 50,
    max_len: int = 3000,
) -> float:
    """
    Fraction of chunks whose text length is within sane bounds.

    v2: applies graduated giant-chunk penalties:
      - Any chunk > 8 000 chars  → 0.4 deduction from final score
      - Any chunk > 15 000 chars → 0.7 deduction from final score
    Penalty is applied to the sane-fraction score, not to overall directly,
    so the metric stays in [0, 1].
    """
    if not chunks:
        return 0.0

    sane = 0
    has_giant  = False
    has_megachunk = False

    for c in chunks:
        length = len(c.get("text", ""))
        if min_len <= length <= max_len:
            sane += 1
        if length > 15_000:
            has_megachunk = True
        elif length > 8_000:
            has_giant = True

    base = sane / len(chunks)

    if has_megachunk:
        base = max(0.0, base - 0.70)
    elif has_giant:
        base = max(0.0, base - 0.40)

    return base


def score_noise_ratio(chunks: List[Dict[str, Any]]) -> float:
    """1 - noise_fraction.  Noise = page-number lines, separator lines, micro-chunks."""
    if not chunks:
        return 0.0
    noise_patterns = [
        r"^\s*\d{1,4}\s*$",      # standalone page numbers
        r"^[\s_\-=]+$",           # separator / blank lines
        r"^.{1,25}$",             # extremely short strings
    ]
    noise_count = sum(
        1 for c in chunks
        if any(re.search(p, c.get("text", "").strip()) for p in noise_patterns)
    )
    return 1.0 - (noise_count / len(chunks))


def score_chapter_coverage(chunks: List[Dict[str, Any]]) -> float:
    """Fraction of chunks whose chapter metadata is not 'Unknown'."""
    if not chunks:
        return 0.0
    known = sum(
        1 for c in chunks
        if "Unknown" not in str(c.get("metadata", {}).get("chapter", "Unknown"))
    )
    return known / len(chunks)


# ── Continuity sequence boundary detection ───────────────────────────

def _is_new_sequence(prev_meta: Dict[str, Any], curr_meta: Dict[str, Any]) -> bool:
    """
    Return True when a legitimate numbering restart is expected.
    Triggers on chapter, part, or ORDER transitions.
    """
    for key in ("chapter", "part"):
        prev_val = str(prev_meta.get(key, ""))
        curr_val = str(curr_meta.get(key, ""))
        if prev_val and curr_val and prev_val != curr_val:
            return True
    # ORDER transition (SCHEDULE_RULES / CPC-style docs)
    prev_strat = str(prev_meta.get("hierarchy_path", ""))
    curr_strat = str(curr_meta.get("hierarchy_path", ""))
    if "ORDER" in prev_strat and "ORDER" in curr_strat and prev_strat != curr_strat:
        return True
    return False


def score_section_continuity(chunks: List[Dict[str, Any]]) -> float:
    """
    Checks that section/rule numbers are broadly monotonically increasing.
    Penalises jumbled output caused by wrong split points.

    v2: resets the running sequence when chapter, part, or ORDER transitions
    are detected, since legal numbering legitimately restarts at such boundaries.
    """
    if not chunks:
        return 0.5  # neutral

    sequence: List[int] = []
    prev_meta: Dict[str, Any] = {}
    increases = 0
    total_comparisons = 0

    for c in chunks:
        meta = c.get("metadata", {})
        sec  = str(meta.get("section_number", meta.get("rule", "")))
        m    = re.match(r"^(\d+)", sec)

        if m:
            num = int(m.group(1))
            # If this chunk starts a new structural sequence, reset
            if sequence and _is_new_sequence(prev_meta, meta):
                sequence = []

            if sequence:
                if num >= sequence[-1]:
                    increases += 1
                total_comparisons += 1

            sequence.append(num)

        prev_meta = meta

    if total_comparisons == 0:
        return 0.5  # too few data points — neutral

    return increases / total_comparisons


# ─────────────────────────────────────────────────────────────────────
# Duplicate chunk penalty helper
# ─────────────────────────────────────────────────────────────────────

def _normalise_prefix(text: str, length: int = 80) -> str:
    """Normalise a text prefix for duplicate detection."""
    t = unicodedata.normalize("NFKD", text[:length]).lower()
    return re.sub(r"\s+", " ", t).strip()


def _score_duplicate_penalty(chunks: List[Dict[str, Any]]) -> float:
    """
    Return a penalty (0.0 = no penalty, up to 0.5) based on the fraction
    of chunks that share an identical normalised text prefix.
    Repeated headers / fallback boilerplate inflates this.
    """
    if len(chunks) < 4:
        return 0.0

    prefixes: List[str] = [_normalise_prefix(c.get("text", "")) for c in chunks]
    from collections import Counter
    counts = Counter(prefixes)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    dup_fraction = duplicate_count / len(chunks)

    # Scale: 10% duplicates → 0.05 penalty, 50%+ → 0.25 penalty
    return min(dup_fraction * 0.5, 0.25)


# ─────────────────────────────────────────────────────────────────────
# Quality band helper
# ─────────────────────────────────────────────────────────────────────

def _quality_band(overall: float) -> str:
    if overall >= 0.85:
        return "excellent"
    if overall >= 0.70:
        return "pass"
    if overall >= 0.60:
        return "near_pass"
    if overall >= 0.45:
        return "weak"
    return "fail"


# ─────────────────────────────────────────────────────────────────────
# Composite scorer
# ─────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "section_capture_rate": 0.30,
    "chapter_coverage":     0.20,
    "chunk_length_sanity":  0.25,
    "noise_ratio":          0.15,
    "section_continuity":   0.10,
}


def compute_quality_score(
    chunks:  List[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Compute per-metric scores and a final weighted overall score.

    Returns a dict like:
    {
        "section_capture_rate": 0.85,
        "chapter_coverage": 0.72,
        "chunk_length_sanity": 0.90,
        "noise_ratio": 0.95,
        "section_continuity": 0.80,
        "overall": 0.86,
        "quality_band": "excellent",   ← new in v2
    }

    overall and all per-metric keys are unchanged — backward compatible.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    metrics: Dict[str, float] = {
        "section_capture_rate": score_section_capture_rate(chunks),
        "chapter_coverage":     score_chapter_coverage(chunks),
        "chunk_length_sanity":  score_chunk_length_sanity(chunks),
        "noise_ratio":          score_noise_ratio(chunks),
        "section_continuity":   score_section_continuity(chunks),
    }

    weighted_sum = sum(metrics[k] * w.get(k, 0.0) for k in metrics)
    # Apply duplicate penalty to overall (does not touch individual metrics)
    dup_penalty   = _score_duplicate_penalty(chunks)
    overall       = max(0.0, min(1.0, weighted_sum - dup_penalty))

    metrics["overall"]      = overall
    metrics["quality_band"] = _quality_band(overall)
    return metrics
