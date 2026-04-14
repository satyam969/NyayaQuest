"""
utils/scoring.py — Quality scoring utilities for chunk evaluation.

Five metrics are computed and combined with configurable weights.
Overall score ≥ 0.70 means the parse quality is acceptable.
"""

import re
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────
# Individual metric functions
# Each accepts the full list of chunk dicts and returns a float in [0, 1].
# ─────────────────────────────────────────────────────────────────────

def score_section_capture_rate(chunks: List[Dict[str, Any]], expected_min: int = 5) -> float:
    """Fraction of chunks that carry a valid (non-N/A) section number."""
    if not chunks:
        return 0.0
    valid = sum(
        1 for c in chunks
        if c.get("metadata", {}).get("section_number", "N/A")
        not in ("N/A", "", None, "Unknown")
    )
    rate = valid / len(chunks)
    # Penalise when absolute count is suspiciously low
    if valid < expected_min:
        rate *= 0.5
    return min(rate, 1.0)


def score_chunk_length_sanity(
    chunks: List[Dict[str, Any]],
    min_len: int = 50,
    max_len: int = 3000,
) -> float:
    """Fraction of chunks whose text length is within sane bounds."""
    if not chunks:
        return 0.0
    sane = sum(
        1 for c in chunks
        if min_len <= len(c.get("text", "")) <= max_len
    )
    return sane / len(chunks)


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


def score_section_continuity(chunks: List[Dict[str, Any]]) -> float:
    """
    Checks that section numbers are broadly monotonically increasing.
    Penalises jumbled output caused by wrong split points.
    """
    nums: List[int] = []
    for c in chunks:
        sec = str(c.get("metadata", {}).get("section_number", ""))
        m = re.match(r"^(\d+)", sec)
        if m:
            nums.append(int(m.group(1)))

    if len(nums) < 2:
        return 0.5  # too few data points — neutral

    increases = sum(1 for i in range(1, len(nums)) if nums[i] >= nums[i - 1])
    return increases / (len(nums) - 1)


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
    chunks: List[Dict[str, Any]],
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
    }
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    metrics: Dict[str, float] = {
        "section_capture_rate": score_section_capture_rate(chunks),
        "chapter_coverage":     score_chapter_coverage(chunks),
        "chunk_length_sanity":  score_chunk_length_sanity(chunks),
        "noise_ratio":          score_noise_ratio(chunks),
        "section_continuity":   score_section_continuity(chunks),
    }
    metrics["overall"] = sum(metrics[k] * w.get(k, 0.0) for k in metrics if k != "overall")
    return metrics
