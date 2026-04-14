"""
stage1_detection/segmenter.py — Multi-act PDF segmenter.

Some PDFs (e.g. the Labour Codes Compendium) bundle several statutes in one
file.  This module detects act boundaries and splits the text into separate
segments, each processed independently by the pipeline.

If only one act is found the original text is returned as a single segment.
"""

import re
from typing import List, Dict, Any


# Pattern that starts a new statutory act (e.g. "THE WAGES CODE, 2019")
_ACT_BOUNDARY_RE = re.compile(
    r"(?=\nTHE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}\s*\n)",
    re.IGNORECASE,
)


def segment_document(text: str) -> List[Dict[str, Any]]:
    """
    Split text into independent act segments.

    Returns
    -------
    list of dicts:  {title: str, text: str, segment_index: int}
    """
    boundaries = [m.start() for m in _ACT_BOUNDARY_RE.finditer(text)]

    if len(boundaries) <= 1:
        return [
            {
                "title": _extract_title(text),
                "text": text,
                "segment_index": 0,
            }
        ]

    segments: List[Dict[str, Any]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        seg_text = text[start:end].strip()
        segments.append(
            {
                "title": _extract_title(seg_text),
                "text": seg_text,
                "segment_index": i,
            }
        )

    return segments


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_title(text: str) -> str:
    """Best-effort title extraction from the first 2 000 characters."""
    # "THE [NAME] ACT, YEAR"
    m = re.search(
        r"THE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}", text[:2000], re.IGNORECASE
    )
    if m:
        return m.group(0).strip()

    # First non-empty line
    for line in text[:500].splitlines():
        line = line.strip()
        if len(line) > 10:
            return line

    return "Unknown Document"
