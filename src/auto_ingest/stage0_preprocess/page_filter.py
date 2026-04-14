"""
stage0_preprocess/page_filter.py — Page-level filtering.

Removes:
  - Table of Contents / Arrangement of Sections
  - Blank pages and standalone page numbers
  - Repeated running headers (e.g. act title printed on every page)

Returns clean body text ready for structural detection.
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Content-start anchors — ordered from most to least reliable.
# The first one that fires determines where real content begins.
# ─────────────────────────────────────────────────────────────────────

_CONTENT_START_ANCHORS = [
    # CHAPTER I + PRELIMINARY (skipping the TOC SECTIONS part)
    r"\nCHAPTER\s+[IVXLCDM]+\s*\n+PRELIMINARY\s*\n(?!\s*SECTIONS)",
    # Real PRELIMINARY block (if no CHAPTER I is above it)
    r"\nPRELIMINARY\s*\n(?!\s*SECTIONS)",
    # CHAPTER + Section 1
    r"\nCHAPTER\s+[IVXLCDM]+\s*\n+1\.\s+Short title",
    # Standard Section 1 opening
    r"\n\s*1\.\s*\(1\)\s*This Act may be called",
    r"\n\s*1\.\s+Short title",
    # Broad preamble opener
    r"\bWhereas\s+it\s+is\s+expedient\b",
]

_TOC_MARKERS = [
    r"ARRANGEMENT OF SECTIONS",
    r"TABLE OF CONTENTS",
    r"CONTENTS\s*\n\s*SECTIONS",
    r"LIST OF SECTIONS",
]


def remove_toc(text: str) -> str:
    """
    Strip the Table of Contents and return text starting from real content.

    Strategy (in order):
    1. Try every content-start anchor and return from the first hit.
    2. Look for a second occurrence of "1. Short title" (first is in TOC).
    3. Find a TOC marker, then seek the next CHAPTER or section 1 after it.
    4. Return original text if nothing fires.
    """
    # Strategy 1: content-start anchors
    for anchor in _CONTENT_START_ANCHORS:
        m = re.search(anchor, text, re.IGNORECASE)
        if m:
            return text[m.start():]

    # Strategy 2: second occurrence of "1. Short title"
    hits = list(re.finditer(r"\n\s*1\.\s+Short title", text, re.IGNORECASE))
    if len(hits) >= 2:
        start_idx = hits[1].start()
        lookbehind = text[max(0, start_idx - 150):start_idx]
        chap_match = list(re.finditer(r"\nCHAPTER\s+[IVXLCDM]+\s*\n+(?:PRELIMINARY\s*\n+)?", lookbehind, re.IGNORECASE))
        if chap_match:
            return text[max(0, start_idx - 150) + chap_match[-1].start():]
        pre_match = list(re.finditer(r"\nPRELIMINARY\s*\n+", lookbehind, re.IGNORECASE))
        if pre_match:
            return text[max(0, start_idx - 150) + pre_match[-1].start():]
        return text[start_idx:]

    # Strategy 3: skip past any TOC block
    for marker in _TOC_MARKERS:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            after = text[m.end():]
            real = re.search(r"\nCHAPTER\s+I|\n\s*1\.\s", after, re.IGNORECASE)
            if real:
                return after[real.start():]

    return text  # nothing matched — return as-is


def filter_blank_pages(text: str) -> str:
    """
    Collapse regions consisting of only whitespace or standalone page numbers.
    """
    # Remove bare page-number lines
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Collapse 3+ consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def remove_page_headers(
    text: str,
    doc_title: Optional[str] = None,
) -> str:
    """
    Remove repeated running headers (document title printed on every page).
    Only removes when doc_title is provided; does nothing otherwise.
    """
    if doc_title:
        escaped = re.escape(doc_title)
        text = re.sub(escaped, "", text, flags=re.IGNORECASE)
    return text
