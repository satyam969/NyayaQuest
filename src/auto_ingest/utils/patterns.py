"""
utils/patterns.py — Generic regex patterns for Indian statutory law documents.
Patterns are ordered from most specific to most general.
Not hardcoded to any single document; all patterns are reusable across doc types.

v2 additions (targeted, backward-compatible):
  - Precompiled versions of WINDOW_SCORE_TEMPLATES patterns (COMPILED_WINDOW_TEMPLATES)
  - Weighted marker support: tuple ("pattern", weight) alongside plain strings
  - Improved COMPENDIUM signals
  - All existing constants unchanged
"""

import re
from typing import Any, Dict, List, Tuple, Union

# ─────────────────────────────────────────────────────────────────────
# Section Patterns
# Split the document text into individual section blocks.
# Each is a lookahead (?=...) so re.split() preserves the delimiters.
# ─────────────────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    # Most strict: handles plain "1. ", "[1. ", "*[1. " AND footnote-bracket "1[9. "
    # (the digit+bracket prefix common in CPC and other codified Indian statutes)
    r"(?=\n\s*(?:\d*\[)?\*?\[?(\d{1,3}[A-Z]?)\.(?:\s+[A-Z\[]|\s*\())",
    # Moderate: any numbered section followed by whitespace (with footnote-bracket prefix)
    r"(?=\n\s*(?:\d*\[)?\*?\[?(\d{1,3}[A-Z]?)\.\s)",
    # Loose: bracket-wrapped section numbers  [1.
    r"(?=\n\s*\[(\d{1,3}[A-Z]?)\.\s)",
]

# ─────────────────────────────────────────────────────────────────────
# Chapter Patterns  (capture group 1 = chapter id, group 2 = title)
# ─────────────────────────────────────────────────────────────────────
CHAPTER_PATTERNS = [
    # CHAPTER I\nPRELIMINARY  or  CHAPTER II\nRIGHT TO INFORMATION
    r"(CHAPTER\s+[IVXLCDM]+)\s*\n+(.*?)(?=\n|\Z)",
    # CHAPTER 1 - Title
    r"(CHAPTER\s+\d+)\s*[-\u2013\u2014]\s*(.*?)(?=\n|\Z)",
    # lowercase: Chapter I\nTitle
    r"(?i)(Chapter\s+[IVXLCDM]+)\s*\n+(.*?)(?=\n|\Z)",
]

# ─────────────────────────────────────────────────────────────────────
# Part Patterns  (capture group 1 = part id, group 2 = title)
# ─────────────────────────────────────────────────────────────────────
PART_PATTERNS = [
    r"(PART\s+[IVXLCDM]+)\s*\n+(.*?)(?=\n|\Z)",
    r"(PART\s+[A-Z])\s*[-\u2013\u2014]\s*(.*?)(?=\n|\Z)",
]

# ─────────────────────────────────────────────────────────────────────
# Schedule / Appendix Patterns
# ─────────────────────────────────────────────────────────────────────
SCHEDULE_PATTERNS = [
    r"\n(THE\s+(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\s+SCHEDULE)",
    r"\n(SCHEDULE\s+[IVXLCDM]+)",
    r"\n(SCHEDULE\s+\d+)",
    r"\n(THE\s+SCHEDULE)",
    r"\nAPPENDIX\s+[A-D]\s*\n",
]

# ─────────────────────────────────────────────────────────────────────
# Order / Rule Patterns (CPC First Schedule style)
# ─────────────────────────────────────────────────────────────────────
ORDER_PATTERNS = [
    r"(\[?ORDER\s+[IVXLCDM]+[A-Z]?[\s\S]*?)(?=\n\[?ORDER\s+[IVXLCDM]+[A-Z]?|\Z)",
]

RULE_PATTERNS = [
    r"(?=\n\s*\*?\[?\d+[A-Z]?\.(?:\s+|[A-Z]))",
]

# ─────────────────────────────────────────────────────────────────────
# TOC Detection Patterns
# ─────────────────────────────────────────────────────────────────────
TOC_PATTERNS = [
    r"ARRANGEMENT OF SECTIONS",
    r"TABLE OF CONTENTS",
    r"CONTENTS\s*\n",
    r"SECTIONS\s*\n\s*PAGE",
    r"LIST OF SECTIONS",
]

# ─────────────────────────────────────────────────────────────────────
# Noise / Footer Keyword Patterns
# ─────────────────────────────────────────────────────────────────────
FOOTER_KW_PATTERNS = [
    r"(Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.)",
    r"amended in its application",
    r"extended to .{1,40} by Act",
    r"Extended to the .{1,40} by",
    r"THE GAZETTE OF INDIA",
    r"MINISTRY OF LAW AND JUSTICE",
]

STRUCTURAL_PROFILES: dict[str, Any] = {
    "GAZETTE_ACT": {
        "patterns": [
            r"(?m)^\s*\d{1,3}[A-Z]?\.\s+[A-Z][^\n]{3,}[\.—\-]",
            r"(?m)^\s*\d{1,3}[A-Z]?\.\s+[A-Z]",
            r"(?m)^\s*\d{1,3}[A-Z]?\.[\u2014\u2013—]",
        ],
        "anti_patterns": [
            r"ORDER\s+[IVXLCDM]+",
        ],
    },
    "CODIFIED_ACT": {
        "global_patterns": [
            r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
            r"Code of Civil Procedure",
            r"THE FIRST SCHEDULE",
        ],
        "global_bonus": 0.6,
        "patterns": [
            r"(?m)^\s*\*?\[?\d{1,3}[A-Z]?\.(?:\s+[A-Z]|\s*\[)",
            r"(?m)^\s*\*?\[?\d{1,3}[A-Z]?\.\s",
        ],
        "anti_patterns": [],
    },
    "SCHEDULE_RULES": {
        "patterns": [
            r"(?m)^ORDER\s+[IVXLCDM]+[A-Z]?",
            r"(?m)^\[?ORDER\s+[IVXLCDM]+",
        ],
        "anti_patterns": [],
    },
    "BARE_ACT": {
        "patterns": [
            r"(?m)^\d{1,3}[A-Z]?\.\s+[A-Z][a-z]",
            r"(?m)^\d{1,3}[A-Z]?\.\s*\(",
        ],
        "anti_patterns": [
            r"(?m)\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
        ],
    },
    "COMPENDIUM": {
        "patterns": [
            r"(?m)CHAPTER\s+I\s*\n\s*PRELIMINARY",
            r"(?m)1\.\s*\(1\)\s*This Act may be called",
        ],
        "anti_patterns": [],
        "min_occurrences": 2,
    },
}

# ─────────────────────────────────────────────────────────────────────
# Window Score Templates
# Used exclusively by the semantic multi-window classifier in detector.py.
# Each template has four lists of markers:
#   title_markers     — strong identity signals in the title/cover region
#   structure_markers — density signals counted across the body
#   tail_markers      — signals expected near the document tail
#   anti_markers      — presence penalises this type
#
# Markers may be plain strings OR ("pattern", weight) tuples.
# Weight defaults to 1.0 when a plain string is used.
# Detector code that uses this should call get_marker_weight() to
# read weights; existing callers that just iterate strings are
# unaffected if they only use the first element of tuples.
#
# NOTE: STRUCTURAL_PROFILES (above) is NOT modified; parser stages still
#       import it directly for section-pattern selection.
# ─────────────────────────────────────────────────────────────────────

# Type alias for weighted marker entries
MarkerEntry = Union[str, Tuple[str, float]]


def get_marker_pattern(entry: MarkerEntry) -> str:
    """Return the regex pattern string from a plain or weighted marker entry."""
    return entry[0] if isinstance(entry, tuple) else entry


def get_marker_weight(entry: MarkerEntry) -> float:
    """Return the weight for a marker entry (default 1.0 for plain strings)."""
    return float(entry[1]) if isinstance(entry, tuple) else 1.0


WINDOW_SCORE_TEMPLATES: dict[str, dict] = {

    # ── Gazette Act ───────────────────────────────────────────────────
    # Published in the Gazette of India; enacted by Parliament.
    # Diagnostic: Gazette header, Ministry block, "An Act to" preamble.
    "GAZETTE_ACT": {
        "title_markers": [
            (r"THE\s+GAZETTE\s+OF\s+INDIA", 2.0),        # very strong signal
            r"MINISTRY\s+OF\s+LAW\s+AND\s+JUSTICE",
            r"MINISTRY\s+OF\s+(LABOUR|FINANCE|HOME|COMMERCE|HEALTH)",
            r"An\s+Act\s+to\b",
            r"No\.\s*\d+\s+of\s+\d{4}",
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}",
            r"Be\s+it\s+enacted\s+by",
            r"EXTRAORDINARY",
            r"PART\s+II[—\-]Section\s+1",
        ],
        "structure_markers": [
            r"(?m)^\s*\d{1,3}[A-Z]?\.\s+[A-Z]",
            r"(?m)^\s*\d{1,3}[A-Z]?\.\s*\(",
            r"(?m)^\s*\(\s*\d+\s*\)",
            r"(?m)^CHAPTER\s+[IVXLCDM]+",
            r"(?m)^PART\s+[IVXLCDM]+",
            r"\bshall\s+mean\b",
            r"\bnotwithstanding\b",
            r"\bprovided\s+that\b",
        ],
        "tail_markers": [
            r"(?m)^THE\s+SCHEDULE",
            r"(?m)^SCHEDULE\s+[IVXLCDM\d]",
            r"Statement\s+of\s+Objects\s+and\s+Reasons",
            r"(?m)^\s*\d{1,3}[A-Z]?\.\s+[A-Z]",
        ],
        "anti_markers": [
            r"(?m)^ORDER\s+[IVXLCDM]+",
            r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
        ],
    },

    # ── Codified Act ──────────────────────────────────────────────────
    # Professionally codified edition with amendment annotations inline.
    # Diagnostic: [Ins. by / Subs. by / Omitted] footnote markers.
    "CODIFIED_ACT": {
        "title_markers": [
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+(?:ACT|CODE),?\s+\d{4}",
            r"Code\s+of\s+Civil\s+Procedure",
            r"Indian\s+Penal\s+Code",
            r"Civil\s+Procedure\s+Code",
            r"Code\s+of\s+Criminal\s+Procedure",
        ],
        "structure_markers": [
            (r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by", 1.5),   # very strong codified signal
            r"(?m)^\s*\*?\[?\d{1,3}[A-Z]?\.\s",
            r"(?m)^CHAPTER\s+[IVXLCDM]+",
            r"(?m)^PART\s+[IVXLCDM]+",
            r"w\.e\.f\.",
            r"(?m)^\s*\(\s*\d+\s*\)",
            r"A\.O\.\s+\d{4}",
        ],
        "tail_markers": [
            (r"(?m)^THE\s+FIRST\s+SCHEDULE", 1.5),  # uniquely strong for codified
            r"(?m)^THE\s+SECOND\s+SCHEDULE",
            (r"(?m)^ORDER\s+[IVXLCDM]+", 1.5),      # CPC Orders in tail
            r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
        ],
        "anti_markers": [
            r"THE\s+GAZETTE\s+OF\s+INDIA",
        ],
    },

    # ── Schedule Rules ────────────────────────────────────────────────
    # Subordinate legislation: rules, orders, regulations, forms.
    # Diagnostic: Rule numbering, FORM headers, dense schedule tail.
    "SCHEDULE_RULES": {
        "title_markers": [
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+RULES,?\s+\d{4}",
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+REGULATIONS,?\s+\d{4}",
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+ORDERS?,?\s+\d{4}",
            (r"In\s+exercise\s+of\s+the\s+powers?\s+conferred", 1.8),  # near-unique signal
            (r"(?m)^ORDER\s+[IVXLCDM]+[A-Z]?", 1.5),
            r"(?m)^\[?ORDER\s+[IVXLCDM]+",
        ],
        "structure_markers": [
            r"(?m)^Rule\s+\d+",
            r"(?m)^\d+\.\s+[A-Z][a-z]",
            r"(?m)^Order\s+[IVXLCDM]+",
            r"(?m)^Rule\s+[IVXLCDM]+",
            r"(?m)^FORM\s+[A-Z\d]",
            r"(?m)^Schedule\s+[IVXLCDM\d]",
            r"(?m)^SCHEDULE\s+[IVXLCDM\d]",
            r"(?m)^\s*\(\s*[a-z]\s*\)",
        ],
        "tail_markers": [
            r"(?m)^FORM\s+[A-Z\d]",
            r"(?m)^SCHEDULE",
            r"(?m)^ANNEXURE",
            r"(?m)^APPENDIX",
            r"(?m)^ORDER\s+[IVXLCDM]+",
            r"(?m)^Rule\s+\d+",
        ],
        "anti_markers": [
            r"THE\s+GAZETTE\s+OF\s+INDIA",
            r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
        ],
    },

    # ── Bare Act ──────────────────────────────────────────────────────
    # Clean statutory text without Gazette headers or amendment annotations.
    # Diagnostic: Numbered sections, chapter hierarchy, no annotation noise.
    "BARE_ACT": {
        "title_markers": [
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}",
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+CODE,?\s+\d{4}",
            r"An\s+Act\s+to\b",
            r"Be\s+it\s+enacted\s+by",
            r"No\.\s*\d+\s+of\s+\d{4}",
        ],
        "structure_markers": [
            r"(?m)^\d{1,3}[A-Z]?\.\s+[A-Z][a-z]",
            r"(?m)^\d{1,3}[A-Z]?\.\s*\(",
            r"(?m)^CHAPTER\s+[IVXLCDM]+",
            r"(?m)^PART\s+[IVXLCDM]+",
            r"(?m)^\s*\(\s*\d+\s*\)",
            r"\bshall\s+mean\b",
            r"\bnotwithstanding\b",
        ],
        "tail_markers": [
            r"(?m)^THE\s+SCHEDULE",
            r"(?m)^SCHEDULE\s+[IVXLCDM\d]",
            r"(?m)^\d{1,3}[A-Z]?\.\s+[A-Z][a-z]",
        ],
        "anti_markers": [
            r"THE\s+GAZETTE\s+OF\s+INDIA",
            r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",
            r"(?m)^ORDER\s+[IVXLCDM]+",
        ],
    },

    # ── Compendium ────────────────────────────────────────────────────
    # Multiple independent Acts bound together.
    # Improved signals: look for repeated preambles and multiple Act titles.
    "COMPENDIUM": {
        "title_markers": [
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+(?:ACT|CODE),?\s+\d{4}",
            r"CHAPTER\s+I\s*\n\s*PRELIMINARY",
        ],
        "structure_markers": [
            (r"CHAPTER\s+I\s*\n\s*PRELIMINARY", 2.0),           # repeated = strong compendium signal
            (r"(?m)^1\.\s*\(1\)\s*This\s+(?:Act|Code)\s+may\s+be\s+called", 2.0),
            r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}",      # multiple act titles
            r"Short\s+title[,\.]?\s+extent\s+and\s+commencement",
        ],
        "tail_markers": [
            (r"CHAPTER\s+I\s*\n\s*PRELIMINARY", 2.0),           # appears again near tail
            r"(?m)^1\.\s*\(1\)\s*This\s+(?:Act|Code)\s+may\s+be\s+called",
        ],
        "anti_markers": [],
    },
}


# ─────────────────────────────────────────────────────────────────────
# Precompiled window template patterns (performance optimisation)
# ─────────────────────────────────────────────────────────────────────
# Maps doc_type -> window_type -> list of (compiled_pattern, weight) tuples
# Detector code should use COMPILED_WINDOW_TEMPLATES instead of compiling
# strings from WINDOW_SCORE_TEMPLATES on every PDF call.
#
# Usage:
#   for compiled_pat, weight in COMPILED_WINDOW_TEMPLATES["GAZETTE_ACT"]["title_markers"]:
#       if compiled_pat.search(text):
#           score += weight
# ─────────────────────────────────────────────────────────────────────

def _compile_window_templates(
    templates: Dict[str, Dict[str, List[MarkerEntry]]],
) -> Dict[str, Dict[str, List[Tuple[re.Pattern, float]]]]:
    """
    Pre-compile all WINDOW_SCORE_TEMPLATES patterns at module load time.
    Returns a nested dict: doc_type → window_type → [(compiled, weight), ...]
    """
    compiled: Dict[str, Dict[str, List[Tuple[re.Pattern, float]]]] = {}
    for doc_type, windows in templates.items():
        compiled[doc_type] = {}
        for window_name, markers in windows.items():
            compiled_markers: List[Tuple[re.Pattern, float]] = []
            for entry in markers:
                pattern = get_marker_pattern(entry)
                weight  = get_marker_weight(entry)
                try:
                    flags = re.IGNORECASE | re.MULTILINE
                    # Patterns that already embed (?m) or (?i) flags don't need
                    # re.MULTILINE / re.IGNORECASE added again — re handles it.
                    compiled_markers.append((re.compile(pattern, flags), weight))
                except re.error:
                    # Fallback: add as non-compiled sentinel so callers can skip
                    pass
            compiled[doc_type][window_name] = compiled_markers
    return compiled


COMPILED_WINDOW_TEMPLATES = _compile_window_templates(WINDOW_SCORE_TEMPLATES)
