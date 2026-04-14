"""
utils/patterns.py — Generic regex patterns for Indian statutory law documents.
Patterns are ordered from most specific to most general.
Not hardcoded to any single document; all patterns are reusable across doc types.
"""

# ─────────────────────────────────────────────────────────────────────
# Section Patterns
# Split the document text into individual section blocks.
# Each is a lookahead (?=...) so re.split() preserves the delimiters.
# ─────────────────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    # Most strict: numbered section followed by capital letter or opening paren
    # Handles: \n 1. Title  /  \n 1A. Title  /  \n *[15A. Title
    r"(?=\n\s*\*?\[?(\d{1,3}[A-Z]?)\.(?:\s+[A-Z\[]|\s*\())",
    # Moderate: numbered section followed by any whitespace
    r"(?=\n\s*(\d{1,3}[A-Z]?)\.\s)",
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

# ─────────────────────────────────────────────────────────────────────
# Document-type Signature Patterns
# Each type maps to a list of regex patterns whose presence indicates
# the document is of that type.
# Confidence = (hits / len(patterns)) for the winning type.
# ─────────────────────────────────────────────────────────────────────
DOC_TYPE_SIGNATURES: dict[str, list[str]] = {
    "CPC": [
        r"Code of Civil Procedure",
        r"ORDER\s+[IVXLCDM]+",
        r"Civil Procedure",
        r"\bdecree\b",
        r"\bexecution\b",
        r"\bplaint\b",
    ],
    "CrPC": [
        r"Code of Criminal Procedure",
        r"Criminal Procedure",
        r"\bMagistrate\b",
        r"\bcognizance\b",
        r"\bbail\b",
        r"\bFirst Information Report\b",
    ],
    "IPC": [
        r"Indian Penal Code",
        r"Penal Code",
        r"\bpunishment\b",
        r"\boffence\b",
        r"\bimprisonment\b",
        r"\bfine\b",
    ],
    "BNS": [
        r"Bharatiya Nyaya Sanhita",
        r"\bSanhita\b",
        r"BNS\b",
    ],
    "RTI": [
        r"Right to Information",
        r"\bRTI\b",
        r"Public Information Officer",
        r"Central Information Commission",
        r"\btransparency\b",
    ],
    "CPA": [
        r"Consumer Protection",
        r"consumer dispute",
        r"deficiency in service",
        r"\bNCDRC\b",
        r"District Commission",
        r"unfair trade practice",
    ],
    "LABOUR": [
        r"Labour Code",
        r"Code on Wages",
        r"Industrial Relations",
        r"Social Security",
        r"Occupational Safety",
        r"\bworker\b",
        r"\bemployer\b",
        r"\bwage\b",
    ],
    "GENERIC": [],  # catch-all — always scores 0
}
