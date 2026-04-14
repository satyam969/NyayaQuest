"""
stage3_parsing/regex_parser.py — Regex-based stateful parser.

Pipeline:
  1. Split text into raw section blocks using the selected pattern.
  2. Walk blocks in order, tracking current_chapter / current_part via
     a lightweight state machine.
  3. Extract section_number + section_title from each block.
  4. Return a list of chunk dicts ready for quality evaluation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..utils.patterns import CHAPTER_PATTERNS, PART_PATTERNS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Text splitting
# ─────────────────────────────────────────────────────────────────────

def split_by_pattern(text: str, pattern: str) -> List[str]:
    """Split text with a lookahead pattern; return non-empty parts."""
    try:
        parts = re.split(pattern, text)
        return [p for p in parts if p.strip()]
    except re.error as e:
        logger.error(f"Regex split error (pattern={pattern!r}): {e}")
        return [text]


# ─────────────────────────────────────────────────────────────────────
# State machine helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_heading(
    text: str,
    patterns: List[str],
    current: str,
) -> Tuple[str, str]:
    """
    Detect a structural heading (chapter or part) in text.
    Returns (heading_string, text_with_heading_removed).
    """
    for pattern in patterns:
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) >= 2:
                    heading = f"{groups[0].strip()} - {groups[1].strip()}"
                elif groups:
                    heading = groups[0].strip()
                else:
                    heading = m.group(0).strip()
                cleaned = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
                return heading, cleaned
        except re.error:
            continue
    return current, text


def extract_chapter(text: str, current: str = "Unknown Chapter") -> Tuple[str, str]:
    return _extract_heading(text, CHAPTER_PATTERNS, current)


def extract_part(text: str, current: str = "Unknown Part") -> Tuple[str, str]:
    return _extract_heading(text, PART_PATTERNS, current)


# ─────────────────────────────────────────────────────────────────────
# Section metadata extraction
# ─────────────────────────────────────────────────────────────────────

_SEC_HEAD_PATTERNS = [
    # "1. Title.—(1) content" (em-dash variants)
    r"^(\d+[A-Z]?)\.\s*(.*?)(?:\.[\u2014\u2013]|\.[-]|\.\s*\n)(.*)",
    # "1. Title\ncontent"
    r"^(\d+[A-Z]?)\.\s*([^\n(]{0,200})\n(.*)",
    # "1. (1) content" (no separate title line)
    r"^(\d+[A-Z]?)\.\s*(\(.*)",
    # "[1. Title" — bracket-wrapped
    r"^\[(\d+[A-Z]?)\.\s*(.*)",
    # "*[1A. Title" — asterisk + bracket
    r"^\*\[(\d+[A-Z]?)\.\s*(.*)",
]


def extract_section_meta(block: str) -> Optional[Dict[str, Any]]:
    """
    Extract section_number, section_title, and content from a raw block.
    Returns None when the block does not look like a numbered section.
    """
    block = block.strip()

    for pat in _SEC_HEAD_PATTERNS:
        m = re.match(pat, block, re.DOTALL)
        if not m:
            continue

        groups = m.groups()
        sec_num = groups[0]

        if len(groups) == 3:
            sec_title = groups[1].strip()[:200]
            content = groups[2].strip()
        elif len(groups) == 2:
            raw = groups[1].strip()
            lines = raw.split("\n", 1)
            sec_title = lines[0].strip()[:200]
            content = lines[1].strip() if len(lines) > 1 else raw
        else:
            sec_title = ""
            content = block

        return {
            "section_number": sec_num,
            "section_title": sec_title,
            "content": content,
        }

    return None


# ─────────────────────────────────────────────────────────────────────
# Main parse function
# ─────────────────────────────────────────────────────────────────────

def parse_with_regex(
    text: str,
    section_pattern: str,
    doc_title: str = "Unknown",
    doc_type: str = "GENERIC",
    features: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Full regex parse → list of chunk dicts.

    Each chunk dict:
    {
      "text":     str,   # rich text with embedded header
      "metadata": dict   # section_number, chapter, part, …
    }
    """
    if features is None:
        features = {}

    raw_blocks = split_by_pattern(text, section_pattern)
    logger.info(f"[regex_parser] {len(raw_blocks)} raw blocks from pattern")

    chunks: List[Dict[str, Any]] = []
    current_chapter = "Unknown Chapter"
    current_part = "Unknown Part"
    chunk_index = 0

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Update context from structural headings embedded in the block
        current_chapter, block = extract_chapter(block, current_chapter)
        current_part, block = extract_part(block, current_part)

        if not block.strip():
            continue

        meta = extract_section_meta(block)

        if meta and len(meta["content"]) >= 30:
            chunk_index += 1
            chunks.append(
                {
                    "text": (
                        f"[{doc_title}] [{current_chapter}] "
                        f"Section {meta['section_number']} \u2014 {meta['section_title']}\n"
                        f"{meta['content']}"
                    ),
                    "metadata": {
                        "doc_title":       doc_title,
                        "doc_type":        doc_type,
                        "section_number":  meta["section_number"],
                        "section_title":   meta["section_title"][:200],
                        "chapter":         current_chapter,
                        "part":            current_part,
                        "hierarchy_path":  f"{current_chapter} > Section {meta['section_number']}",
                        "chunk_index":     chunk_index,
                        "parse_strategy":  "regex",
                    },
                }
            )
        elif len(block) >= 50:
            # Unrecognised block — preserve as raw fallback chunk
            chunk_index += 1
            chunks.append(
                {
                    "text": f"[{doc_title}] [{current_chapter}]\n{block}",
                    "metadata": {
                        "doc_title":      doc_title,
                        "doc_type":       doc_type,
                        "section_number": "N/A",
                        "section_title":  "N/A",
                        "chapter":        current_chapter,
                        "part":           current_part,
                        "hierarchy_path": current_chapter,
                        "chunk_index":    chunk_index,
                        "parse_strategy": "regex_fallback",
                    },
                }
            )

    logger.info(
        f"[regex_parser] produced {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c['metadata']['section_number'] != 'N/A')} with section IDs)"
    )
    return chunks
