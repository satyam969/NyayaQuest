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


def extract_order(text: str, current: str = "Unknown Order") -> Tuple[str, str]:
    # Match headers like `[ORDER I` or `ORDER XXI` at start of line
    return _extract_heading(text, [r"(?m)^\[?(ORDER\s+[IVXLCDM]+[A-Z]?)\b"], current)


# ─────────────────────────────────────────────────────────────────────
# Section metadata extraction
# ─────────────────────────────────────────────────────────────────────

_SEC_HEAD_PATTERNS = [
    # "1[9. Title.—(1) content" — footnote-digit + bracket prefix (CPC/codified acts)
    r"^\d+\[(\d+[A-Z]?)\.\s*([^\n]{0,200}?)(?:\.[\u2014\u2013]|\.[\-]+|[\-]{2,}|\.\s*\n)(.*)",
    r"^\d+\[(\d+[A-Z]?)\.\s*([^\n(]{0,200})\n(.*)",
    r"^\d+\[(\d+[A-Z]?)\.\s*(\(.*)",
    # "1. Title.—(1) content" (em-dash variants, .-- or -- separators)
    r"^(\d+[A-Z]?)\.\s*([^\n]{0,200}?)(?:\.[\u2014\u2013]|\.[\-]+|[\-]{2,}|\.\s*\n)(.*)",
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
    current_order = "Unknown Order"
    in_schedule = False        # only flip to order mode after THE FIRST SCHEDULE
    chunk_index = 0

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Reset schedule mode if we see body text headers like PRELIMINARY or PART I.
        # This prevents the Table of Contents' mention of the Schedule from trapping
        # the entire document in "Order mode".
        if re.search(r"^\s*(?:PRELIMINARY|PART\s+[IVX]+)\s*$", block, re.IGNORECASE | re.MULTILINE):
            in_schedule = False
            current_order = "Unknown Order"

        # Detect schedule boundary (CPC First Schedule) — enables order mode
        # Must have parsed at least one section (chunk_index > 0) to avoid tripping in the TOC.
        if not in_schedule and chunk_index > 0 and re.search(r"^\s*THE\s+FIRST\s+SCHEDULE\s*$", block, re.IGNORECASE | re.MULTILINE):
            in_schedule = True

        # Update context from structural headings embedded in the block
        current_chapter, block = extract_chapter(block, current_chapter)
        current_part, block = extract_part(block, current_part)
        # pending_order: extract ORDER heading if in schedule, but only USE it
        # for classification AFTER the current block (prevents section 158 from
        # being mislabeled as a rule just because ORDER I is in the same block).
        pending_order = current_order
        if in_schedule:
            pending_order, block = extract_order(block, current_order)

        if not block.strip():
            continue

        meta = extract_section_meta(block)

        if meta and len(meta["content"]) >= 30:
            chunk_index += 1
            if current_order != "Unknown Order":  # order carried from a PREVIOUS block
                chunks.append(
                    {
                        "text": (
                            f"[{doc_title}] [{current_order}] "
                            f"Rule {meta['section_number']} \u2014 {meta['section_title']}\n"
                            f"{meta['content']}"
                        ),
                        "metadata": {
                            "doc_title":       doc_title,
                            "doc_type":        doc_type,
                            "type":            "rule",
                            "order":           current_order,
                            "rule":            meta["section_number"],
                            "rule_title":      meta["section_title"][:200],
                            "chapter":         current_chapter,
                            "part":            current_part,
                            "hierarchy_path":  f"{current_chapter} > {current_order} > Rule {meta['section_number']}",
                            "chunk_index":     chunk_index,
                            "parse_strategy":  "regex",
                        },
                    }
                )
            else:
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
                            "type":            "section",
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

        # Advance order state for next iteration (after if/elif so syntax is valid)
        current_order = pending_order

    logger.info(
        f"[regex_parser] produced {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c['metadata'].get('section_number', c['metadata'].get('rule', 'N/A')) != 'N/A')} with section/rule IDs)"
    )
    return chunks
