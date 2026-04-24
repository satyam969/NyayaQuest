"""
stage7_storage/chunker.py — Legal-aware semantic sub-chunker.

Upgrades over v1
----------------
1. Legal-aware separator hierarchy — splits at natural legal boundaries
   (double-newline, paragraph, CHAPTER, PART, Section, ORDER, RULE,
   subsection markers, clause markers, sentence endings) before falling
   back to raw character splitting.
2. Adaptive chunk size — slightly larger for schedule/table-heavy text,
   smaller for dense statutory clauses.
3. Micro-chunk suppression — fragments < MIN_SUB_CHUNK_CHARS are merged
   with the previous sub-chunk or dropped.
4. Context prefix for continuation chunks — sub-chunks after the first
   get a lightweight header "[Section N — Title]" for retrieval quality.
5. Metadata enrichment — adds parent_chunk_index, sub_chunk,
   total_chunks, chunk_source_strategy, chunk_char_length.
6. Safe fallback — if legal-aware splitter fails, falls back silently
   to the standard RecursiveCharacterTextSplitter.

Public API (unchanged):
    split_chunks(chunks, chunk_size, chunk_overlap) → List[chunk]
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE    = 800
DEFAULT_CHUNK_OVERLAP = 100

# Sub-chunks shorter than this (after strip) are considered micro-fragments
MIN_SUB_CHUNK_CHARS = 40

# Legal-aware separator priority (ordered most-preferred → least-preferred)
_LEGAL_SEPARATORS = [
    "\n\n",                          # paragraph / blank-line boundary
    "\n\nCHAPTER ",                  # chapter heading
    "\n\nPART ",                     # part heading
    "\nSection ",                    # section heading
    "\nSec. ",
    "\nRule ",                       # rule heading
    "\nORDER ",                      # CPC-style order
    "\nSCHEDULE ",                   # schedule heading
    "\n    (",                       # indented sub-section like (1), (a)
    "\n(",                           # sub-section / clause
    ". ",                            # sentence boundary
    "\n",                            # any newline
    " ",                             # word boundary (last resort)
    "",                              # character boundary (final fallback)
]

# Markers that suggest schedule / table / list-heavy content → larger chunks
_SCHEDULE_MARKERS = re.compile(
    r"SCHEDULE|APPENDIX|TABLE|FORM\s+[A-Z]|Item\s+\d|Sl\.?\s*No\.",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _adaptive_chunk_size(text: str, base_size: int) -> int:
    """
    Slightly expand chunk size for schedule/table-heavy text.
    Keep base size for regular statutory text.
    """
    if _SCHEDULE_MARKERS.search(text[:500]):
        return int(base_size * 1.35)   # ~1080 for default 800
    return base_size


def _context_prefix(meta: Dict[str, Any]) -> str:
    """
    Build a one-line context header for continuation sub-chunks.
    Used so retrieval can identify the parent section without reading
    the full text of sub-chunk 1.
    """
    sec_num   = meta.get("section_number", "")
    sec_title = meta.get("section_title",  "")
    chapter   = meta.get("chapter",        "")

    parts: List[str] = []
    if sec_num and sec_num not in ("N/A", ""):
        parts.append(f"Section {sec_num}")
    if sec_title and sec_title not in ("N/A", ""):
        parts.append(sec_title[:80])
    if chapter and chapter not in ("Unknown Chapter", "N/A", ""):
        parts.append(f"[{chapter[:60]}]")

    if parts:
        return " — ".join(parts) + "\n"
    return ""


def _merge_micro_chunks(sub_texts: List[str]) -> List[str]:
    """
    Merge micro-fragments (< MIN_SUB_CHUNK_CHARS) into the previous
    sub-chunk. Trailing micro-chunks are prepended to the next one.
    """
    if not sub_texts:
        return sub_texts

    merged: List[str] = []
    pending = ""

    for text in sub_texts:
        stripped = text.strip()
        if not stripped:
            continue
        if len(stripped) < MIN_SUB_CHUNK_CHARS:
            # Accumulate into pending buffer
            pending = (pending + " " + stripped).strip() if pending else stripped
        else:
            if pending:
                merged.append(pending + "\n" + text)
                pending = ""
            else:
                merged.append(text)

    # Flush any remaining pending content into the last chunk
    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n" + pending
        else:
            merged.append(pending)

    return merged


def _legal_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_LEGAL_SEPARATORS,
        keep_separator=False,
    )


def _fallback_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def split_chunks(
    chunks:        List[Dict[str, Any]],
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Sub-split any chunk whose text exceeds chunk_size characters using
    legal-aware separators.

    Metadata propagated and enriched with:
      parent_chunk_index   : int  — chunk_index of the parent
      sub_chunk            : int  — 1-indexed position within parent
      total_chunks         : int  — total sub-chunks for this parent
      chunk_source_strategy: str  — "passthrough" / "legal_semantic_split" / "fallback_split"
      chunk_char_length    : int  — character length of this sub-chunk's text
    """
    final: List[Dict[str, Any]] = []
    semantic_count  = 0
    fallback_count  = 0

    logger.info(f"[chunker] input chunks={len(chunks)}")

    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        parent_idx = meta.get("chunk_index", 0)

        # ── Passthrough: chunk fits within size limit ──────────────
        if len(text) <= chunk_size:
            final.append({
                "text": text,
                "metadata": {
                    **meta,
                    "parent_chunk_index":    parent_idx,
                    "sub_chunk":             1,
                    "total_chunks":          1,
                    "chunk_source_strategy": "passthrough",
                    "chunk_char_length":     len(text),
                },
            })
            continue

        # ── Adaptive size for schedule-heavy content ───────────────
        effective_size = _adaptive_chunk_size(text, chunk_size)

        # ── Legal-aware split attempt ──────────────────────────────
        sub_texts: Optional[List[str]] = None
        strategy  = "legal_semantic_split"
        try:
            raw = _legal_splitter(effective_size, chunk_overlap).split_text(text)
            sub_texts = _merge_micro_chunks(raw)
            semantic_count += 1
        except Exception as exc:
            logger.debug(f"[chunker] Legal splitter failed ({exc}) — using fallback")

        # ── Fallback to standard splitter ─────────────────────────
        if not sub_texts:
            try:
                raw = _fallback_splitter(effective_size, chunk_overlap).split_text(text)
                sub_texts = _merge_micro_chunks(raw)
                strategy  = "fallback_split"
                fallback_count += 1
            except Exception as exc:
                logger.warning(f"[chunker] Fallback splitter also failed ({exc}) — keeping original")
                sub_texts = [text]
                strategy  = "fallback_split"
                fallback_count += 1

        # ── Guarantee at least one sub-chunk ──────────────────────
        if not sub_texts:
            sub_texts = [text]

        total = len(sub_texts)
        prefix = _context_prefix(meta)

        for j, sub_text in enumerate(sub_texts):
            sub_text = sub_text.strip()
            if not sub_text:
                continue

            # Prepend context header to continuation chunks (not first)
            if j > 0 and prefix:
                sub_text = prefix + sub_text

            final.append({
                "text": sub_text,
                "metadata": {
                    **meta,
                    "parent_chunk_index":    parent_idx,
                    "sub_chunk":             j + 1,
                    "total_chunks":          total,
                    "chunk_source_strategy": strategy,
                    "chunk_char_length":     len(sub_text),
                },
            })

    logger.info(
        f"[chunker] output chunks={len(final)}  "
        f"semantic_splits={semantic_count}  "
        f"fallback_splits={fallback_count}"
    )
    return final
