"""
stage6_fallback/fallback_engine.py — 3-tier fallback system.

Guarantees that the pipeline NEVER returns zero chunks, even for
completely unrecognised documents.

Tier 1 — Brute-force regex sweep
    Try every SECTION_PATTERN against the text; keep whichever gives
    the highest count of section-identified chunks.

Tier 2 — Generic character-split
    RecursiveCharacterTextSplitter at 800-char chunks; assigns default
    N/A metadata.  Never fails.

Tier 3 (optional, requires LLM) — LLM extraction on a sample
    Ask the LLM to pull out sections from the first 3 000 characters
    as structured JSON.  Merge LLM chunks with Tier 2 tail.

A quarantine record is appended to `auto_ingest_quarantine.jsonl` so
operators can review difficult documents.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.patterns import SECTION_PATTERNS
from ..stage3_parsing.regex_parser import parse_with_regex

logger = logging.getLogger(__name__)

QUARANTINE_FILE = "auto_ingest_quarantine.jsonl"


# ─────────────────────────────────────────────────────────────────────
# Quarantine log
# ─────────────────────────────────────────────────────────────────────

def _log_quarantine(
    pdf_path: str,
    reason: str,
    metrics: Dict[str, float],
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pdf_path":  pdf_path,
        "reason":    reason,
        "metrics":   metrics,
    }
    try:
        with open(QUARANTINE_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.warning(f"Quarantine logged: {pdf_path} | {reason}")
    except OSError as e:
        logger.error(f"Could not write quarantine record: {e}")


# ─────────────────────────────────────────────────────────────────────
# Tier 1 — brute-force regex sweep
# ─────────────────────────────────────────────────────────────────────

def tier1_brute_force_regex(
    text: str,
    doc_title: str,
) -> List[Dict[str, Any]]:
    """
    Try every known SECTION_PATTERN.
    Return the result set that identifies the most sections.
    """
    best_chunks: List[Dict[str, Any]] = []
    best_id_count = 0

    for pattern in SECTION_PATTERNS:
        try:
            chunks = parse_with_regex(text, pattern, doc_title, "GENERIC")
            id_count = sum(
                1 for c in chunks
                if c["metadata"].get("section_number", "N/A") not in ("N/A", "", None)
            )
            if id_count > best_id_count:
                best_id_count = id_count
                best_chunks   = chunks
        except Exception as exc:
            logger.debug(f"Tier 1 pattern failed ({pattern!r}): {exc}")
            continue

    for c in best_chunks:
        c["metadata"]["parse_strategy"] = "fallback_tier1"

    logger.info(
        f"Fallback Tier 1: {len(best_chunks)} chunks "
        f"({best_id_count} section-identified)"
    )
    return best_chunks


# ─────────────────────────────────────────────────────────────────────
# Tier 2 — generic character split (never fails)
# ─────────────────────────────────────────────────────────────────────

def tier2_generic_split(
    text: str,
    doc_title: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Last-resort splitter.  Always returns ≥ 1 chunk.
    Assigns default N/A metadata so downstream stages never break.
    """
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    raw_chunks = splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []

    for i, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue
        chunks.append(
            {
                "text": f"[{doc_title}]\n{chunk_text}",
                "metadata": {
                    "doc_title":      doc_title,
                    "doc_type":       "GENERIC",
                    "section_number": "N/A",
                    "section_title":  "N/A",
                    "chapter":        "Unknown Chapter",
                    "part":           "Unknown Part",
                    "hierarchy_path": "Unknown",
                    "chunk_index":    i + 1,
                    "parse_strategy": "fallback_tier2",
                },
            }
        )

    logger.info(f"Fallback Tier 2: {len(chunks)} generic chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Tier 3 — LLM extraction on sample (optional)
# ─────────────────────────────────────────────────────────────────────

def tier3_llm_extraction(
    llm,
    text: str,
    doc_title: str,
    sample_size: int = 3000,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to extract a JSON array of {section, title, content}
    from a text sample.  Returns None on any failure.
    """
    sample = text[:sample_size]
    prompt = (
        f"Extract all numbered sections from this legal document text.\n"
        f"Document: {doc_title}\n\nText:\n---\n{sample}\n---\n\n"
        f"Return a JSON array:\n"
        f'[{{"section": "1", "title": "Short title", "content": "..."}}]\n'
        f"Return ONLY the JSON array, no explanation."
    )

    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        m = re.search(r"\[[\s\S]*\]", content)
        if not m:
            return None

        data   = json.loads(m.group(0))
        chunks = []
        for i, item in enumerate(data):
            sec_num   = str(item.get("section", "N/A"))
            sec_title = str(item.get("title",   ""))[:200]
            body      = str(item.get("content", ""))
            chunks.append(
                {
                    "text": f"[{doc_title}] Section {sec_num} \u2014 {sec_title}\n{body}",
                    "metadata": {
                        "doc_title":      doc_title,
                        "doc_type":       "GENERIC",
                        "section_number": sec_num,
                        "section_title":  sec_title,
                        "chapter":        "Unknown Chapter",
                        "part":           "Unknown Part",
                        "hierarchy_path": f"Section {sec_num}",
                        "chunk_index":    i + 1,
                        "parse_strategy": "fallback_tier3_llm",
                    },
                }
            )

        logger.info(f"Fallback Tier 3 (LLM): {len(chunks)} extracted chunks")
        return chunks

    except Exception as exc:
        logger.error(f"Fallback Tier 3 LLM extraction failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def run_fallback(
    text: str,
    doc_title: str,
    pdf_path: str,
    metrics: Dict[str, float],
    llm=None,
) -> List[Dict[str, Any]]:
    """
    Execute the fallback cascade.  Always returns at least one chunk.

    Flow:
      Tier 1 → if ≥ 5 section-ID'd chunks → return
      Tier 2 → always produces chunks (anchor)
      Tier 3 (if LLM available) → merge LLM sample + Tier 2 tail
    """
    _log_quarantine(pdf_path, "Fallback engine activated", metrics)

    # Tier 1
    t1_chunks = tier1_brute_force_regex(text, doc_title)
    id_count  = sum(
        1 for c in t1_chunks
        if c["metadata"].get("section_number", "N/A") not in ("N/A", "", None)
    )
    if id_count >= 5:
        logger.info("Fallback Tier 1 sufficient — returning Tier 1 chunks")
        return t1_chunks

    # Tier 2 is our safety net — always computed
    logger.info("Fallback escalating to Tier 2")
    t2_chunks = tier2_generic_split(text, doc_title)

    if llm is not None:
        logger.info("Fallback attempting Tier 3 LLM extraction")
        t3_chunks = tier3_llm_extraction(llm, text, doc_title)
        if t3_chunks and len(t3_chunks) >= 3:
            # Replace the first N tier-2 chunks with better LLM chunks
            combined = t3_chunks + t2_chunks[len(t3_chunks):]
            logger.info(
                f"Fallback Tier 3+2 combined: "
                f"{len(t3_chunks)} LLM + {len(t2_chunks) - len(t3_chunks)} generic"
            )
            return combined

    return t2_chunks
