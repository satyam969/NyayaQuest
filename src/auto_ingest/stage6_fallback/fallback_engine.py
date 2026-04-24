"""
stage6_fallback/fallback_engine.py — 3-tier fallback system (upgraded).

Guarantees the pipeline NEVER returns zero chunks, even for completely
unrecognised documents.

Tier order (revised)
--------------------
Tier 1 — Quality-scored regex/rule sweep
    Try every SECTION_PATTERNS + RULE_PATTERNS candidate.
    Evaluate each with Stage 4 metrics (not just ID count).
    Accept immediately if overall >= TIER1_PASS_THRESHOLD.

Tier 3 — LLM structured extraction (optional, if llm available)
    Multi-window sampling (head + dense + tail) for richer context.
    LLM returns JSON array of {section, title, content}.
    Validated before accepting; merged with Tier 2 tail for coverage.

Tier 2 — Generic character-split (final never-fail guarantee)
    RecursiveCharacterTextSplitter, adaptive chunk size by doc length.
    Always returns >= 1 chunk.

A quarantine record is appended to `auto_ingest_quarantine.jsonl`.

Public API (unchanged):
    tier1_brute_force_regex(text, doc_title)            → List[chunk]
    tier2_generic_split(text, doc_title, ...)           → List[chunk]
    tier3_llm_extraction(llm, text, doc_title, ...)     → Optional[List[chunk]]
    run_fallback(text, doc_title, pdf_path, metrics, llm) → List[chunk]
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.patterns import SECTION_PATTERNS, RULE_PATTERNS
from ..utils.scoring import compute_quality_score
from ..stage3_parsing.regex_parser import parse_with_regex

logger = logging.getLogger(__name__)

QUARANTINE_FILE = "auto_ingest_quarantine.jsonl"

# ── Quality thresholds ────────────────────────────────────────────────────────
TIER1_PASS_THRESHOLD: float = 0.70   # accept Tier 1 immediately if met
TIER3_MIN_CHUNKS:     int   = 3      # minimum LLM chunks to be usable

# ── Sampling sizes for Tier 3 ─────────────────────────────────────────────────
_HEAD_SIZE  = 4_000
_DENSE_SIZE = 5_000
_DENSE_STEP = 3_000
_DENSE_CAP  = 15
_TAIL_SIZE  = 3_000

# ── Structural density markers for dense-window selection ─────────────────────
_DENSITY_PATS = [
    r"(?m)^\s*\d{1,3}[A-Z]?\.\s",
    r"(?m)^CHAPTER\s+[IVXLCDM]+",
    r"(?m)^PART\s+[IVXLCDM]+",
    r"(?m)^Rule\s+\d+",
    r"(?m)^ORDER\s+[IVXLCDM]+",
    r"(?m)^SCHEDULE",
]

# ── Adaptive Tier 2 chunk sizing ─────────────────────────────────────────────
_T2_SHORT_THRESHOLD = 30_000    # chars
_T2_LONG_THRESHOLD  = 150_000   # chars
_T2_SHORT_SIZE      = 600
_T2_DEFAULT_SIZE    = 800
_T2_LONG_SIZE       = 1_200
_T2_OVERLAP         = 100


# ─────────────────────────────────────────────────────────────────────
# Quarantine log
# ─────────────────────────────────────────────────────────────────────

def _log_quarantine(
    pdf_path:    str,
    reason:      str,
    metrics:     Dict[str, float],
    extra:       Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pdf_path":  pdf_path,
        "reason":    reason,
        "metrics":   metrics,
    }
    if extra:
        record.update(extra)
    try:
        with open(QUARANTINE_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.warning(f"[fallback] Quarantine logged: {pdf_path} | {reason}")
    except OSError as e:
        logger.error(f"[fallback] Could not write quarantine record: {e}")


# ─────────────────────────────────────────────────────────────────────
# Sampling helper (shared by Tier 3)
# ─────────────────────────────────────────────────────────────────────

def _marker_count(text: str) -> int:
    return sum(len(re.findall(p, text)) for p in _DENSITY_PATS)


def _dense_block(text: str) -> str:
    n = len(text)
    if n <= _DENSE_SIZE:
        return text
    best_start, best_score = 0, -1
    positions = list(range(0, n - _DENSE_SIZE + 1, _DENSE_STEP))[:_DENSE_CAP]
    if n > _DENSE_SIZE:
        positions.append(n - _DENSE_SIZE)
    for start in set(positions):
        score = _marker_count(text[start : start + _DENSE_SIZE])
        if score > best_score:
            best_score, best_start = score, start
    return text[best_start : best_start + _DENSE_SIZE]


def _multi_window_sample(text: str) -> str:
    head  = text[:_HEAD_SIZE]
    dense = _dense_block(text)
    tail  = text[-_TAIL_SIZE:] if len(text) > _TAIL_SIZE else text
    parts: List[str] = []
    seen: set = set()
    for part in (head, dense, tail):
        if part.strip() and part not in seen:
            parts.append(part)
            seen.add(part)
    return "\n\n[...]\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Tier 1 — Quality-scored regex/rule sweep
# ─────────────────────────────────────────────────────────────────────

def tier1_brute_force_regex(
    text:      str,
    doc_title: str,
) -> List[Dict[str, Any]]:
    """
    Try every SECTION_PATTERNS + RULE_PATTERNS candidate.
    Select the winner by Stage 4 composite quality score (not raw ID count).
    Returns the best chunk list found (may be empty if all patterns crash).
    """
    best_chunks:  List[Dict[str, Any]] = []
    best_score:   float                = -1.0
    best_capture: int                  = 0

    all_patterns = list(SECTION_PATTERNS) + list(RULE_PATTERNS)

    for pattern in all_patterns:
        try:
            chunks = parse_with_regex(text, pattern, doc_title, "GENERIC")
            if not chunks:
                continue
            metrics  = compute_quality_score(chunks)
            score    = metrics.get("overall", 0.0)
            capture  = sum(
                1 for c in chunks
                if c["metadata"].get("section_number", "N/A") not in ("N/A", "", None)
            )
            # Better score wins; tiebreak on more identified sections
            if score > best_score or (score == best_score and capture > best_capture):
                best_score   = score
                best_chunks  = chunks
                best_capture = capture
        except Exception as exc:
            logger.debug(f"[fallback] Tier 1 pattern failed ({pattern!r}): {exc}")
            continue

    for c in best_chunks:
        meta = c.setdefault("metadata", {})
        meta["parse_strategy"]      = "fallback_tier1"
        meta["fallback_tier"]       = 1
        meta["fallback_confidence"] = "medium"

    logger.info(
        f"[fallback] Tier 1: {len(best_chunks)} chunks  "
        f"score={best_score:.3f}  capture={best_capture}"
    )
    return best_chunks


# ─────────────────────────────────────────────────────────────────────
# Tier 2 — Generic character split (never fails)
# ─────────────────────────────────────────────────────────────────────

def tier2_generic_split(
    text:          str,
    doc_title:     str,
    chunk_size:    int = 0,    # 0 = auto-select based on doc length
    chunk_overlap: int = _T2_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Last-resort splitter. Always returns >= 1 chunk.
    Chunk size is adaptive: smaller for short docs, larger for long docs.
    Assigns default N/A metadata so downstream stages never break.
    """
    if chunk_size == 0:
        n = len(text)
        if n < _T2_SHORT_THRESHOLD:
            chunk_size = _T2_SHORT_SIZE
        elif n > _T2_LONG_THRESHOLD:
            chunk_size = _T2_LONG_SIZE
        else:
            chunk_size = _T2_DEFAULT_SIZE

    splitter   = RecursiveCharacterTextSplitter(
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
                    "doc_title":          doc_title,
                    "doc_type":           "GENERIC",
                    "section_number":     "N/A",
                    "section_title":      "N/A",
                    "chapter":            "Unknown Chapter",
                    "part":               "Unknown Part",
                    "hierarchy_path":     "Unknown",
                    "chunk_index":        i + 1,
                    "parse_strategy":     "fallback_tier2",
                    "fallback_tier":      2,
                    "fallback_confidence": "low",
                },
            }
        )

    # Absolute guarantee: if splitter somehow produced nothing, emit one chunk
    if not chunks:
        chunks = [
            {
                "text": f"[{doc_title}]\n{text[:chunk_size]}",
                "metadata": {
                    "doc_title":          doc_title,
                    "doc_type":           "GENERIC",
                    "section_number":     "N/A",
                    "section_title":      "N/A",
                    "chapter":            "Unknown Chapter",
                    "part":               "Unknown Part",
                    "hierarchy_path":     "Unknown",
                    "chunk_index":        1,
                    "parse_strategy":     "fallback_tier2_emergency",
                    "fallback_tier":      2,
                    "fallback_confidence": "low",
                },
            }
        ]

    logger.info(
        f"[fallback] Tier 2: {len(chunks)} generic chunks  "
        f"chunk_size={chunk_size}"
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Tier 3 — LLM structured extraction (optional)
# ─────────────────────────────────────────────────────────────────────

def tier3_llm_extraction(
    llm,
    text:        str,
    doc_title:   str,
    sample_size: int = 0,  # 0 = use multi-window sampling
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to extract a JSON array of {section, title, content}
    from a representative multi-window sample.
    Returns None on any failure or if output is unusable.
    """
    # Multi-window sampling: head + dense structural block + tail
    sample = _multi_window_sample(text) if sample_size == 0 else text[:sample_size]

    prompt = f"""You are a legal document parser specialising in Indian statutory law.
Extract all numbered sections or rules from this document sample.

Document: {doc_title}

Recognised numbering formats:
  - "1.", "1A.", "12B." (plain section numbers)
  - "Section 1", "Rule 1", "Rule 1A"
  - "[1.", "*[1." (codified/annotated statutes)
  - "ORDER I RULE 1" (CPC-style)

Text:
---
{sample[:10_000]}
---

Return a JSON array ONLY — no markdown fences, no explanation:
[{{"section": "1", "title": "Short title", "content": "Body text of the section..."}},
 {{"section": "2", "title": "Definitions", "content": "In this Act..."}}]

Rules:
- "section" must be the number string ("1", "12A", "Rule 3").
- "title" is the short heading (max 200 chars). Use "" if absent.
- "content" is the section body text (do NOT truncate — include full text).
- Return ONLY valid JSON. No markdown. No explanation."""

    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(f"[fallback] Tier 3 LLM call failed: {exc}")
        return None

    # Extract JSON array from response
    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        logger.warning("[fallback] Tier 3: no JSON array found in LLM response")
        return None

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"[fallback] Tier 3: JSON decode failed — {e}")
        return None

    if not isinstance(data, list) or len(data) < TIER3_MIN_CHUNKS:
        logger.warning(
            f"[fallback] Tier 3: too few items ({len(data) if isinstance(data, list) else 0})"
        )
        return None

    # Validate for obvious hallucination: all identical sections = bad
    sec_nums = [str(item.get("section", "")) for item in data if isinstance(item, dict)]
    if len(set(sec_nums)) < max(1, len(sec_nums) // 2):
        logger.warning("[fallback] Tier 3: suspected hallucination (too many duplicate section nums)")
        return None

    chunks: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        sec_num   = str(item.get("section", "N/A"))
        sec_title = str(item.get("title",   ""))[:200]
        body      = str(item.get("content", ""))
        if not body.strip():
            continue
        chunks.append(
            {
                "text": f"[{doc_title}] Section {sec_num} \u2014 {sec_title}\n{body}",
                "metadata": {
                    "doc_title":          doc_title,
                    "doc_type":           "GENERIC",
                    "section_number":     sec_num,
                    "section_title":      sec_title,
                    "chapter":            "Unknown Chapter",
                    "part":               "Unknown Part",
                    "hierarchy_path":     f"Section {sec_num}",
                    "chunk_index":        i + 1,
                    "parse_strategy":     "fallback_tier3_llm",
                    "fallback_tier":      3,
                    "fallback_confidence": "medium_high",
                },
            }
        )

    if len(chunks) < TIER3_MIN_CHUNKS:
        logger.warning(f"[fallback] Tier 3: only {len(chunks)} usable chunks after filtering")
        return None

    logger.info(f"[fallback] Tier 3 (LLM): {len(chunks)} extracted chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Tier 3 + Tier 2 region-aware merge
# ─────────────────────────────────────────────────────────────────────

def _merge_tier3_tier2(
    t3_chunks: List[Dict[str, Any]],
    t2_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge Tier 3 (LLM-extracted) chunks with Tier 2 (generic) tail chunks.

    Strategy:
    - Tier 3 covers the sampled windows (head + dense + tail).
    - Tier 2 covers the full document uniformly.
    - Estimate how many Tier 2 chunks represent "covered" regions by
      character coverage, then append the uncovered remainder.
    - This avoids duplicate leading content.
    """
    if not t3_chunks:
        return t2_chunks
    if not t2_chunks:
        return t3_chunks

    # Estimate how many Tier 2 chunks the sampled regions cover.
    # Sampled text is roughly (_HEAD_SIZE + _DENSE_SIZE + _TAIL_SIZE) chars.
    sampled_chars = _HEAD_SIZE + _DENSE_SIZE + _TAIL_SIZE
    total_chars   = sum(len(c.get("text", "")) for c in t2_chunks)
    if total_chars == 0:
        return t3_chunks

    covered_fraction  = min(sampled_chars / total_chars, 0.85)
    t2_covered_count  = max(1, int(len(t2_chunks) * covered_fraction))
    t2_remainder      = t2_chunks[t2_covered_count:]

    combined = t3_chunks + t2_remainder
    logger.info(
        f"[fallback] Tier 3+2 merge: {len(t3_chunks)} LLM + "
        f"{len(t2_remainder)} generic tail  (skipped {t2_covered_count} covered)"
    )
    return combined


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def run_fallback(
    text:      str,
    doc_title: str,
    pdf_path:  str,
    metrics:   Dict[str, float],
    llm=None,
) -> List[Dict[str, Any]]:
    """
    Execute the fallback cascade. Always returns at least one chunk.

    Tier order:
        Tier 1 → quality-scored regex sweep → accept if overall >= 0.70
        Tier 3 → LLM extraction (if llm available and Tier 1 insufficient)
        Tier 2 → generic character split (final guarantee)
    """
    logger.info(f"[fallback] Activated for: {doc_title}")

    chosen_tier  = 2
    best_score   = metrics.get("overall", 0.0)
    final_chunks: List[Dict[str, Any]] = []

    # ── Tier 1: quality-scored regex sweep ───────────────────────────
    t1_chunks: List[Dict[str, Any]] = []
    t1_score  = 0.0
    try:
        t1_chunks = tier1_brute_force_regex(text, doc_title)
        if t1_chunks:
            t1_metrics = compute_quality_score(t1_chunks)
            t1_score   = t1_metrics.get("overall", 0.0)

            if t1_score >= TIER1_PASS_THRESHOLD:
                logger.info(
                    f"[fallback] Tier 1 accepted (score={t1_score:.3f} >= "
                    f"{TIER1_PASS_THRESHOLD})"
                )
                _log_quarantine(
                    pdf_path, "Tier 1 regex rescue",
                    metrics,
                    {"chosen_tier": 1, "returned_chunk_count": len(t1_chunks),
                     "best_score": t1_score, "doc_title": doc_title,
                     "text_length": len(text)},
                )
                return t1_chunks

            logger.info(
                f"[fallback] Tier 1 insufficient (score={t1_score:.3f} < "
                f"{TIER1_PASS_THRESHOLD}) — escalating"
            )
    except Exception as exc:
        logger.error(f"[fallback] Tier 1 crashed: {exc} — skipping")

    # ── Tier 2: always compute as safety net ─────────────────────────
    try:
        t2_chunks = tier2_generic_split(text, doc_title)
    except Exception as exc:
        logger.error(f"[fallback] Tier 2 crashed: {exc} — using minimal chunk")
        t2_chunks = [
            {
                "text": f"[{doc_title}]\n{text[:800]}",
                "metadata": {
                    "doc_title": doc_title, "doc_type": "GENERIC",
                    "section_number": "N/A", "section_title": "N/A",
                    "chapter": "Unknown Chapter", "part": "Unknown Part",
                    "hierarchy_path": "Unknown", "chunk_index": 1,
                    "parse_strategy": "fallback_emergency",
                    "fallback_tier": 2, "fallback_confidence": "low",
                },
            }
        ]

    final_chunks = t2_chunks
    chosen_tier  = 2

    # ── Tier 3: LLM extraction (only if Tier 1 was insufficient) ─────
    if llm is not None:
        logger.info("[fallback] Attempting Tier 3 LLM extraction")
        try:
            t3_chunks = tier3_llm_extraction(llm, text, doc_title)
            if t3_chunks:
                t3_metrics = compute_quality_score(t3_chunks)
                t3_score   = t3_metrics.get("overall", 0.0)
                logger.info(f"[fallback] Tier 3 accepted: {len(t3_chunks)} chunks  score={t3_score:.3f}")

                final_chunks = _merge_tier3_tier2(t3_chunks, t2_chunks)
                chosen_tier  = 3
            else:
                logger.info("[fallback] Tier 3 returned nothing — using Tier 2")
        except Exception as exc:
            logger.error(f"[fallback] Tier 3 crashed: {exc} — using Tier 2")
    else:
        logger.info("[fallback] No LLM available — Tier 2 is final result")

    # ── Guarantee non-empty output ────────────────────────────────────
    if not final_chunks:
        logger.error("[fallback] All tiers produced empty output — emitting emergency chunk")
        final_chunks = tier2_generic_split(text[:5_000], doc_title, chunk_size=800)

    # ── Quarantine log ────────────────────────────────────────────────
    _log_quarantine(
        pdf_path,
        f"Fallback Tier {chosen_tier} used",
        metrics,
        {
            "doc_title":           doc_title,
            "text_length":         len(text),
            "chosen_tier":         chosen_tier,
            "returned_chunk_count": len(final_chunks),
            "best_score":          best_score,
        },
    )

    logger.info(
        f"[fallback] Complete: tier={chosen_tier}  "
        f"chunks={len(final_chunks)}  doc={doc_title}"
    )
    return final_chunks
