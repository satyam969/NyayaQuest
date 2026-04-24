"""
stage2_strategy/schema_strategy.py — LLM-based schema generation with
3-attempt sample pre-validation.

Changes from v1
---------------
1. Multi-window sampling (head 5K + dense 6K + tail 4K) replaces text[:3000].
2. Confidence-tiered doc_type hints — weak prior when confidence is low.
3. Feature-structured natural-language hints instead of raw dict dump.
4. 3-attempt sample pre-validation loop (all on ~15K sample, NOT full PDF):
     Attempt 1 — Initial schema  → validate on sample
     Attempt 2 — PATCH using failure evidence → validate on sample
     Attempt 3 — FRESH broad schema → validate on sample
   Returns best schema found; full parse happens outside (orchestrator).
5. generate_schema() preserved exactly for backward compatibility.
6. New public API: generate_validated_schema() for the SCHEMA branch.

Schema shape (unchanged)
{
  "section_pattern":       "<Python lookahead regex for re.split()>",
  "chapter_pattern":       "<regex capturing (chapter_id, title) or null>",
  "part_pattern":          "<regex capturing (part_id, title) or null>",
  "title_extract_pattern": "<regex with groups (section_num, title)>",
  "hierarchy":             ["chapter", "part", "section"],
  "flags":                 { "has_definitions": bool, ... },
  "metadata_defaults":     { "year": "2005" }
}
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Sampling constants
# ─────────────────────────────────────────────────────────────────────

_HEAD_SIZE  = 5_000
_DENSE_SIZE = 6_000
_DENSE_STEP = 3_000
_DENSE_CAP  = 20          # max sliding-window positions to evaluate
_TAIL_SIZE  = 4_000

# ─────────────────────────────────────────────────────────────────────
# Pre-validation thresholds (lenient — it's a sample, not full doc)
# ─────────────────────────────────────────────────────────────────────

_PRE_MIN_CHUNKS  = 3
_PRE_MIN_CAPTURE = 0.40
_PRE_MIN_OVERALL = 0.55
_PRE_MAX_CHUNK   = 10_000   # chars — any chunk larger = pattern not splitting

# ─────────────────────────────────────────────────────────────────────
# Structural density markers for dense-window selection
# ─────────────────────────────────────────────────────────────────────

_DENSITY_PATS: List[str] = [
    r"(?m)^\s*\d{1,3}[A-Z]?\.\s",
    r"(?m)^CHAPTER\s+[IVXLCDM]+",
    r"(?m)^PART\s+[IVXLCDM]+",
    r"(?m)^Rule\s+\d+",
    r"(?m)^ORDER\s+[IVXLCDM]+",
    r"(?m)^SCHEDULE",
    r"(?m)^Article\s+\d+",
]


# ─────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────

def _marker_count(text: str) -> int:
    """Count structural marker hits in a text window."""
    return sum(len(re.findall(p, text)) for p in _DENSITY_PATS)


def _dense_block(text: str) -> str:
    """Return the 6K-char window with the highest structural marker density."""
    n = len(text)
    if n <= _DENSE_SIZE:
        return text

    best_start, best_score = 0, -1
    positions = list(range(0, n - _DENSE_SIZE + 1, _DENSE_STEP))[:_DENSE_CAP]
    # Always include a window anchored at the very end
    if n > _DENSE_SIZE:
        positions.append(n - _DENSE_SIZE)

    for start in set(positions):
        score = _marker_count(text[start : start + _DENSE_SIZE])
        if score > best_score:
            best_score, best_start = score, start

    return text[best_start : best_start + _DENSE_SIZE]


def _build_windows(text: str) -> Tuple[str, str, str]:
    head  = text[:_HEAD_SIZE]
    dense = _dense_block(text)
    tail  = text[-_TAIL_SIZE:] if len(text) > _TAIL_SIZE else text
    return head, dense, tail


def _combined_sample(text: str) -> str:
    """Combine head + dense + tail, deduplicated, separated by marker."""
    head, dense, tail = _build_windows(text)
    seen: List[str] = []
    for part in (head, dense, tail):
        if part.strip() and part not in seen:
            seen.append(part)
    return "\n\n[...]\n\n".join(seen)


# ─────────────────────────────────────────────────────────────────────
# Prompt-hint builders
# ─────────────────────────────────────────────────────────────────────

def _confidence_hint(doc_type: str, confidence: float, features: Dict[str, Any]) -> str:
    score_bd = features.get("score_breakdown", {})
    if confidence >= 0.70:
        return f"Document Type (high confidence): {doc_type}"
    if confidence >= 0.40:
        return (
            f"Document Type (uncertain — treat as hint only): "
            f"{doc_type}  (confidence: {confidence:.2f})"
        )
    # Very low confidence — show ranked candidates
    if score_bd:
        ranked = sorted(score_bd.items(), key=lambda x: x[1], reverse=True)[:3]
        cands = ", ".join(f"{t} ({s:.2f})" for t, s in ranked if s > 0)
        return (
            f"Document type is UNCERTAIN (confidence: {confidence:.2f}). "
            f"Ranked candidates: {cands}. "
            f"Infer structure primarily from the text sample, not these labels."
        )
    return f"Document type is UNCERTAIN (confidence: {confidence:.2f}). Infer from text."


def _feature_hints(features: Dict[str, Any]) -> str:
    lines: List[str] = []
    if features.get("has_chapters"):
        lines.append("- Has CHAPTER headings (Roman numerals: CHAPTER I, II, ...)")
    if features.get("has_parts"):
        lines.append("- Has PART divisions (PART I, PART A, ...)")
    if features.get("has_orders"):
        lines.append("- Has ORDER headings (ORDER I, II, ...) — may be a schedule/rules doc")
    if features.get("has_schedules"):
        lines.append("- Has SCHEDULE sections")
    if features.get("has_definitions"):
        lines.append("- Has definitions section")
    n = features.get("approx_section_count", 0)
    if n > 100:
        lines.append(f"- Large statute (~{n} sections) — likely codified or long-form act")
    elif n > 0:
        lines.append(f"- ~{n} sections detected")
    return "\n".join(lines) if lines else "No strong structural signals detected."


# ─────────────────────────────────────────────────────────────────────
# Prompt builders (3 variants)
# ─────────────────────────────────────────────────────────────────────

def build_schema_prompt(
    text_sample: str,
    doc_type: str,
    features: Dict[str, Any],
    confidence: float = 1.0,
) -> str:
    """Attempt 1 — initial schema generation prompt."""
    type_hint  = _confidence_hint(doc_type, confidence, features)
    feat_hints = _feature_hints(features)

    return f"""You are a legal document structure analyst specialising in Indian statutory law.
Analyse this document sample and return a JSON parsing schema.

{type_hint}

Structural Signals:
{feat_hints}

Document Sample (representative windows — head, body, tail):
---
{text_sample[:12_000]}
---

Return a JSON object with EXACTLY these fields:
{{
  "section_pattern":       "<Python lookahead regex for re.split() to split into sections>",
  "chapter_pattern":       "<Python regex capturing (chapter_id, title) or null>",
  "part_pattern":          "<Python regex capturing (part_id, title) or null>",
  "title_extract_pattern": "<Python regex with groups (section_num, title)>",
  "hierarchy":             ["chapter", "part", "section"],
  "flags": {{
    "has_definitions": true,
    "has_schedules":   false,
    "has_orders":      false,
    "has_parts":       false
  }},
  "metadata_defaults": {{ "year": "<4-digit year as string>" }}
}}

Rules:
- section_pattern MUST use a lookahead (?=...) so re.split() keeps delimiters.
- Split ONLY at top-level sections, never at sub-clauses like (1)(a).
- Account for amendment brackets: e.g. "[12A. " → use optional \\\\d*\\\\[ prefix.
- Chapters use ROMAN NUMERALS (CHAPTER I, II, X) — use [IVXLCDM]+ NOT \\\\d+.
- title_extract_pattern must stop at first '--', '\u2014', or '\\\\n'. Avoid (.*).
- All backslashes in regex strings MUST be double-escaped (\\\\n not \\n).
- Return ONLY the JSON object — no markdown fences, no explanation."""


def build_patch_prompt(
    schema: Dict[str, Any],
    failure_report: Dict[str, Any],
    text_sample: str,
) -> str:
    """Attempt 2 — patch existing schema using failure evidence."""
    issues  = "\n".join(f"  - {i}" for i in failure_report.get("issues", ["(none)"]))
    metrics = failure_report.get("metrics", {})
    samples = "\n---\n".join(failure_report.get("raw_text_samples", [])[:2])

    return f"""A parsing schema FAILED sample validation for an Indian statutory document.

Current Schema:
{json.dumps(schema, indent=2)}

Failure Metrics:
  section_capture_rate : {metrics.get('section_capture_rate', 0):.2f}  (target >= 0.40)
  chunk_length_sanity  : {metrics.get('chunk_length_sanity',  0):.2f}
  section_continuity   : {metrics.get('section_continuity',   0):.2f}
  overall              : {metrics.get('overall',              0):.2f}  (target >= 0.55)

Issues Detected:
{issues}

Failed Chunk Samples:
---
{samples[:1_500]}
---

Text Sample:
---
{text_sample[:2_000]}
---

Output a JSON PATCH with ONLY the fields that need changing.
Rules:
- section_pattern MUST use a lookahead (?=...).
- All regex must be valid Python re syntax with double-escaped backslashes.
- Only include fields that genuinely need updating.
- Return ONLY the JSON patch object — no markdown, no explanation."""


def build_fresh_prompt(
    text_sample: str,
    failure_summary: str,
    features: Dict[str, Any],
) -> str:
    """Attempt 3 — discard prior schema, generate fresh simple/broad schema."""
    feat_hints = _feature_hints(features)

    return f"""A parsing schema for an Indian statutory document has FAILED after a repair attempt.

DISCARD all prior schema assumptions. Generate a fresh, simple, broad schema from scratch.
Prefer simpler patterns over specific ones. When in doubt, use broader regex.

Structural signals from document:
{feat_hints}

Previous failure summary: {failure_summary}

Document Sample:
---
{text_sample[:8_000]}
---

Return a JSON object with EXACTLY these fields:
{{
  "section_pattern":       "<simple broad lookahead regex>",
  "chapter_pattern":       "<simple chapter regex or null>",
  "part_pattern":          null,
  "title_extract_pattern": "<simple title regex>",
  "hierarchy":             ["chapter", "section"],
  "flags": {{
    "has_definitions": false,
    "has_schedules":   false,
    "has_orders":      false,
    "has_parts":       false
  }},
  "metadata_defaults": {{ "year": "" }}
}}

Guidance for simple schemas:
- Use (?=\\\\n\\\\s*\\\\d{{1,3}}[A-Z]?\\\\.\\\\s) as section_pattern when uncertain.
- Use (CHAPTER\\\\s+[IVXLCDM]+)\\\\s*\\\\n([^\\\\n]*) for chapter_pattern if chapters exist.
- Use ^(\\\\d+[A-Z]?)\\\\.\\\\s*([^\\\\n\u2014.{{}}]{{0,150}}) for title_extract_pattern.
- Return ONLY the JSON object — no markdown, no explanation."""


# ─────────────────────────────────────────────────────────────────────
# Response parsing (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────

def parse_schema_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract and validate the JSON schema from the LLM response."""
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
    json_str = m.group(1) if m else None

    if not json_str:
        m2 = re.search(r"\{[\s\S]*\}", response_text)
        json_str = m2.group(0) if m2 else None

    if not json_str:
        logger.warning("Schema parse: no JSON object found in LLM response")
        return None

    try:
        schema = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Schema parse: JSON decode error — {e}")
        return None

    for field in ("section_pattern", "hierarchy"):
        if field not in schema:
            logger.warning(f"Schema parse: missing required field '{field}'")
            return None

    try:
        re.compile(schema["section_pattern"])
    except re.error as e:
        logger.warning(f"Schema parse: section_pattern invalid regex — {e}")
        return None

    return schema


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _llm_call(llm, prompt: str) -> Optional[Dict[str, Any]]:
    """Invoke LLM, parse response, return schema or None."""
    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        return parse_schema_response(content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def _quick_validate(
    schema: Dict[str, Any],
    sample_text: str,
    doc_title: str,
) -> Tuple[bool, Dict[str, float], List[Dict[str, Any]]]:
    """
    Run SchemaChunker on sample_text and apply lightweight quality checks.
    Returns (passed, metrics, chunks).

    Thresholds are intentionally lenient — this is a fast pre-validation
    on ~15K chars, not a full-document quality gate.
    """
    # Lazy imports — avoids circular import at module load time
    from ..stage3_parsing.schema_chunker import SchemaChunker  # noqa: PLC0415
    from ..utils.scoring import compute_quality_score          # noqa: PLC0415

    try:
        chunks = SchemaChunker(schema, doc_title).parse(sample_text)
    except Exception as e:
        logger.warning(f"Pre-validation chunker error: {e}")
        return False, {"overall": 0.0}, []

    if not chunks:
        return False, {"overall": 0.0, "chunk_count": 0.0}, []

    metrics = compute_quality_score(chunks)
    metrics["chunk_count"]   = float(len(chunks))
    max_len = max((len(c.get("text", "")) for c in chunks), default=0)
    metrics["max_chunk_len"] = float(max_len)

    has_giant = max_len > _PRE_MAX_CHUNK
    passed = (
        len(chunks)                          >= _PRE_MIN_CHUNKS
        and metrics["section_capture_rate"]  >= _PRE_MIN_CAPTURE
        and metrics["overall"]               >= _PRE_MIN_OVERALL
        and not has_giant
    )

    logger.debug(
        f"Pre-val: chunks={len(chunks)}  "
        f"capture={metrics['section_capture_rate']:.2f}  "
        f"overall={metrics['overall']:.2f}  "
        f"max_len={max_len}  giant={has_giant}  passed={passed}"
    )
    return passed, metrics, chunks


def _failure_report(
    chunks: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Build a lightweight failure report for the patch prompt."""
    from ..stage4_evaluation.critic import analyze_failures  # noqa: PLC0415
    return analyze_failures(chunks, metrics)


def _failure_summary(metrics: Dict[str, float]) -> str:
    """One-line description of the worst failure mode."""
    problems: List[str] = []
    if metrics.get("section_capture_rate", 1.0) < _PRE_MIN_CAPTURE:
        problems.append("low section capture (pattern not matching section numbers)")
    if metrics.get("max_chunk_len", 0) > _PRE_MAX_CHUNK:
        problems.append("giant chunks (section_pattern not splitting correctly)")
    if metrics.get("section_continuity", 1.0) < 0.4:
        problems.append("non-sequential section numbers (wrong split boundaries)")
    if metrics.get("overall", 1.0) < _PRE_MIN_OVERALL:
        problems.append(f"low overall score ({metrics.get('overall', 0):.2f})")
    return "; ".join(problems) if problems else "general quality failure"


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def generate_schema(
    llm,
    text: str,
    doc_type: str,
    features: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Send a schema-generation prompt to the LLM and return the parsed schema.
    Returns None on any failure (LLM unavailable, bad JSON, invalid regex, …).

    Preserved exactly for backward compatibility (HYBRID strategy uses this).
    Uses multi-window sampling but no pre-validation loop.
    """
    sample = _combined_sample(text)
    prompt = build_schema_prompt(sample, doc_type, features, confidence=1.0)
    return _llm_call(llm, prompt)


def generate_validated_schema(
    llm,
    text: str,
    doc_type: str,
    confidence: float,
    features: Dict[str, Any],
    doc_title: str = "Unknown",
) -> Optional[Dict[str, Any]]:
    """
    Generate a schema and pre-validate it on sample windows using a
    3-attempt evidence-driven retry loop — all on ~15K sample text,
    NOT on the full PDF.

    Attempt 1 — Initial schema → validate on sample
    Attempt 2 — PATCH using failure metrics + bad chunk evidence
    Attempt 3 — FRESH broad schema (discard prior assumptions)

    Returns the highest-scoring schema found across all attempts,
    even if none passed the validation threshold (Stage 4/5/6 remain
    as safety nets after the full parse).

    Returns None only if every LLM call fails entirely.

    Parameters
    ----------
    llm        : LangChain LLM instance
    text       : full cleaned document text
    doc_type   : doc-type label from Stage 1 (used as weak prior)
    confidence : Stage 1 confidence — governs how strongly doc_type is used
    features   : structural flags from Stage 1
    doc_title  : document label for SchemaChunker prefixes
    """
    sample_text = _combined_sample(text)

    best_schema:  Optional[Dict[str, Any]] = None
    best_score:   float                    = -1.0
    last_metrics: Dict[str, float]         = {}
    last_chunks:  List[Dict[str, Any]]     = []

    for attempt in range(3):
        logger.info(f"[schema] Pre-validation attempt {attempt + 1}/3")

        # ── Build attempt-specific prompt ────────────────────────────
        if attempt == 0:
            prompt = build_schema_prompt(sample_text, doc_type, features, confidence)

        elif attempt == 1:
            if best_schema is None:
                # Attempt 1's LLM call failed entirely — retry initial prompt
                prompt = build_schema_prompt(sample_text, doc_type, features, confidence)
            else:
                report = _failure_report(last_chunks, last_metrics)
                prompt = build_patch_prompt(best_schema, report, sample_text)

        else:  # attempt == 2
            summary = _failure_summary(last_metrics)
            prompt  = build_fresh_prompt(sample_text, summary, features)

        # ── LLM call ─────────────────────────────────────────────────
        schema = _llm_call(llm, prompt)
        if schema is None:
            logger.warning(f"[schema] Attempt {attempt + 1}: no valid schema returned")
            continue

        # ── Sample pre-validation ─────────────────────────────────────
        passed, metrics, chunks = _quick_validate(schema, sample_text, doc_title)
        score = metrics.get("overall", 0.0)

        logger.info(
            f"[schema] Attempt {attempt + 1}: score={score:.3f}  "
            f"capture={metrics.get('section_capture_rate', 0):.2f}  "
            f"chunks={int(metrics.get('chunk_count', 0))}  "
            f"passed={passed}"
        )

        if score > best_score:
            best_score   = score
            best_schema  = schema
            last_metrics = metrics
            last_chunks  = chunks

        if passed:
            logger.info(f"[schema] Pre-validation PASSED on attempt {attempt + 1}")
            return best_schema

    # All 3 attempts done
    if best_schema is not None:
        logger.warning(
            f"[schema] Pre-validation: 3 attempts exhausted, best_score={best_score:.3f}. "
            f"Returning best schema — full parse + Stage 4/5 will evaluate further."
        )
    else:
        logger.error("[schema] All 3 LLM attempts failed to return any valid schema.")

    return best_schema
