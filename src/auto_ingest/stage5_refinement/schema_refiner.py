"""
stage5_refinement/schema_refiner.py — Full-document schema patch generator.

Role in the new architecture
-----------------------------
Stage 2 (generate_validated_schema) already ran a 3-attempt intelligent
retry loop on sample windows before the full parse.  By the time this
module is called, the schema has ALREADY passed sample validation but
failed on the full document.

Therefore the failure is likely a full-document-specific issue:
  - Late-stage numbering changes (section 100+ use different format)
  - Schedules / Orders near the tail not matched by section_pattern
  - Chapter transitions far into the document
  - Long-range continuity drift from a slightly-too-broad split
  - Appendix / Form noise in the tail inflating noise_ratio

The prompt reflects this context: we are NOT asking for a complete
schema redesign — we are asking for a targeted patch to an already
mostly-working schema.

Public API (unchanged):
    build_refinement_prompt(schema, failure_analysis, text_sample) → str
    apply_patch(schema, patch)                                      → dict
    refine_schema(llm, schema, failure_analysis, text_sample)       → Optional[dict]
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex fields that must compile if present in a patch
_REGEX_FIELDS = (
    "section_pattern",
    "chapter_pattern",
    "part_pattern",
    "title_extract_pattern",
)


# ─────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────

def _failure_guidance(issues: List[str], metrics: Dict[str, float]) -> str:
    """
    Build targeted fix guidance based on which metrics failed and what
    issues the critic detected. This gives the LLM concrete direction
    instead of a vague 'it failed' message.
    """
    hints: List[str] = []

    capture = metrics.get("section_capture_rate", 1.0)
    cont    = metrics.get("section_continuity",   1.0)
    length  = metrics.get("chunk_length_sanity",  1.0)
    noise   = metrics.get("noise_ratio",          1.0)
    chap    = metrics.get("chapter_coverage",     1.0)

    if capture < 0.50:
        hints.append(
            "SECTION CAPTURE LOW: section_pattern is missing many sections in the "
            "full document. Consider whether the later parts of the document use a "
            "slightly different numbering format (e.g. '100A.' vs '[100A.') or if "
            "the pattern is over-anchored to the document head."
        )
    if length < 0.50:
        if any("giant" in i.lower() or "too long" in i.lower() for i in issues):
            hints.append(
                "GIANT CHUNKS: section_pattern is missing split points in the middle "
                "or tail of the document. Increase split sensitivity — make the "
                "section_pattern slightly broader or add an alternate boundary."
            )
        else:
            hints.append(
                "CHUNK LENGTH ISSUE: many chunks are too short (over-split) or too "
                "long (under-split). Adjust the section_pattern boundary."
            )
    if cont < 0.50:
        hints.append(
            "CONTINUITY BROKEN: section numbers are not monotonically increasing. "
            "The section_pattern may be splitting at sub-clause boundaries like (1), "
            "(2) instead of top-level sections. Tighten the pattern to avoid "
            "matching sub-sections."
        )
    if noise > 0.0 and noise < 0.60:
        hints.append(
            "HIGH NOISE: many chunks contain only page numbers, headers, or appendix "
            "boilerplate near the document tail. Consider whether the section_pattern "
            "can be tightened to avoid splitting appendix/form sections."
        )
    if chap < 0.40:
        hints.append(
            "CHAPTER COLLAPSE: chapter_pattern is not tracking chapters through the "
            "full document. The pattern may only match the initial chapters and miss "
            "later ones (e.g. CHAPTER X vs CHAPTER I–V). Verify [IVXLCDM]+ covers "
            "all Roman numerals present."
        )

    if not hints:
        hints.append(
            "GENERAL QUALITY FAILURE: overall score is below threshold. Review the "
            "section_pattern for edge cases that appear in the later half of the document."
        )

    return "\n\n".join(hints)


def build_refinement_prompt(
    schema:           Dict[str, Any],
    failure_analysis: Dict[str, Any],
    text_sample:      str,
) -> str:
    """
    Build the full-document schema refinement prompt.

    Key context communicated to the LLM:
    - The schema already passed SAMPLE validation in Stage 2.
    - It failed on the FULL document — so the issue is likely in the
      middle/tail, not the head.
    - We need a PATCH only, not a full regeneration.
    """
    issues  = failure_analysis.get("issues", ["(no specific issues listed)"])
    metrics = failure_analysis.get("metrics", {})
    samples = failure_analysis.get("raw_text_samples", [])

    issues_text = "\n".join(f"  - {i}" for i in issues)
    samples_text = "\n---\n".join(s[:600] for s in samples[:3])
    guidance     = _failure_guidance(issues, metrics)
    schema_json  = json.dumps(schema, indent=2)

    return f"""You are a legal document parsing expert specialising in Indian statutory PDFs.

CONTEXT:
This schema was generated and VALIDATED on a sample of the document in an earlier stage.
It passed sample quality checks but FAILED when applied to the FULL PDF.
This means the failure is likely in the middle or tail of the document, not the head.

Do NOT redesign the schema. Output a PATCH with ONLY the fields that need changing.

Current Schema:
{schema_json}

Full-Document Quality Failure:
  section_capture_rate : {metrics.get('section_capture_rate', 0):.2f}  (target >= 0.70)
  chunk_length_sanity  : {metrics.get('chunk_length_sanity',  0):.2f}
  section_continuity   : {metrics.get('section_continuity',   0):.2f}
  chapter_coverage     : {metrics.get('chapter_coverage',     0):.2f}
  noise_ratio          : {metrics.get('noise_ratio',          0):.2f}
  overall              : {metrics.get('overall',              0):.2f}  (target >= 0.70)

Critic Issues Detected:
{issues_text}

Targeted Fix Guidance (based on which metrics failed):
{guidance}

Representative Failed Chunk Samples (from the full document):
---
{samples_text[:2_000]}
---

Reference Text Sample (for pattern testing):
---
{text_sample[:2_500]}
---

Output a JSON PATCH with ONLY the fields that need changing.
Example (if only section_pattern needs fixing):
{{"section_pattern": "(?=\\\\n\\\\s*\\\\d{{1,3}}[A-Z]?\\\\.\\\\s)"}}

Rules:
- section_pattern MUST use a lookahead (?=...) for re.split().
- NEVER split at sub-clauses like (1), (2), (a), (b).
- All regex must be valid Python re syntax with double-escaped backslashes.
- Only include fields that genuinely need updating (minimal patch).
- Return ONLY the JSON patch object — no markdown fences, no explanation."""


# ─────────────────────────────────────────────────────────────────────
# Patch application
# ─────────────────────────────────────────────────────────────────────

def apply_patch(
    schema: Dict[str, Any],
    patch:  Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-merge patch into schema (nested dicts are merged, not replaced)."""
    updated = dict(schema)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = {**updated[key], **value}
        else:
            updated[key] = value
    return updated


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def refine_schema(
    llm,
    schema:           Dict[str, Any],
    failure_analysis: Dict[str, Any],
    text_sample:      str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a schema patch using full-document failure evidence and
    return the merged (patched) schema.

    Returns None on any failure so the caller can gracefully fall through
    to Stage 6 without an exception.
    """
    prompt = build_refinement_prompt(schema, failure_analysis, text_sample)
    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[stage5] Schema refiner LLM call failed: {e}")
        return None

    # ── Extract JSON patch ───────────────────────────────────────────
    m        = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    json_str = m.group(1) if m else None
    if not json_str:
        m2       = re.search(r"\{[\s\S]*\}", content)
        json_str = m2.group(0) if m2 else None

    if not json_str:
        logger.warning("[stage5] Schema refiner: no JSON patch found in LLM response")
        return None

    try:
        patch = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[stage5] Schema refiner: JSON decode error — {e}")
        return None

    # ── Validate and drop invalid regex fields ───────────────────────
    for field in _REGEX_FIELDS:
        if field in patch and patch[field]:
            try:
                re.compile(patch[field])
            except re.error as e:
                logger.warning(
                    f"[stage5] Schema refiner: {field!r} is invalid regex "
                    f"— dropping from patch ({e})"
                )
                patch.pop(field)

    if not patch:
        logger.warning("[stage5] Schema refiner: patch is empty after validation — nothing to apply")
        return None

    merged = apply_patch(schema, patch)
    logger.info(f"[stage5] Schema refiner: patch applied — fields: {list(patch.keys())}")
    return merged
