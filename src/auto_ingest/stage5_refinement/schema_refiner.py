"""
stage5_refinement/schema_refiner.py — LLM-based schema patch generator.

Takes a failing schema + failure analysis report and asks the LLM to
output a JSON patch containing only the fields that need changing.
The patch is validated (regex fields compiled) then merged onto the
existing schema using apply_patch().
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Regex fields that must compile if present in a patch
_REGEX_FIELDS = (
    "section_pattern",
    "chapter_pattern",
    "part_pattern",
    "title_extract_pattern",
)


def build_refinement_prompt(
    schema: Dict[str, Any],
    failure_analysis: Dict[str, Any],
    text_sample: str,
) -> str:
    issues_text = "\n".join(
        f"  - {i}" for i in failure_analysis.get("issues", ["(no specific issues listed)"])
    )
    samples = "\n---\n".join(failure_analysis.get("raw_text_samples", [])[:2])
    schema_json = json.dumps(schema, indent=2)

    return f"""You are a legal document parsing expert.
A regex-based parsing schema FAILED quality evaluation for an Indian statutory PDF.

Current Schema:
{schema_json}

Failure Issues:
{issues_text}

Failed Chunk Samples (text that did NOT parse correctly):
---
{samples[:2000]}
---

Document Text Sample:
---
{text_sample[:2000]}
---

Output a JSON PATCH with ONLY the fields that need changing.
Example (if only section_pattern needs fixing):
{{"section_pattern": "(?=\\\\n\\\\s*\\\\d{{1,3}}[A-Z]?\\\\.\\\\s)"}}

Rules:
- section_pattern MUST use a lookahead (?=...) for re.split().
- All regex must be valid Python re syntax.
- Only include fields that genuinely need updating.
- Return ONLY the JSON patch object — no markdown, no explanation."""


def apply_patch(
    schema: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-merge patch into schema (nested dicts are merged, not replaced)."""
    updated = dict(schema)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = {**updated[key], **value}
        else:
            updated[key] = value
    return updated


def refine_schema(
    llm,
    schema: Dict[str, Any],
    failure_analysis: Dict[str, Any],
    text_sample: str,
) -> Optional[Dict[str, Any]]:
    """
    Use the LLM to generate a schema patch and return the merged schema.
    Returns None on any failure so the caller can gracefully fall through.
    """
    prompt = build_refinement_prompt(schema, failure_analysis, text_sample)
    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Schema refinement LLM call failed: {e}")
        return None

    # Extract JSON patch
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    json_str = m.group(1) if m else None
    if not json_str:
        m2 = re.search(r"\{[\s\S]*\}", content)
        json_str = m2.group(0) if m2 else None

    if not json_str:
        logger.warning("Schema refiner: no JSON patch found in LLM response")
        return None

    try:
        patch = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Schema refiner: JSON decode error — {e}")
        return None

    # Validate regex fields
    for field in _REGEX_FIELDS:
        if field in patch and patch[field]:
            try:
                re.compile(patch[field])
            except re.error as e:
                logger.warning(f"Schema refiner: refined {field!r} is invalid regex — dropping it ({e})")
                patch.pop(field)

    merged = apply_patch(schema, patch)
    logger.info(f"Schema refiner: patch applied — changed fields: {list(patch.keys())}")
    return merged
