"""
stage2_strategy/schema_strategy.py — LLM-based schema generation.

Sends a document sample to the LLM and parses back a JSON schema that
describes how to split the document into sections.  The schema drives
the SchemaChunker in Stage 3.

Schema shape:
{
  "section_pattern":       "<lookahead regex for re.split()>",
  "chapter_pattern":       "<regex or null>",
  "part_pattern":          "<regex or null>",
  "title_extract_pattern": "<regex to pull section number + title>",
  "hierarchy":             ["chapter", "part", "section"],
  "flags": {
      "has_definitions": bool,
      "has_schedules":   bool,
      "has_orders":      bool,
      "has_parts":       bool
  },
  "metadata_defaults": {
      "law_code": "RTI",
      "year":     "2005"
  }
}
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────

def build_schema_prompt(
    text_sample: str,
    doc_type: str,
    features: Dict[str, Any],
) -> str:
    feature_lines = "\n".join(
        f"  {k}: {v}"
        for k, v in features.items()
        if k != "score_breakdown"
    )
    return f"""You are a legal document structure analyst specialising in Indian statutory law.
Analyse this document sample and return a JSON parsing schema.

Document Type Detected: {doc_type}
Structural Features:
{feature_lines}

Document Sample (first 3000 chars):
---
{text_sample[:3000]}
---

Return a JSON object with EXACTLY these fields:
{{
  "section_pattern":       "<Python lookahead regex for re.split() to split into sections>",
  "chapter_pattern":       "<Python regex capturing (chapter_id, title) or null>",
  "part_pattern":          "<Python regex capturing (part_id, title) or null>",
  "title_extract_pattern": "<Python regex with groups (section_num, title) matching ONLY the short heading before any body text>",
  "hierarchy":             ["chapter", "part", "section"],
  "flags": {{
    "has_definitions": true,
    "has_schedules":   false,
    "has_orders":      false,
    "has_parts":       false
  }},
  "metadata_defaults": {{
    "year":     "<4-digit year as string>"
  }}
}}

Rules:
- CRITICAL: section_pattern MUST use a lookahead (?=...) so re.split() keeps delimiters. It MUST ONLY split at top-level sections and NEVER at sub-sections enclosed in parentheses like "(1)". Because Indian statutes often prefix sections with amendment brackets or footnote numbers (e.g. "[12A. " or "1[12. "), the regex MUST account for optional leading digits and brackets before the main section number. Example: "(?=\\\\n\\\\s*(?:\\\\d*\\\\[)?\\\\d+[A-Z]*\\\\.\\\\s)"
- CRITICAL: Indian statutes use ROMAN NUMERALS for chapters (e.g. CHAPTER I, CHAPTER II, CHAPTER X). Your chapter_pattern MUST match Roman numerals using [IVXLCDM]+ NOT \\\\d+. Example: "CHAPTER\\\\s+([IVXLCDM]+)\\\\s*\\\\n([^\\\\n]*)"
- CRITICAL: title_extract_pattern must capture ONLY the short title (stopping at the first occurrence of '--', '\u2014', '.' or '\\\\n'). Do NOT use (.*) as it is too greedy. Example: "^(\\\\d+[A-Z]?)\\\\.\\\\s*([^.\\\\n\\\\-]{0,150})"
- CRITICAL: You are generating JSON. You MUST double-escape all backslashes in your regex strings. For example, use \\\\n instead of \\n, and \\\\s instead of \\s. Failure to double-escape will break the JSON parser.
- All regex must be valid Python re syntax.
- Return ONLY the JSON object — no markdown fences, no explanation."""


# ─────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────

def parse_schema_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract and validate the JSON schema from the LLM response."""
    # Try ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
    json_str = m.group(1) if m else None

    if not json_str:
        # Bare JSON object
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

    # Required field check
    for field in ("section_pattern", "hierarchy"):
        if field not in schema:
            logger.warning(f"Schema parse: missing required field '{field}'")
            return None

    # Validate section_pattern compiles
    try:
        re.compile(schema["section_pattern"])
    except re.error as e:
        logger.warning(f"Schema parse: section_pattern invalid regex — {e}")
        return None

    return schema


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
    """
    prompt = build_schema_prompt(text, doc_type, features)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return parse_schema_response(content)
    except Exception as e:
        logger.error(f"Schema generation LLM call failed: {e}")
        return None
