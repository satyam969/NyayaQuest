"""
stage4_evaluation/critic.py — Failure analyst.

Analyses the failing chunk set and produces a structured diagnostic
report that is fed into the schema-refinement loop (Stage 5).
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def analyze_failures(
    chunks: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Produce a human-readable and machine-readable failure report.

    Returns a dict with:
      issues          : List[str]   — textual issue descriptions
      sample_failures : List[dict]  — up to 5 representative bad chunks
      raw_text_samples: List[str]   — raw text of up to 3 bad chunks
      metrics         : dict        — the scores that triggered the failure
      total_chunks    : int
      failed_section_count : int
      failed_short_count   : int
    """
    failed_sec   = [c for c in chunks if c["metadata"].get("section_number", "N/A") == "N/A"]
    failed_short = [c for c in chunks if len(c.get("text", "")) < 50]

    issues: List[str] = []

    if metrics.get("section_capture_rate", 1.0) < 0.5:
        issues.append(
            "Low section capture rate — section_pattern likely does not match "
            "this document's numbering format."
        )
    if metrics.get("chunk_length_sanity", 1.0) < 0.5:
        issues.append(
            "Many chunks are too short or too long — split boundaries may be incorrect."
        )
    if metrics.get("noise_ratio", 1.0) < 0.5:
        issues.append(
            "High noise ratio — many chunks appear to be headers, "
            "page numbers, or blank regions."
        )
    if metrics.get("section_continuity", 1.0) < 0.4:
        issues.append(
            "Section numbers are not monotonically increasing — "
            "the parser may be mis-splitting the document."
        )
    if metrics.get("chapter_coverage", 1.0) < 0.3:
        issues.append(
            "Very few chunks have chapter context — chapter_pattern may be wrong or missing."
        )

    sample_failures = []
    for c in (failed_sec + failed_short)[:5]:
        sample_failures.append(
            {
                "text_preview": c.get("text", "")[:300],
                "metadata":     c.get("metadata", {}),
            }
        )

    raw_samples = [c.get("text", "")[:500] for c in failed_sec[:3]]

    report = {
        "issues":               issues,
        "sample_failures":      sample_failures,
        "raw_text_samples":     raw_samples,
        "metrics":              metrics,
        "total_chunks":         len(chunks),
        "failed_section_count": len(failed_sec),
        "failed_short_count":   len(failed_short),
    }

    logger.info(
        f"Critic: {len(issues)} issue(s) identified | "
        f"failed_sections={len(failed_sec)} / {len(chunks)}"
    )
    return report
