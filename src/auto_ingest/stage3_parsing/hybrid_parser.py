"""
stage3_parsing/hybrid_parser.py — Hybrid parser.

Runs both regex and schema parsers and returns whichever produces
more section-identified chunks (higher section_capture coverage).
"""

import logging
from typing import List, Dict, Any, Optional

from .regex_parser  import parse_with_regex
from .schema_chunker import SchemaChunker

logger = logging.getLogger(__name__)


def parse_hybrid(
    text: str,
    section_pattern: str,
    schema: Optional[Dict[str, Any]],
    doc_title: str = "Unknown",
    doc_type: str  = "GENERIC",
    features: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Run regex first, then schema (if available).
    Return the result set with the higher section-ID coverage.

    When section counts are equal, prefer schema chunks (richer metadata).
    """
    if features is None:
        features = {}

    regex_chunks = parse_with_regex(text, section_pattern, doc_title, doc_type, features)
    logger.info(f"[hybrid] regex → {len(regex_chunks)} chunks")

    if not schema:
        return regex_chunks

    sc = SchemaChunker(schema, doc_title)
    schema_chunks = sc.parse(text)
    logger.info(f"[hybrid] schema → {len(schema_chunks)} chunks")

    def _id_count(chunks):
        return sum(
            1 for c in chunks
            if c["metadata"].get("section_number", "N/A") not in ("N/A", "", None)
        )

    regex_ids  = _id_count(regex_chunks)
    schema_ids = _id_count(schema_chunks)

    logger.info(f"[hybrid] regex_ids={regex_ids}  schema_ids={schema_ids}")

    if schema_ids >= regex_ids:
        for c in schema_chunks:
            c["metadata"]["parse_strategy"] = "hybrid_schema"
        return schema_chunks
    else:
        for c in regex_chunks:
            c["metadata"]["parse_strategy"] = "hybrid_regex"
        return regex_chunks
