"""
stage3_parsing/schema_chunker.py — Schema-driven stateful chunker.

Consumes a schema dict (from Stage 2 LLM generation) and runs a state
machine over the document text to produce structured chunks.

State tracked per chunk:
  current_chapter, current_part, current_section
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_SEC_PATTERN = r"(?=\n\s*\d{1,3}[A-Z]?\.\s)"
_DEFAULT_TITLE_RE    = r"^(\d+[A-Z]?)\.\s*([^\n(]{0,200})"


class SchemaChunker:
    """
    Stateful parser driven by an LLM-generated schema dict.

    Usage
    -----
    chunker = SchemaChunker(schema, doc_title="RTI 2005")
    chunks  = chunker.parse(text)
    """

    def __init__(self, schema: Dict[str, Any], doc_title: str = "Unknown"):
        self.schema      = schema
        self.doc_title   = doc_title

        self.sec_pattern   = schema.get("section_pattern",       _DEFAULT_SEC_PATTERN)
        self.chap_pattern  = schema.get("chapter_pattern")
        self.part_pattern  = schema.get("part_pattern")
        self.title_re      = schema.get("title_extract_pattern", _DEFAULT_TITLE_RE)
        self.hierarchy     = schema.get("hierarchy", ["chapter", "section"])
        self.meta_defaults = schema.get("metadata_defaults", {})

        # State
        self.current_chapter = "Unknown Chapter"
        self.current_part    = "Unknown Part"
        self.current_section = "0"
        self.chunk_index     = 0

    # ── State updaters ────────────────────────────────────────────────

    def _apply_heading(
        self,
        text: str,
        pattern: str | None,
        current: str,
    ) -> Tuple[str, str]:
        """Try to extract a heading using `pattern`; update `current` and strip it."""
        if not pattern:
            return current, text
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
                text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
                return heading, text
        except re.error as e:
            logger.debug(f"SchemaChunker heading pattern error: {e}")
        return current, text

    # ── Section head extraction ───────────────────────────────────────

    def _extract_sec_head(self, text: str) -> Tuple[str, str, str]:
        """
        Parse (section_number, section_title, content_body) from a block.
        Falls back to generic patterns when the schema's title_re fails.
        Title is always trimmed at the first '--', em-dash, or newline.
        """
        def _clean_title(raw: str) -> str:
            """Trim title at the first body-text separator."""
            for sep in ("\u2014", ".—", ".--", "--", "\n"):
                idx = raw.find(sep)
                if idx != -1:
                    raw = raw[:idx]
            return raw.strip()[:200]

        try:
            m = re.match(self.title_re, text, re.DOTALL)
            if m:
                groups = m.groups()
                sec_num   = groups[0] if groups else "N/A"
                sec_title = _clean_title(groups[1]) if len(groups) > 1 else ""
                content   = text[m.end():].strip()
                # If content is empty the regex consumed the whole block — re-derive
                if not content:
                    rest = text[text.find(".") + 1:].strip() if "." in text else text
                    content = rest
                return sec_num, sec_title, content
        except re.error as e:
            logger.debug(f"SchemaChunker title_re error: {e}")

        # Generic fallback
        first_line, *rest = text.split("\n", 1)
        m2 = re.match(r"^(?:\d*\[)?\[?(\d+[A-Z]*)\.\s*(.*)", first_line.strip())
        if m2:
            return m2.group(1), _clean_title(m2.group(2)), (rest[0].strip() if rest else "")

        return "N/A", "", text

    # ── Main parse ────────────────────────────────────────────────────

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Split text by schema section_pattern and produce chunk dicts."""
        try:
            raw_blocks = re.split(self.sec_pattern, text)
        except re.error as e:
            logger.error(f"SchemaChunker: section_pattern failed ({e}), using text as one block")
            raw_blocks = [text]

        raw_blocks = [b for b in raw_blocks if b.strip()]
        logger.info(f"[schema_chunker] {len(raw_blocks)} raw blocks")

        chunks: List[Dict[str, Any]] = []
        law_code = self.doc_title
        year     = self.meta_defaults.get("year", "")
        label    = f"{law_code} {year}".strip() if year else law_code

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            next_chapter, block = self._apply_heading(
                block, self.chap_pattern, self.current_chapter
            )
            next_part, block = self._apply_heading(
                block, self.part_pattern, self.current_part
            )

            if not block.strip():
                # Block was solely headings, or empty. Apply state immediately for the next block.
                self.current_chapter = next_chapter
                self.current_part = next_part
                continue

            sec_num, sec_title, content = self._extract_sec_head(block)
            self.current_section = sec_num
            self.chunk_index += 1

            chunks.append(
                {
                    "text": (
                        f"[{label}] [{self.current_chapter}] "
                        f"Section {sec_num} \u2014 {sec_title}\n"
                        f"{content}"
                    ),
                    "metadata": {
                        "doc_title":       self.doc_title,
                        "doc_type":        self.meta_defaults.get("doc_type", "GENERIC"),
                        "law_code":        law_code,
                        "year":            year,
                        "section_number":  sec_num,
                        "section_title":   sec_title[:200],
                        "chapter":         self.current_chapter,
                        "part":            self.current_part,
                        "hierarchy_path":  f"{self.current_chapter} > Section {sec_num}",
                        "chunk_index":     self.chunk_index,
                        "parse_strategy":  "schema",
                    },
                }
            )
            
            self.current_chapter = next_chapter
            self.current_part = next_part

        logger.info(
            f"[schema_chunker] produced {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c['metadata']['section_number'] != 'N/A')} with IDs)"
        )
        return chunks
