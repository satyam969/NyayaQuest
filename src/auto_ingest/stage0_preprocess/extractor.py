"""
stage0_preprocess/extractor.py — PDF text extraction via PyMuPDF.

Two-layer footer removal strategy (battle-tested on CPC, RTI, CPA, Labour Codes):
  Layer 1 — Separator line: find horizontal drawing > 50pt wide in lower 40% of page.
             Discard every text LINE whose top edge is below that y-coordinate.
  Layer 2 — Keyword scan from bottom-up: strip lingering footnote lines even when
             Layer 1 found the separator (some blocks straddle the boundary).
"""

import re
import fitz  # PyMuPDF
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────
# Raw page extraction (returns list of page dicts)
# ─────────────────────────────────────────────────────────────────────

def extract_raw_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all pages from a PDF.
    Returns a list of dicts: {page_num, text, height}.
    """
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        pages.append({
            "page_num": i,
            "text": page.get_text("text"),
            "height": page.rect.height,
        })
    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────────────
# Footer-aware extraction (main export)
# ─────────────────────────────────────────────────────────────────────

_FOOTNOTE_KW_RE = re.compile(
    r"(Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|"
    r"amended in its application|extended to .{1,40} by Act|"
    r"extended to the .{1,40} by|vide notification|Gazette of India)",
    re.IGNORECASE,
)


def extract_text_without_footers(
    pdf_path: str,
    min_font_size: float = 8.0,
) -> str:
    """
    Full text extraction with footer removal applied page-by-page.
    Returns the joined text of all pages.
    """
    doc = fitz.open(pdf_path)
    page_texts: List[str] = []

    for page in doc:
        page_h = page.rect.height

        # ── Layer 1: locate the separator line ───────────────────────
        valid_lines = []
        for d in page.get_drawings():
            r = d["rect"]
            is_horizontal = abs(r.y0 - r.y1) < 2
            is_wide_enough = r.width > 50 and r.width < page.rect.width * 0.45
            in_lower_half = r.y0 > page_h * 0.30
            if is_horizontal and is_wide_enough and in_lower_half:
                valid_lines.append(r.y0)
                
        sep_y: float | None = None
        # Exclude tables which have numerous horizontal grid lines
        if 0 < len(valid_lines) <= 2:
            sep_y = min(valid_lines)

        # ── Extract text at LINE level, applying separator cut ────────
        kept_lines: List[str] = []
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                if sep_y is not None and line["bbox"][1] >= sep_y:
                    continue  # below separator → footer
                line_text = "".join(
                    s["text"]
                    for s in line["spans"]
                    if s.get("size", 0) >= min_font_size
                )
                kept_lines.append(line_text)

        page_text = "\n".join(kept_lines)

        # ── Layer 2: keyword scan from the bottom up ──────────────────
        lines = page_text.split("\n")
        i = len(lines) - 1
        in_footer = True
        while i >= 0 and in_footer:
            stripped = lines[i].strip()
            if not stripped:
                i -= 1
                continue
            is_numbered = bool(re.match(r"^\d{1,2}\.\s", stripped))
            has_bracket = "[" in stripped
            has_kw = bool(_FOOTNOTE_KW_RE.search(stripped))

            if is_numbered and has_kw and not has_bracket:
                del lines[i]
                i -= 1
            elif stripped.startswith("*") and has_kw:
                i -= 1
            else:
                in_footer = False

        page_texts.append("\n".join(lines[: i + 1]))

    doc.close()
    return "\n".join(page_texts)


# ─────────────────────────────────────────────────────────────────────
# Text cleaning (encoding artefacts, Gazette headers, footnotes)
# ─────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Generic cleaning for Indian legal PDFs.
    Handles common encoding artefacts, Gazette headers, page numbers,
    editorial footnotes, and excess whitespace.
    """
    # Encoding artefacts
    text = text.replace("ù", " ")
    text = text.replace("\uf0b7", " ")    # bullet
    text = text.replace("\u2013", "-")    # en-dash
    text = text.replace("\u2014", "--")   # em-dash
    text = text.replace("\u2019", "'")    # right single quote
    text = text.replace("\u201c", '"')    # left double quote
    text = text.replace("\u201d", '"')    # right double quote

    # Standalone page numbers
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

    # Horizontal rules / underscores
    text = re.sub(r"_+", "", text)

    # Collapse excess inline spaces (preserve newlines)
    text = re.sub(r" {2,}", " ", text)

    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common Gazette-of-India running headers
    text = re.sub(r"(?im)^.*THE GAZETTE OF INDIA.*$", "", text)
    text = re.sub(r"(?im)^.*MINISTRY OF LAW AND JUSTICE.*$", "", text)
    text = re.sub(r"SEC\.\s*\d+[A-Z]?(\(\w+\))?\]?", "", text, flags=re.IGNORECASE)

    # Remove amendment / editorial footnotes embedded in the body
    text = re.sub(
        r"\n\s*\d{1,2}\.\s+(?=[^\[\n]*?"
        r"(Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|"
        r"amended in its application|extended to .{1,40} by Act|"
        r"extended to the .{1,40} by|vide notification|Gazette of India)"
        r")[^\n]+(?:\n(?!\s*(?:\(\w+\)\s|\d+\.\s))[^\n]+)*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()
