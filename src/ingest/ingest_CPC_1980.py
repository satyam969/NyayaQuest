import os
import glob
import re
import fitz                          # PyMuPDF — used directly for footer removal
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = "data/legal_pdfs"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"

# ── TUNABLE PARAMETERS ────────────────────────────────────────────────────────
# Larger chunks preserve more statutory context per retrieval hit.
# Overlap ensures a clause that straddles a boundary appears in both chunks.
CHUNK_SIZE    = 800   # was 400 — doubled to keep full provisos / sub-clauses intact
CHUNK_OVERLAP = 100   # was 50


# ---------------- CLEAN TEXT ----------------
def clean_text(text):

    # Normalize the "ù" PDF encoding artifact used as separator after section headings
    text = text.replace("ù", " ")
    # Other common PDF encoding artifacts (dashes, bullets)
    text = text.replace("\uf0b7", " ")   # bullet
    text = text.replace("\u2013", "-")   # en-dash
    text = text.replace("\u2014", "--")  # em-dash

    # Remove standalone page numbers (a lone digit(s) on its own line)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # Remove underscores / horizontal rules
    text = re.sub(r'_+', '', text)

    # Collapse excessive internal spaces (but keep newlines)
    text = re.sub(r' {2,}', ' ', text)

    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Covers: amendment notices, substitution notes, and editorial state-amendment notes
    # like "1. This Act has been amended in its application to Assam..."
    # The [^\[\n]*? and \d{1,2} checks ensure we don't accidentally match and delete 
    # repealed sections like "68. [Title...] - Rep. by..." which contain brackets.
    text = re.sub(
        r'\n\s*\d{1,2}\.\s+(?=[^\[\n]*?'
        r'(Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|'
        r'amended in its application|extended to .{1,40} by Act|'
        r'extended to the .{1,40} by)'
        r')[^\n]+',
        '',
        text, flags=re.IGNORECASE
    )

    return text.strip()


# ---------------- FOOTER-AWARE TEXT EXTRACTION ----------------
def extract_text_without_footers(pdf_path):
    """
    Extract the full text of a PDF page-by-page using native fitz,
    stripping legal footnotes before joining pages.

    Two-layer strategy (data-driven from PDF analysis):
    ────────────────────────────────────────────────────
    Layer 1 — Separator line (covers ~91% of pages):
      The CPC PDF draws a short horizontal vector line (~144 pts wide)
      between the main text and the footnote block.  We locate it with
      page.get_drawings() and discard every text block whose top edge
      (bbox[1]) is BELOW that line's y-coordinate.

    Layer 2 — Keyword fallback (remaining ~9% of pages):
      If no separator line is found, scan the extracted text lines from
      the bottom upward and strip lines that look like legal footnotes:
      numbered lines containing 'Subs.', 'Ins.', 'w.e.f.', 'A.O.' etc.
    """
    doc = fitz.open(pdf_path)
    page_texts = []

    for page in doc:
        # ── Layer 1: find the separator line ──────────────────────────────
        drawings = page.get_drawings()
        # A separator line is horizontal (height < 2 pts), at least 50 pts wide,
        # and sits in the lower half of the page (y > 40% of page height)
        page_h = page.rect.height
        sep_y = None
        for d in drawings:
            r = d['rect']
            is_horizontal = abs(r.y0 - r.y1) < 2
            is_wide_enough = r.width > 50
            in_lower_half  = r.y0 > page_h * 0.40
            if is_horizontal and is_wide_enough and in_lower_half:
                # Use the top-most qualifying line (there is usually only one)
                if sep_y is None or r.y0 < sep_y:
                    sep_y = r.y0

        # ── Extract text blocks — filter at LINE level ────────────────────
        # IMPORTANT: PyMuPDF sometimes groups main text AND footnotes into the
        # same block (Block 20 on page 35 is a real example: top=655, bot=769,
        # meaning it spans from main text above the separator all the way down
        # into the footnote zone below it).
        # Filtering at block level (checking block_top) misses these cases.
        # We must check each LINE's y-position individually.
        blocks = page.get_text('dict')['blocks']
        kept_lines = []
        for b in blocks:
            if 'lines' not in b:
                continue

            for line in b['lines']:
                line_top = line['bbox'][1]

                # Drop this line if its top edge is below (or at) the separator
                if sep_y is not None and line_top >= sep_y:
                    continue

                line_text = ''.join(s['text'] for s in line['spans'] if s['size'] >= 8.0)
                kept_lines.append(line_text)

        page_text = '\n'.join(kept_lines)

        # ── Layer 2: keyword scan — runs on ALL pages ──────────────────────
        # Even when Layer 1 found a separator line, a few footnote blocks can
        # still slip through if their bbox top is slightly above the line's y.
        # Scanning from the bottom up on the already-filtered text is cheap
        # and guarantees no footnotes leak into the chunks.
        lines = page_text.split('\n')
        i = len(lines) - 1
        in_footer = True
        while i >= 0 and in_footer:
            stripped = lines[i].strip()
            if not stripped:
                i -= 1
                continue
            is_numbered     = bool(re.match(r'^\d{1,2}\.\s', stripped))
            has_bracket     = '[' in stripped
            has_footnote_kw = bool(re.search(
                r'(Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|'
                r'amended in its application|extended to .{1,40} by Act|'
                r'extended to the .{1,40} by)',
                stripped, re.IGNORECASE
            ))

            if is_numbered and has_footnote_kw and not has_bracket:
                del lines[i]
                i -= 1
            elif stripped.startswith('*') and has_footnote_kw:
                i -= 1
            else:
                in_footer = False
        page_text = '\n'.join(lines[:i + 1])

        page_texts.append(page_text)

    doc.close()
    return '\n'.join(page_texts)


# ---------------- REMOVE TOC ----------------
def remove_toc(text):
    """
    Skip the Table of Contents pages.
    Real content starts at the PRELIMINARY block NOT followed by 'SECTIONS'.
    """
    match = re.search(r'\nPRELIMINARY\s*\n(?!SECTIONS)', text)
    if match:
        print(f"DEBUG: Real PRELIMINARY found at char {match.start()}")
        return text[match.start():]

    # Fallback: second occurrence of '1.  Short title'
    matches = list(re.finditer(r'\n\s*1\.\s+Short title', text))
    if len(matches) >= 2:
        print(f"DEBUG: Using 2nd 'Section 1. Short title' at char {matches[1].start()}")
        return text[matches[1].start():]

    print("DEBUG: TOC removal fallback — returning full text")
    return text


# ---------------- SPLIT MAIN TEXT FROM FIRST SCHEDULE ----------------
def remove_appendix(text):
    """
    Splits text into (main_sections_text, first_schedule_text).
    The First Schedule contains Orders I–LI and is ingested separately.
    Appendix A–D (forms/templates) are stripped by extract_orders().
    """
    match = re.search(r'\nTHE FIRST SCHEDULE', text, re.IGNORECASE)
    if match:
        return text[:match.start()], text[match.start():]

    # Fallback: if First Schedule header is missing, stop at Appendix
    match = re.search(r'\nAPPENDIX\s+[A-D]', text, re.IGNORECASE)
    if match:
        return text[:match.start()], ""

    return text, ""


# ---------------- SPLIT SECTIONS ----------------
def split_sections(text):
    """Split the text into a list of section strings."""
    # Splits on pattern: newline, optional spaces, optional asterisk, optional bracket, digits, optional A-Z, dot, optional space/capital
    # e.g. " 1. ", "\n2A. ", "\n[89. ", "\n119.Unauthorized", "\n*[15A."
    sections = re.split(r'(?=\n\s*\*?\[?\d{1,3}[A-Z]?\.(?:\s+|[A-Z]))', text)
    return [s for s in sections if s.strip()]


# ---------------- EXTRACT PART ----------------
def extract_part(section, current_part):
    """Detect PART headings inside a section block and update current_part."""
    match = re.search(r'(PART\s+[IVXLCDM]+)\s*\n+(.*)', section)
    if match:
        part_title = match.group(2).strip()
        part = f"{match.group(1)} - {part_title}"
        # Strip the PART heading from the section body
        section = re.sub(r'PART\s+[IVXLCDM]+\s*\n+.*?\n', '', section, count=1).strip()
        return part, section
    return current_part, section


# ---------------- DETECT DEFINITIONS ----------------
def is_definition_section(content):
    has_clauses  = re.search(r'\(\d+\)', content)
    has_quotes   = re.search(r'[\""\u201c\u201d].+?[\""\u201c\u201d]', content)
    has_keywords = re.search(r'\b(means|includes)\b', content, re.IGNORECASE)
    return has_clauses and has_quotes and has_keywords


def extract_definitions(section_number, section_title, content):
    """
    Parse numbered sub-clauses from a definitions section.
    Returns one record per defined term, with the parent section number intact.
    """
    # Pattern: (N) "term" [optional text] means/includes/include ...
    # We capture everything after the quotes as the "description" part to ensure nothing is lost.
    matches = re.findall(
        r'\n?\s*\((\d+)\)\s*[\""\u201c](.*?)[\""\u201d](.*?)(means|includes|include)\b\s*(.*?)(?=\n\s*\(\d+\)|$)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    results = []
    for clause, term, pre_verb, verb, post_verb in matches:
        desc_clean = (pre_verb + verb + " " + post_verb).strip()
        # Clean up standalone asterisks left over from deleted repeal text 
        desc_clean = re.sub(r'[\s\*]+$', '', desc_clean)
        results.append({
            "clause": clause,
            "term": term.strip(),
            "content": desc_clean
        })
    return results


# ---------------- EXTRACT ORDERS (First Schedule) ----------------
def extract_orders(schedule_text):
    """Extract ORDER I, ORDER II … ORDER LI blocks from the First Schedule.

    The schedule_text is truncated at the first APPENDIX A/B/C/D heading
    before the regex runs.  Without this, ORDER LI (the last order) would
    greedily consume all Appendix A–D template content that follows it,
    causing form templates to be ingested as rule chunks.
    """
    # Hard-stop before Appendix A / B / C / D standalone headings.
    # Must be ALL-CAPS on its own line to avoid matching inline citations
    # like "...in Form No. 3 in Appendix C, with such variations..."
    appendix_match = re.search(r'\nAPPENDIX\s+[A-D]\s*\n', schedule_text)
    if appendix_match:
        schedule_text = schedule_text[:appendix_match.start()]
        print(f"DEBUG: Appendix boundary found at char {appendix_match.start()} — truncating schedule text")
    else:
        print("DEBUG: No APPENDIX A-D boundary found in schedule text")

    return re.findall(
        r'(\[?ORDER\s+[IVXLCDM]+[A-Z]?[\s\S]*?)(?=\n\[?ORDER\s+[IVXLCDM]+[A-Z]?|\Z)',
        schedule_text
    )


# ---------------- EXTRACT RULES ----------------
def extract_rules(order_text):
    return re.split(r'(?=\n\s*\*?\[?\d+[A-Z]?\.(?:\s+|[A-Z]))', order_text)


# ---------------- PREPROCESS ----------------
def preprocess_document(pdf_path):

    # Extract full text with footers stripped (two-layer: separator line + keyword fallback)
    full_text = extract_text_without_footers(pdf_path)

    # ── CLEAN ──────────────────────────────────────────────────────────
    full_text = clean_text(full_text)

    print("\nDEBUG: BEFORE TOC REMOVAL:\n", full_text[:300])

    # ── REMOVE TOC ─────────────────────────────────────────────────────
    full_text = remove_toc(full_text)
    print("\nDEBUG: AFTER TOC REMOVAL:\n", full_text[:300])

    # ── SPLIT MAIN SECTIONS FROM FIRST SCHEDULE ─────────────────────────
    main_text, schedule_text = remove_appendix(full_text)
    print(f"DEBUG: Main text length: {len(main_text)}, Schedule text length: {len(schedule_text)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    basename = os.path.basename(pdf_path)
    results  = []
    current_part = "Unknown Part"

    # ================================================================
    # PART A — SECTIONS (main body, Sections 1–158)
    # ================================================================
    raw_sections = split_sections(main_text)
    print(f"\nDEBUG: Total raw sections found: {len(raw_sections)}")

    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue

        current_part, sec = extract_part(sec, current_part)

        # Allow optional asterisk/bracket and optional spacing for typos like '119.Unauthorized'
        match = re.match(r'^\*?\[?(\d+[A-Z]?)\.\s*(.*)', sec, re.DOTALL)
        if not match:
            continue

        section_number = match.group(1)
        raw_content    = match.group(2).strip()

        # Extract the section title (first line before any sub-clause / newline)
        title_match  = re.match(r'^([^\n\(]+)', raw_content)
        section_title = title_match.group(1).strip() if title_match else ""

        # Skip stub entries (TOC remnant — section title only, no substance)
        if len(raw_content) < 50:
            continue

        # ── DEFINITIONS ──────────────────────────────────────────────
        if is_definition_section(raw_content):
            defs = extract_definitions(section_number, section_title, raw_content)

            if defs:
                for d in defs:
                    # Prefix ALWAYS contains parent section number for retrieval
                    text = (
                        f"[CPC 1908] [{current_part}] Section {section_number} — {section_title}\n"
                        f"Clause ({d['clause']}) \"{d['term']}\": {d['content']}"
                    )
                    results.append({
                        "text": text,
                        "metadata": {
                            "type":    "definition",
                            "section": section_number,   # parent section (e.g. "2")
                            "clause":  d["clause"],
                            "term":    d["term"],
                            "part":    current_part,
                            "source":  basename
                        }
                    })
                continue  # definitions handled — skip normal chunking

        # ── NORMAL SECTIONS ──────────────────────────────────────────
        chunks = splitter.split_text(raw_content)
        for i, chunk in enumerate(chunks):
            # Always embed section number + title in every chunk for retrieval
            text = (
                f"[CPC 1908] [{current_part}] Section {section_number} — {section_title}\n"
                f"{chunk}"
            )
            results.append({
                "text": text,
                "metadata": {
                    "type":         "section",
                    "section":      section_number,
                    "section_title": section_title,
                    "part":         current_part,
                    "chunk_index":  i + 1,
                    "total_chunks": len(chunks),
                    "source":       basename
                }
            })

    # ================================================================
    # PART B — ORDERS & RULES (First Schedule)
    # ================================================================
    if schedule_text:
        orders = extract_orders(schedule_text)
        print(f"DEBUG: Orders found: {len(orders)}")

        for order_block in orders:
            order_header = order_block.split('\n')[0]
            order_match = re.search(r'^\[?ORDER\s+([IVXLCDM]+[A-Z]?)\s+(.*)', order_header, re.IGNORECASE)
            if not order_match:
                continue

            order_number = order_match.group(1)

            # Extract the Order title (line after ORDER N header)
            title_line = re.search(
                r'ORDER\s+[IVXLCDM]+[A-Z]?\s*\n+(.*?)(?:\nRULES|\n\d+\.)', order_block
            )
            order_title = title_line.group(1).strip() if title_line else ""

            rules = extract_rules(order_block)

            for rule_text in rules:
                rule_text = rule_text.strip()
                if not rule_text:
                    continue

                rule_match = re.match(r'^\*?\[?(\d+[A-Z]?)\.\s*(.*)', rule_text, re.DOTALL)
                if not rule_match:
                    continue

                rule_number = rule_match.group(1)
                rule_content = rule_match.group(2).strip()

                # Extract rule title
                rule_title_match = re.match(r'^([^\n\.]+)', rule_content)
                rule_title = rule_title_match.group(1).strip() if rule_title_match else ""

                if len(rule_content) < 20:
                    continue

                chunks = splitter.split_text(rule_content)
                for i, chunk in enumerate(chunks):
                    # Embed order number + rule number + titles in every chunk
                    text = (
                        f"[CPC 1908] Order {order_number} — {order_title}\n"
                        f"Rule {rule_number} — {rule_title}\n"
                        f"{chunk}"
                    )
                    results.append({
                        "text": text,
                        "metadata": {
                            "type":        "rule",
                            "order":       order_number,
                            "order_title": order_title,
                            "rule":        rule_number,
                            "rule_title":  rule_title,
                            "chunk_index": i + 1,
                            "total_chunks": len(chunks),
                            "source":      basename
                        }
                    })
    else:
        print("DEBUG: No First Schedule text found — skipping Orders/Rules ingestion")

    return results


# ---------------- INGEST ----------------
def ingest_cpc():

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    for pdf in pdf_files:
        if "cpc" not in pdf.lower():
            continue

        print(f"\nProcessing {pdf}...")

        chunks = preprocess_document(pdf)

        print(f"\nTotal chunks created: {len(chunks)}")

        if len(chunks) == 0:
            print("ERROR: No chunks extracted — check regex or debug output")
            continue

        ids  = [f"cpc_{i}" for i in range(len(chunks))]
        docs = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        for i in range(0, len(docs), 8):
            collection.add(
                ids=ids[i:i + 8],
                documents=docs[i:i + 8],
                metadatas=metas[i:i + 8]
            )

        print(f"Ingestion complete — {len(chunks)} chunks added to '{COLLECTION_NAME}'\n")


if __name__ == "__main__":
    ingest_cpc()