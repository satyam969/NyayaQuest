import os
import glob
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader
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

    return text.strip()


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
    """
    match = re.search(r'\nTHE FIRST SCHEDULE', text, re.IGNORECASE)
    if match:
        return text[:match.start()], text[match.start():]

    match = re.search(r'\nAPPENDIX', text, re.IGNORECASE)
    if match:
        return text[:match.start()], ""

    return text, ""


# ---------------- SPLIT SECTIONS ----------------
def split_sections(text):
    """
    Split at each new section heading: '1.', '2A.', '10B.' etc.
    Uses a lookahead so the delimiter is kept at the start of each chunk.
    """
    return re.split(r'(?=\n\s*\d{1,3}[A-Z]?\.\s)', text)


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


# ---------------- SEMANTIC SYNONYM LINE ----------------
def _semantic_header(section_number: str, section_title: str) -> str:
    """
    Returns a short 'Common queries:' line with alternative phrasings of the
    section title.  This boosts both BM25 and vector recall for short sections
    whose statutory text alone is too brief to rank well.
    """
    title = section_title.lower()
    synonyms = [section_title]  # always include the official title

    # ── Section-specific synonym map (catches known retrieval failures) ──
    _SYNONYMS = {
        "38":  ["court by which decree may be executed", "executing court", "which court executes decree", "power to execute decree"],
        "35B": ["costs for causing delay", "adjournment costs", "shall not be allowed further steps", "delay costs not paid"],
        "47":  ["questions determined by executing court", "no suit bar executing court", "executing court determines questions"],
        "52":  ["execution against legal representative", "personal liability legal rep", "execution after death judgment-debtor"],
        "21A": ["bar on suit to set aside decree", "no suit place of suing jurisdiction", "challenge decree jurisdiction"],
        "34":  ["interest on principal sum adjudged", "court may order interest", "date of suit date of decree interest"],
    }
    if section_number in _SYNONYMS:
        synonyms.extend(_SYNONYMS[section_number])
    else:
        # Generic: strip punctuation from title and add as extra term
        synonyms.append(re.sub(r'[^a-zA-Z0-9 ]', '', section_title))

    return "Common queries: " + " | ".join(synonyms)


# ---------------- EXTRACT DEFINITIONS ----------------
def extract_definitions(section_number, section_title, content):
    """
    Parse numbered sub-clauses from a definitions section.
    Returns one record per defined term, with the parent section number intact.
    """
    # Pattern: (N) "term" means/includes ...
    matches = re.findall(
        r'\((\d+)\)\s*[\""\u201c](.*?)[\""\u201d]\s*(means|includes)\s*(.*?)(?=\(\d+\)|$)',
        content,
        re.DOTALL
    )
    results = []
    for clause, term, _, desc in matches:
        desc_clean = desc.strip()
        results.append({
            "clause": clause,
            "term": term.strip(),
            "content": desc_clean
        })
    return results


# ---------------- EXTRACT ORDERS (First Schedule) ----------------
def extract_orders(schedule_text):
    """Extract ORDER I, ORDER II … blocks from the First Schedule text."""
    return re.findall(
        r'(ORDER\s+[IVXLCDM]+[\s\S]*?)(?=\nORDER\s+[IVXLCDM]+|\Z)',
        schedule_text
    )


# ---------------- EXTRACT RULES ----------------
def extract_rules(order_text):
    return re.split(r'(?=\n\s*\d+[A-Z]?\.\s)', order_text)


# ---------------- PREPROCESS ----------------
def preprocess_document(pdf_path):

    loader = PyMuPDFLoader(pdf_path)
    pages  = loader.load()

    full_text = "\n".join([p.page_content for p in pages])

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
    sections = split_sections(main_text)
    print(f"\nDEBUG: Total raw sections found: {len(sections)}")

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        current_part, sec = extract_part(sec, current_part)

        match = re.match(r'^(\d+[A-Z]?)\.\s+(.*)', sec, re.DOTALL)
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
                            "type":           "definition",
                            "law_code":       "CPC",
                            "year":           "1908",
                            "section":        section_number,   # parent section (e.g. "2")
                            "section_number": section_number,   # alias for DB Explorer
                            "clause":         d["clause"],
                            "term":           d["term"],
                            "chapter":        current_part,
                            "part":           current_part,
                            "source":         basename
                        }
                    })
                continue  # definitions handled — skip normal chunking

        # ── NORMAL SECTIONS ──────────────────────────────────────────
        chunks = splitter.split_text(raw_content)
        for i, chunk in enumerate(chunks):
            # Always embed section number + title + semantic synonyms in every chunk for retrieval
            synonym_line = _semantic_header(section_number, section_title)
            text = (
                f"[CPC 1908] [{current_part}] Section {section_number} — {section_title}\n"
                f"{synonym_line}\n"
                f"{chunk}"
            )
            results.append({
                "text": text,
                "metadata": {
                    "type":           "section",
                    "law_code":       "CPC",
                    "year":           "1908",
                    "section":        section_number,
                    "section_number": section_number,   # alias for DB Explorer
                    "section_title":  section_title,
                    "chapter":        current_part,
                    "part":           current_part,
                    "chunk_index":    i + 1,
                    "total_chunks":   len(chunks),
                    "source":         basename
                }
            })

    # ================================================================
    # PART B — ORDERS & RULES (First Schedule)
    # ================================================================
    if schedule_text:
        orders = extract_orders(schedule_text)
        print(f"DEBUG: Orders found: {len(orders)}")

        for order_block in orders:
            order_match = re.search(r'ORDER\s+([IVXLCDM]+)', order_block)
            if not order_match:
                continue

            order_number = order_match.group(1)

            # Extract the Order title (line after ORDER N header)
            title_line = re.search(
                r'ORDER\s+[IVXLCDM]+\s*\n+(.*?)(?:\nRULES|\n\d+\.)', order_block
            )
            order_title = title_line.group(1).strip() if title_line else ""

            rules = extract_rules(order_block)

            for rule in rules:
                rule = rule.strip()
                if not rule:
                    continue

                match = re.match(r'^(\d+[A-Z]?)\.\s+(.*)', rule, re.DOTALL)
                if not match:
                    continue

                rule_number = match.group(1)
                rule_content = match.group(2).strip()

                # Extract rule title
                rule_title_match = re.match(r'^([^\n\.]+)', rule_content)
                rule_title = rule_title_match.group(1).strip() if rule_title_match else ""

                if len(rule_content) < 20:
                    continue

                chunks = splitter.split_text(rule_content)
                for i, chunk in enumerate(chunks):
                    # Combine Order + Rule + both titles in one BM25-searchable header line
                    text = (
                        f"[CPC 1908] Order {order_number} Rule {rule_number} — {order_title} — {rule_title}\n"
                        f"Common queries: Order {order_number} Rule {rule_number} {order_title} {rule_title} CPC 1908\n"
                        f"{chunk}"
                    )
                    results.append({
                        "text": text,
                        "metadata": {
                            "type":           "rule",
                            "law_code":       "CPC",
                            "year":           "1908",
                            "order":          order_number,
                            "section_number": f"Order {order_number} Rule {rule_number}",  # for DB Explorer
                            "chapter":        f"Order {order_number} — {order_title}",
                            "order_title":    order_title,
                            "rule":           rule_number,
                            "rule_title":     rule_title,
                            "chunk_index":    i + 1,
                            "total_chunks":   len(chunks),
                            "source":         basename
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