import os
import re
import fitz
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
PDF_PATH = "data/legal_pdfs/Consumer Protection Act 2019_.pdf"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"
LAW_CODE = "CPA"
YEAR = "2019"


# -------------------------------
# 1. LOAD + CLEAN PDF (AGGRESSIVE)
# -------------------------------
def load_and_clean_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text("text")
        full_text += text + "\n"

    # --- CLEANING ---

    # 1. Remove ANY line containing "THE GAZETTE OF INDIA" 
    full_text = re.sub(r'(?im)^.*THE GAZETTE OF INDIA.*$', '', full_text)

    # 2. Remove generalized Part/Section tags left behind 
    full_text = re.sub(r'\[?PART\s+[IVX]+[—\-\]]?', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'SEC\.\s*\d+[A-Z]?(\(\w+\))?\]?', '', full_text, flags=re.IGNORECASE)

    # 3. Remove arrangement section ONLY if clearly present
    full_text = re.sub(
        r'ARRANGEMENT OF SECTIONS.*?CHAPTER I',
        'CHAPTER I',
        full_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 4. Remove repeated act titles
    full_text = re.sub(
        r'THE CONSUMER PROTECTION ACT,?\s*2019',
        '',
        full_text,
        flags=re.IGNORECASE
    )

    # 5. Remove standalone page numbers
    full_text = re.sub(r'^\s*\d+\s*$', '', full_text, flags=re.MULTILINE)

    # 6. Normalize whitespace
    full_text = re.sub(r'\r', '\n', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # ---------------------------------------------------------
    # 7. THE AGGRESSIVE PREAMBLE SLICER
    # ---------------------------------------------------------
    # Erase the specific margin note that scrambles page 1
    full_text = re.sub(r'Short title,?\s*extent,?\s*commencement\s*and\s*application\.?', '', full_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Erase the arrangement of sections and all preamble garbage
    # We look strictly for "CHAPTER I" followed by "PRELIMINARY"
    start_match = re.search(r'CHAPTER\s+I\s*\n*PRELIMINARY', full_text, flags=re.IGNORECASE)
    
    # If the parser split them up, fallback to the exact Section 1 text
    if not start_match:
        start_match = re.search(r'1\.\s*\(1\)\s*This Act may be called', full_text, flags=re.IGNORECASE)

    if start_match:
        full_text = full_text[start_match.start():]
    # ---------------------------------------------------------

    return full_text.strip()


# -------------------------------
# 2. SPLIT BY SECTIONS (ROBUST)
# -------------------------------
def split_by_section(text):
    """
    Matches:
    1. ...
    12. ...
    104. ...
    2A. ...
    """
    pattern = r'(?=\n\s*\d{1,3}[A-Z]?\.\s)'
    chunks = re.split(pattern, text)
    return chunks


# -------------------------------
# 3. PARSE SECTION (BULLETPROOF)
# -------------------------------
def parse_section(section_text):
    """
    Extracts ONLY the section number and keeps the entire text strictly intact.
    Drops title extraction to prevent splitting words like "sub-section".
    """
    match = re.match(r'^(\d+[A-Z]?)\.\s*(.+)', section_text, re.DOTALL)
    if not match:
        return None

    section_num = match.group(1)
    content = match.group(2).strip()

    # Return "N/A" for the title to keep metadata structure intact
    return section_num, "N/A", content


# -------------------------------
# 4. INGEST PIPELINE
# -------------------------------
def ingest_cpa():
    print(f"🚀 Starting Ingestion for {LAW_CODE} {YEAR}...")

    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found!")
        return

    # Load
    full_text = load_and_clean_pdf(PDF_PATH)

    # Split
    raw_sections = split_by_section(full_text)
    print(f"📄 Found {len(raw_sections)} raw blocks")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    enriched_sections = []
    current_chapter = "Unknown Chapter"

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # -------------------------------
        # Detect Chapter
        # -------------------------------
        chap_match = re.search(
            r'(CHAPTER\s+[IVXLCDM]+)\s*\n+(.*)',
            section,
            re.IGNORECASE
        )

        if chap_match:
            current_chapter = f"{chap_match.group(1).upper()} - {chap_match.group(2).strip()}"
            section = re.sub(
                r'CHAPTER\s+[IVXLCDM]+\s*\n+.*',
                '',
                section,
                flags=re.IGNORECASE
            ).strip()

        # -------------------------------
        # Parse Section
        # -------------------------------
        parsed = parse_section(section)

        if parsed:
            section_num, section_title, content = parsed

            sub_chunks = splitter.split_text(content)

            for i, sub in enumerate(sub_chunks):
                # Clean enriched text without [N/A]
                enriched_text = (
                    f"[{LAW_CODE} {YEAR}] "
                    f"[{current_chapter}] "
                    f"Section {section_num}: {sub}"
                )

                enriched_sections.append({
                    "text": enriched_text,
                    "metadata": {
                        "law_code": LAW_CODE,
                        "year": YEAR,
                        "chapter": current_chapter,
                        "section_number": section_num,
                        "section_title": section_title, # Stays "N/A"
                        "chunk_index": i + 1,
                        "total_chunks": len(sub_chunks),
                        "source": os.path.basename(PDF_PATH)
                    }
                })

        else:
            # Fallback chunk
            sub_chunks = splitter.split_text(section)

            for i, sub in enumerate(sub_chunks):
                enriched_text = (
                    f"[{LAW_CODE} {YEAR}] "
                    f"[{current_chapter}] {sub}"
                )

                enriched_sections.append({
                    "text": enriched_text,
                    "metadata": {
                        "law_code": LAW_CODE,
                        "year": YEAR,
                        "chapter": current_chapter,
                        "section_number": "N/A",
                        "section_title": "N/A",
                        "chunk_index": i + 1,
                        "total_chunks": len(sub_chunks),
                        "source": os.path.basename(PDF_PATH)
                    }
                })

    print(f"✅ Generated {len(enriched_sections)} chunks")

    # -------------------------------
    # 5. STORE IN CHROMA
    # -------------------------------
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    emb_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    ids = [f"{LAW_CODE}_{i}" for i in range(len(enriched_sections))]
    documents = [s["text"] for s in enriched_sections]
    metadatas = [s["metadata"] for s in enriched_sections]

    batch_size = 100

    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
        print(f"📦 Added batch {i // batch_size + 1}")

    print(f"🎉 Ingestion Complete! Total chunks: {collection.count()}")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    ingest_cpa()