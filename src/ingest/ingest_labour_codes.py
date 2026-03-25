import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
SOURCE_PDF_PATH = "data/legal_pdfs/A Compendium on new Four Labour Codes-labor law.pdf"
OUTPUT_PDF_DIR = "data/legal_pdfs/labour_codes"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"

# Exact page ranges based on the compendium index (PyMuPDF is 0-indexed)
CODES_METADATA = [
    {"name": "Code_on_Wages_2019", "start": 8, "end": 36, "code": "COW", "year": "2019"},
    {"name": "Industrial_Relations_Code_2020", "start": 37, "end": 92, "code": "IRC", "year": "2020"},
    {"name": "Code_on_Social_Security_2020", "start": 93, "end": 208, "code": "CSS", "year": "2020"},
    {"name": "OSH_Code_2020", "start": 209, "end": 294, "code": "OSH", "year": "2020"},
]

# -------------------------------
# 1. SPLIT MASTER PDF INTO 4
# -------------------------------
def split_compendium():
    print("✂️ Slicing compendium into 4 separate Code PDFs...")
    os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)
    generated_files = []
    
    doc = fitz.open(SOURCE_PDF_PATH)
    
    for info in CODES_METADATA:
        out_path = os.path.join(OUTPUT_PDF_DIR, f"{info['name']}.pdf")
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=info['start'], to_page=info['end'])
        new_doc.save(out_path)
        new_doc.close()
        
        generated_files.append({
            "path": out_path,
            "law_code": info["code"],
            "year": info["year"],
            "filename": f"{info['name']}.pdf"
        })
        print(f"   -> Created: {out_path}")
        
    doc.close()
    return generated_files

# -------------------------------
# 2. LOAD + CLEAN PDF 
# -------------------------------
def load_and_clean_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    # Clean headers, footers, and redundant tags
    full_text = re.sub(r'(?im)^.*THE GAZETTE OF INDIA.*$', '', full_text)
    full_text = re.sub(r'(?im)^.*MINISTRY OF LAW AND JUSTICE.*$', '', full_text)
    full_text = re.sub(r'\[?PART\s+[IVX]+[—\-\]]?', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'SEC\.\s*\d+[A-Z]?(\(\w+\))?\]?', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'^\s*\d+\s*$', '', full_text, flags=re.MULTILINE)
    
    full_text = re.sub(r'\r', '\n', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Aggressive preamble slicer: Jump straight to Chapter I
    start_match = re.search(r'CHAPTER\s+I\s*\n*PRELIMINARY', full_text, flags=re.IGNORECASE)
    if not start_match:
        start_match = re.search(r'1\.\s*\(1\)\s*This Act may be called', full_text, flags=re.IGNORECASE)

    if start_match:
        full_text = full_text[start_match.start():]

    return full_text.strip()

# -------------------------------
# 3. SPLIT BY SECTIONS & SCHEDULES
# -------------------------------
def split_by_section(text):
    """
    Matches numbered sections OR Schedule headers to prevent massive blobs.
    """
    pattern = r'(?=\n\s*\d{1,3}[A-Z]?\.\s|\n\s*THE\s+[A-Z]+\s+SCHEDULE|\n\s*SCHEDULE\b)'
    chunks = re.split(pattern, text, flags=re.IGNORECASE)
    return chunks

# -------------------------------
# 4. PARSE SECTION OR SCHEDULE
# -------------------------------
def parse_section(section_text):
    section_text = section_text.strip()
    
    # Check if it's a Schedule
    schedule_match = re.match(r'^(THE\s+[A-Z]+\s+SCHEDULE|SCHEDULE)\s*(.*)', section_text, re.DOTALL | re.IGNORECASE)
    if schedule_match:
        schedule_name = schedule_match.group(1).upper().strip()
        content = schedule_match.group(2).strip()
        return schedule_name, "N/A", content

    # Check if it's a standard numbered Section
    section_match = re.match(r'^(\d+[A-Z]?)\.\s*(.+)', section_text, re.DOTALL)
    if section_match:
        section_num = section_match.group(1)
        content = section_match.group(2).strip()
        return section_num, "N/A", content

    return None

# -------------------------------
# 5. INGEST PIPELINE
# -------------------------------
def ingest_all_codes():
    if not os.path.exists(SOURCE_PDF_PATH):
        print(f"❌ Error: '{SOURCE_PDF_PATH}' not found in the current directory!")
        print("Please ensure the compendium PDF is in the exact same folder as this script.")
        return

    # A: Split the PDF
    pdf_files = split_compendium()
    
    # B: Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # try:
    #     client.delete_collection(name=COLLECTION_NAME)
    #     print("🗑️ Cleaned up old database collection for a fresh ingest.")
    # except Exception:
    #     pass

    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    all_documents = []
    all_metadatas = []
    all_ids = []
    global_chunk_idx = 0

    # C: Process each generated PDF
    for file_info in pdf_files:
        print(f"🚀 Processing {file_info['law_code']} {file_info['year']}...")
        
        full_text = load_and_clean_pdf(file_info['path'])
        raw_sections = split_by_section(full_text)
        
        current_chapter = "Unknown Chapter"

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            # Detect Chapter
            chap_match = re.search(r'(CHAPTER\s+[IVXLCDM]+)\s*\n+(.*)', section, re.IGNORECASE)
            if chap_match:
                current_chapter = f"{chap_match.group(1).upper()} - {chap_match.group(2).strip()}"
                section = re.sub(r'CHAPTER\s+[IVXLCDM]+\s*\n+.*', '', section, flags=re.IGNORECASE).strip()

            parsed = parse_section(section)
            
            if parsed:
                section_num, section_title, content = parsed

                # Override Chapter for Schedules
                if "SCHEDULE" in section_num.upper():
                    effective_chapter = "Schedules"
                else:
                    effective_chapter = current_chapter

                sub_chunks = splitter.split_text(content)

                for i, sub in enumerate(sub_chunks):
                    enriched_text = f"[{file_info['law_code']} {file_info['year']}] [{effective_chapter}] {section_num}: {sub}"
                    all_documents.append(enriched_text)
                    all_metadatas.append({
                        "law_code": file_info["law_code"],
                        "year": file_info["year"],
                        "chapter": effective_chapter, 
                        "section_number": section_num,
                        "chunk_index": i + 1,
                        "total_chunks": len(sub_chunks),
                        "source": file_info["filename"]
                    })
                    all_ids.append(f"{file_info['law_code']}_{global_chunk_idx}")
                    global_chunk_idx += 1
            else:
                # Fallback chunk for unparsed blocks
                sub_chunks = splitter.split_text(section)
                for i, sub in enumerate(sub_chunks):
                    enriched_text = f"[{file_info['law_code']} {file_info['year']}] [{current_chapter}] {sub}"
                    all_documents.append(enriched_text)
                    all_metadatas.append({
                        "law_code": file_info["law_code"],
                        "year": file_info["year"],
                        "chapter": current_chapter,
                        "section_number": "N/A",
                        "chunk_index": i + 1,
                        "total_chunks": len(sub_chunks),
                        "source": file_info["filename"]
                    })
                    all_ids.append(f"{file_info['law_code']}_{global_chunk_idx}")
                    global_chunk_idx += 1

    # D: Batch Insert into Chroma
    print("💾 Saving chunks to ChromaDB (this may take a minute while it downloads the embedding model)...")
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        collection.add(
            ids=all_ids[i:i + batch_size],
            documents=all_documents[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size]
        )
    
    print(f"🎉 Ingestion Complete! Total chunks added across 4 Labour Codes: {collection.count()}")

if __name__ == "__main__":
    ingest_all_codes()