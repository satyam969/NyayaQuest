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

CODES_METADATA = [
    {"name": "Code_on_Wages_2019", "start": 8, "end": 36, "code": "COW", "year": "2019"},
    {"name": "Industrial_Relations_Code_2020", "start": 37, "end": 92, "code": "IRC", "year": "2020"},
    {"name": "Code_on_Social_Security_2020", "start": 93, "end": 209, "code": "CSS", "year": "2020"},
    {"name": "OSH_Code_2020", "start": 210, "end": 294, "code": "OSH", "year": "2020"},
]

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
        
    doc.close()
    return generated_files

# -------------------------------
# 1. LOAD + CLEAN (WITH OMNI-CROP)
# -------------------------------
def load_and_clean_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        rect = page.rect
        # 🛡️ THE OMNI-CROP 🛡️
        # Crop 72 points (1 inch) from Top/Bottom to destroy page headers and footers
        # Crop 50 points from Left/Right to destroy sidebars/margin notes
        clip_box = fitz.Rect(50, 72, rect.width - 50, rect.height - 72)
        
        full_text += page.get_text("text", clip=clip_box) + "\n"

    # Standard cleanups
    full_text = re.sub(r'(?im)^.*THE GAZETTE OF INDIA.*$', '', full_text)
    full_text = re.sub(r'(?im)^.*MINISTRY OF LAW AND JUSTICE.*$', '', full_text)
    full_text = re.sub(r'\[?PART\s+[IVX]+[—\-\]]?', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'SEC\.\s*\d+[A-Z]?(\(\w+\))?\]?', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'^\s*\d+\s*$', '', full_text, flags=re.MULTILINE)
    
    full_text = re.sub(r'\r', '\n', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # 🛡️ BULLETPROOF INDEX SKIPPER 🛡️
    # Look for the exact phrase that starts the actual law body, skipping the Table of Contents entirely.
    start_match = re.search(r'1\.\s*\(\s*1\s*\)\s*This\s+(Act|Code)\s+may\s+be\s+called', full_text, flags=re.IGNORECASE)
    if start_match:
        preceding_text = full_text[:start_match.start()]
        last_chapter_idx = preceding_text.upper().rfind("CHAPTER")
        if last_chapter_idx != -1 and (start_match.start() - last_chapter_idx) < 200:
            full_text = full_text[last_chapter_idx:]
        else:
            full_text = full_text[start_match.start():]

    return full_text.strip()

# -------------------------------
# 2. SPLIT BY SECTIONS & CHAPTERS
# -------------------------------
def split_by_section(text):
    # Splits exactly when a real Chapter or Section starts
    pattern = r'(?=\n\s*CHAPTER\s+[IVXLCDM]+\b|\n\s*\d{1,3}[A-Z]?\.\s|\n\s*THE\s+[A-Z]+\s+SCHEDULE|\n\s*SCHEDULE\b)'
    chunks = re.split(pattern, text, flags=re.IGNORECASE)
    return chunks

def parse_section(section_text):
    section_text = section_text.strip()
    
    schedule_match = re.match(r'^(THE\s+[A-Z]+\s+SCHEDULE|SCHEDULE)\s*(.*)', section_text, re.DOTALL | re.IGNORECASE)
    if schedule_match:
        return schedule_match.group(1).upper().strip(), "N/A", schedule_match.group(2).strip()

    section_match = re.match(r'^(\d+[A-Z]?)\.\s*(.+)', section_text, re.DOTALL)
    if section_match:
        return section_match.group(1), "N/A", section_match.group(2).strip()

    return None

# -------------------------------
# 3. INGEST PIPELINE
# -------------------------------
def ingest_all_codes():
    if not os.path.exists(SOURCE_PDF_PATH):
        print(f"❌ Error: {SOURCE_PDF_PATH} not found!")
        return

    pdf_files = split_compendium()
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    all_documents = []
    all_metadatas = []
    all_ids = []
    global_chunk_idx = 0

    for file_info in pdf_files:
        print(f"🚀 Processing {file_info['law_code']} {file_info['year']}...")
        
        full_text = load_and_clean_pdf(file_info['path'])
        raw_sections = split_by_section(full_text)
        
        current_chapter = "Unknown Chapter"

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            # CHAPTER EXTRACTION
            chap_match = re.search(r'^(CHAPTER\s+[IVXLCDM]+)\s*\n+(.*)', section, re.IGNORECASE)
            if chap_match:
                chapter_title = chap_match.group(2).split('\n')[0].strip()
                current_chapter = f"{chap_match.group(1).upper()} - {chapter_title}"
                
                section = re.sub(r'^CHAPTER\s+[IVXLCDM]+\s*\n+[^\n]*', '', section, flags=re.IGNORECASE).strip()
                
                if not section:
                    continue

            parsed = parse_section(section)
            
            if parsed:
                section_num, section_title, content = parsed

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

    print("💾 Saving chunks to ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        collection.add(
            ids=all_ids[i:i + batch_size],
            documents=all_documents[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size]
        )
    
    print(f"🎉 Ingestion Complete! Total chunks added: {collection.count()}")

if __name__ == "__main__":
    ingest_all_codes()