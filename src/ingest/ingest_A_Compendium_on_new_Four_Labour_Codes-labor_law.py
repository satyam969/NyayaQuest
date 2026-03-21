import os
import re
import fitz
import hashlib
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
PDF_PATH = "data/legal_pdfs/A_Compendium_on_new_Four_Labour_Codes-labor_law.pdf"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"
SOURCE_NAME = os.path.basename(PDF_PATH)

# Primary statutory markers
CODE_MARKERS = [
    r"THE CODE ON WAGES, 2019",
    r"THE INDUSTRIAL RELATIONS CODE, 2020",
    r"THE CODE ON SOCIAL SECURITY, 2020",
    r"THE OCCUPATIONAL SAFETY, HEALTH AND WORKING CONDITIONS CODE, 2020"
]

def load_and_clean_pdf(pdf_path):
    """Loads PDF via PyMuPDF and scrubs standard Gazette formatting noise."""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text("text") + "\n"
        
    # 1. Remove Gazette Headers and Footers
    full_text = re.sub(r'THE GAZETTE OF INDIA.*?EXTRAORDINARY', '', full_text, flags=re.IGNORECASE)
    full_text = re.sub(r'MINISTRY OF LABOUR & EMPLOYMENT', '', full_text, flags=re.IGNORECASE)
    
    # 2. Remove standalone page numbers
    full_text = re.sub(r'^\s*\d+\s*$', '', full_text, flags=re.MULTILINE)
    
    # 3. Remove excessive visual underscores and multiple newlines
    full_text = re.sub(r'_+', '', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text.strip()

def generate_collision_proof_id(metadata, text):
    """Generates a 100% unique hash using deep metadata to prevent DuplicateIDError."""
    # Combining all metadata fields + the actual text ensures no two chunks ever share an ID
    unique_string = f"{metadata['act_name']}_{metadata['chapter']}_{metadata['section_number']}_{metadata['chunk_index']}_{text}"
    # Using the full 64-character SHA-256 hash guarantees zero collisions
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def split_into_codes(text):
    """Splits the 295-page text into the 4 primary Labour Codes."""
    pattern = r'(?=' + '|'.join(CODE_MARKERS) + r')'
    raw_codes = re.split(pattern, text, flags=re.IGNORECASE)
    
    parsed_codes = []
    for block in raw_codes:
        block = block.strip()
        if not block: continue
        
        current_code_name = "Labour Codes Introduction / General"
        for marker in CODE_MARKERS:
            if re.search(r'^' + marker, block, re.IGNORECASE):
                current_code_name = marker.replace(r'\s+', ' ').title()
                break
                
        parsed_codes.append({"code_name": current_code_name, "text": block})
        
    return parsed_codes

def process_compendium():
    print(f"🚀 Starting Ingestion Pipeline for {SOURCE_NAME}...")
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found!")
        return

    full_text = load_and_clean_pdf(PDF_PATH)
    codes_data = split_into_codes(full_text)
    
    # Balanced Chunking: 800 covers most legal sub-clauses perfectly.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    enriched_sections = []

    for code_dict in codes_data:
        code_name = code_dict["code_name"]
        code_text = code_dict["text"]
        
        chapter_pattern = r'(?=\n(?:CHAPTER\s+[IVXLCDM]+|(?:THE\s+)?[A-Z]*\s*SCHEDULE\s*[A-ZIVX]*))'
        chapters_raw = re.split(chapter_pattern, code_text)
        
        for chapter_block in chapters_raw:
            chapter_block = chapter_block.strip()
            if not chapter_block: continue
            
            current_chapter = "Preliminary / General"
            chap_match = re.search(r'^(CHAPTER\s+[IVXLCDM]+|(?:THE\s+)?[A-Z]*\s*SCHEDULE\s*[A-ZIVX]*)\s*\n+(.*?)(?=\n|$)', chapter_block)
            if chap_match:
                current_chapter = f"{chap_match.group(1).strip()} - {chap_match.group(2).strip()}"
                chapter_block = re.sub(r'^(CHAPTER\s+[IVXLCDM]+|(?:THE\s+)?[A-Z]*\s*SCHEDULE\s*[A-ZIVX]*)\s*\n+.*?(?=\n|$)', '', chapter_block).strip()

            section_pattern = r'(?=\n\s*\d+[A-Z]?\.\s)'
            sections_raw = re.split(section_pattern, chapter_block)
            
            for section in sections_raw:
                section = section.strip()
                if not section: continue
                
                # Robust Title Extraction
                match = re.match(r'^(\d+[A-Z]?)\.\s*(.*?)(?:[\.\-—\u2013\u2014]{1,3}|\n)(.*)', section, re.DOTALL)
                
                if match:
                    section_num = match.group(1)
                    section_title = match.group(2).strip()
                    content = match.group(3).strip()
                    
                    if not content and len(section_title) > 50:
                        content = section_title
                        section_title = "Unknown"
                        
                    sub_chunks = text_splitter.split_text(content)
                else:
                    section_num = "Schedule/General"
                    section_title = "N/A"
                    sub_chunks = text_splitter.split_text(section)

                total_chunks = len(sub_chunks)
                for i, sub in enumerate(sub_chunks):
                    # Context Injection inside the text
                    title_str = f" [{section_title}]" if section_title != "Unknown" and section_title != "N/A" else ""
                    enriched_text = f"[{code_name}] [{current_chapter}] Section {section_num}{title_str}: {sub}"
                    
                    metadata = {
                        "act_name": code_name,
                        "domain": "Labor Law",
                        "chapter": current_chapter,
                        "section_number": section_num,
                        "section_title": section_title,
                        "chunk_index": i + 1,
                        "total_chunks": total_chunks,
                        "source": SOURCE_NAME
                    }
                    
                    # Generate the collision-proof ID
                    doc_id = generate_collision_proof_id(metadata, enriched_text)
                    
                    enriched_sections.append({
                        "id": doc_id,
                        "text": enriched_text,
                        "metadata": metadata
                    })

    print(f"  Successfully processed {len(enriched_sections)} balanced chunks.")

    # ------------------------------------------------------------------------
    # CHROMADB UPSERTION
    # ------------------------------------------------------------------------
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    ids = [s["id"] for s in enriched_sections]
    texts = [s["text"] for s in enriched_sections]
    metadatas = [s["metadata"] for s in enriched_sections]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"    Upserted batch {i//batch_size + 1}...")

    print(f"✅ Ingestion Complete! Total safe chunks in DB: {collection.count()}")

if __name__ == "__main__":
    process_compendium()