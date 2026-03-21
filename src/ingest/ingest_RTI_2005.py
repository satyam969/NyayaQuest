import os
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
PDF_PATH = "data/legal_pdfs/RTI_2005.pdf"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"
LAW_CODE = "RTI"
YEAR = "2005"

import fitz

def load_and_clean_pdf(pdf_path):
    """Loads PDF with PyMuPDF block extraction to scrub footnotes based on font size."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        in_footnotes = False
        for b in blocks:
            if "lines" not in b: continue
            
            first_span = None
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        first_span = span
                        break
                if first_span: break
                
            if first_span:
                text = first_span["text"].strip()
                size = first_span["size"]
                # Footnotes: start with a digit and size <= 9.5
                if size <= 9.5 and re.match(r'^\W*\d+\.', text):
                    in_footnotes = True
                    
            if not in_footnotes:
                block_text = ""
                for line in b["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += "\n"
                full_text += block_text + "\n"
                
    # 0. Strip the entire preamble/amending acts list
    start_str = "An Act to provide for setting out the practical regime"
    start_idx = full_text.find(start_str)
    if start_idx != -1:
        title_idx = full_text.rfind("THE RIGHT TO INFORMATION ACT, 2005", 0, start_idx)
        if title_idx != -1:
            full_text = full_text[title_idx:]
        else:
            full_text = full_text[start_idx:]
            
    # 1. Remove the "ARRANGEMENT OF SECTIONS"
    full_text = re.sub(r'ARRANGEMENT OF SECTIONS.*?(?=CHAPTER I)', '', full_text, flags=re.DOTALL)
    
    # 2. Remove repetitive headers
    full_text = re.sub(r'THE RIGHT TO INFORMATION ACT, 2005', '', full_text)
    
    # 3. Remove standalone page numbers
    full_text = re.sub(r'^\s*\d+\s*$', '', full_text, flags=re.MULTILINE)
    
    # 4. Remove excessive underscores and whitespace
    full_text = re.sub(r'_+', '', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text.strip()

def split_by_section(text):
    """
    Splits the document text by 'Sections'.
    Pattern matches digits followed by a dot at start of line: "1. ", "12. " etc.
    """
    pattern = r'(?=\n\s*\d+\.\s)'
    chunks = re.split(pattern, text)
    return chunks

def ingest_rti():
    print(f"🚀 Starting Ingestion for {LAW_CODE} {YEAR}...")
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found!")
        return

    # Load and Clean
    full_text = load_and_clean_pdf(PDF_PATH)
    
    # Split into Sections
    raw_sections = split_by_section(full_text)
    print(f"  Found {len(raw_sections)} raw section blocks.")
    
    # Secondary splinter for long sections
    secondary_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    enriched_sections = []
    current_chapter = "Unknown Chapter"
    
    for section in raw_sections:
        section = section.strip()
        if not section: continue
        
        # 1. Update Chapter (CHAPTER II - RIGHT TO INFORMATION...)
        # Looks for "CHAPTER <ROMAN> \n <TITLE>"
        chap_match = re.search(r'(CHAPTER\s*[IVXLCDMivxlcdm]+)\s*\n+(.*?)(?=\n|$)', section)
        if chap_match:
            current_chapter = f"{chap_match.group(1)} - {chap_match.group(2).strip()}"
            # Remove the chapter text from the section content
            section = re.sub(r'CHAPTER\s*[IVXLCDMivxlcdm]+\s*\n+.*?(?=\n|$)', '', section).strip()
            
        # 2. Extract Section Number and Title
        # Matches patterns like: "4. Obligations of public authorities.—(1)..." with em-dash handling
        match = re.match(r'^(\d+[A-Z]?)\.\s*(.*?)(?:\.\u2014|\.\u2013|\.—|\.-|\.\s*-|\n)(.*)', section, re.DOTALL)
        if match:
            section_num = match.group(1)
            section_title = match.group(2).strip()
            content = match.group(3).strip()
            
            # If the content ended up empty but the title was long, maybe it wasn't a standard title split
            if not content and len(section_title) > 50:
                content = section_title
                section_title = "Unknown"
            
            sub_chunks = secondary_splitter.split_text(content)
            total_chunks = len(sub_chunks)
            
            for i, sub in enumerate(sub_chunks):
                # Metadata-rich prefix for LLM awareness
                title_str = f" [{section_title}]" if section_title != "Unknown" else ""
                enriched_text = f"[{LAW_CODE} {YEAR}] [{current_chapter}] Section {section_num}{title_str}: {sub}"
                
                enriched_sections.append({
                    "text": enriched_text,
                    "metadata": {
                        "law_code": LAW_CODE,
                        "year": YEAR,
                        "chapter": current_chapter,
                        "section_number": section_num,
                        "section_title": section_title,
                        "chunk_index": i + 1,
                        "total_chunks": total_chunks,
                        "source": os.path.basename(PDF_PATH)
                    }
                })
        else:
            # Fallback for introductory text or schedules
            if len(section) > 50:
                sub_chunks = secondary_splitter.split_text(section)
                for i, sub in enumerate(sub_chunks):
                    enriched_text = f"[{LAW_CODE} {YEAR}] [{current_chapter}] {sub}"
                    enriched_sections.append({
                        "text": enriched_text,
                        "metadata": {
                            "law_code": LAW_CODE,
                            "year": YEAR,
                            "chapter": current_chapter,
                            "section_number": "N/A",
                            "chunk_index": i + 1,
                            "total_chunks": len(sub_chunks),
                            "source": os.path.basename(PDF_PATH)
                        }
                    })

    # Total enriched chunks
    print(f"  Generated {len(enriched_sections)} enriched chunks.")

    # ChromaDB Persistence
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Using the same embedding function as BNS (BGE-Small)
    # Note: SentenceTransformerEmbeddingFunction needs the sentence-transformers package
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    # Prepare for batch addition
    ids = [f"{LAW_CODE}_{i}" for i in range(len(enriched_sections))]
    documents = [s["text"] for s in enriched_sections]
    metadatas = [s["metadata"] for s in enriched_sections]

    # Add to collection in batches to avoid overwhelming
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"    Added batch {i//batch_size + 1}")

    print(f"✅ Ingestion Complete for {LAW_CODE} 2005! Total chunks in collection: {collection.count()}")

if __name__ == "__main__":
    ingest_rti()
