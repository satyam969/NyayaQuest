import os
import glob
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader

# Configuration
PDF_DIR = "data/legal_pdfs"
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"

def clean_text(text):
    """Clean up header/footer noise from the PDF text."""
    # Remove the massive Hindi/English Gazette initial publication header spanning across Page 1 and 2
    text = re.sub(r'vlk/kkj\.k.*?CG-DL-E-\d+-\d+', '', text, flags=re.DOTALL)
    
    # Remove all the repetitive underscore lines used for visual formatting
    text = re.sub(r'_+', '', text)
    
    # Remove "THE GAZETTE OF INDIA EXTRAORDINARY" and page number notations (e.g., "[Part II—" or "Sec. 1]")
    text = re.sub(r'THE GAZETTE OF INDIA EXTRAORDINARY', '', text)
    text = re.sub(r'\[?Part II—', '', text)
    text = re.sub(r'Sec\.\s*\d+\]?', '', text)
    
    # Remove standalone small text lines, page numbers and excessive whitespace
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Empty lines with just a number
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def split_by_section(text):
    """
    Splits the document text by legally defined 'Sections' or 'Articles'.
    """
    pattern = r'(?=\n\s*\d+\.\s)'
    chunks = re.split(pattern, text)
    
    # We don't just clean them; we want to preserve the raw strings to find Chapters
    return chunks

def preprocess_document(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    
    full_text = "\n".join([page.page_content for page in data])
    full_text = clean_text(full_text)
    
    raw_sections = split_by_section(full_text)
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # The splitter ensures it NEVER cuts words in half. It splits on ["\n\n", "\n", " ", ""] in that order.
    secondary_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    # Parse law_code and year from filename (e.g., "BNS_2023.pdf" -> "BNS", "2023")
    basename = os.path.basename(pdf_path)
    name_match = re.match(r'^([A-Za-z]+)_(\d{4})', basename)
    law_code = name_match.group(1).upper() if name_match else "UNKNOWN"
    year = name_match.group(2) if name_match else "Unknown"

    enriched_sections = []
    current_chapter = "Unknown Chapter"
    
    for section in raw_sections:
        section = section.strip()
        if not section: continue
        
        # 1. Statefully Update Chapter
        chap_match = re.search(r'(CHAPTER\s*[IVXLCDMivxlcdm]+)\s*\n+(.*?)(?=\n|$)', section)
        if chap_match:
            current_chapter = f"{chap_match.group(1)} - {chap_match.group(2).strip()}"
            section = re.sub(r'CHAPTER\s*[IVXLCDMivxlcdm]+\s*\n+.*?(?=\n|$)', '', section).strip()
            
        # 2. Extract Section Number and clean text
        match = re.match(r'^(\d+[A-Z]?)\.\s*(.*)', section, re.DOTALL)
        if match:
            section_num = match.group(1)
            content = match.group(2).strip()
            
            sub_chunks = secondary_splitter.split_text(content)
            total_chunks = len(sub_chunks)
            
            for i, sub in enumerate(sub_chunks):
                enriched_text = f"[{law_code} {year}] [{current_chapter}] Section {section_num}: {sub}"
                
                enriched_sections.append({
                    "text": enriched_text,
                    "metadata": {
                        "law_code": law_code,
                        "year": year,
                        "chapter": current_chapter,
                        "section_number": section_num,
                        "chunk_index": i + 1,
                        "total_chunks": total_chunks,
                        "source": basename
                    }
                })
        else:
            if len(section) > 100:
                sub_chunks = secondary_splitter.split_text(section)
                for i, sub in enumerate(sub_chunks):
                    enriched_sections.append({
                        "text": sub,
                        "metadata": {
                            "law_code": law_code,
                            "year": year,
                            "chapter": current_chapter,
                            "section_number": "general",
                            "chunk_index": i + 1,
                            "total_chunks": len(sub_chunks),
                            "source": basename
                        }
                    })
            
    return enriched_sections


def ingest_structural():
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    # Reset collection for a clean re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        print(f"Processing {basename} structurally...")
        
        try:
            structural_chunks = preprocess_document(pdf_path)
            print(f"  Extracted {len(structural_chunks)} structural sections.")
            
            if not structural_chunks:
                print(f"  Warning: No structural chunks found for {basename}. Check formatting.")
                continue

            ids = [f"{basename}_sec_{chunk['metadata']['section_number']}_{i}" for i, chunk in enumerate(structural_chunks)]
            texts = [chunk["text"] for chunk in structural_chunks]
            metadatas = [chunk["metadata"] for chunk in structural_chunks]
            
            # Batch add to Chroma (reduced batch size for memory safety)
            batch_size = 4
            for i in range(0, len(texts), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=texts[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            
            print(f"  Successfully added {basename} to collection.")
            
        except Exception as e:
            print(f"  Failed to process {basename}: {e}")

if __name__ == "__main__":
    ingest_structural()
