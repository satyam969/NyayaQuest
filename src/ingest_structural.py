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
    # Remove large blocks of whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove common PDF footer patterns (like page numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()

def split_by_section(text):
    """
    Splits the document text by legally defined 'Sections' or 'Articles'.
    Assumes sections start with a number followed by a period and a space or newline.
    For BNS, it typically looks like: "103. (1) Whoever commits murder..."
    """
    # Look for patterns like "\n103. " or "\nSection 103. "
    # We use a lookahead assertion (?=\n\d+\.) so we split before the number but keep it in the next chunk.
    pattern = r'(?=\n\s*\d+\.\s)'
    chunks = re.split(pattern, text)
    
    # Clean up chunks and remove empty ones
    cleaned_chunks = [clean_text(chunk) for chunk in chunks if chunk.strip()]
    return cleaned_chunks

def preprocess_document(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    
    # Initially join all pages to treat the document as a continuous text stream
    # This helps in crossing page boundaries when searching for sections.
    full_text = "\n".join([page.page_content for page in data])
    
    sections = split_by_section(full_text)
    
    # Secondary splitter for dangerously large sections (like schedules)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    secondary_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)

    # Post-process: inject context
    enriched_sections = []
    for section in sections:
        # Try to extract the section number if it starts with one
        match = re.match(r'^\s*(\d+)\.\s*(.*)', section, re.DOTALL)
        if match:
            section_num = match.group(1)
            content = match.group(2)
            # Prepend a clear header so the AI knows what this chunk represents
            enriched_text = f"Section {section_num}: {content}"
            
            # Split this section if it's too large
            sub_chunks = secondary_splitter.split_text(enriched_text)
            for sub in sub_chunks:
                enriched_sections.append({
                    "text": sub,
                    "metadata": {
                        "section_number": section_num,
                        "source": os.path.basename(pdf_path)
                    }
                })
        else:
            if len(section) > 100: # Ignore tiny random strings
                sub_chunks = secondary_splitter.split_text(section)
                for sub in sub_chunks:
                    enriched_sections.append({
                        "text": sub,
                        "metadata": {
                            "section_number": "general",
                            "source": os.path.basename(pdf_path)
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
