# File: src/verify_cpa.py

import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import the new Hybrid Retriever
from hybrid_retriever import HybridRetriever

DB_DIR = "./chroma_db_groq_legal"

def evaluate_retrieval(retriever, query: str, category: str, k: int = 3):
    print(f"\n{'='*70}")
    print(f"📂 CATEGORY: {category}")
    print(f"🔍 QUERY: '{query}'")
    print(f"{'='*70}")

    # Temporarily adjust the retriever's k-value if needed for this specific query
    retriever.k = k
    
    # Perform the Hybrid Search
    results = retriever.invoke(query)

    if not results:
        print("❌ No results found. Check your ingestion script.")
        return

    print(f"✅ Retrieved Top {k} Chunks (Ranked by RRF Fusion):\n")
    
    for i, doc in enumerate(results):
        print(f"--- Rank {i+1} ---")
        print(f"Metadata: Law: {doc.metadata.get('law_code')} | Section: {doc.metadata.get('section_number')}")
        
        # Print a preview of the text (first 250 chars)
        text_preview = doc.page_content.replace('\n', ' ')[:250]
        print(f"Text Preview: {text_preview}...\n")

if __name__ == "__main__":
    print("Initializing Database and Building BM25 Index... Please wait.")
    
    # 1. Load Embeddings and Vector Store
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    try:
        vector_db = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=embedding_model,
            collection_name="legal_knowledge"
        )
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    # 2. Build the Hybrid Retriever ONCE
    hybrid_retriever = HybridRetriever.from_vector_store(
        vector_store=vector_db,
        k=3,
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    # A dictionary of test cases categorized by the type of retrieval challenge
    test_suite = [
        # 1. Direct Section Lookups (Tests if hybrid search fixes the "Section 104" bug)
        ("What does Section 104 of the Consumer Protection Act say about regulations?", "Direct Section Reference"),
        ("Explain the provisions of Section 21 regarding penalties for misleading advertisements.", "Direct Section Reference"),
        
        # 2. Conceptual & Definitional (Tests semantic understanding)
        ("What is the legal definition of an 'advertisement'?", "Definitions"),
        ("Does the definition of a 'consumer' include a person who buys goods for a commercial purpose like resale?", "Exclusion Criteria"),
        ("Who has the authority to establish the Central Consumer Protection Council?", "Authorities"),
        
        # 3. Pecuniary Jurisdiction (Tests ability to find specific financial thresholds)
        ("What is the pecuniary jurisdiction of the District Commission? Up to what value of goods can they entertain complaints?", "Jurisdiction / Financials"),
        ("Within how many days must an appeal be filed before the State Commission against a District Commission's order?", "Procedural Timelines"),
        
        # 4. New 2019 Act Features (Tests specific modern clauses)
        ("Under what circumstances is a 'product seller' (who is not the manufacturer) held liable in a product liability action?", "Product Liability"),
        ("Can a celebrity or brand ambassador be penalized for endorsing a misleading advertisement? How can they avoid it?", "Endorsements & Penalties"),
        ("What is the procedure for settling consumer disputes through the newly introduced Mediation process?", "Mediation Mechanism"),
        
        # 5. Tricky / Edge Cases (Tests if the retriever gets confused by exceptions)
        ("If a consumer signs an 'unfair contract' for a flat worth 3 Crore Rupees, which Commission has the original jurisdiction to hear the complaint?", "Edge Case / Unfair Contracts") 
    ]
    
    print(f"\nStarting Retrieval Evaluation for {len(test_suite)} test cases...\n")
    
    for query, category in test_suite:
        evaluate_retrieval(hybrid_retriever, query, category)