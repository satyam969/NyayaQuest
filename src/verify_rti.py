import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configuration
CHROMA_DIR = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"

def verify_rti():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
    # query for RTI documents
    results = collection.get(
        where={"law_code": "RTI"},
        limit=5,
        include=['documents', 'metadatas']
    )
    
    print(f"Total RTI chunks found: {len(results['ids'])}")
    
    for i in range(len(results['documents'])):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Content Preview: {results['documents'][i][:200]}...")

if __name__ == "__main__":
    verify_rti()
