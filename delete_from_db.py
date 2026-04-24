import argparse
import chromadb

def delete_from_db(doc_title: str, chroma_dir="chroma_db_groq_legal", collection_name="legal_knowledge"):
    """Delete all chunks belonging to a specific doc_title from ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        
        # Check how many chunks match
        results = collection.get(where={"doc_title": doc_title}, include=["metadatas"])
        count = len(results.get("ids", []))
        
        if count == 0:
            print(f"No chunks found for doc_title: '{doc_title}'.")
            
            # Print available titles to help the user
            all_results = collection.get(include=["metadatas"])
            all_titles = set(m.get("doc_title", "Unknown") for m in all_results.get("metadatas", []))
            print(f"Available doc_titles in DB: {all_titles}")
            return

        # Perform the deletion
        collection.delete(where={"doc_title": doc_title})
        print(f"✅ Successfully deleted {count} chunks for doc_title: '{doc_title}'.")

    except Exception as e:
        print(f"❌ Error communicating with ChromaDB: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a PDF's ingestion from ChromaDB by its doc_title")
    parser.add_argument("--title", required=True, help="The doc_title to delete (e.g. 'CPC 1908' or 'PRELIMINARY')")
    args = parser.parse_args()
    
    delete_from_db(args.title)
