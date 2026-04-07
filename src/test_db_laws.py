import chromadb

client = chromadb.PersistentClient(path="chroma_db_groq_legal")
collection = client.get_collection("legal_knowledge")

print(f"Total chunks in DB: {collection.count()}")

results = collection.get(include=["metadatas"])
laws = set()
for m in results["metadatas"]:
    if not m: continue
    code = m.get("law_code")
    act = m.get("act_name")
    
    if code: laws.add(f"law_code: {code}")
    if act: laws.add(f"act_name: {act}")

print("\n--- Unique Laws Found in DB ---")
for l in sorted(list(laws)):
    print(l)
print("-------------------------------")
