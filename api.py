import os
import shutil
import zipfile
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from nyayaquest_main import NyayaQuest
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from hybrid_retriever import HybridRetriever

from src.db_utils import get_user_conversations, create_conversation, get_conversation_history, db
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NyayaQuest API")

# Enable CORS for the Vite frontend and deployed Vercel app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Embedding class (same as app.py)
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Initialize AI Engine (Single global instance to save memory)
groq_api_key = os.getenv('GROQ_API_KEY')
chroma_dir = os.getenv('CHROMA_PERSIST_DIR', 'chroma_db_groq_legal')

<<<<<<< HEAD
# --- Auto-Setup ChromaDB ---
# Priority: 1) Already extracted, 2) Local zip in repo, 3) Download from Google Drive
if not os.path.exists(chroma_dir):
    print(f"[SETUP] ChromaDB not found at '{chroma_dir}'.")

    # Check for zip file already present in the repo (uploaded via HF web UI)
    local_zip = os.path.join(os.path.dirname(__file__), 'chroma_db_groq_legal.zip')
    extract_to = os.path.dirname(chroma_dir) if os.path.dirname(chroma_dir) else "."

    if os.path.exists(local_zip):
        print(f"[SETUP] Found local zip at '{local_zip}'. Extracting...")
        try:
            with zipfile.ZipFile(local_zip, 'r') as zf:
                zf.extractall(extract_to)
            print(f"[SETUP] ChromaDB extracted successfully to '{chroma_dir}'!")
        except Exception as e:
            print(f"[SETUP ERROR] Failed to extract local zip: {e}")
    else:
        # Fall back to downloading from Google Drive
        print("[SETUP] No local zip found. Downloading from Google Drive...")
        try:
            import gdown
            gdrive_id = "1TXr1pW-qBU5vLHekxjgb6IQ4BLeI4twq"
            zip_path = "/tmp/chroma_db.zip"
            gdown.download(id=gdrive_id, output=zip_path, quiet=False)
            print(f"[SETUP] Extracting to '{chroma_dir}'...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            os.remove(zip_path)
            print(f"[SETUP] ChromaDB ready at '{chroma_dir}'!")
        except Exception as e:
            print(f"[SETUP ERROR] Failed to download ChromaDB: {e}")
else:
    print(f"[SETUP] ChromaDB already exists at '{chroma_dir}'. Skipping setup.")
=======
# --- Auto-Download ChromaDB from Google Drive if missing ---
# This runs on first boot (e.g. on Hugging Face where /data is empty).
# After the first run, the data persists in the volume and this is skipped.
if not os.path.exists(chroma_dir):
    print(f"[SETUP] ChromaDB not found at '{chroma_dir}'. Downloading from Google Drive...")
    try:
        import gdown
        gdrive_id = "1TXr1pW-qBU5vLHekxjgb6IQ4BLeI4twq"
        zip_path = "/tmp/chroma_db.zip"
        gdown.download(id=gdrive_id, output=zip_path, quiet=False)
        
        print(f"[SETUP] Extracting database to '{chroma_dir}'...")
        # Extract to parent directory so the folder name is preserved
        extract_to = os.path.dirname(chroma_dir) or "."
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        os.remove(zip_path)
        print(f"[SETUP] ChromaDB ready at '{chroma_dir}'!")
    except Exception as e:
        print(f"[SETUP ERROR] Failed to download ChromaDB: {e}")

>>>>>>> d7885ad15579824de9db11f04c5279f2140de3fb

llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, groq_api_key=groq_api_key)
embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings, collection_name="legal_knowledge")
hybrid_retriever = HybridRetriever.from_vector_store(vector_store, k=20, vector_weight=0.5, bm25_weight=0.5)

law_engine = NyayaQuest(llm, embeddings, vector_store, hybrid_retriever=hybrid_retriever)

@app.get("/")
async def root():
    return {"status": "online", "message": "NyayaQuest AI Engine is running perfectly!"}

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    context: List[Dict[str, Any]]
    thread_id: str

class ConversationCreate(BaseModel):
    user_id: str

class AuthRequest(BaseModel):
    email: str
    password: str

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "NyayaQuest API is running"}

@app.get("/api/conversations/{user_id}")
def get_conversations(user_id: str):
    try:
        convos = get_user_conversations(user_id)
        return {"conversations": convos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
def create_new_conversation(data: ConversationCreate):
    try:
        new_id = create_conversation(data.user_id)
        return {"thread_id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from src.auth_utils import sign_in, sign_up

@app.post("/api/auth/signup")
def register_user(req: AuthRequest):
    res = sign_up(req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Signup failed"))
    return res

@app.post("/api/auth/signin")
def login_user(req: AuthRequest):
    res = sign_in(req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Login failed"))
    return res

@app.get("/api/conversations/{user_id}/{thread_id}")
def get_chat_history(user_id: str, thread_id: str):
    try:
        history_docs = get_conversation_history(thread_id)
        # Format for frontend
        formatted_history = []
        for m in history_docs:
            msg_entry = {"role": m["role"], "content": m["content"]}
            if "metadata" in m and m.get("metadata") and "context" in m["metadata"]:
                msg_entry["context"] = m["metadata"]["context"]
            formatted_history.append(msg_entry)
            
        # Sync Backend History for engine
        if hasattr(law_engine.cache, 'set_chat_history'):
            law_engine.cache.set_chat_history(thread_id, formatted_history)
            
        return {"history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Call the engine
        result, context, updated_history = law_engine.conversational(request.message, request.thread_id)
        
        # Serialize context documents for frontend
        serialized_context = []
        if context:
            for doc in context:
                serialized_context.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
                
        return ChatResponse(
            response=result,
            context=serialized_context,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally with: uvicorn api:app --reload
