import os
import shutil
import zipfile
import uuid
import threading
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


# --- Auto-Setup ChromaDB ---
# Priority: 1) Already extracted, 2) Local zip in repo, 3) Download from Google Drive
if not os.path.exists(chroma_dir):
    print(f"[SETUP] ChromaDB not found at '{chroma_dir}'.")

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

# ─────────────────────────────────────────────────────────────────────────────
# PDF Ingestion Endpoints
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_PDFS_DIR = os.path.join(os.path.dirname(__file__), "data", "legal_pdfs")
os.makedirs(LEGAL_PDFS_DIR, exist_ok=True)

# In-memory job store  {job_id: job_dict}
_jobs: Dict[str, Any] = {}
_jobs_lock = threading.Lock()

ingest_logger = logging.getLogger("ingest")


def _append_log(job_id: str, level: str, stage: int, msg: str) -> None:
    """Thread-safe log append into the job store."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "stage": stage, "msg": msg}
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["log_lines"].append(entry)


def _run_pipeline_job(job_id: str, pdf_path: str, no_llm: bool) -> None:
    """Run the ingestion pipeline in a background thread and update job state."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        # Import here so the main module loads fast
        from auto_ingest.orchestrator import run_pipeline
        from auto_ingest.config import CHROMA_DIR, COLLECTION_NAME, LLM_MODEL

        _append_log(job_id, "info", 0, f"Pipeline started — {os.path.basename(pdf_path)}")

        # Capture pipeline result
        # run_pipeline returns List[Dict] — one dict per segment
        segments = run_pipeline(
            pdf_path=pdf_path,
            chroma_dir=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            llm_model=None if no_llm else LLM_MODEL,
            no_llm=no_llm,
        )

        total_chunks = sum(s.get("chunks_stored", 0) for s in segments)
        scores = [s.get("final_metrics", {}).get("overall", 0)
                  for s in segments if s.get("final_metrics")]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0
        used_fallback = any(s.get("used_fallback") for s in segments)

        # Log each segment result
        for s in segments:
            flag = " [FALLBACK]" if s.get("used_fallback") else ""
            _append_log(job_id, "pass" if not s.get("used_fallback") else "warn", 7,
                f"{s.get('segment_title','?')}: {s.get('chunks_stored',0)} chunks | "
                f"score={s.get('final_metrics',{}).get('overall',0):.3f} | "
                f"strategy={s.get('strategy_used','?')}{flag}")

        _append_log(job_id, "pass", 7,
            f"Complete — {total_chunks} total chunks | avg_score={avg_score} | fallback={used_fallback}")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "total_chunks":  total_chunks,
                "avg_score":     avg_score,
                "used_fallback": used_fallback,
                "segments":      len(segments),
            }

    except Exception as exc:
        ingest_logger.exception(f"Pipeline failed for job {job_id}: {exc}")
        _append_log(job_id, "fail", -1, f"Pipeline ERROR: {exc}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"]  = str(exc)


def _new_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":    job_id,
            "filename":  filename,
            "status":    "queued",
            "log_lines": [],
            "result":    None,
            "error":     None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    # Keep last 50 jobs only
    with _jobs_lock:
        if len(_jobs) > 50:
            oldest = sorted(_jobs.keys(),
                            key=lambda k: _jobs[k]["created_at"])[:len(_jobs) - 50]
            for k in oldest:
                del _jobs[k]
    return job_id


# ── Upload PDF file ───────────────────────────────────────────────────────────

@app.post("/api/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    no_llm: bool = Form(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    save_path = os.path.join(LEGAL_PDFS_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = _new_job(file.filename)
    _append_log(job_id, "info", 0, f"Saved to data/legal_pdfs/{file.filename}")
    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, save_path, no_llm),
        daemon=True,
    ).start()

    return {"job_id": job_id, "filename": file.filename, "status": "queued"}


# ── Ingest from URL ───────────────────────────────────────────────────────────

class IngestUrlRequest(BaseModel):
    url: str
    no_llm: bool = False

@app.post("/api/ingest/url")
async def ingest_url(req: IngestUrlRequest):
    import httpx
    url = req.url.strip()
    if not url.lower().endswith(".pdf") and "pdf" not in url.lower():
        raise HTTPException(status_code=400, detail="URL does not appear to be a PDF")

    filename = url.split("/")[-1].split("?")[0]
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    save_path = os.path.join(LEGAL_PDFS_DIR, filename)

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}")

    job_id = _new_job(filename)
    _append_log(job_id, "info", 0, f"Downloaded {filename} → data/legal_pdfs/")
    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, save_path, req.no_llm),
        daemon=True,
    ).start()

    return {"job_id": job_id, "filename": filename, "status": "queued"}


# ── Poll job status ───────────────────────────────────────────────────────────

@app.get("/api/ingest/status/{job_id}")
def ingest_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── List recent jobs ─────────────────────────────────────────────────────────

@app.get("/api/ingest/jobs")
def list_ingest_jobs():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}

