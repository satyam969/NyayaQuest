"""
NyayaQuest API — Production-grade FastAPI backend.

Changes from original:
  - Health & readiness probes (/health, /ready)
  - Rate limiting via slowapi (10/min on /api/chat)
  - Structured JSON logging via structlog
  - Request ID middleware for distributed tracing
  - Sentry error tracking (conditional on SENTRY_DSN)
  - CORS restricted to ALLOWED_ORIGINS env var
  - Admin-gated endpoints via API key
  - Prometheus metrics at /metrics
  - Externalized secrets (no hardcoded keys)
  - Pydantic response models for typed contracts
"""

import os
import shutil
import zipfile
import uuid
import threading
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from nyayaquest_main import NyayaQuest
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from hybrid_retriever import HybridRetriever

from src.db_utils import get_user_conversations, create_conversation, get_conversation_history, db
from dotenv import load_dotenv

# ── Structured Logging ───────────────────────────────────────────────
from logging_config import log

# ── Rate Limiting ────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Sentry ───────────────────────────────────────────────────────────
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# ── Prometheus Metrics ───────────────────────────────────────────────
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ── Middleware & Dependencies ────────────────────────────────────────
from middleware.request_id import RequestIDMiddleware
from dependencies import require_admin

# ── Intent Router (pre-RAG guard) ────────────────────────────────────
from intent_router import classify_intent, QueryIntent
import intent_router
from intent_handlers import handle_non_legal_intent

load_dotenv()

# ── Sentry Init (conditional) ────────────────────────────────────────
if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1,
        environment=os.getenv("ENVIRONMENT", "production"),
        release=os.getenv("GIT_COMMIT", "unknown"),
    )
    log.info("sentry_initialized", environment=os.getenv("ENVIRONMENT", "production"))

# ── App Init ─────────────────────────────────────────────────────────
app = FastAPI(title="NyayaQuest API")

# ── Rate Limiter ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ── Middleware ───────────────────────────────────────────────────────
app.add_middleware(RequestIDMiddleware)

# CORS — restricted to allowlist from env
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

# Browsers reject allow_origins=["*"] when allow_credentials=True.
# If no origins are configured, we fall back to standard local dev origins.
if not ALLOWED_ORIGINS:
    allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
else:
    allow_origins = ALLOWED_ORIGINS

log.info("cors_configured", allow_origins=allow_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics ───────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# Custom application metrics
llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["model", "status"],
)
llm_latency = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["model"],
)
cache_hits = Counter(
    "cache_hits_total",
    "Cache hits",
    ["cache_layer"],
)
retrieval_depth = Histogram(
    "retrieval_depth_docs",
    "Number of documents retrieved",
)
rerank_scores = Histogram(
    "rerank_scores",
    "Cross-encoder rerank scores",
)

# ── Intent Classification Metrics ────────────────────────────────────
intent_classifications_total = Counter(
    "intent_classifications_total",
    "Total intent classifications",
    ["intent", "method"],
)
intent_classification_latency = Histogram(
    "intent_classification_latency_seconds",
    "Intent classification latency",
    ["method"],
)
rag_pipeline_skipped_total = Counter(
    "rag_pipeline_skipped_total",
    "Total requests where RAG was skipped due to non-legal intent",
    ["intent"],
)


# ── Custom Embedding Class ───────────────────────────────────────────
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ── Initialize AI Engine ─────────────────────────────────────────────
groq_api_key = os.getenv('GROQ_API_KEY')
chroma_dir = os.getenv('CHROMA_PERSIST_DIR', 'chroma_db_groq_legal')


# --- Auto-Setup ChromaDB ---
# Priority: 1) Already extracted, 2) Local zip in repo, 3) Download from Google Drive
if not os.path.exists(chroma_dir):
    log.info("chromadb_setup_started", chroma_dir=chroma_dir)

    local_zip = os.path.join(os.path.dirname(__file__), 'chroma_db_groq_legal.zip')
    extract_to = os.path.dirname(chroma_dir) if os.path.dirname(chroma_dir) else "."

    if os.path.exists(local_zip):
        log.info("chromadb_extracting_local_zip", path=local_zip)
        try:
            with zipfile.ZipFile(local_zip, 'r') as zf:
                zf.extractall(extract_to)
            log.info("chromadb_extracted_successfully", chroma_dir=chroma_dir)
        except Exception as e:
            log.error("chromadb_extraction_failed", error=str(e))
    else:
        log.info("chromadb_downloading_from_gdrive")
        try:
            import gdown
            gdrive_id = os.getenv("GOOGLE_DRIVE_ARTIFACT_ID", "1TXr1pW-qBU5vLHekxjgb6IQ4BLeI4twq")
            zip_path = "/tmp/chroma_db.zip"
            gdown.download(id=gdrive_id, output=zip_path, quiet=False)
            log.info("chromadb_extracting_download", chroma_dir=chroma_dir)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            os.remove(zip_path)
            log.info("chromadb_ready", chroma_dir=chroma_dir)
        except Exception as e:
            log.error("chromadb_download_failed", error=str(e))
else:
    log.info("chromadb_already_exists", chroma_dir=chroma_dir)



llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, groq_api_key=groq_api_key)
embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings, collection_name="legal_knowledge")
hybrid_retriever = HybridRetriever.from_vector_store(vector_store, k=20, vector_weight=0.5, bm25_weight=0.5)

law_engine = NyayaQuest(llm, embeddings, vector_store, hybrid_retriever=hybrid_retriever)

# ── Intent Router Configuration ──────────────────────────────────────
_classification_threshold = float(
    os.getenv("CLASSIFICATION_THRESHOLD", intent_router.CONFIDENCE_THRESHOLD)
)
intent_router.CONFIDENCE_THRESHOLD = _classification_threshold
log.info("intent_router_configured", threshold=_classification_threshold)


# ── Health & Readiness Probes ────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — is the process alive?"""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe — can we serve traffic?"""
    checks = {
        "vectorstore": _check_vectorstore(),
        "llm": _check_llm(),
        "redis": _check_redis(),
    }
    all_ok = all(checks.values())
    return JSONResponse(
        content={"ready": all_ok, "checks": checks},
        status_code=status.HTTP_200_OK if all_ok
                    else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def _check_vectorstore() -> bool:
    try:
        return vector_store._collection.count() > 0
    except Exception:
        return False


def _check_llm() -> bool:
    try:
        return bool(os.getenv("GROQ_API_KEY"))
    except Exception:
        return False


def _check_redis() -> bool:
    try:
        return law_engine.cache.redis_client.ping() if law_engine.cache.use_redis else True
    except Exception:
        return True  # in-memory fallback exists


# ── Root ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "message": "NyayaQuest AI Engine is running perfectly!"}


# ── Pydantic Models ─────────────────────────────────────────────────

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


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "NyayaQuest API is running"}

@app.get("/api/conversations/{user_id}")
def get_conversations(user_id: str):
    try:
        convos = get_user_conversations(user_id)
        return {"conversations": convos}
    except Exception as e:
        log.exception("get_conversations_failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
def create_new_conversation(data: ConversationCreate):
    try:
        new_id = create_conversation(data.user_id)
        return {"thread_id": new_id}
    except Exception as e:
        log.exception("create_conversation_failed", user_id=data.user_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

from src.auth_utils import sign_in, sign_up

@app.post("/api/auth/signup")
@limiter.limit("20/minute")
def register_user(request: Request, req: AuthRequest):
    res = sign_up(req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Signup failed"))
    log.info("user_signup", email=req.email)
    return res

@app.post("/api/auth/signin")
@limiter.limit("20/minute")
def login_user(request: Request, req: AuthRequest):
    res = sign_in(req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Login failed"))
    log.info("user_signin", email=req.email)
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
        log.exception("get_chat_history_failed", user_id=user_id, thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, chat_request: ChatRequest):
    query = chat_request.message.strip()
    log.info(
        "chat_request_received",
        user_id=chat_request.user_id,
        thread_id=chat_request.thread_id,
        query_length=len(query),
    )

    # ─────────────────────────────────────────
    # STEP 1: Intent classification (pre-RAG guard)
    # ─────────────────────────────────────────
    import time as _time
    _cls_start = _time.time()
    classification = classify_intent(query, llm=llm)
    _cls_elapsed = _time.time() - _cls_start

    # Record Prometheus metrics
    intent_classifications_total.labels(
        intent=classification.intent.value,
        method=classification.method,
    ).inc()
    intent_classification_latency.labels(
        method=classification.method,
    ).observe(_cls_elapsed)

    log.info(
        "intent_classified",
        user_id=chat_request.user_id,
        intent=classification.intent.value,
        confidence=classification.confidence,
        method=classification.method,
        latency_ms=round(_cls_elapsed * 1000, 1),
    )

    # ─────────────────────────────────────────
    # STEP 2: Route based on intent
    # ─────────────────────────────────────────

    # Non-legal intent → canned response, skip RAG entirely
    if classification.intent != QueryIntent.LEGAL_QUERY:
        canned = handle_non_legal_intent(classification.intent)
        rag_pipeline_skipped_total.labels(
            intent=classification.intent.value,
        ).inc()
        log.info(
            "rag_skipped",
            user_id=chat_request.user_id,
            intent=classification.intent.value,
        )
        return ChatResponse(
            response=canned["answer"],
            context=[],
            thread_id=chat_request.thread_id,
        )

    # ─────────────────────────────────────────
    # STEP 3: Legal query → full RAG pipeline
    # ─────────────────────────────────────────
    try:
        result, context, updated_history = law_engine.conversational(
            query, chat_request.thread_id
        )

        # Serialize context documents for frontend
        serialized_context = []
        if context:
            for doc in context:
                serialized_context.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                })

        log.info(
            "chat_request_completed",
            user_id=chat_request.user_id,
            doc_count=len(serialized_context),
        )

        return ChatResponse(
            response=result,
            context=serialized_context,
            thread_id=chat_request.thread_id,
        )
    except Exception as e:
        log.exception(
            "chat_request_failed",
            user_id=chat_request.user_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin Endpoints ──────────────────────────────────────────────────

@app.get("/api/admin/cost", dependencies=[Depends(require_admin)])
def cost_report():
    """Admin endpoint to check daily API costs."""
    llm_count = llm_calls_total._metrics.get(("llama-3.3-70b", "success"), None)
    llm_count_val = int(llm_count._value.get()) if llm_count else 0

    # Groq LLaMA-3.3-70b pricing (approximate)
    cost_per_call = 0.0004  # USD
    estimated_cost_usd = llm_count_val * cost_per_call

    cache_hit_metric = cache_hits._metrics.get(("redis",), None)
    cache_hit_val = int(cache_hit_metric._value.get()) if cache_hit_metric else 0

    return {
        "date": datetime.utcnow().isoformat(),
        "llm_calls_today": llm_count_val,
        "estimated_cost_usd": round(estimated_cost_usd, 2),
        "cache_hits": cache_hit_val,
    }


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
        pipeline_kwargs: Dict[str, Any] = {
            "pdf_path": pdf_path,
            "chroma_dir": CHROMA_DIR,
            "collection_name": COLLECTION_NAME,
            "no_llm": no_llm,
        }
        if not no_llm and LLM_MODEL:
            pipeline_kwargs["llm_model"] = str(LLM_MODEL)
        segments = run_pipeline(**pipeline_kwargs)

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
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    save_path = os.path.join(LEGAL_PDFS_DIR, filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = _new_job(filename)
    _append_log(job_id, "info", 0, f"Saved to data/legal_pdfs/{filename}")
    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, save_path, no_llm),
        daemon=True,
    ).start()

    return {"job_id": job_id, "filename": filename, "status": "queued"}


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

