import threading
import logging
from cache import RedisCache
from langchain.schema import HumanMessage, AIMessage
from chains import get_rag_chain
from prompts import SYSTEM_PROMPT, QA_PROMPT
from src.db_utils import add_message_to_db, update_conversation_title
from src.title_gen import generate_conversation_title
from logging_config import log
from verification import verify_answer_grounding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

# ── Prometheus Metrics (imported from api.py at runtime) ─────────────
# We import lazily to avoid circular imports during module init.
_metrics = None

def _get_metrics():
    global _metrics
    if _metrics is None:
        try:
            from api import llm_calls_total, llm_latency, cache_hits
            _metrics = {
                "llm_calls_total": llm_calls_total,
                "llm_latency": llm_latency,
                "cache_hits": cache_hits,
            }
        except ImportError:
            _metrics = {}
    return _metrics


class NyayaQuest:
    """
    NyayaQuest is a conversational AI interface that leverages a retrieval-augmented generation (RAG) pipeline 
    to answer user queries based on vector search and LLM responses. It supports Redis-based caching to improve 
    performance and stores session-based chat histories in memory and Firestore.
    """
    store = {}
    store_lock = threading.Lock()

    def __init__(self, llm, embeddings, vector_store, redis_url="redis://localhost:6379/0", hybrid_retriever=None):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.cache = RedisCache(redis_url)

    def get_session_history(self, session_id):
        with NyayaQuest.store_lock:
            if session_id not in NyayaQuest.store:
                NyayaQuest.store[session_id] = self.cache.get_chat_history(session_id)
                log.info("session_history_created", session_id=session_id)
            else:
                log.debug("session_history_reused", session_id=session_id)
        return NyayaQuest.store[session_id]

    def conversational(self, query, session_id):
        """
        Handles a query from a user within a session:
        - Returns cached response if available.
        - Otherwise, runs full RAG pipeline and updates history.
        - Persists to Firestore if session_id is a valid conversation ID.
        """
        metrics = _get_metrics()

        cache_key = self.cache.make_cache_key(query, session_id)
        cached_answer = self.cache.get(cache_key)
        
        # We always want the updated context, so on cache hit we still need to know history
        # but for simplicity, we return empty context on cache hits for now.
        if cached_answer:
            log.info("cache_hit", cache_key=cache_key, session_id=session_id)
            if "cache_hits" in metrics:
                metrics["cache_hits"].labels(cache_layer="redis").inc()
            chat_history = self.get_session_history(session_id).messages
            return cached_answer, [], chat_history

        log.info("cache_miss", cache_key=cache_key, session_id=session_id)

        rag_chain = get_rag_chain(self.llm, self.vector_store, SYSTEM_PROMPT, QA_PROMPT, hybrid_retriever=self.hybrid_retriever)

        chat_history_obj = self.get_session_history(session_id)
        messages = chat_history_obj.messages
        is_first_message = len(messages) == 0

        # Track LLM call with Prometheus metrics
        import time
        start_time = time.time()
        try:
            response = rag_chain.invoke(
                {"input": query, "chat_history": messages},
                config={"configurable": {"session_id": session_id}},
            )
            latency = time.time() - start_time
            if "llm_calls_total" in metrics:
                metrics["llm_calls_total"].labels(model="llama-3.3-70b", status="success").inc()
            if "llm_latency" in metrics:
                metrics["llm_latency"].labels(model="llama-3.3-70b").observe(latency)
            log.info("llm_call_completed", session_id=session_id, latency_ms=round(latency * 1000, 1))
        except Exception as e:
            if "llm_calls_total" in metrics:
                metrics["llm_calls_total"].labels(model="llama-3.3-70b", status="error").inc()
            log.exception("llm_call_failed", session_id=session_id, error=str(e))
            raise

        answer = response['answer']
        context = response.get('context', [])

        # ── Answer Verification ──────────────────────────────────────
        try:
            retrieved_chunks = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in context
            ]
            verification = verify_answer_grounding(answer, retrieved_chunks)
            if verification["warning"]:
                log.warning(
                    "ungrounded_claims_detected",
                    session_id=session_id,
                    grounding_score=verification["grounding_score"],
                    unsupported=verification["unsupported_claims"],
                )
            else:
                log.info(
                    "answer_grounded",
                    session_id=session_id,
                    grounding_score=verification["grounding_score"],
                )
        except Exception as e:
            log.warning("verification_failed", session_id=session_id, error=str(e))

        # Update chat history (In-Memory/Redis)
        chat_history_obj.add_user_message(query)
        chat_history_obj.add_ai_message(answer)

        # 💾 Persistence in Firestore
        if session_id and not session_id.startswith("guest_") and not session_id.startswith("test_"):
            # Generate title if it's the very first message
            if is_first_message:
                try:
                    new_title = generate_conversation_title(self.llm, query)
                    update_conversation_title(session_id, new_title)
                except Exception as e:
                    log.warning("title_generation_failed", session_id=session_id, error=str(e))
            
            # Save to Firestore (including context for sources)
            context_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata} 
                for doc in context
            ]
            add_message_to_db(session_id, "user", query)
            add_message_to_db(session_id, "assistant", answer, metadata={"context": context_data})

        # Cache the answer
        self.cache.set(cache_key, answer)

        return answer, context, chat_history_obj.messages
