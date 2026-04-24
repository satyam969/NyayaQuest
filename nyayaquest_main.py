import threading
import logging
from cache import RedisCache
from langchain.schema import HumanMessage, AIMessage
from chains import get_rag_chain
from prompts import SYSTEM_PROMPT, QA_PROMPT
from src.db_utils import add_message_to_db, update_conversation_title
from src.title_gen import generate_conversation_title

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

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
                logging.info(f"Created new chat history for session_id: {session_id}")
            else:
                logging.debug(f"Using existing chat history for session_id: {session_id}")
        return NyayaQuest.store[session_id]

    def conversational(self, query, session_id):
        """
        Handles a query from a user within a session:
        - Returns cached response if available.
        - Otherwise, runs full RAG pipeline and updates history.
        - Persists to Firestore if session_id is a valid conversation ID.
        """
        cache_key = self.cache.make_cache_key(query, session_id)
        cached_answer = self.cache.get(cache_key)
        
        # We always want the updated context, so on cache hit we still need to know history
        # but for simplicity, we return empty context on cache hits for now.
        if cached_answer:
            logging.info(f"Cache hit for key: {cache_key}")
            chat_history = self.get_session_history(session_id).messages
            return cached_answer, [], chat_history

        logging.info(f"Cache miss for key: {cache_key}. Generating new answer.")

        rag_chain = get_rag_chain(self.llm, self.vector_store, SYSTEM_PROMPT, QA_PROMPT, hybrid_retriever=self.hybrid_retriever)

        chat_history_obj = self.get_session_history(session_id)
        messages = chat_history_obj.messages
        is_first_message = len(messages) == 0

        response = rag_chain.invoke(
            {"input": query, "chat_history": messages},
            config={"configurable": {"session_id": session_id}},
        )

        answer = response['answer']
        context = response.get('context', [])

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
                    logging.warning(f"Failed to generate title: {e}")
            
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
