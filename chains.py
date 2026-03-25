import logging
from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranker — loaded ONCE at module import time (avoids per-request overhead)
# ---------------------------------------------------------------------------
# logger.info("Loading BAAI/bge-reranker-base cross-encoder …")
# _RERANKER = CrossEncoder("BAAI/bge-reranker-base")
# logger.info("Reranker loaded successfully.")

_RERANKER = None

def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        logger.info("Loading reranker model …")
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker loaded successfully.")
    return _RERANKER


def rerank_documents(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:  # TUNABLE: top_n
    """
    Re-rank *docs* against *query* using the BGE cross-encoder and return the
    top-*top_n* documents ordered by descending relevance score.

    Preserves all original document metadata (section number, act name, etc.).
    """
    if not docs:
        return docs

    pairs = [(query, doc.page_content) for doc in docs]
    # scores = _RERANKER.predict(pairs, batch_size=32)

    reranker = get_reranker()
    scores = reranker.predict(pairs, batch_size=32)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # ✅ Attach score to metadata (IMPORTANT)
    for score, doc in ranked:
        doc.metadata["rerank_score"] = float(score)

    # 🔍 DEBUG: Print top reranked documents
    for i, (score, doc) in enumerate(ranked[:top_n]):
        logger.info(
            "Rank %d | Score: %.4f | Section: %s | Text: %s",
            i + 1,
            float(score),
            doc.metadata.get("section"),
            doc.page_content[:150].replace("\n", " ")
        )

    top_docs = [doc for _, doc in ranked[:top_n]]

    logger.info(
        "Reranker: %d candidates → top-%d selected (scores: %s)",
        len(docs),
        top_n,
        [round(float(s), 4) for s, _ in ranked[:top_n]],
    )
    return top_docs


class RerankingRetriever(BaseRetriever):
    """
    Thin LangChain BaseRetriever wrapper that:
      1. Delegates retrieval to an inner retriever (MultiQueryRetriever).
     
     
     
     
     
     
     
     
           2. Applies cross-encoder reranking via rerank_documents().
      3. Returns only the top-n reranked documents to the rest of the chain.

    This keeps the reranking logic completely isolated from the rest of the
    pipeline — nothing outside this class or rerank_documents() needs to change.
    """

    inner_retriever: BaseRetriever
    query: str = ""          # set dynamically on each call
    top_n: int = 5          # TUNABLE: number of docs returned to the LLM

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 1. Retrieve candidate pool from MultiQueryRetriever
        raw_docs = self.inner_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        logger.info("RerankingRetriever: retrieved %d raw documents.", len(raw_docs))

        # 2. Apply cross-encoder reranking
        return rerank_documents(query, raw_docs, top_n=self.top_n)


def get_rag_chain(llm, vector_store, system_prompt, qa_prompt, hybrid_retriever=None):
    from langchain.retrievers import MultiQueryRetriever
    from langchain.prompts import PromptTemplate

    # Custom Multi-Query prompting to prioritize formal Statutes and Sections
    template = """You are a senior Indian legal researcher with expertise in the Bharatiya Nyaya Sanhita (BNS) 2023 and the Code of Civil Procedure, 1908 (CPC).
    The database contains chunks prefixed with [LAW_CODE YEAR] [CHAPTER] Section NUMBER or [CPC 1908] Order ROMAN_NUMERAL — Title.

    Rewrite the user's question into 4 different versions:
    1. A formal Section lookup (e.g., "Section 103 BNS punishment for murder" or "Section 35B CPC costs for causing delay")
    2. An exact statutory phrase (e.g., "Whoever commits murder shall be punished" or "shall not be allowed to take further steps")
    3. A chapter-aware legal research query (e.g., "Chapter VI offences affecting human body" or "CPC Part I Suits in General costs adjournment")
    4. A CPC Order/Rule lookup if applicable (e.g., "Order XXXVII Rule 2 summary suit leave to defend CPC 1908" or "Order I Rule 1 joined as plaintiffs CPC"); if not a CPC Order/Rule question, write a synonym/paraphrase of the original question instead.

    Original question: {question}
    Generate only the 4 versions:"""

    mq_prompt = PromptTemplate(input_variables=["question"], template=template)

    # Use hybrid retriever if provided, otherwise fall back to vector-only
    if hybrid_retriever is not None:
        base_retriever = hybrid_retriever
    else:
        base_retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 20}
        )
    
    # Multi-Query with statutory prompt
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, prompt=mq_prompt
    )

    # ── RERANKING LAYER ────────────────────────────────────────────────────
    # Inserted EXACTLY between Retriever → LLM.
    # The LLM only ever receives the top-5 reranked documents.
    reranking_retriever = RerankingRetriever(
        inner_retriever=mq_retriever,
        top_n=5,   # TUNABLE: reduce to 3 for speed, increase to 8 for recall
    )
    # ── END RERANKING LAYER ────────────────────────────────────────────────

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, reranking_retriever, contextualize_q_prompt   # ← uses reranked retriever
    )

    qa_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", qa_prompt),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain