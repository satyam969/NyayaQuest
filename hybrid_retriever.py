"""
Hybrid Retriever: Vector Search + BM25
=======================================
Combines dense vector retrieval (semantic similarity) with sparse BM25 retrieval 
(exact keyword matching) using Reciprocal Rank Fusion (RRF).

This dramatically improves retrieval for legal queries where exact terms like
"murder", "theft", "Section 103" are critical AND the semantic meaning matters.
"""

import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines:
    1. Dense Vector Search (from ChromaDB via LangChain)
    2. Sparse BM25 Keyword Search (built from the same corpus)
    
    Results are fused using Reciprocal Rank Fusion (RRF).
    """
    
    vector_retriever: object = Field(description="LangChain vector retriever")
    bm25: object = Field(default=None, description="BM25 index")
    corpus_docs: List[Document] = Field(default_factory=list, description="All documents for BM25")
    k: int = Field(default=20, description="Number of results to return")
    vector_weight: float = Field(default=0.5, description="Weight for vector results (0-1)")
    bm25_weight: float = Field(default=0.5, description="Weight for BM25 results (0-1)")

    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_vector_store(cls, vector_store, k=20, vector_weight=0.5, bm25_weight=0.5):
        """
        Build the hybrid retriever from a LangChain vector store.
        Loads ALL documents from the store to build the BM25 index.
        """
        # 1. Build the vector retriever
        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # 2. Load ALL documents from ChromaDB for BM25 indexing
        collection = vector_store._collection
        all_data = collection.get(include=['documents', 'metadatas'])
        
        corpus_docs = []
        for i, (doc_text, meta) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
            corpus_docs.append(Document(
                page_content=doc_text,
                metadata=meta if meta else {}
            ))
        
        # 3. Tokenize corpus for BM25 (only if documents exist)
        if len(corpus_docs) > 0:
            tokenized_corpus = [cls._tokenize(doc.page_content) for doc in corpus_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            print(f"  [SUCCESS] BM25 index built with {len(corpus_docs)} documents")
        else:
            bm25 = None
            print("  [WARNING] Vector store is empty. BM25 index skipping.")
        
        return cls(
            vector_retriever=vector_retriever,
            bm25=bm25,
            corpus_docs=corpus_docs,
            k=k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple legal-aware tokenization."""
        # Lowercase + split on non-alphanumeric (keeps section numbers intact)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    @staticmethod
    def _is_section_specific_query(query: str) -> bool:
        """
        Detect queries that reference an explicit Section or Order number.
        These benefit from higher BM25 weight since BM25 excels at exact term matching.
        Examples: 'Section 35B', 'Section 21A', 'Order XXXVII', 'Section 34'
        """
        return bool(re.search(
            r'section\s+\d+[A-Z]?|order\s+[IVXLCDM]+',
            query,
            re.IGNORECASE
        ))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Main retrieval method called by LangChain.
        1. Get top-k from vector search
        2. Get top-k from BM25
        3. Fuse using Reciprocal Rank Fusion (with adaptive weights)
        """
        # Adaptive weights: boost BM25 for section/order-specific queries
        if self._is_section_specific_query(query):
            vec_w  = 0.3
            bm25_w = 0.7
        else:
            vec_w  = self.vector_weight
            bm25_w = self.bm25_weight

        # Vector search results
        vector_docs = self.vector_retriever.invoke(query)

        # Fallback if BM25 is not initialized
        if self.bm25 is None or not self.corpus_docs:
            return vector_docs[:self.k]

        # BM25 search results
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k BM25 indices
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        bm25_docs = [self.corpus_docs[i] for i in top_bm25_indices]

        # Reciprocal Rank Fusion with adaptive weights
        fused = self._reciprocal_rank_fusion(vector_docs, bm25_docs, vec_w=vec_w, bm25_w=bm25_w)

        return fused[:self.k]
    
    def _reciprocal_rank_fusion(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        rrf_k: int = 60,
        vec_w: float = None,
        bm25_w: float = None,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) to combine two ranked lists.
        Score = weight * 1/(rrf_k + rank)

        vec_w / bm25_w allow per-call weight overrides (adaptive weighting).
        Higher rrf_k means less emphasis on top ranks (smoother fusion).
        """
        if vec_w is None:
            vec_w = self.vector_weight
        if bm25_w is None:
            bm25_w = self.bm25_weight

        doc_scores = {}  # page_content -> (score, doc)

        # Score vector results
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            score = vec_w * (1.0 / (rrf_k + rank + 1))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc)
            else:
                doc_scores[key] = (score, doc)

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            score = bm25_w * (1.0 / (rrf_k + rank + 1))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc)
            else:
                doc_scores[key] = (score, doc)

        # Sort by fused score (descending)
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

        return [doc for _, doc in sorted_docs]
