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
        
        # 3. Tokenize corpus for BM25
        tokenized_corpus = [cls._tokenize(doc.page_content) for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"  ✅ BM25 index built with {len(corpus_docs)} documents")
        
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
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Main retrieval method called by LangChain.
        1. Get top-k from vector search
        2. Get top-k from BM25
        3. Fuse using Reciprocal Rank Fusion
        """
        # Vector search results
        vector_docs = self.vector_retriever.invoke(query)
        
        # BM25 search results
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k BM25 indices
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        bm25_docs = [self.corpus_docs[i] for i in top_bm25_indices]
        
        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_docs, bm25_docs)
        
        return fused[:self.k]
    
    def _reciprocal_rank_fusion(self, vector_docs: List[Document], bm25_docs: List[Document], rrf_k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) to combine two ranked lists.
        Score = weight * 1/(rrf_k + rank)
        
        Higher rrf_k means less emphasis on top ranks (smoother fusion).
        """
        doc_scores = {}  # page_content -> (score, doc)
        
        # Score vector results
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            score = self.vector_weight * (1.0 / (rrf_k + rank + 1))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc)
            else:
                doc_scores[key] = (score, doc)
        
        # Score BM25 results  
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            score = self.bm25_weight * (1.0 / (rrf_k + rank + 1))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + score, doc)
            else:
                doc_scores[key] = (score, doc)
        
        # Sort by fused score (descending)
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in sorted_docs]
