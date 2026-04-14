"""
stage7_storage/storage.py — ChromaDB storage with idempotent upsert.

Key design decisions:
  - Deterministic chunk IDs via SHA-256 hash of content + key metadata.
    Re-ingesting the same document never creates duplicates (upsert).
  - Metadata sanitization: ChromaDB only accepts str / int / float / bool.
    None → "N/A", everything else → str().
  - Batched upsert at BATCH_SIZE (100) to stay within memory limits.
"""

import hashlib
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

CHROMA_DIR       = "chroma_db_groq_legal"
COLLECTION_NAME  = "legal_knowledge"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
BATCH_SIZE       = 100


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def make_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Deterministic 32-char hex ID.
    Derived from doc_title + section_number + chunk_index + sub_chunk
    + first 100 chars of text.  Collision probability is negligible.
    """
    meta = chunk.get("metadata", {})
    key  = "|".join(
        [
            str(meta.get("doc_title",      "")),
            str(meta.get("section_number", "")),
            str(meta.get("chunk_index",    "")),
            str(meta.get("sub_chunk",      "")),
            chunk.get("text", "")[:100],
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata values are ChromaDB-safe (str/int/float/bool).
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = "N/A"
        else:
            clean[k] = str(v)
    return clean


# ─────────────────────────────────────────────────────────────────────
# Collection factory
# ─────────────────────────────────────────────────────────────────────

def get_collection(
    chroma_dir:      str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_dir)
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
    )


# ─────────────────────────────────────────────────────────────────────
# Main storage function
# ─────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: List[Dict[str, Any]],
    chroma_dir:      str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    """
    Upsert chunks into ChromaDB.  Returns the number of chunks stored.
    Uses upsert (not add) so re-ingestion is safe and idempotent.
    """
    if not chunks:
        logger.warning("store_chunks: empty chunk list — nothing to store")
        return 0

    collection = get_collection(chroma_dir, collection_name, embedding_model)

    ids       = [make_chunk_id(c)                        for c in chunks]
    documents = [c["text"]                               for c in chunks]
    metadatas = [sanitize_metadata(c.get("metadata", {})) for c in chunks]

    stored = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids   = ids[i:i + BATCH_SIZE]
        batch_docs  = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        try:
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            stored += len(batch_ids)
            logger.info(
                f"Stored batch {i // BATCH_SIZE + 1}: {len(batch_ids)} chunks "
                f"(collection total ≈ {collection.count()})"
            )
        except Exception as exc:
            logger.error(f"Storage batch {i // BATCH_SIZE + 1} failed: {exc}")

    return stored
