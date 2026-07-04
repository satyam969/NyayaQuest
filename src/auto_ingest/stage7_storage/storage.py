"""
stage7_storage/storage.py — ChromaDB storage with idempotent upsert.

Upgrades over v1
----------------
1. Normalised chunk IDs — text is lowercased + whitespace-collapsed before
   hashing so minor formatting changes don't create duplicate logical chunks.
   Segment index and doc path hash added when available.
2. Rich metadata sanitization — strings trimmed to 500 chars; lists/dicts
   serialised to JSON strings; None → "N/A".
3. Collection caching — module-level dict avoids recreating the embedding
   function and collection on every call.
4. Consistent constants — BATCH_SIZE sourced from config (100), not a
   local override (20).
5. Smart batch retry — failed batches retry at 1/5 size, then singly.
   No chunk is silently lost due to one bad embed.
6. No per-batch collection.count() — only logged once at the end.
7. Storage metadata — stored_at timestamp, pipeline_version, embedding_model
   injected into every chunk's metadata before upsert.
8. Normalised text hash — lightweight content fingerprint stored as
   normalized_text_hash for corpus diagnostics.

Public API (unchanged):
    make_chunk_id(chunk)                               → str
    sanitize_metadata(meta)                            → dict
    get_collection(chroma_dir, collection_name, ...)   → Collection
    store_chunks(chunks, chroma_dir, collection_name, embedding_model) → int
"""

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..config import BATCH_SIZE, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "2.0"

# ── Collection cache ─────────────────────────────────────────────────────────
_COLLECTION_CACHE: Dict[Tuple[str, str, str], chromadb.Collection] = {}


# ─────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """
    Stable normalisation for hashing purposes only (not stored).
    Lowercases, strips Unicode modifiers, collapses whitespace.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _short_hash(value: str, length: int = 16) -> str:
    """Return the first `length` hex chars of a SHA-256 digest."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


# ─────────────────────────────────────────────────────────────────────
# Chunk ID
# ─────────────────────────────────────────────────────────────────────

def make_chunk_id(chunk: Dict[str, Any]) -> str:
    """
    Deterministic 32-char hex ID.

    Components (joined with '|'):
      doc_title · section_number · chunk_index · sub_chunk
      · segment_index (if present) · normalised first 120 chars of text
      · optional source path hash

    Normalisation of text ensures minor whitespace / formatting changes
    do not create duplicate logical chunk IDs.
    """
    meta = chunk.get("metadata", {})
    text = chunk.get("text", "")

    normalised_prefix = _normalise_text(text[:120])

    parts = [
        str(meta.get("doc_title",      "")),
        str(meta.get("section_number", "")),
        str(meta.get("chunk_index",    "")),
        str(meta.get("sub_chunk",      "")),
        str(meta.get("segment_index",  "")),
        normalised_prefix,
    ]

    # Optional: include a short hash of the source PDF path for cross-doc
    # disambiguation when the same section exists in multiple editions
    pdf_path = meta.get("pdf_path", "")
    if pdf_path:
        parts.append(_short_hash(pdf_path, 8))

    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────
# Metadata sanitization
# ─────────────────────────────────────────────────────────────────────

_MAX_META_STR_LEN = 500   # ChromaDB chokes on very long string values


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata values are ChromaDB-safe (str / int / float / bool).

    Rules:
      - str  → trimmed to MAX_META_STR_LEN
      - int / float / bool → kept as-is
      - None  → "N/A"
      - list / dict → JSON-serialised string (trimmed)
      - anything else → str() (trimmed)
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            clean[k] = v
        elif isinstance(v, str):
            clean[k] = v[:_MAX_META_STR_LEN]
        elif v is None:
            clean[k] = "N/A"
        elif isinstance(v, (list, dict)):
            try:
                clean[k] = json.dumps(v)[:_MAX_META_STR_LEN]
            except (TypeError, ValueError):
                clean[k] = str(v)[:_MAX_META_STR_LEN]
        else:
            clean[k] = str(v)[:_MAX_META_STR_LEN]
    return clean


# ─────────────────────────────────────────────────────────────────────
# Collection factory with caching
# ─────────────────────────────────────────────────────────────────────

def get_collection(
    chroma_dir:      str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """
    Get or create the ChromaDB collection.

    Results are cached in-process by (chroma_dir, collection_name,
    embedding_model) so the embedding model is not reloaded on every call.
    """
    cache_key = (chroma_dir, collection_name, embedding_model)
    if cache_key in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[cache_key]

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        col    = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
        )
        _COLLECTION_CACHE[cache_key] = col
        logger.info(
            f"[storage] Collection ready: '{collection_name}' "
            f"@ {chroma_dir}  model={embedding_model}"
        )
        return col
    except Exception as exc:
        logger.error(f"[storage] Failed to create collection '{collection_name}': {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────
# Storage metadata injection
# ─────────────────────────────────────────────────────────────────────

def _inject_storage_meta(
    meta:            Dict[str, Any],
    embedding_model: str,
) -> Dict[str, Any]:
    """
    Inject pipeline-level metadata before upsert.
    Preserves all existing keys; only adds missing ones.
    """
    now = datetime.now(timezone.utc).isoformat()
    injected = {
        "stored_at":        now,
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model":  embedding_model,
    }
    # Only set if not already present (don't overwrite doc-specific values)
    for k, v in injected.items():
        if k not in meta:
            meta[k] = v
    return meta


def _normalised_text_hash(text: str) -> str:
    """Lightweight content fingerprint for corpus diagnostics."""
    return _short_hash(_normalise_text(text), 16)


# ─────────────────────────────────────────────────────────────────────
# Batch upsert with retry
# ─────────────────────────────────────────────────────────────────────

def _upsert_batch(
    collection: chromadb.Collection,
    ids:        List[str],
    documents:  List[str],
    metadatas:  List[Dict[str, Any]],
    batch_num:  int,
) -> int:
    """
    Upsert a single batch.  On failure, retries with smaller sub-batches
    (size // 5), then individually.  Returns the number successfully stored.
    """
    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)
    except Exception as exc:
        logger.warning(
            f"[storage] Batch {batch_num} failed ({len(ids)} chunks): {exc} "
            f"— retrying as smaller sub-batches"
        )

    # Retry at 1/5 size
    sub_size = max(1, len(ids) // 5)
    stored   = 0
    for i in range(0, len(ids), sub_size):
        s_ids  = ids[i : i + sub_size]
        s_docs = documents[i : i + sub_size]
        s_meta = metadatas[i : i + sub_size]
        try:
            collection.upsert(ids=s_ids, documents=s_docs, metadatas=s_meta)
            stored += len(s_ids)
        except Exception as exc2:
            logger.warning(
                f"[storage] Sub-batch failed ({len(s_ids)} chunks): {exc2} "
                f"— retrying individually"
            )
            # Final fallback: individual inserts
            for idx, doc, meta in zip(s_ids, s_docs, s_meta):
                try:
                    collection.upsert(ids=[idx], documents=[doc], metadatas=[meta])
                    stored += 1
                except Exception as exc3:
                    logger.error(f"[storage] Single-chunk upsert failed (id={idx}): {exc3}")

    return stored


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks:          List[Dict[str, Any]],
    chroma_dir:      str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    """
    Upsert chunks into ChromaDB.  Returns the number of chunks stored.

    Design:
    - Deterministic IDs → idempotent; re-ingesting never duplicates.
    - Storage metadata injected before upsert (stored_at, pipeline_version).
    - Normalised text hash stored for corpus diagnostics.
    - Batched at BATCH_SIZE (from config); failed batches retry gracefully.
    - collection.count() called once at end (not per-batch).
    """
    if not chunks:
        logger.warning("[storage] store_chunks: empty chunk list — nothing to store")
        return 0

    collection = get_collection(chroma_dir, collection_name, embedding_model)
    logger.info(f"[storage] Storing {len(chunks)} chunks into '{collection_name}'")

    ids:       List[str]            = []
    documents: List[str]            = []
    metadatas: List[Dict[str, Any]] = []

    for c in chunks:
        text = c.get("text", "")
        meta = dict(c.get("metadata", {}))

        # Inject pipeline metadata and content fingerprint
        meta = _inject_storage_meta(meta, embedding_model)
        meta["normalized_text_hash"] = _normalised_text_hash(text)

        ids.append(make_chunk_id(c))
        documents.append(text)
        metadatas.append(sanitize_metadata(meta))

    stored    = 0
    batch_num = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch_num += 1
        b_ids  = ids[i : i + BATCH_SIZE]
        b_docs = documents[i : i + BATCH_SIZE]
        b_meta = metadatas[i : i + BATCH_SIZE]

        n = _upsert_batch(collection, b_ids, b_docs, b_meta, batch_num)
        stored += n
        logger.info(f"[storage] Batch {batch_num}: stored={n}/{len(b_ids)}")

    try:
        final_count = collection.count()
        logger.info(
            f"[storage] Complete: stored={stored}/{len(chunks)}  "
            f"collection_total={final_count}"
        )
    except Exception:
        logger.info(f"[storage] Complete: stored={stored}/{len(chunks)}")

    return stored
