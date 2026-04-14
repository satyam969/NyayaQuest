"""
stage7_storage/chunker.py — Final sub-chunker.

For each incoming chunk, applies RecursiveCharacterTextSplitter when
the text exceeds chunk_size.  Preserves and propagates all metadata,
adding sub_chunk and total_chunks fields.
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE    = 800
DEFAULT_CHUNK_OVERLAP = 100


def split_chunks(
    chunks: List[Dict[str, Any]],
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Sub-split any chunk whose text exceeds chunk_size characters.

    Metadata is propagated and enriched with:
      sub_chunk    : int  — 1-indexed position within the parent chunk
      total_chunks : int  — total sub-chunks for this parent
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        if len(text) <= chunk_size:
            final.append(
                {
                    "text":     text,
                    "metadata": {**meta, "sub_chunk": 1, "total_chunks": 1},
                }
            )
        else:
            sub_texts = splitter.split_text(text)
            for j, sub_text in enumerate(sub_texts):
                final.append(
                    {
                        "text": sub_text,
                        "metadata": {
                            **meta,
                            "sub_chunk":    j + 1,
                            "total_chunks": len(sub_texts),
                        },
                    }
                )

    logger.info(
        f"[chunker] {len(final)} final chunks "
        f"(from {len(chunks)} pre-split chunks)"
    )
    return final
