"""
Answer verification layer for NyayaQuest.

Post-processes every LLM answer to verify that statutory citations
(Section X, Article Y, Order Z) actually appear in the retrieved chunks.
If not, flags them as unsupported — improving trust and traceability.
"""

import re
from typing import List


def verify_answer_grounding(
    answer: str,
    retrieved_chunks: List[dict],
) -> dict:
    """
    Check if the LLM answer is grounded in retrieved chunks.

    Args:
        answer: The LLM-generated answer text.
        retrieved_chunks: List of dicts with at least a "content" or
                          "page_content" key containing the chunk text.

    Returns:
        Dict with:
            - grounding_score (float 0-1): fraction of cited provisions found
            - unsupported_claims (list[str]): citations not in any chunk
            - warning (bool): True if any unsupported claims exist
    """
    # Build a single searchable string from all retrieved chunks
    chunk_texts = []
    for c in retrieved_chunks:
        text = c.get("content") or c.get("page_content", "")
        chunk_texts.append(text)
    chunk_text = " ".join(chunk_texts).lower()

    # Extract statutory citations from the answer
    # Matches patterns like: Section 103, Article 21, Order XXXVII, Rule 2
    citations_in_answer = re.findall(
        r"(?:section|article|order|rule)\s+[\w]+",
        answer.lower(),
    )

    if not citations_in_answer:
        return {
            "grounding_score": 1.0,
            "unsupported_claims": [],
            "warning": False,
        }

    unsupported = []
    for cite in citations_in_answer:
        if cite not in chunk_text:
            unsupported.append(cite)

    grounding_score = 1.0 - (len(unsupported) / max(len(citations_in_answer), 1))

    return {
        "grounding_score": round(grounding_score, 3),
        "unsupported_claims": unsupported,
        "warning": len(unsupported) > 0,
    }
