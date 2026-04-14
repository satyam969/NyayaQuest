"""
stage5_refinement/retry_controller.py — Schema refinement retry loop.

Runs up to MAX_RETRIES rounds of:
  parse → evaluate → (if fail) criticise → refine schema → repeat

Returns the best chunk set found across all attempts.
"""

import logging
from typing import Dict, Any, List, Tuple

from ..stage3_parsing.schema_chunker  import SchemaChunker
from ..stage4_evaluation.quality_evaluator import evaluate_chunks
from ..stage4_evaluation.critic       import analyze_failures
from .schema_refiner                  import refine_schema

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def schema_refinement_loop(
    llm,
    initial_schema: Dict[str, Any],
    text: str,
    doc_title: str,
    text_sample: str,
    threshold: float = 0.70,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], bool]:
    """
    Iteratively refine a schema until quality passes or retries exhausted.

    Parameters
    ----------
    llm            : LangChain LLM instance
    initial_schema : starting schema dict from Stage 2
    text           : full document text
    doc_title      : document label for chunk prefixes
    text_sample    : short text sample for the refinement prompt
    threshold      : quality score threshold

    Returns
    -------
    best_chunks  : List[dict]  — chunk list with highest overall score
    best_metrics : Dict        — metrics for the best attempt
    converged    : bool        — True if any attempt passed the threshold
    """
    schema = initial_schema
    best_chunks:  List[Dict[str, Any]] = []
    best_metrics: Dict[str, float]     = {"overall": 0.0}

    for attempt in range(MAX_RETRIES + 1):
        chunker = SchemaChunker(schema, doc_title)
        chunks  = chunker.parse(text)
        passed, metrics = evaluate_chunks(chunks, threshold)

        logger.info(
            f"Refinement attempt {attempt}: "
            f"score={metrics['overall']:.3f}  passed={passed}"
        )

        if metrics["overall"] > best_metrics["overall"]:
            best_chunks  = chunks
            best_metrics = metrics

        if passed:
            return best_chunks, best_metrics, True

        if attempt >= MAX_RETRIES:
            logger.warning(f"Refinement: exhausted {MAX_RETRIES} retries without convergence")
            break

        # Analyse failures and ask LLM for a patch
        failure_report = analyze_failures(chunks, metrics)
        refined = refine_schema(llm, schema, failure_report, text_sample)

        if refined is None:
            logger.warning(f"Refinement attempt {attempt + 1}: refiner returned None — stopping early")
            break

        schema = refined

    return best_chunks, best_metrics, False
