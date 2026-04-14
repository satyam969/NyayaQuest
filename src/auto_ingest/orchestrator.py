"""
orchestrator.py — Autonomous Hybrid PDF Ingestion Pipeline (NyayaQuest)

Pipeline stages
───────────────
Stage 0  Extract text with footer removal; clean; filter TOC / blank pages.
Stage 1  Detect document type (heuristic); segment multi-act PDFs.
Stage 2  Select strategy (REGEX / HYBRID / SCHEMA) per detection confidence.
Stage 3  Parse with selected strategy → list of chunk dicts.
Stage 4  Evaluate quality (5-metric weighted score, threshold 0.70).
Stage 5  If quality fails → Schema-refinement loop (max 3 retries).
Stage 6  If still failing → 3-tier Fallback Engine (never-fail guarantee).
Stage 7  Sub-split long chunks; upsert into ChromaDB (idempotent).

CLI
───
python -m auto_ingest.orchestrator --pdf path/to/file.pdf [options]

Options
───────
  --chroma-dir   DIR       ChromaDB persistence path  [chroma_db_groq_legal]
  --collection   NAME      Collection name            [legal_knowledge]
  --llm-model    MODEL     Groq model for LLM steps   [llama-3.3-70b-versatile]
  --threshold    FLOAT     Quality-pass threshold      [0.70]
  --no-llm                 Regex-only mode (skip all LLM calls)
  --verbose                Enable DEBUG logging
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ── Stage imports ─────────────────────────────────────────────────────────────
from .stage0_preprocess.extractor   import extract_text_without_footers, clean_text
from .stage0_preprocess.page_filter import filter_blank_pages, remove_toc

from .stage1_detection.detector   import detect_document_type
from .stage1_detection.segmenter  import segment_document

from .stage2_strategy.hybrid_selector import Strategy, select_strategy
from .stage2_strategy.regex_strategy  import select_section_pattern
from .stage2_strategy.schema_strategy import generate_schema

from .stage3_parsing.regex_parser  import parse_with_regex
from .stage3_parsing.schema_chunker import SchemaChunker
from .stage3_parsing.hybrid_parser  import parse_hybrid

from .stage4_evaluation.quality_evaluator import QUALITY_THRESHOLD, evaluate_chunks
from .stage4_evaluation.critic            import analyze_failures

from .stage5_refinement.retry_controller import schema_refinement_loop

from .stage6_fallback.fallback_engine import run_fallback

from .stage7_storage.chunker import split_chunks
from .stage7_storage.storage import store_chunks

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(model: str = "llama-3.3-70b-versatile"):
    """Instantiate a Groq LLM from the GROQ_API_KEY env-var."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM features disabled")
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, api_key=api_key)
    except Exception as exc:
        logger.error(f"LLM init failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment processor
# ─────────────────────────────────────────────────────────────────────────────

def ingest_segment(
    text: str,
    segment_title: str,
    pdf_path: str,
    doc_type: str,
    confidence: float,
    features: Dict[str, Any],
    llm,
    chroma_dir: str,
    collection_name: str,
    threshold: float,
) -> Dict[str, Any]:
    """
    Run stages 2–7 for a single document segment.

    Returns a result summary dict.
    """
    result: Dict[str, Any] = {
        "segment_title": segment_title,
        "strategy_used": None,
        "chunks_stored": 0,
        "final_metrics": {},
        "converged":     False,
        "used_fallback": False,
    }

    # ── Stage 2: strategy selection ──────────────────────────────────────────
    strategy = select_strategy(confidence, features)
    result["strategy_used"] = strategy.value
    logger.info(
        f"[{segment_title}] strategy={strategy.value}  confidence={confidence:.2f}"
    )

    section_pattern = select_section_pattern(text, doc_type, features)
    schema: Optional[Dict[str, Any]] = None
    chunks: List[Dict[str, Any]] = []
    metrics: Dict[str, float]    = {}

    # ── Stage 3 + 4: parse → evaluate ────────────────────────────────────────
    try:
        if strategy == Strategy.REGEX:
            chunks = parse_with_regex(text, section_pattern, segment_title, doc_type, features)
            passed, metrics = evaluate_chunks(chunks, threshold)

        elif strategy == Strategy.HYBRID:
            if llm:
                schema = generate_schema(llm, text, doc_type, features)
            chunks = parse_hybrid(text, section_pattern, schema, segment_title, doc_type, features)
            passed, metrics = evaluate_chunks(chunks, threshold)

        else:  # SCHEMA
            if llm:
                schema = generate_schema(llm, text, doc_type, features)
            if schema:
                chunker = SchemaChunker(schema, segment_title)
                chunks  = chunker.parse(text)
            else:
                logger.warning(f"[{segment_title}] Schema gen failed — falling back to regex")
                chunks = parse_with_regex(text, section_pattern, segment_title, doc_type, features)
            passed, metrics = evaluate_chunks(chunks, threshold)

    except Exception as exc:
        logger.error(f"[{segment_title}] Parse stage raised: {exc}")
        passed  = False
        metrics = {"overall": 0.0}

    logger.info(
        f"[{segment_title}] initial parse: {len(chunks)} chunks  "
        f"score={metrics.get('overall', 0):.3f}"
    )

    # ── Stage 5: schema refinement loop ──────────────────────────────────────
    if not passed and llm:
        logger.info(f"[{segment_title}] Quality below threshold → schema refinement")
        if schema is None:
            schema = generate_schema(llm, text, doc_type, features)

        if schema:
            text_sample = text[:3000]
            ref_chunks, ref_metrics, converged = schema_refinement_loop(
                llm, schema, text, segment_title, text_sample, threshold
            )
            if ref_metrics.get("overall", 0) > metrics.get("overall", 0):
                chunks  = ref_chunks
                metrics = ref_metrics
                passed  = converged
                result["converged"] = converged
                logger.info(
                    f"[{segment_title}] Refinement result: "
                    f"score={metrics['overall']:.3f}  converged={converged}"
                )

    # ── Stage 6: fallback ─────────────────────────────────────────────────────
    if not passed or not chunks:
        logger.warning(f"[{segment_title}] Activating fallback engine")
        result["used_fallback"] = True
        fb_chunks = run_fallback(text, segment_title, pdf_path, metrics, llm)
        if fb_chunks:
            chunks  = fb_chunks
            _, metrics = evaluate_chunks(chunks, threshold)

    # ── Stage 7: sub-split + store ────────────────────────────────────────────
    final_chunks = split_chunks(chunks)
    stored       = store_chunks(final_chunks, chroma_dir, collection_name)

    result["chunks_stored"] = stored
    result["final_metrics"] = metrics

    logger.info(
        f"[{segment_title}] ✓ {stored} chunks stored  "
        f"final_score={metrics.get('overall', 0):.3f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path:        str,
    chroma_dir:      str   = "chroma_db_groq_legal",
    collection_name: str   = "legal_knowledge",
    llm_model:       str   = "llama-3.3-70b-versatile",
    threshold:       float = QUALITY_THRESHOLD,
    no_llm:          bool  = False,
) -> List[Dict[str, Any]]:
    """
    Ingest a single PDF through the full pipeline.

    Returns a list of per-segment result dicts (one per detected act).
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    logger.info("═" * 60)
    logger.info(f"Auto-Ingest Pipeline  →  {pdf_path}")
    logger.info("═" * 60)

    # LLM
    llm = None if no_llm else _get_llm(llm_model)

    # ── Stage 0 ───────────────────────────────────────────────────────────────
    logger.info("Stage 0: extracting + cleaning text …")
    raw_text = extract_text_without_footers(pdf_path)
    text     = clean_text(raw_text)
    text     = filter_blank_pages(text)
    text     = remove_toc(text)
    logger.info(f"Stage 0 complete — {len(text):,} chars after cleaning")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    logger.info("Stage 1: detecting document type …")
    doc_type, confidence, features = detect_document_type(text)
    logger.info(
        f"Stage 1 — doc_type={doc_type}  confidence={confidence:.2f}  "
        f"sections≈{features.get('approx_section_count', '?')}"
    )

    logger.info("Stage 1: segmenting …")
    segments = segment_document(text)
    logger.info(f"Stage 1 — {len(segments)} segment(s) found")

    # ── Stages 2–7 per segment ────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    for seg in segments:
        logger.info(f"{'─' * 50}")
        logger.info(f"Processing: {seg['title']}")
        result = ingest_segment(
            text            = seg["text"],
            segment_title   = seg["title"],
            pdf_path        = pdf_path,
            doc_type        = doc_type,
            confidence      = confidence,
            features        = features,
            llm             = llm,
            chroma_dir      = chroma_dir,
            collection_name = collection_name,
            threshold       = threshold,
        )
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(r["chunks_stored"] for r in all_results)
    logger.info("═" * 60)
    logger.info(
        f"Pipeline complete — {total} total chunks stored "
        f"across {len(segments)} segment(s)"
    )
    for r in all_results:
        flag = " [FALLBACK]" if r["used_fallback"] else ""
        logger.info(
            f"  • {r['segment_title']}: strategy={r['strategy_used']}  "
            f"chunks={r['chunks_stored']}  "
            f"score={r['final_metrics'].get('overall', 0):.3f}{flag}"
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m auto_ingest.orchestrator",
        description="NyayaQuest — Autonomous Legal PDF → ChromaDB ingestion pipeline",
    )
    p.add_argument("--pdf",         required=True,
                   help="Path to the PDF file to ingest")
    p.add_argument("--chroma-dir",  default="chroma_db_groq_legal",
                   help="ChromaDB persistence directory")
    p.add_argument("--collection",  default="legal_knowledge",
                   help="ChromaDB collection name")
    p.add_argument("--llm-model",   default="llama-3.3-70b-versatile",
                   help="Groq model for LLM-based steps")
    p.add_argument("--threshold",   type=float, default=QUALITY_THRESHOLD,
                   help="Quality score threshold (0.0–1.0)")
    p.add_argument("--no-llm",      action="store_true",
                   help="Disable all LLM features (regex-only mode)")
    p.add_argument("--verbose",     action="store_true",
                   help="Enable DEBUG-level logging")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_pipeline(
        pdf_path        = args.pdf,
        chroma_dir      = args.chroma_dir,
        collection_name = args.collection,
        llm_model       = args.llm_model,
        threshold       = args.threshold,
        no_llm          = args.no_llm,
    )

    total = sum(r["chunks_stored"] for r in results)
    print(f"\n✅  Ingestion complete — {total} chunks stored")
    for r in results:
        flag = "  [FALLBACK]" if r["used_fallback"] else ""
        score = r["final_metrics"].get("overall", 0)
        print(
            f"  • {r['segment_title']}\n"
            f"    strategy={r['strategy_used']}  chunks={r['chunks_stored']}  "
            f"score={score:.3f}{flag}"
        )


if __name__ == "__main__":
    main()
