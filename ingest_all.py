"""
ingest_all.py — Bulk ingestion runner for NyayaQuest.

Adds src/ to sys.path so auto_ingest resolves as a package,
then runs the full pipeline on every known legal PDF.

Usage (from d:\\NyayaQuest):
    python ingest_all.py
    python ingest_all.py --no-llm          # regex-only, no LLM calls
    python ingest_all.py --verbose         # DEBUG logging
    python ingest_all.py --threshold 0.65  # relax quality gate
    python ingest_all.py --pdf path/to/one.pdf  # single file
"""

import argparse
import os
import sys

# ── Fix import path ────────────────────────────────────────────────────────────
# auto_ingest lives under src/, so we need src/ on the path.
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_ingest.orchestrator import run_pipeline  # noqa: E402

# ── Legal PDFs Directory ───────────────────────────────────────────────────────
PDF_DIR = "data/legal_pdfs"


def main():
    parser = argparse.ArgumentParser(
        description="NyayaQuest — Ingest all legal PDFs into ChromaDB"
    )
    parser.add_argument(
        "--pdf",
        help="Ingest a single PDF instead of all known documents",
    )
    parser.add_argument(
        "--chroma-dir",
        default="chroma_db_groq_legal",
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection",
        default="legal_knowledge",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--llm-model",
        default="llama-3.3-70b-versatile",
        help="Groq model for LLM-assisted stages",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Quality score threshold  [0.0–1.0]",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM features (regex-only, fastest)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Build file list
    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = [os.path.join(PDF_DIR, name) for name in os.listdir(PDF_DIR) if name.lower().endswith(".pdf")]

    # Filter to existing files only
    existing = [p for p in pdfs if os.path.exists(p)]
    missing  = [p for p in pdfs if not os.path.exists(p)]

    if missing:
        print(f"WARNING: Skipping {len(missing)} missing file(s):")
        for m in missing:
            print(f"   - {m}")

    if not existing:
        print("ERROR: No PDF files found to ingest.")
        sys.exit(1)

    print(f"\n>> Ingesting {len(existing)} PDF(s) into '{args.collection}'")
    print(f"   LLM: {'disabled' if args.no_llm else args.llm_model}")
    print(f"   Quality threshold: {args.threshold}")
    print("=" * 60)

    grand_total = 0
    all_results = []

    for pdf_path in existing:
        print(f"\n[PDF] {os.path.basename(pdf_path)}")
        results = run_pipeline(
            pdf_path        = pdf_path,
            chroma_dir      = args.chroma_dir,
            collection_name = args.collection,
            llm_model       = args.llm_model,
            threshold       = args.threshold,
            no_llm          = args.no_llm,
        )
        for r in results:
            grand_total += r["chunks_stored"]
            all_results.append((os.path.basename(pdf_path), r))

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ALL DONE -- {grand_total} total chunks stored\n")
    for pdf_name, r in all_results:
        flag  = "  [FALLBACK]" if r["used_fallback"] else ""
        score = r["final_metrics"].get("overall", 0)
        print(
            f"  {pdf_name}\n"
            f"    └─ {r['segment_title']}: strategy={r['strategy_used']}  "
            f"chunks={r['chunks_stored']}  score={score:.3f}{flag}"
        )


if __name__ == "__main__":
    main()
