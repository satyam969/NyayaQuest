"""
config.py — All tunable constants for the autonomous ingestion pipeline.
Single source of truth: no magic numbers anywhere else.
"""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800    # characters per chunk (covers full provisos / sub-clauses)
CHUNK_OVERLAP = 100    # characters of overlap between consecutive chunks

# ── Embedding & Storage ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR      = "chroma_db_groq_legal"
COLLECTION_NAME = "legal_knowledge"
BATCH_SIZE      = 100  # ChromaDB insertion batch size

# ── Quality Thresholds ────────────────────────────────────────────────────────
# Centralised so Stage 4/5/6 routing decisions use consistent values.
NEAR_PASS_THRESHOLD    = 0.66  # accept parse without Stage 5 retry
SEVERE_FAIL_THRESHOLD  = 0.55  # below this → always trigger Stage 6 fallback
MAX_SCHEMA_ATTEMPTS    = 3     # max LLM attempts in generate_validated_schema()

# ── Structure Detection ───────────────────────────────────────────────────────
SAMPLE_PAGES              = 12     # evenly-spaced pages used for doc profiling
MIN_SECTION_RATIO         = 0.10  # fraction of sampled pages that must match a pattern
HYBRID_CONFIDENCE_THRESH  = 0.6   # below this → enable hybrid parsing (primary + secondary)
PATTERN_MIN_SCORE         = 0.10  # below this → skip to next fallback tier

# ── Dynamic Re-evaluation ─────────────────────────────────────────────────────
REEVAL_WINDOW       = 30   # re-check regex quality every N sections parsed
REEVAL_DROP_THRESH  = 0.25 # if match ratio drops below this → re-run strategy

# ── LLM Control ──────────────────────────────────────────────────────────────
USE_LLM                   = True   # master switch; set False for fully offline mode
LLM_ONLY_IF_CONFIDENCE_BELOW = 0.5 # use LLM tiebreaker only when confidence < this
LLM_MODEL                 = "llama-3.3-70b-versatile"    # Groq — replaces decommissioned llama-3.1-8b-instant
LLM_TEMPERATURE           = 0.0
LLM_MAX_TOKENS            = 2048
LLM_BATCH_CHARS           = 6000  # characters sent per LLM extraction batch (Tier 3)

# ── Extraction / Noise Filtering ──────────────────────────────────────────────
MIN_FONT_SIZE        = 8.0   # lines with smaller fonts are dropped (footnotes, captions)
SEP_LINE_MIN_WIDTH   = 50    # pts — minimum width for a horizontal separator line
SEP_LINE_ZONE        = 0.40  # separator must be in the lower X% of the page

# ── Quarantine ────────────────────────────────────────────────────────────────
QUARANTINE_LOG = "data/quarantine_log.jsonl"
MIN_ACCEPTABLE_CHUNKS = 10   # below this → always quarantine

# ── Footnote keyword patterns (universal across all GazetteIndia PDFs) ────────
FOOTNOTE_KEYWORDS = (
    r"Subs\.|Ins\.|Omitted|w\.e\.f\.|ibid\.|A\.O\.|Rep\.|"
    r"amended in its application|extended to .{1,40} by Act|"
    r"extended to the .{1,40} by"
)

# ── Universal noise patterns ──────────────────────────────────────────────────
GAZETTE_NOISE_PATTERNS = [
    r"vlk/kkj\.k.*?CG-DL-E-\d+-\d+",           # Hindi Gazette header (BNS-style)
    r"THE GAZETTE OF INDIA EXTRAORDINARY",
    r"\[?Part II[—\-]",
    r"Sec\.\s*\d+\]?",
    r"MINISTRY OF LAW AND JUSTICE",
    r"MINISTRY OF LABOUR",
]
