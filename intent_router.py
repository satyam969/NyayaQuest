"""
Query intent classification for NyayaQuest.
Confidence-based routing: heuristic first, LLM fallback for ambiguous cases.
"""

import re
import hashlib
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from logging_config import log


# ─────────────────────────────────────────────
# Intent categories
# ─────────────────────────────────────────────

class QueryIntent(str, Enum):
    LEGAL_QUERY = "legal_query"
    GREETING = "greeting"
    META_QUERY = "meta_query"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


@dataclass
class ClassificationResult:
    intent: QueryIntent
    confidence: float
    method: str
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reasoning": self.reasoning,
        }


# ─────────────────────────────────────────────
# Heuristic vocabulary
# ─────────────────────────────────────────────

GREETINGS = {
    "hi", "hii", "hiii", "hey", "heyy", "hello", "helo", "hola",
    "namaste", "namaskar", "sup", "yo", "gm", "gn", "hey there",
    "hi there", "good morning", "good evening", "good afternoon",
}

# Strong signals — high confidence for legal intent
STRONG_LEGAL_INDICATORS = {
    "section", "act", "law", "legal", "court", "judgment",
    "constitution", "article", "clause", "punishment", "penalty",
    "bail", "petition", "writ", "bns", "cpc", "ipc", "rti",
    "crpc", "offence", "offense", "liable", "arrest", "warrant",
    "tribunal", "convict", "acquit", "sentence", "prosecution",
    "advocate", "lawyer", "attorney", "counsel", "litigant",
}

# Weak signals — need multiple hits for high confidence
WEAK_LEGAL_INDICATORS = {
    "rights", "duties", "fine", "notice", "complaint",
    "consumer", "labour", "labor", "employment", "contract",
    "agreement", "property", "tenant", "landlord", "divorce",
    "marriage", "custody", "harassment", "dispute", "compensation",
    "settlement", "dowry", "cheque", "check", "bounce",
}

# Meta query patterns (regex)
META_PATTERNS = [
    r"^what\s+(can|do)\s+you",
    r"^who\s+(are|made)\s+you",
    r"^how\s+(do|does)\s+(you|this|it)\s+work",
    r"^what\s+is\s+(this|nyayaquest)",
    r"^help$",
    r"^features?$",
    r"^how\s+to\s+use",
    r"^instructions?$",
    r"^commands?$",
    r"^tell\s+me\s+about\s+(you|yourself|nyayaquest)",
]

# Out-of-scope: multi-word phrases (substring match is safe)
OUT_OF_SCOPE_PHRASES = {
    "capital of", "president of", "prime minister of",
    "who won", "when did", "how tall",
    "share price",
    "us law", "american law", "uk law", "british law",
    "chinese law", "european law",
}

# Out-of-scope: single words (matched with word boundaries to avoid
# false positives like "code" matching "labour code")
OUT_OF_SCOPE_WORDS = {
    "weather", "temperature", "forecast",
    "sports", "cricket", "football", "olympics",
    "movie", "song", "actor", "actress",
    "recipe", "cook", "food",
    "python", "javascript", "programming",
    "stock", "bitcoin", "immigration",
}

# Pre-compiled regex for single-word OOS matching
_OOS_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in OUT_OF_SCOPE_WORDS) + r")\b"
)


# ─────────────────────────────────────────────
# Heuristic classifier (fast, deterministic)
# ─────────────────────────────────────────────

def classify_heuristic(query: str) -> ClassificationResult:
    """
    Fast rule-based classification with confidence scoring.
    Runs in <1ms — no external calls.
    """
    q = query.lower().strip()
    words = q.split()

    # ── Empty or nonsense ──
    if not q or len(q) < 2:
        return ClassificationResult(
            QueryIntent.AMBIGUOUS, 0.50, "heuristic",
            "empty or too short",
        )

    # ── Exact greeting match ──
    if q in GREETINGS:
        return ClassificationResult(
            QueryIntent.GREETING, 0.99, "heuristic",
            "exact greeting match",
        )

    # ── Very short queries (2-3 chars) — likely noise/greeting ──
    if len(q) < 4:
        return ClassificationResult(
            QueryIntent.GREETING, 0.85, "heuristic",
            "very short — likely greeting/noise",
        )

    # ── Meta pattern match ──
    for pattern in META_PATTERNS:
        if re.search(pattern, q):
            return ClassificationResult(
                QueryIntent.META_QUERY, 0.95, "heuristic",
                "matched meta pattern",
            )

    # ── Out-of-scope indicators ──
    oos_phrase_hits = sum(1 for p in OUT_OF_SCOPE_PHRASES if p in q)
    oos_word_hits = len(_OOS_WORD_PATTERN.findall(q))
    oos_hits = oos_phrase_hits + oos_word_hits
    if oos_hits >= 1:
        # But check for legal override — "US law in India" might still be legal
        legal_context = any(w in q for w in ["india", "indian", "domestic"])
        if not legal_context:
            return ClassificationResult(
                QueryIntent.OUT_OF_SCOPE, 0.85, "heuristic",
                "out-of-scope indicators found",
            )

    # ── Legal indicator scoring ──
    strong_hits = sum(1 for w in STRONG_LEGAL_INDICATORS if w in q)
    weak_hits = sum(1 for w in WEAK_LEGAL_INDICATORS if w in q)
    legal_score = (strong_hits * 2) + weak_hits

    # ── Very clear legal query ──
    if strong_hits >= 2:
        return ClassificationResult(
            QueryIntent.LEGAL_QUERY, 0.95, "heuristic",
            f"multiple strong legal indicators: {strong_hits}",
        )

    # ── Strong legal signal with adequate query length ──
    if strong_hits >= 1 and len(words) >= 4:
        return ClassificationResult(
            QueryIntent.LEGAL_QUERY, 0.90, "heuristic",
            "strong legal indicator + adequate length",
        )

    # ── Weak legal signals ──
    if legal_score >= 2 and len(words) >= 4:
        return ClassificationResult(
            QueryIntent.LEGAL_QUERY, 0.80, "heuristic",
            f"weak legal signals: score={legal_score}",
        )

    # ── Greeting followed by legal query ──
    starts_with_greeting = any(
        q.startswith(g + " ") for g in ["hi", "hii", "hello", "hey"]
    )
    if starts_with_greeting:
        rest = q.split(" ", 1)[1] if " " in q else ""
        if any(w in rest for w in STRONG_LEGAL_INDICATORS):
            return ClassificationResult(
                QueryIntent.LEGAL_QUERY, 0.90, "heuristic",
                "greeting + legal continuation",
            )
        return ClassificationResult(
            QueryIntent.GREETING, 0.75, "heuristic",
            "greeting with non-legal continuation",
        )

    # ── Uncertain — low confidence, delegate to LLM ──
    return ClassificationResult(
        QueryIntent.AMBIGUOUS, 0.30, "heuristic",
        f"no strong signals — score={legal_score}, len={len(words)}",
    )


# ─────────────────────────────────────────────
# LLM classifier (for ambiguous cases)
# ─────────────────────────────────────────────

CLASSIFICATION_PROMPT = """You are a query classifier for NyayaQuest, an AI legal research assistant specialized in Indian statutory law.

The covered acts are:
- Bharatiya Nyaya Sanhita (BNS) 2023
- Code of Civil Procedure (CPC) 1908
- Right to Information Act (RTI) 2005
- Consumer Protection Act 2019
- Labour Codes

Classify the query below into EXACTLY ONE category:

- LEGAL_QUERY: Question about Indian statutory law, sections, rights, procedures, penalties, or any covered act
- GREETING: Casual hello, greeting, or small talk without a legal question
- META_QUERY: Question about the assistant itself (capabilities, how it works, who made it)
- OUT_OF_SCOPE: Not about Indian law (weather, sports, coding, foreign laws, general knowledge)
- AMBIGUOUS: Unclear intent, too vague to route confidently

Query: "{query}"

Respond with ONLY the category name in uppercase. No explanation. Nothing else."""


def classify_llm(query: str, llm) -> ClassificationResult:
    """
    LLM-based classification for ambiguous cases.
    Uses a cheap, fast LLM call — no retrieval.
    """
    try:
        response = llm.invoke(CLASSIFICATION_PROMPT.format(query=query))
        result_text = (
            response.content if hasattr(response, "content")
            else str(response)
        ).strip().upper()

        # Parse response
        for intent in QueryIntent:
            if intent.value.upper() in result_text:
                return ClassificationResult(
                    intent, 0.90, "llm",
                    f"LLM classified as {intent.value}",
                )

        # Unparseable — safe fallback
        log.warning(
            "llm_classification_unparseable",
            query=query[:80],
            raw_output=result_text[:100],
        )
        return ClassificationResult(
            QueryIntent.LEGAL_QUERY, 0.50, "llm",
            "unparseable, defaulting to legal",
        )
    except Exception as e:
        log.warning("llm_classification_failed", error=str(e))
        # Safe fallback — try RAG anyway
        return ClassificationResult(
            QueryIntent.LEGAL_QUERY, 0.40, "llm_error",
            f"LLM error, defaulting to legal: {type(e).__name__}",
        )


# ─────────────────────────────────────────────
# LLM classification cache (dedupes repeated queries)
# ─────────────────────────────────────────────

_llm_cache: dict[str, ClassificationResult] = {}
_CACHE_MAX_SIZE = 5000


def _cache_key(query: str) -> str:
    """Normalize and hash query for cache key."""
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def _cache_get(query: str) -> Optional[ClassificationResult]:
    return _llm_cache.get(_cache_key(query))


def _cache_put(query: str, result: ClassificationResult) -> None:
    if len(_llm_cache) >= _CACHE_MAX_SIZE:
        # Simple LRU-ish — clear half
        for key in list(_llm_cache.keys())[:_CACHE_MAX_SIZE // 2]:
            del _llm_cache[key]
    _llm_cache[_cache_key(query)] = result


# ─────────────────────────────────────────────
# Public API — confidence-based routing
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.70


def classify_intent(query: str, llm=None) -> ClassificationResult:
    """
    Main entry point for intent classification.

    Strategy:
    1. Try fast heuristic first
    2. If confidence >= threshold, use it (85% of cases)
    3. Otherwise delegate to LLM (15% of cases)
    4. Cache LLM classifications for repeated queries
    5. Log everything for offline improvement

    Args:
        query: User's raw query text
        llm: LangChain-compatible LLM for fallback (optional)

    Returns:
        ClassificationResult with intent, confidence, method, reasoning
    """
    # ── Guard against empty input ──
    if not query or not query.strip():
        return ClassificationResult(
            QueryIntent.AMBIGUOUS, 1.0, "heuristic",
            "empty query",
        )

    # ── Fast heuristic pass ──
    heuristic_result = classify_heuristic(query)

    # ── High confidence — trust the heuristic ──
    if heuristic_result.confidence >= CONFIDENCE_THRESHOLD:
        log.info(
            "intent_classified",
            query=query[:80],
            **heuristic_result.to_dict(),
        )
        return heuristic_result

    # ── Low confidence — check LLM cache first ──
    cached = _cache_get(query)
    if cached is not None:
        log.info(
            "intent_classified_cached",
            query=query[:80],
            **cached.to_dict(),
        )
        return cached

    # ── LLM fallback ──
    if llm is not None:
        start = time.time()
        llm_result = classify_llm(query, llm)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        _cache_put(query, llm_result)
        log.info(
            "intent_classified",
            query=query[:80],
            heuristic_was=heuristic_result.intent.value,
            heuristic_confidence=heuristic_result.confidence,
            llm_latency_ms=elapsed_ms,
            **llm_result.to_dict(),
        )
        return llm_result

    # ── No LLM available — return heuristic result with warning ──
    log.warning(
        "intent_low_confidence_no_llm",
        query=query[:80],
        **heuristic_result.to_dict(),
    )
    return heuristic_result
