"""
stage1_detection/detector.py — Semantic multi-window document-type detector.

Strategy
--------
Instead of scoring only the first 5 000 characters (which often contains
cover pages, Gazette headers, TOC, OCR junk, or boilerplate), three
semantically meaningful windows are extracted from the cleaned text:

  Window 1 — Title Block
      The earliest region that contains a meaningful statutory title anchor
      (ACT / CODE / RULES / "No. X of YEAR" / "An Act to …").
      Extracted as ±TITLE_WINDOW_SIZE//2 chars around the anchor.

  Window 2 — Densest Structural Block
      A sliding window (DENSE_WINDOW_SIZE, 50 % overlap, capped at
      DENSE_MAX_WINDOWS iterations) scanning the full text.  The window
      with the highest count of structural-marker regex hits is selected.

  Window 3 — Last Structured Block
      Scans backward from the end to find the last region that still
      contains legal structure markers (sections / schedules / rules /
      chapters).  Extracts up to TAIL_WINDOW_SIZE chars ending there.

Each window is scored against WINDOW_SCORE_TEMPLATES for all known doc
types.  Per-window scores are merged with fixed weights:

    final = 0.35 * title_score + 0.45 * dense_score + 0.20 * tail_score

Anti-pattern penalties and boost rules are then applied before the
winner is selected.

Small-doc shortcut
------------------
If len(text) < SMALL_DOC_THRESHOLD (12 000 chars) the whole text is
used as Window 1; Windows 2 and 3 are skipped (score weight → 1.0).

Public API (backward compatible)
---------------------------------
detect_document_type(text, debug=False)
  → (doc_type: str, confidence: float, features: dict)

  features always contains all legacy keys PLUS:
    "debug_scores": { ... }   (populated only when debug=True or always
                               as an empty dict otherwise)
"""

import re
from typing import Tuple, Dict, Any, List, Optional

from ..utils.patterns import STRUCTURAL_PROFILES, WINDOW_SCORE_TEMPLATES


# ─────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────

SMALL_DOC_THRESHOLD = 12_000   # chars — use whole text as W1 only
TITLE_WINDOW_SIZE   = 5_000    # chars extracted around title anchor
DENSE_WINDOW_SIZE   = 5_000    # sliding window size for W2
DENSE_WINDOW_STEP   = 2_500    # 50 % overlap step
DENSE_MAX_WINDOWS   = 30       # cap to bound worst-case scan time
TAIL_WINDOW_SIZE    = 5_000    # chars for the last structured block

# Weighted merge: (title, dense, tail)
_W_TITLE = 0.35
_W_DENSE = 0.45
_W_TAIL  = 0.20

# Boost / penalty magnitudes
_BOOST_LARGE  = 0.20
_BOOST_MEDIUM = 0.12
_BOOST_SMALL  = 0.06
_PENALTY      = 0.25

# Regex anchors used to locate the title block
_TITLE_ANCHOR_RE = re.compile(
    r"""
    (?:
        THE\s+[A-Z][A-Z\s,'\-]{3,}(?:ACT|CODE|RULES|REGULATIONS),?\s+\d{4}
      | No\.\s*\d{1,3}\s+of\s+\d{4}
      | An\s+Act\s+to\b
      | \bCODE\b
      | \bACT\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Structural markers used to score Window 2 density
_DENSE_MARKERS: List[str] = [
    r"(?m)^\s*\d{1,3}[A-Z]?\.\s",          # section numbers
    r"(?m)^CHAPTER\s+[IVXLCDM]+",
    r"(?m)^PART\s+[IVXLCDM]+",
    r"(?m)^Rule\s+\d+",
    r"(?m)^ORDER\s+[IVXLCDM]+",
    r"(?m)^SCHEDULE",
    r"(?m)^Article\s+\d+",
    r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by",  # amendment annotations
]

# Markers used to locate the last structured region (Window 3)
# Note: no re.VERBOSE here to avoid inline-flag conflicts on Python 3.11.
_TAIL_ANCHOR_RE = re.compile(
    r"(?:SCHEDULE|FORM\s+[A-Z\d]|APPENDIX|ANNEXURE"
    r"|^Rule\s+\d+"
    r"|^CHAPTER\s+[IVXLCDM]+"
    r"|^\s*\d{1,3}[A-Z]?\.\s)",
    re.IGNORECASE | re.MULTILINE,
)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def detect_document_type(
    text: str,
    debug: bool = False,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Score text against structural format signatures using a 3-window
    semantic classifier.

    Parameters
    ----------
    text  : str   — cleaned, full document text
    debug : bool  — if True, per-window scores / extracted windows /
                    penalties / boosts are included in features["debug_scores"]

    Returns
    -------
    doc_type   : str   — best-matching type (e.g. "GAZETTE_ACT")
    confidence : float — normalised score in [0, 1]
    features   : dict  — structural flags + score breakdown
    """

    # ── 1. Extract the three semantic windows ─────────────────────────
    is_small = len(text) < SMALL_DOC_THRESHOLD

    w1_text = extract_title_block(text)
    w2_text = text if is_small else extract_densest_structure_block(text)
    w3_text = "" if is_small else extract_last_structured_block(text)

    # ── 2. Score each window against every template ───────────────────
    doc_types = list(WINDOW_SCORE_TEMPLATES.keys())

    w1_scores = {dt: score_template(w1_text, WINDOW_SCORE_TEMPLATES[dt]) for dt in doc_types}
    w2_scores = {dt: score_template(w2_text, WINDOW_SCORE_TEMPLATES[dt]) for dt in doc_types}
    w3_scores = {dt: score_template(w3_text, WINDOW_SCORE_TEMPLATES[dt]) for dt in doc_types} \
                if w3_text else {dt: 0.0 for dt in doc_types}

    # ── 3. Merge scores ───────────────────────────────────────────────
    if is_small:
        merged = {dt: w1_scores[dt] for dt in doc_types}
    else:
        merged = merge_scores(w1_scores, w2_scores, w3_scores)

    # ── 4. Legacy STRUCTURAL_PROFILES fallback boost ──────────────────
    # Run the old first-5K scorer and blend it in for types whose
    # template scores are very low (acts as a safety net).
    # COMPENDIUM is excluded from blending because its boost logic is
    # deferred (compendium-v2); blending it causes false positives on
    # single-act Bare Acts and Codified Acts.
    legacy = _legacy_profile_scores(text)
    for dt in doc_types:
        if dt == "COMPENDIUM":
            continue   # TODO(compendium-v2): re-enable when tiered boost is ready
        if merged.get(dt, 0.0) < 0.10 and legacy.get(dt, 0.0) > 0.0:
            merged[dt] = max(merged[dt], legacy[dt] * 0.5)

    # ── 5. Penalties and boosts ───────────────────────────────────────
    penalties: List[str] = []
    boosts: List[str]    = []
    merged = apply_penalties_and_boosts(text, merged, penalties, boosts)

    # ── 6. Resolve winner ─────────────────────────────────────────────
    best_type   = max(merged, key=merged.__getitem__) if merged else "GENERIC"
    raw_conf    = merged.get(best_type, 0.0)
    confidence  = min(raw_conf, 1.0)

    if confidence == 0.0:
        best_type  = "GENERIC"
        confidence = 0.0

    # ── 7. Structural feature flags (backward compatible) ─────────────
    features = _extract_structural_features(text)
    features["score_breakdown"] = {
        dt: round(merged.get(dt, 0.0), 4) for dt in doc_types
    }

    if debug:
        features["debug_scores"] = {
            "window_1_title": {dt: round(w1_scores[dt], 4) for dt in doc_types},
            "window_2_dense": {dt: round(w2_scores[dt], 4) for dt in doc_types},
            "window_3_tail":  {dt: round(w3_scores[dt], 4) for dt in doc_types},
            "merged":         {dt: round(merged.get(dt, 0.0), 4) for dt in doc_types},
            "legacy_profile": {dt: round(legacy.get(dt, 0.0), 4) for dt in doc_types},
            "penalties":      penalties,
            "boosts":         boosts,
            "is_small_doc":   is_small,
            "w1_length":      len(w1_text),
            "w2_length":      len(w2_text),
            "w3_length":      len(w3_text),
        }
    else:
        features["debug_scores"] = {}

    return best_type, confidence, features


# ─────────────────────────────────────────────────────────────────────
# Window Extractors
# ─────────────────────────────────────────────────────────────────────

def extract_title_block(text: str) -> str:
    """
    Locate the earliest meaningful statutory title anchor and return a
    window of ±TITLE_WINDOW_SIZE//2 characters around it.

    Falls back to the first TITLE_WINDOW_SIZE characters if no anchor
    is found (covers plain bare-act PDFs that open directly with a title).
    """
    half = TITLE_WINDOW_SIZE // 2

    # Search within the first 20 % of the document (or 15 000 chars max)
    search_zone = text[: min(len(text), max(15_000, len(text) // 5))]
    m = _TITLE_ANCHOR_RE.search(search_zone)

    if m:
        anchor = m.start()
        start  = max(0, anchor - half)
        end    = min(len(text), anchor + half)
        return text[start:end]

    # Fallback: first TITLE_WINDOW_SIZE chars
    return text[:TITLE_WINDOW_SIZE]


def extract_densest_structure_block(text: str) -> str:
    """
    Slide a window of DENSE_WINDOW_SIZE chars across the full text with
    DENSE_WINDOW_STEP overlap (50 %).  Return the window with the highest
    cumulative count of structural-marker regex matches.

    Caps at DENSE_MAX_WINDOWS iterations to bound runtime on very long docs.
    """
    n       = len(text)
    step    = DENSE_WINDOW_STEP
    wsize   = DENSE_WINDOW_SIZE

    best_start = 0
    best_score = -1

    positions = range(0, n - wsize + 1, step)
    # Apply window cap
    positions = list(positions)[:DENSE_MAX_WINDOWS]
    # Always include a window near the end
    if n > wsize:
        positions.append(n - wsize)

    for start in positions:
        window = text[start: start + wsize]
        score  = _count_structural_hits(window)
        if score > best_score:
            best_score = score
            best_start = start

    return text[best_start: best_start + wsize]


def extract_last_structured_block(text: str) -> str:
    """
    Scan backward from the end of the document to find the last position
    that still contains a legal structure marker.  Return the final
    TAIL_WINDOW_SIZE characters ending at that position.

    Falls back to the last TAIL_WINDOW_SIZE chars if no marker is found.
    """
    # Search in the last 40 % of the document
    tail_zone_start = max(0, len(text) - max(30_000, len(text) // 3))
    tail_zone       = text[tail_zone_start:]

    # Find all marker positions (we want the last one)
    last_pos: Optional[int] = None
    for m in _TAIL_ANCHOR_RE.finditer(tail_zone):
        last_pos = m.start()

    if last_pos is not None:
        abs_pos = tail_zone_start + last_pos
        start   = max(0, abs_pos - TAIL_WINDOW_SIZE + 200)
        return text[start: abs_pos + 200]

    # Fallback: just the last TAIL_WINDOW_SIZE chars
    return text[-TAIL_WINDOW_SIZE:]


# ─────────────────────────────────────────────────────────────────────
# Template Scorer
# ─────────────────────────────────────────────────────────────────────

def score_template(window: str, template: Dict[str, List[str]]) -> float:
    """
    Score a text window against a single WINDOW_SCORE_TEMPLATES entry.

    Scoring formula
    ---------------
    • title_markers:     each hit contributes 1 / len(markers)  → max 1.0
    • structure_markers: each hit contributes 1 / len(markers)  → max 1.0
    • tail_markers:      each hit contributes 1 / len(markers)  → max 1.0
    • Combined average of the three sub-scores                  → [0, 1]
    • Anti-markers:      any hit zeroes the combined score.

    Empty window → 0.0
    """
    if not window:
        return 0.0

    # Anti-marker check (immediate veto)
    for ap in template.get("anti_markers", []):
        if re.search(ap, window, re.IGNORECASE):
            return 0.0

    def _subscore(patterns: List[str]) -> float:
        if not patterns:
            return 0.0
        hits = sum(
            1 for p in patterns if re.search(p, window, re.IGNORECASE | re.MULTILINE)
        )
        return hits / len(patterns)

    title_s     = _subscore(template.get("title_markers", []))
    structure_s = _subscore(template.get("structure_markers", []))
    tail_s      = _subscore(template.get("tail_markers", []))

    # Simple average of the three sub-scores
    raw = (title_s + structure_s + tail_s) / 3.0
    return min(raw, 1.0)


# ─────────────────────────────────────────────────────────────────────
# Score Merger
# ─────────────────────────────────────────────────────────────────────

def merge_scores(
    w1: Dict[str, float],
    w2: Dict[str, float],
    w3: Dict[str, float],
) -> Dict[str, float]:
    """
    Weighted merge of per-window scores.

        merged[type] = 0.35 * w1 + 0.45 * w2 + 0.20 * w3
    """
    doc_types = set(w1) | set(w2) | set(w3)
    return {
        dt: (
            _W_TITLE * w1.get(dt, 0.0)
            + _W_DENSE * w2.get(dt, 0.0)
            + _W_TAIL  * w3.get(dt, 0.0)
        )
        for dt in doc_types
    }


# ─────────────────────────────────────────────────────────────────────
# Penalties and Boosts
# ─────────────────────────────────────────────────────────────────────

def apply_penalties_and_boosts(
    text: str,
    scores: Dict[str, float],
    penalties: List[str],
    boosts: List[str],
) -> Dict[str, float]:
    """
    Apply document-level anti-pattern penalties and structural boost rules
    to the merged scores dict (mutates a copy, returns it).

    Penalty rules
    -------------
    1. Gazette tokens present but NO numbered sections found → penalise GAZETTE_ACT
    2. Bare Act title present but no CHAPTER / PART hierarchy → penalise BARE_ACT
    3. SCHEDULE_RULES title but no Rule / Schedule / ORDER markers → penalise SCHEDULE_RULES

    Boost rules
    -----------
    A. Gazette header + "Be it enacted by" → boost GAZETTE_ACT
    B. [Ins./Subs./Omitted] annotations present + CHAPTER hierarchy → boost CODIFIED_ACT
    C. "In exercise of the powers conferred" → boost SCHEDULE_RULES
    D. Dense SCHEDULE / FORM / ANNEXURE tail → boost SCHEDULE_RULES

    COMPENDIUM boosts are DEFERRED (compendium-v2).
    """
    s = dict(scores)  # work on a copy

    has_gazette_token  = bool(re.search(r"THE\s+GAZETTE\s+OF\s+INDIA", text, re.IGNORECASE))
    has_sections       = bool(re.search(r"(?m)^\s*\d{1,3}[A-Z]?\.\s+[A-Z]", text))
    has_chapters       = bool(re.search(r"(?m)^CHAPTER\s+[IVXLCDM]+", text, re.IGNORECASE))
    has_parts          = bool(re.search(r"(?m)^PART\s+[IVXLCDM]+", text, re.IGNORECASE))
    has_annotations    = bool(re.search(r"\[(?:Ins\.|Subs\.|Omitted|Rep\.)\s+by", text))
    has_be_it_enacted  = bool(re.search(r"Be\s+it\s+enacted\s+by", text, re.IGNORECASE))
    has_in_exercise    = bool(re.search(
        r"In\s+exercise\s+of\s+the\s+powers?\s+conferred", text, re.IGNORECASE
    ))
    has_rule_markers   = bool(re.search(r"(?m)^Rule\s+\d+", text, re.IGNORECASE))
    has_order_markers  = bool(re.search(r"(?m)^ORDER\s+[IVXLCDM]+", text))
    has_schedule_tail  = bool(re.search(
        r"(?:SCHEDULE|FORM\s+[A-Z\d]|ANNEXURE|APPENDIX)",
        text[-8000:], re.IGNORECASE | re.DOTALL,
    ))

    # ── COMPENDIUM suppression (deferred logic) ───────────────────────
    # Count distinct ACT/CODE titles across the full text.
    # With fewer than 2, COMPENDIUM should score near zero.
    # TODO(compendium-v2): replace this with tiered boost logic.
    act_titles = re.findall(
        r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+(?:ACT|CODE),?\s+\d{4}",
        text, re.IGNORECASE,
    )
    distinct_acts = len(set(t.strip().upper() for t in act_titles))
    if distinct_acts < 2:
        s["COMPENDIUM"] = max(0.0, s.get("COMPENDIUM", 0.0) - 0.80)
        penalties.append(
            f"COMPENDIUM: only {distinct_acts} distinct act title(s) found "
            f"(need ≥2 for compendium detection)"
        )

    # ── Penalty 1: Gazette tokens but no sections ─────────────────────
    if has_gazette_token and not has_sections:
        s["GAZETTE_ACT"] = max(0.0, s.get("GAZETTE_ACT", 0.0) - _PENALTY)
        penalties.append("GAZETTE_ACT: gazette token but no numbered sections")

    # ── Penalty 2: Bare Act title but no hierarchy ────────────────────
    bare_title = bool(re.search(
        r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+ACT,?\s+\d{4}", text[:8000], re.IGNORECASE
    ))
    if bare_title and not has_chapters and not has_parts:
        s["BARE_ACT"] = max(0.0, s.get("BARE_ACT", 0.0) - _PENALTY)
        penalties.append("BARE_ACT: act title found but no CHAPTER/PART hierarchy")

    # ── Penalty 3: Rules doc label but no rule/schedule markers ──────
    rules_title = bool(re.search(
        r"(?m)^THE\s+[A-Z][A-Z\s,'\-]+RULES,?\s+\d{4}", text[:8000], re.IGNORECASE
    ))
    if rules_title and not has_rule_markers and not has_order_markers:
        s["SCHEDULE_RULES"] = max(0.0, s.get("SCHEDULE_RULES", 0.0) - _PENALTY)
        penalties.append("SCHEDULE_RULES: rules title but no rule/order markers in body")

    # ── Boost A: Gazette header + enactment phrase ────────────────────
    if has_gazette_token and has_be_it_enacted and has_sections:
        s["GAZETTE_ACT"] = min(1.0, s.get("GAZETTE_ACT", 0.0) + _BOOST_LARGE)
        boosts.append("GAZETTE_ACT: gazette header + 'Be it enacted by' + sections")

    # ── Boost B: Amendment annotations + chapter hierarchy ───────────
    if has_annotations and has_chapters:
        s["CODIFIED_ACT"] = min(1.0, s.get("CODIFIED_ACT", 0.0) + _BOOST_LARGE)
        boosts.append("CODIFIED_ACT: [Ins./Subs.] annotations + CHAPTER hierarchy")
    elif has_annotations:
        s["CODIFIED_ACT"] = min(1.0, s.get("CODIFIED_ACT", 0.0) + _BOOST_SMALL)
        boosts.append("CODIFIED_ACT: [Ins./Subs.] annotations present")

    # ── Boost C: "In exercise of powers conferred" ────────────────────
    if has_in_exercise:
        s["SCHEDULE_RULES"] = min(1.0, s.get("SCHEDULE_RULES", 0.0) + _BOOST_MEDIUM)
        boosts.append("SCHEDULE_RULES: 'In exercise of powers conferred' phrase")

    # ── Boost D: Schedule/Form heavy tail ────────────────────────────
    if has_schedule_tail and (has_rule_markers or has_order_markers):
        s["SCHEDULE_RULES"] = min(1.0, s.get("SCHEDULE_RULES", 0.0) + _BOOST_MEDIUM)
        boosts.append("SCHEDULE_RULES: schedule/form-heavy tail + rule/order markers")

    # ── Boost E: CHAPTER + SECTION hierarchy → BARE_ACT / CODIFIED ───
    if has_chapters and has_sections and not has_annotations and not has_gazette_token:
        s["BARE_ACT"] = min(1.0, s.get("BARE_ACT", 0.0) + _BOOST_SMALL)
        boosts.append("BARE_ACT: clean chapter+section hierarchy, no annotations/gazette")

    # TODO(compendium-v2): add COMPENDIUM tiered boost logic here

    return s


# ─────────────────────────────────────────────────────────────────────
# Legacy profile scorer (safety net / blending)
# ─────────────────────────────────────────────────────────────────────

def _legacy_profile_scores(text: str) -> Dict[str, float]:
    """
    Run the original first-5000-chars STRUCTURAL_PROFILES scorer.
    Used as a safety-net blending signal for types with very low
    window template scores.
    """
    sample = text[:5000]
    scores: Dict[str, float] = {}

    for doc_type, profile in STRUCTURAL_PROFILES.items():
        patterns     = profile.get("patterns", [])
        anti_patterns = profile.get("anti_patterns", [])

        if any(re.search(ap, sample, re.IGNORECASE) for ap in anti_patterns):
            scores[doc_type] = 0.0
            continue

        if not patterns:
            scores[doc_type] = 0.0
            continue

        hits             = 0
        total_occurrences = 0
        for p in patterns:
            matches = re.findall(p, sample, re.IGNORECASE)
            if matches:
                hits += 1
                total_occurrences += len(matches)

        min_occ = profile.get("min_occurrences", 1)
        if total_occurrences < min_occ:
            scores[doc_type] = 0.0
            continue

        base_score = hits / len(patterns)

        # Global patterns (e.g. CODIFIED_ACT full-text bonus)
        gp     = profile.get("global_patterns", [])
        gbonus = profile.get("global_bonus", 0.0)
        if gp and gbonus > 0:
            if any(re.search(g, text, re.IGNORECASE) for g in gp):
                base_score += gbonus

        scores[doc_type] = min(base_score, 1.0)

    return scores


# ─────────────────────────────────────────────────────────────────────
# Structural density helper
# ─────────────────────────────────────────────────────────────────────

def _count_structural_hits(window: str) -> int:
    """Count total regex hits for dense structural markers in a window."""
    count = 0
    for p in _DENSE_MARKERS:
        count += len(re.findall(p, window, re.IGNORECASE | re.MULTILINE))
    return count


# ─────────────────────────────────────────────────────────────────────
# Structural feature flags (unchanged legacy API)
# ─────────────────────────────────────────────────────────────────────

def _extract_structural_features(text: str) -> Dict[str, Any]:
    """Detect high-level structural markers present in the document."""
    return {
        "has_chapters": bool(
            re.search(r"CHAPTER\s+[IVXLCDM]+", text, re.IGNORECASE)
        ),
        "has_parts": bool(
            re.search(r"PART\s+[IVXLCDM]+", text, re.IGNORECASE)
        ),
        "has_orders": bool(
            re.search(r"\bORDER\s+[IVXLCDM]+\b", text)
        ),
        "has_schedules": bool(
            re.search(
                r"(THE FIRST SCHEDULE|THE SECOND SCHEDULE|"
                r"SCHEDULE [IVXLCDM]|THE SCHEDULE)",
                text,
                re.IGNORECASE,
            )
        ),
        "approx_section_count": len(
            re.findall(r"\n\s*\d{1,3}[A-Z]?\.\s", text)
        ),
        "has_definitions": bool(
            re.search(r'\bmeans\b.{0,200}?\binclude', text, re.IGNORECASE | re.DOTALL)
        ),
        "approx_char_count": len(text),
    }
