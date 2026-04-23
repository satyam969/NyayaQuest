"""
tests/stage1/test_detector.py — Unit tests for the semantic multi-window detector.

Run with:
    cd d:\\NyayaQuest
    python -m pytest tests/stage1/test_detector.py -v

Each test builds a minimal synthetic text fixture that mimics the structure of a
real Indian statutory PDF, then asserts:
  • doc_type  == expected type
  • confidence >= minimum acceptable threshold
  • features  contains all legacy keys
  • debug_scores is populated when debug=True
"""

import sys
import os
import pytest

# Ensure the src package is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from auto_ingest.stage1_detection.detector import (
    detect_document_type,
    extract_title_block,
    extract_densest_structure_block,
    extract_last_structured_block,
    score_template,
    merge_scores,
    apply_penalties_and_boosts,
)
from auto_ingest.utils.patterns import WINDOW_SCORE_TEMPLATES


# ─────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────

def _make_gazette_act() -> str:
    """Simulates a Gazette of India extraordinary issue with a new act."""
    header = (
        "THE GAZETTE OF INDIA EXTRAORDINARY\n"
        "PART II—Section 1\n"
        "MINISTRY OF LAW AND JUSTICE\n"
        "No. 28 of 2022\n"
        "THE DIGITAL PERSONAL DATA PROTECTION ACT, 2022\n"
        "An Act to provide for the processing of digital personal data.\n"
        "Be it enacted by Parliament in the Seventy-third Year of the Republic.\n\n"
    )
    sections = ""
    for i in range(1, 40):
        sections += (
            f"\n{i}. Short title of section {i}.— "
            f"This section shall mean the following provision shall apply. "
            f"Notwithstanding anything contained herein, provided that the authority "
            f"may issue directions as deemed fit.\n"
            f"   (1) Sub-clause one of section {i}.\n"
            f"   (2) Sub-clause two of section {i}.\n"
        )
    tail = (
        "\nTHE SCHEDULE\n"
        "Statement of Objects and Reasons\n"
        "This Act is enacted to protect the digital personal data of citizens.\n"
    )
    return header + sections + tail


def _make_bare_act() -> str:
    """Simulates a clean Bare Act with no Gazette headers or annotations."""
    title = (
        "THE RIGHT TO INFORMATION ACT, 2005\n"
        "No. 22 of 2005\n"
        "An Act to provide for setting out the practical regime of right to information.\n"
        "Be it enacted by Parliament in the Fifty-sixth Year of the Republic.\n\n"
        "CHAPTER I\nPRELIMINARY\n\n"
    )
    sections = ""
    for i in range(1, 35):
        sections += (
            f"\n{i}. {'Short title' if i == 1 else f'Section {i} heading'}.  "
            f"In this Act, unless the context otherwise requires, "
            f"the expression shall mean the appropriate authority. "
            f"Notwithstanding anything contained in any other law.\n"
            f"   ({i})(a) clause a.\n"
            f"   ({i})(b) clause b.\n"
        )
    tail = "\nSCHEDULE I\nList of intelligence and security organisations.\n"
    return title + sections + tail


def _make_codified_act() -> str:
    """Simulates a codified edition with [Ins./Subs.] amendment annotations."""
    title = (
        "THE CODE OF CIVIL PROCEDURE, 1908\n\n"
        "CHAPTER I\nPRELIMINARY\n\n"
    )
    sections = ""
    for i in range(1, 45):
        sections += (
            f"\n{i}. [Subs. by Act {i + 10} of 19{50 + i}, s. 3, for the original.] "
            f"This section shall mean the following. w.e.f. 1-1-19{60 + i}. "
            f"[Ins. by Act {i + 5} of 20{i:02d}, s. 2.] "
            f"A.O. 1950 — provided that the court may.\n"
            f"   ({i})(1) Sub-section one.\n"
        )
        if i % 8 == 0:
            sections += f"\nCHAPTER {['II','III','IV','V','VI','VII'][i // 8 - 1]}\nHEADING\n"
    tail = (
        "\nTHE FIRST SCHEDULE\n"
        "ORDER I\nParties to Suits\n\n"
        "Rule 1. Who may be joined as plaintiffs.\n"
        "Rule 2. Power of court to order separate trial.\n"
        "\nTHE SECOND SCHEDULE\nFORMS\n"
    )
    return title + sections + tail


def _make_schedule_rules() -> str:
    """Simulates a Rules + Schedules document (subordinate legislation)."""
    title = (
        "THE MOTOR VEHICLES RULES, 1989\n"
        "In exercise of the powers conferred by section 211 of the Motor Vehicles Act, 1988,\n"
        "the Central Government hereby makes the following rules:—\n\n"
    )
    rules = ""
    for i in range(1, 50):
        rules += (
            f"\nRule {i}. {'Short title and commencement' if i == 1 else f'Provision {i}'}.—"
            f"   (a) clause a of rule {i}.\n"
            f"   (b) clause b of rule {i}.\n"
        )
    tail = (
        "\nSCHEDULE I\nFEES\n\n"
        "SCHEDULE II\nFORMS\n\n"
        "FORM A\nApplication for driving licence.\n\n"
        "ANNEXURE\nList of approved testing equipment.\n"
        "APPENDIX\nGlossary of terms.\n"
    )
    return title + rules + tail


def _make_compendium_smoke() -> str:
    """
    Minimal compendium fixture — just enough to avoid crashes.
    Full COMPENDIUM boost logic is deferred to compendium-v2.
    """
    return (
        "THE WAGES CODE, 2019\n\n"
        "CHAPTER I\nPRELIMINARY\n\n"
        "1.(1) This Act may be called the Code on Wages, 2019.\n"
        "2. Definitions.\n\n"
        "THE INDUSTRIAL RELATIONS CODE, 2020\n\n"
        "CHAPTER I\nPRELIMINARY\n\n"
        "1.(1) This Code may be called the Industrial Relations Code, 2020.\n"
    )


def _make_noisy_ocr_act() -> str:
    """
    Simulates a scanned PDF where the first ~6000 chars are OCR garbage /
    Gazette boilerplate, and the real act structure begins mid-document.
    The old first-5000-char detector would fail on this; the new one should succeed.
    """
    ocr_junk = (
        "vlk/kkjk CG-DL-E-2022-001234 jkti=k Hkkjr lk/kkjk\n"
        "4523 THE GAZETTE OF INDIA EXTRAORDINARY PART II Section 1\n"
        "MINISTRY OF LAW AND JUSTICE (Legislative Department)\n"
        "New Delhi the 12th April 2022 CHAITRA 22 1944 (SAKA)\n"
    ) * 20   # ~2 000 chars of noise repeated to push past 5 000

    real_content = (
        "\nTHE ENVIRONMENT PROTECTION ACT, 1986\n"
        "No. 29 of 1986\n"
        "An Act to provide for the protection and improvement of environment.\n"
        "Be it enacted by Parliament in the Thirty-seventh Year of the Republic.\n\n"
        "CHAPTER I\nPRELIMINARY\n\n"
    )
    sections = ""
    for i in range(1, 30):
        sections += (
            f"\n{i}. Section heading {i}.— "
            f"This section shall mean the following. "
            f"Notwithstanding anything, provided that.\n"
            f"   ({i}) sub-clause.\n"
        )
    return ocr_junk + real_content + sections


# ─────────────────────────────────────────────────────────────────────
# Legacy feature keys that must always be present
# ─────────────────────────────────────────────────────────────────────

LEGACY_FEATURE_KEYS = {
    "has_chapters",
    "has_parts",
    "has_orders",
    "has_schedules",
    "approx_section_count",
    "has_definitions",
    "approx_char_count",
    "score_breakdown",
    "debug_scores",
}


def _assert_features(features: dict) -> None:
    """All legacy keys must be present in features."""
    for key in LEGACY_FEATURE_KEYS:
        assert key in features, f"Missing feature key: '{key}'"


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestGazetteAct:
    text = _make_gazette_act()

    def test_doc_type(self):
        doc_type, conf, features = detect_document_type(self.text)
        assert doc_type == "GAZETTE_ACT", (
            f"Expected GAZETTE_ACT, got {doc_type} (conf={conf:.3f})\n"
            f"scores={features['score_breakdown']}"
        )

    def test_confidence(self):
        _, conf, _ = detect_document_type(self.text)
        assert conf >= 0.45, f"Confidence too low: {conf:.3f}"

    def test_legacy_features(self):
        _, _, features = detect_document_type(self.text)
        _assert_features(features)

    def test_debug_mode(self):
        _, _, features = detect_document_type(self.text, debug=True)
        ds = features["debug_scores"]
        assert "window_1_title" in ds
        assert "window_2_dense" in ds
        assert "window_3_tail"  in ds
        assert "penalties" in ds
        assert "boosts"    in ds


class TestBareAct:
    text = _make_bare_act()

    def test_doc_type(self):
        doc_type, conf, features = detect_document_type(self.text)
        assert doc_type == "BARE_ACT", (
            f"Expected BARE_ACT, got {doc_type} (conf={conf:.3f})\n"
            f"scores={features['score_breakdown']}"
        )

    def test_confidence(self):
        _, conf, _ = detect_document_type(self.text)
        assert conf >= 0.40, f"Confidence too low: {conf:.3f}"

    def test_legacy_features(self):
        _, _, features = detect_document_type(self.text)
        _assert_features(features)


class TestCodifiedAct:
    text = _make_codified_act()

    def test_doc_type(self):
        doc_type, conf, features = detect_document_type(self.text)
        assert doc_type == "CODIFIED_ACT", (
            f"Expected CODIFIED_ACT, got {doc_type} (conf={conf:.3f})\n"
            f"scores={features['score_breakdown']}"
        )

    def test_confidence(self):
        _, conf, _ = detect_document_type(self.text)
        assert conf >= 0.45, f"Confidence too low: {conf:.3f}"

    def test_legacy_features(self):
        _, _, features = detect_document_type(self.text)
        _assert_features(features)

    def test_annotations_detected(self):
        _, _, features = detect_document_type(self.text, debug=True)
        boosts = features["debug_scores"]["boosts"]
        assert any("CODIFIED_ACT" in b for b in boosts), (
            f"Expected CODIFIED_ACT boost, got boosts={boosts}"
        )


class TestScheduleRules:
    text = _make_schedule_rules()

    def test_doc_type(self):
        doc_type, conf, features = detect_document_type(self.text)
        assert doc_type == "SCHEDULE_RULES", (
            f"Expected SCHEDULE_RULES, got {doc_type} (conf={conf:.3f})\n"
            f"scores={features['score_breakdown']}"
        )

    def test_confidence(self):
        _, conf, _ = detect_document_type(self.text)
        assert conf >= 0.40, f"Confidence too low: {conf:.3f}"

    def test_legacy_features(self):
        _, _, features = detect_document_type(self.text)
        _assert_features(features)

    def test_in_exercise_boost(self):
        _, _, features = detect_document_type(self.text, debug=True)
        boosts = features["debug_scores"]["boosts"]
        assert any("SCHEDULE_RULES" in b for b in boosts), (
            f"Expected SCHEDULE_RULES boost, got boosts={boosts}"
        )


class TestCompendiumSmoke:
    """Smoke-only: COMPENDIUM boost logic is deferred. Just assert no crash."""
    text = _make_compendium_smoke()

    def test_no_crash(self):
        doc_type, conf, features = detect_document_type(self.text, debug=True)
        # Must return a valid type string, not throw
        assert isinstance(doc_type, str)
        assert 0.0 <= conf <= 1.0
        _assert_features(features)


class TestNoisyOCR:
    text = _make_noisy_ocr_act()

    def test_doc_type(self):
        """
        The key regression test: noisy first 5K must not fool the classifier.
        The new 3-window approach should locate real structure mid-document.
        """
        doc_type, conf, features = detect_document_type(self.text)
        # Should detect as Gazette (Gazette header in noise) or Bare Act — NOT GENERIC
        assert doc_type in ("GAZETTE_ACT", "BARE_ACT"), (
            f"Expected GAZETTE_ACT or BARE_ACT for noisy OCR doc, "
            f"got {doc_type} (conf={conf:.3f})\n"
            f"scores={features['score_breakdown']}"
        )

    def test_confidence_above_zero(self):
        _, conf, _ = detect_document_type(self.text)
        assert conf > 0.0, "Noisy OCR doc should not score zero confidence"

    def test_legacy_features(self):
        _, _, features = detect_document_type(self.text)
        _assert_features(features)


# ─────────────────────────────────────────────────────────────────────
# Unit tests for individual helper functions
# ─────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_score_template_antis_veto(self):
        """Anti-marker should zero the score regardless of positive hits."""
        template = WINDOW_SCORE_TEMPLATES["BARE_ACT"]
        text_with_gazette = (
            "THE GAZETTE OF INDIA EXTRAORDINARY\n"
            "1. Short title.\n2. Definitions.\n3. Application.\n"
        )
        score = score_template(text_with_gazette, template)
        assert score == 0.0, f"Anti-marker veto failed, got score={score}"

    def test_score_template_empty_window(self):
        for name, tmpl in WINDOW_SCORE_TEMPLATES.items():
            assert score_template("", tmpl) == 0.0, f"Empty window should score 0 for {name}"

    def test_merge_scores_weights(self):
        w1 = {"GAZETTE_ACT": 1.0, "BARE_ACT": 0.0}
        w2 = {"GAZETTE_ACT": 0.0, "BARE_ACT": 1.0}
        w3 = {"GAZETTE_ACT": 0.0, "BARE_ACT": 0.0}
        merged = merge_scores(w1, w2, w3)
        # GAZETTE_ACT: 0.35*1 + 0.45*0 + 0.20*0 = 0.35
        assert abs(merged["GAZETTE_ACT"] - 0.35) < 1e-6
        # BARE_ACT:    0.35*0 + 0.45*1 + 0.20*0 = 0.45
        assert abs(merged["BARE_ACT"] - 0.45) < 1e-6

    def test_extract_title_block_finds_act(self):
        text = "Some noise\n" * 100 + "THE ENVIRONMENT ACT, 1986\n" + "body text\n" * 50
        block = extract_title_block(text)
        assert "ENVIRONMENT ACT" in block

    def test_extract_last_structured_block_nonempty(self):
        text = "preamble\n" * 200 + "SCHEDULE\nList of things.\n" * 30
        block = extract_last_structured_block(text)
        assert len(block) > 0
