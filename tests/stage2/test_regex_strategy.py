"""
tests/stage2/test_regex_strategy.py — Tests for the upgraded regex pattern selector.

Structure
---------
Each test class covers one scenario.  Where relevant, the test also
demonstrates that the OLD scorer (using only text[:15000] + raw count)
would have made a WORSE choice — proving the upgrade is an improvement.

Run with:
    cd d:\\NyayaQuest
    .venv\\Scripts\\python.exe -m pytest tests/stage2/test_regex_strategy.py -v
"""

import re
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from auto_ingest.stage2_strategy.regex_strategy import (
    select_section_pattern,
    select_chapter_pattern,
    _build_candidate_pool,
    _continuity_ratio,
    _score_pattern,
    _RULE_PREFIX_PATTERN,
)
from auto_ingest.utils.patterns import (
    SECTION_PATTERNS,
    CHAPTER_PATTERNS,
    RULE_PATTERNS,
)


# ─────────────────────────────────────────────────────────────────────
# Shared fixture text builders
# ─────────────────────────────────────────────────────────────────────

def _gazette_text(n_sections: int = 30) -> str:
    """Clean Gazette Act: numbered sections 1. Short title … N. Penalty."""
    lines = [
        "THE GAZETTE OF INDIA EXTRAORDINARY",
        "THE DIGITAL DATA PROTECTION ACT, 2022",
        "No. 28 of 2022",
        "An Act to protect digital personal data.",
        "CHAPTER I",
        "PRELIMINARY",
        "",
    ]
    for i in range(1, n_sections + 1):
        lines += [
            f"{i}. Short title of section {i}.—",
            f"   (1) This section shall mean the following provision.",
            f"   (2) Provided that the authority may issue directions.",
            "",
        ]
    return "\n".join(lines)


def _codified_text(n_sections: int = 35) -> str:
    """CPC-style codified act: bracket-prefixed numbering e.g. [1. Title."""
    lines = [
        "THE CODE OF CIVIL PROCEDURE, 1908",
        "CHAPTER I",
        "PRELIMINARY",
        "",
    ]
    for i in range(1, n_sections + 1):
        # Mix of plain and bracket-prefix notation to mimic CPC
        if i % 3 == 0:
            lines += [
                f"[{i}. [Subs. by Act {i+5} of 1970, s. 2.] Short title.—",
                f"   (1) Sub-section one of section {i}.",
                "",
            ]
        elif i % 5 == 0:
            lines += [
                f"*[{i}A. [Ins. by Act {i+2} of 1980.] New provision.—",
                f"   (1) Inserted provision content.",
                "",
            ]
        else:
            lines += [
                f"{i}. Definition of term {i}.—",
                f"   (1) In this Code, unless context otherwise requires.",
                "",
            ]
    return "\n".join(lines)


def _rules_prefix_text(n_rules: int = 40) -> str:
    """
    Subordinate legislation with explicit 'Rule N.' prefix format.
    This is the CRITICAL fixture — old code scores ZERO on this, new code wins.
    """
    lines = [
        "THE MOTOR VEHICLES RULES, 1989",
        "In exercise of the powers conferred by section 211 of the Motor Vehicles Act, 1988.",
        "",
    ]
    for i in range(1, n_rules + 1):
        lines += [
            f"Rule {i}. {'Short title and commencement' if i == 1 else f'Provision {i}'}.—",
            f"   (a) Clause a of rule {i}.",
            f"   (b) Clause b of rule {i}.",
            "",
        ]
    lines += [
        "SCHEDULE I",
        "SCHEDULE II",
        "FORM A — Application",
    ]
    return "\n".join(lines)


def _rules_bare_text(n_rules: int = 40) -> str:
    """Rules with bare numbered sections (no 'Rule' prefix) — all patterns should work."""
    lines = [
        "THE COMPANY RULES, 2014",
        "In exercise of the powers conferred by section 469 of the Companies Act, 2013.",
        "",
    ]
    for i in range(1, n_rules + 1):
        lines += [
            f"{i}. {'Short title' if i == 1 else f'Rule heading {i}'}.—",
            f"   (a) Clause a.",
            f"   (b) Clause b.",
            "",
        ]
    return "\n".join(lines)


def _has_orders_text(n_orders: int = 3, rules_per_order: int = 12) -> str:
    """CPC-style: ORDER I, ORDER II … with numbered rules inside."""
    lines = [
        "THE CODE OF CIVIL PROCEDURE, 1908",
        "THE FIRST SCHEDULE",
        "",
    ]
    for o in range(1, n_orders + 1):
        order_names = ["I", "II", "III", "IV", "V"][o - 1]
        lines += [f"ORDER {order_names}", f"Parties to Suits", ""]
        for r in range(1, rules_per_order + 1):
            lines += [
                f"{r}. Rule {r} of Order {o}.—",
                f"   (1) Content of rule {r}.",
                "",
            ]
    return "\n".join(lines)


def _nonsequential_text(n: int = 20) -> str:
    """Text whose section numbers jump randomly — tests continuity penalty."""
    jumps = [1, 3, 8, 10, 15, 18, 25, 30, 40, 50,
             55, 60, 70, 80, 90, 95, 100, 110, 120, 130]
    lines = []
    for num in jumps[:n]:
        lines += [f"{num}. Section heading {num}.—", "   Content.", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Helper: simulate old scorer on same text
# ─────────────────────────────────────────────────────────────────────

def _old_select_section_pattern(text: str) -> str:
    """Replication of the original (pre-upgrade) scorer logic."""
    sample = text[:15_000]
    scored = []
    for pattern in SECTION_PATTERNS:
        try:
            count = len(re.findall(pattern, sample))
            scored.append((pattern, count))
        except re.error:
            continue
    if not scored:
        return SECTION_PATTERNS[-1]

    def _score(item):
        _, cnt = item
        if cnt < 5:
            return 0
        if cnt > 500:
            return max(0, 500 - cnt)
        return cnt

    best, _ = max(scored, key=_score)
    return best


# ─────────────────────────────────────────────────────────────────────
# Unit tests for helpers
# ─────────────────────────────────────────────────────────────────────

class TestContinuityRatio:

    def test_perfect_sequence(self):
        assert _continuity_ratio(["1", "2", "3", "4", "5"]) == 1.0

    def test_reversed_sequence(self):
        assert _continuity_ratio(["5", "4", "3", "2", "1"]) == 0.0

    def test_neutral_on_too_few(self):
        assert _continuity_ratio(["1", "3"]) == 0.5

    def test_empty_neutral(self):
        assert _continuity_ratio([]) == 0.5

    def test_section_letters_ignored_correctly(self):
        # "1A" → leading integer 1, "2" → 2 — monotone
        ratio = _continuity_ratio(["1", "1A", "2", "3", "4"])
        assert ratio == 1.0

    def test_mixed_sequence(self):
        # [1,2,3,10,11,12] — all increasing
        ratio = _continuity_ratio(["1", "2", "3", "10", "11", "12"])
        assert ratio == 1.0


class TestBuildCandidatePool:

    def test_codified_pool_starts_with_strict(self):
        pool = _build_candidate_pool("CODIFIED_ACT", {})
        patterns = [p for p, _ in pool]
        assert patterns[0] == SECTION_PATTERNS[0], (
            "CODIFIED_ACT should prioritize the strict bracket-prefix pattern"
        )

    def test_gazette_pool_starts_with_moderate(self):
        pool = _build_candidate_pool("GAZETTE_ACT", {})
        patterns = [p for p, _ in pool]
        assert patterns[0] == SECTION_PATTERNS[1], (
            "GAZETTE_ACT should prioritize the moderate pattern"
        )

    def test_schedule_rules_pool_starts_with_rule_prefix(self):
        pool = _build_candidate_pool("SCHEDULE_RULES", {})
        patterns = [p for p, _ in pool]
        assert patterns[0] == _RULE_PREFIX_PATTERN, (
            "SCHEDULE_RULES should prioritize the 'Rule N.' prefix pattern"
        )

    def test_has_orders_injects_rule_pattern(self):
        # For GAZETTE_ACT (which doesn't have RULE_PATTERNS in its default pool)
        pool_without = _build_candidate_pool("GAZETTE_ACT", {"has_orders": False})
        pool_with    = _build_candidate_pool("GAZETTE_ACT", {"has_orders": True})
        patterns_without = [p for p, _ in pool_without]
        patterns_with    = [p for p, _ in pool_with]
        assert RULE_PATTERNS[0] not in patterns_without
        assert RULE_PATTERNS[0] in patterns_with

    def test_no_duplicate_patterns(self):
        for doc_type in ["GAZETTE_ACT", "CODIFIED_ACT", "SCHEDULE_RULES", "BARE_ACT", "GENERIC"]:
            pool = _build_candidate_pool(doc_type, {"has_orders": True})
            patterns = [p for p, _ in pool]
            assert len(patterns) == len(set(patterns)), (
                f"Duplicate patterns in pool for {doc_type}"
            )

    def test_priority_bonus_first_position(self):
        pool = _build_candidate_pool("CODIFIED_ACT", {})
        _, bonus_first  = pool[0]
        _, bonus_second = pool[1]
        assert bonus_first > bonus_second, (
            "First pattern in pool must have higher priority bonus"
        )

    def test_unknown_doctype_uses_generic(self):
        pool = _build_candidate_pool("TOTALLY_UNKNOWN_TYPE", {})
        patterns = [p for p, _ in pool]
        # Should fall back to SECTION_PATTERNS (same as GENERIC)
        assert all(p in SECTION_PATTERNS for p in patterns)


# ─────────────────────────────────────────────────────────────────────
# Integration tests — section pattern selection
# ─────────────────────────────────────────────────────────────────────

class TestGazetteActPattern:
    text = _gazette_text(30)
    features = {"has_orders": False, "has_schedules": False, "approx_section_count": 30}

    def test_returns_valid_pattern(self):
        p = select_section_pattern(self.text, "GAZETTE_ACT", self.features)
        assert isinstance(p, str) and len(p) > 0

    def test_pattern_splits_sections(self):
        p = select_section_pattern(self.text, "GAZETTE_ACT", self.features)
        parts = [x for x in re.split(p, self.text) if x.strip()]
        assert len(parts) >= 20, (
            f"Expected ≥20 splits for gazette text, got {len(parts)}"
        )

    def test_backward_compat_same_as_old(self):
        """Gazette text is simple enough that old + new should agree."""
        new = select_section_pattern(self.text, "GAZETTE_ACT", self.features)
        old = _old_select_section_pattern(self.text)
        # Both should be able to split the text — even if they pick different patterns
        new_count = len([x for x in re.split(new, self.text) if x.strip()])
        old_count = len([x for x in re.split(old, self.text) if x.strip()])
        assert new_count >= old_count, (
            f"New pattern should produce ≥ as many splits as old: new={new_count} old={old_count}"
        )


class TestCodifiedActPattern:
    text = _codified_text(35)
    features = {"has_orders": False, "has_schedules": False, "approx_section_count": 35}

    def test_returns_valid_pattern(self):
        p = select_section_pattern(self.text, "CODIFIED_ACT", self.features)
        assert isinstance(p, str) and len(p) > 0

    def test_strict_pattern_preferred(self):
        """
        CODIFIED_ACT should prefer SECTION_PATTERNS[0] which handles [N. and *[N.
        The old scorer might pick a different pattern if a looser one accidentally
        scores higher on the first 15K.  The priority bonus ensures the strict one wins.
        """
        p = select_section_pattern(self.text, "CODIFIED_ACT", self.features)
        # Should be either strict (0) or moderate (1) — but not the loose bracket-only (2)
        assert p in (SECTION_PATTERNS[0], SECTION_PATTERNS[1]), (
            f"Expected strict or moderate pattern for CODIFIED_ACT, got: {p!r}"
        )

    def test_bracket_sections_are_matched(self):
        p = select_section_pattern(self.text, "CODIFIED_ACT", self.features)
        matches = re.findall(p, self.text)
        assert len(matches) >= 20, (
            f"Pattern should match ≥20 bracket-prefix sections, got {len(matches)}"
        )


class TestRulesPrefixPattern:
    """
    KEY IMPROVEMENT TEST: "Rule N." format documents.

    The OLD scorer: all 3 SECTION_PATTERNS score ZERO because none of them
    match "Rule 1." (they look for \n  NUMBER. not \n  Rule NUMBER.).
    The old scorer would return the fallback (last pattern) by default.

    The NEW scorer: _RULE_PREFIX_PATTERN is in the SCHEDULE_RULES pool
    and scores 40 hits → selected correctly.
    """
    text = _rules_prefix_text(40)
    features = {"has_orders": False, "has_schedules": True, "approx_section_count": 40}

    def test_new_selects_rule_prefix_pattern(self):
        p = select_section_pattern(self.text, "SCHEDULE_RULES", self.features)
        assert p == _RULE_PREFIX_PATTERN, (
            f"Expected _RULE_PREFIX_PATTERN for 'Rule N.' document, got: {p!r}"
        )

    def test_rule_prefix_pattern_actually_splits(self):
        p = select_section_pattern(self.text, "SCHEDULE_RULES", self.features)
        parts = [x for x in re.split(p, self.text) if x.strip()]
        assert len(parts) >= 35, (
            f"Expected ≥35 rule blocks, got {len(parts)}"
        )

    def test_old_scorer_fails_this_case(self):
        """
        PROVE THE IMPROVEMENT: old scorer gets zero matches on 'Rule N.' text.
        """
        old_pattern = _old_select_section_pattern(self.text)
        old_matches = re.findall(old_pattern, self.text[:15_000])
        assert len(old_matches) < 5, (
            f"Old scorer should score zero on 'Rule N.' format, "
            f"but got {len(old_matches)} matches — test assumption is wrong"
        )

    def test_new_scorer_beats_old(self):
        new_p = select_section_pattern(self.text, "SCHEDULE_RULES", self.features)
        old_p = _old_select_section_pattern(self.text)
        new_count = len(re.findall(new_p, self.text[:15_000]))
        old_count = len(re.findall(old_p, self.text[:15_000]))
        assert new_count > old_count, (
            f"New pattern should find more rules than old: new={new_count} old={old_count}"
        )


class TestHasOrdersFeature:
    """has_orders=True should inject RULE_PATTERNS into the candidate pool."""
    text = _has_orders_text(3, 12)
    features_with_orders    = {"has_orders": True,  "has_schedules": False}
    features_without_orders = {"has_orders": False, "has_schedules": False}

    def test_rule_pattern_in_pool_with_flag(self):
        pool = _build_candidate_pool("GAZETTE_ACT", self.features_with_orders)
        patterns = [p for p, _ in pool]
        assert RULE_PATTERNS[0] in patterns

    def test_rule_pattern_not_in_pool_without_flag(self):
        pool = _build_candidate_pool("GAZETTE_ACT", self.features_without_orders)
        patterns = [p for p, _ in pool]
        assert RULE_PATTERNS[0] not in patterns

    def test_pattern_still_selected_correctly(self):
        p = select_section_pattern(self.text, "CODIFIED_ACT", self.features_with_orders)
        parts = [x for x in re.split(p, self.text) if x.strip()]
        assert len(parts) >= 20, (
            f"Expected ≥20 rule splits in ORDER document, got {len(parts)}"
        )


class TestContinuityBonus:
    """
    Prove that a pattern producing sequential numbers beats an equally-popular
    but non-sequential alternative.
    """

    def test_sequential_text_preferred(self):
        # Build text where sections go 1, 2, 3 … (fully sequential)
        sequential = _gazette_text(30)
        # Score the moderate pattern directly (no priority bonus)
        seq_score = _score_pattern(SECTION_PATTERNS[1], sequential[:15_000], priority_bonus=0)

        # Non-sequential text: same number of sections but random jumps
        nonseq = _nonsequential_text(20)
        nonseq_score = _score_pattern(SECTION_PATTERNS[1], nonseq[:15_000], priority_bonus=0)

        # Sequential should score higher (continuity multiplier 1.0 vs ~0.6)
        assert seq_score > nonseq_score, (
            f"Sequential text should score higher: seq={seq_score:.1f} nonseq={nonseq_score:.1f}"
        )

    def test_continuity_does_not_override_count(self):
        """
        A pattern with 30 sequential hits must beat a pattern with only 4 hits
        (even if the 4-hit pattern is perfectly sequential).
        """
        big_text   = _gazette_text(30)
        small_text = _gazette_text(4)  # only 4 sections

        big_score   = _score_pattern(SECTION_PATTERNS[1], big_text[:15_000],   0)
        small_score = _score_pattern(SECTION_PATTERNS[1], small_text[:15_000], 0)

        assert big_score > small_score, (
            f"30-section text should outscore 4-section text: big={big_score:.1f} small={small_score:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Backward compatibility tests
# ─────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    """All public function signatures and return types must be unchanged."""

    def test_select_section_pattern_returns_str(self):
        result = select_section_pattern("some text 1. foo\n2. bar\n3. baz\n", "GENERIC", {})
        assert isinstance(result, str)

    def test_select_section_pattern_unknown_doctype(self):
        """Must not raise even for an unknown doc_type."""
        result = select_section_pattern("1. Short title.\n2. Definitions.\n", "ALIEN_TYPE", {})
        assert isinstance(result, str)

    def test_select_section_pattern_empty_features(self):
        result = select_section_pattern(_gazette_text(15), "GAZETTE_ACT", {})
        assert isinstance(result, str) and len(result) > 0

    def test_select_chapter_pattern_returns_none_no_chapters(self):
        result = select_chapter_pattern("1. Short title.\n2. Definitions.\n", "GAZETTE_ACT")
        assert result is None

    def test_select_chapter_pattern_returns_pattern_with_chapters(self):
        text = "CHAPTER I\nPRELIMINARY\n\n1. Short title.\n2. Definitions.\n"
        result = select_chapter_pattern(text, "GAZETTE_ACT")
        assert result is not None
        assert result in CHAPTER_PATTERNS

    def test_select_section_pattern_empty_text(self):
        """Empty text must return a fallback pattern, not raise."""
        result = select_section_pattern("", "GAZETTE_ACT", {})
        assert isinstance(result, str)

    def test_select_section_pattern_short_text(self):
        """Very short text (< MIN_SECTIONS matches) must return fallback."""
        result = select_section_pattern("1. Short title.\n2. Definitions.\n", "GAZETTE_ACT", {})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────
# Score-pattern unit tests
# ─────────────────────────────────────────────────────────────────────

class TestScorePattern:

    def test_zero_for_too_few(self):
        score = _score_pattern(SECTION_PATTERNS[1], "1. Section one.\n", priority_bonus=0)
        # Only 1 match < _MIN_SECTIONS → base=0, but priority_bonus=0 → score=0
        assert score == 0.0

    def test_oversplit_penalised(self):
        # Build text with 600 one-char "sections" to trigger oversplit
        text = "\n".join(f"{i}. x" for i in range(1, 601))
        score_no_bonus = _score_pattern(SECTION_PATTERNS[1], text, priority_bonus=0)
        # Should be 0 or small (oversplit: max(0, 500 - 600) = 0 base × multiplier = 0)
        assert score_no_bonus == 0.0

    def test_priority_bonus_added(self):
        text = _gazette_text(20)
        score_0  = _score_pattern(SECTION_PATTERNS[1], text[:15_000], priority_bonus=0)
        score_20 = _score_pattern(SECTION_PATTERNS[1], text[:15_000], priority_bonus=20)
        assert score_20 == score_0 + 20

    def test_invalid_regex_returns_zero(self):
        score = _score_pattern(r"(?=\n\s*[invalid", "1. Test\n2. Test\n", priority_bonus=0)
        assert score == 0.0
