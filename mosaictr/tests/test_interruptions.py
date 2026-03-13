"""Tests for mosaictr.interruptions module."""

from __future__ import annotations

import pytest

from mosaictr.interruptions import (
    Interruption,
    InterruptionResult,
    detect_interruptions,
    _build_context,
    _find_motif_units,
    _hamming_distance,
)


# ---------------------------------------------------------------------------
# _find_motif_units
# ---------------------------------------------------------------------------

class TestFindMotifUnits:
    def test_pure_sequence(self):
        units = _find_motif_units("CAGCAGCAG", "CAG")
        assert len(units) == 3
        assert all(is_pure for _, is_pure in units)

    def test_with_interruption(self):
        units = _find_motif_units("CAGCAACAG", "CAG")
        assert len(units) == 3
        assert units[0][1] is True
        assert units[1][1] is False  # CAA != CAG
        assert units[2][1] is True

    def test_empty_input(self):
        assert _find_motif_units("", "CAG") == []
        assert _find_motif_units("CAG", "") == []


# ---------------------------------------------------------------------------
# _hamming_distance
# ---------------------------------------------------------------------------

class TestHammingDistance:
    def test_identical(self):
        assert _hamming_distance("CAG", "CAG") == 0

    def test_one_mismatch(self):
        assert _hamming_distance("CAG", "CAA") == 1

    def test_case_insensitive(self):
        assert _hamming_distance("cag", "CAG") == 0

    def test_different_lengths(self):
        assert _hamming_distance("CAG", "CA") == 1  # length diff counts


# ---------------------------------------------------------------------------
# detect_interruptions
# ---------------------------------------------------------------------------

class TestDetectInterruptions:
    def test_pure_sequence(self):
        result = detect_interruptions("CAGCAGCAGCAGCAG", "CAG")
        assert result.n_repeat_units == 5
        assert result.n_pure_units == 5
        assert result.n_interrupted_units == 0
        assert len(result.interruptions) == 0
        assert result.purity == 1.0
        assert result.longest_pure_run == 5

    def test_single_interruption(self):
        # CAG CAG CAG CAA CAG CAG
        seq = "CAGCAGCAGCAACAGCAG"
        result = detect_interruptions(seq, "CAG")
        assert result.n_repeat_units == 6
        assert result.n_pure_units == 5
        assert result.n_interrupted_units == 1
        assert len(result.interruptions) == 1
        assert result.interruptions[0].observed == "CAA"
        assert result.interruptions[0].position == 3

    def test_multiple_interruptions(self):
        # CAG CAA CAG CTG CAG
        seq = "CAGCAACAGCTGCAG"
        result = detect_interruptions(seq, "CAG")
        assert result.n_interrupted_units == 2
        assert len(result.interruptions) == 2

    def test_purity_score(self):
        # 3 pure + 3 interrupted = 6 units, purity = 0.5
        seq = "CAGCAACAGCTGCAGAAA"
        result = detect_interruptions(seq, "CAG")
        assert result.purity == pytest.approx(0.5)

    def test_empty_sequence(self):
        result = detect_interruptions("", "CAG")
        assert result.n_repeat_units == 0
        assert result.total_length == 0

    def test_sequence_shorter_than_motif(self):
        result = detect_interruptions("CA", "CAG")
        assert result.n_repeat_units == 1
        assert result.total_length == 2

    def test_longest_pure_run(self):
        # CAG CAG CAA CAG CAG CAG
        seq = "CAGCAGCAACAGCAGCAG"
        result = detect_interruptions(seq, "CAG")
        assert result.longest_pure_run == 3

    def test_sequence_composition(self):
        seq = "CAGCAGCAACAGCAG"
        result = detect_interruptions(seq, "CAG")
        assert result.sequence_composition["CAG"] == 4
        assert result.sequence_composition["CAA"] == 1


# ---------------------------------------------------------------------------
# InterruptionResult fields
# ---------------------------------------------------------------------------

    def test_trailing_partial_unit_below_half_motif_skipped(self):
        """Trailing base(s) shorter than half the motif should not count as interruption."""
        # CAGCAGC -> 2 pure units + "C" trailing (1 < 3//2=1, but max(1,1)=1, so 1 < 1 is False)
        # Actually for motif_len=3: max(motif_len // 2, 1) = max(1, 1) = 1
        # "C" has len=1, so 1 < 1 is False -> it IS reported
        # But for ATTCT (5bp motif): max(5//2, 1) = max(2, 1) = 2
        # "A" (1bp trailing) -> 1 < 2 -> skipped
        seq = "ATTCTATTCTA"  # 2 pure ATTCT + trailing "A"
        result = detect_interruptions(seq, "ATTCT")
        assert result.n_repeat_units == 3  # includes partial
        assert result.n_pure_units == 2
        # Trailing "A" should be skipped (1 < max(5//2, 1) = 2)
        assert len(result.interruptions) == 0

    def test_trailing_partial_unit_above_half_motif_reported(self):
        """Trailing bases >= half motif length should be reported as interruption."""
        # ATTCTATTCTATT -> 2 pure ATTCT + "ATT" (3 >= max(5//2,1)=2)
        seq = "ATTCTATTCTATT"
        result = detect_interruptions(seq, "ATTCT")
        assert result.n_repeat_units == 3
        assert len(result.interruptions) == 1
        assert result.interruptions[0].observed == "ATT"

    def test_case_insensitive_detection(self):
        """Mixed case in input should work."""
        result = detect_interruptions("cagcagcag", "CAG")
        assert result.n_pure_units == 3
        assert result.purity == 1.0

    def test_long_motif_vntr(self):
        """VNTR-length motif (>6bp) interruption detection."""
        motif = "ATTCTATTCT"  # 10bp
        seq = motif * 3 + "ATTCTATTCA" + motif * 2  # 1 interruption
        result = detect_interruptions(seq, motif)
        assert result.n_repeat_units == 6
        assert result.n_pure_units == 5
        assert len(result.interruptions) == 1


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_context_with_flanking_pure_runs(self):
        units = _find_motif_units("CAGCAGCAACAGCAG", "CAG")
        # units: [("CAG", True), ("CAG", True), ("CAA", False), ("CAG", True), ("CAG", True)]
        context = _build_context(units, 2, "CAG")
        assert "(CAG)2" in context
        assert "CAA" in context

    def test_context_no_flanking_runs(self):
        """Interruption at the only position, no pure flanking."""
        units = [("CAA", False)]
        context = _build_context(units, 0, "CAG")
        assert context == "CAA"

    def test_context_left_flank_only(self):
        units = _find_motif_units("CAGCAGCAA", "CAG")
        # [("CAG",T), ("CAG",T), ("CAA",F)]
        context = _build_context(units, 2, "CAG")
        assert context == "(CAG)2-CAA"

    def test_context_right_flank_only(self):
        units = _find_motif_units("CAACAGCAG", "CAG")
        # [("CAA",F), ("CAG",T), ("CAG",T)]
        context = _build_context(units, 0, "CAG")
        assert context == "CAA-(CAG)2"


# ---------------------------------------------------------------------------
# InterruptionResult fields
# ---------------------------------------------------------------------------

class TestInterruptionResult:
    def test_default_construction(self):
        r = InterruptionResult()
        assert r.total_length == 0
        assert r.n_repeat_units == 0
        assert r.purity == 0.0
        assert r.interruptions == []
        assert r.sequence_composition == {}
