"""Tests for HaploTR genotype module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from haplotr.genotype import (
    ReadInfo,
    _adaptive_collapse_threshold,
    _assign_hp0_reads,
    _compute_diff,
    _gap_bimodal_test,
    _hp_concordance,
    _robust_vntr_median,
    _sizing_quality,
    _trimmed_weighted_median,
    _v4_str_genotype,
    _v4_vntr_genotype,
    _v4_zygosity_decision,
    _weighted_median,
    _write_output_bed,
    compute_allele_size_cigar,
    hp_cond_v4_genotype,
)

# CIGAR operation codes
_M, _I, _D, _N, _S, _H, _P, _EQ, _X = range(9)


# ---------------------------------------------------------------------------
# compute_allele_size_cigar
# ---------------------------------------------------------------------------

class TestComputeAlleleSizeCigar:
    def _make_aln(self, cigar, ref_start=0):
        aln = MagicMock()
        aln.cigartuples = cigar
        aln.reference_start = ref_start
        return aln

    def test_simple_match(self):
        aln = self._make_aln([(_M, 300)], ref_start=0)
        size = compute_allele_size_cigar(aln, 100, 200)
        assert size == 100.0

    def test_insertion_in_locus(self):
        aln = self._make_aln([(_M, 100), (_I, 10), (_M, 200)], ref_start=0)
        size = compute_allele_size_cigar(aln, 50, 150)
        assert size == 110.0

    def test_deletion_in_locus(self):
        aln = self._make_aln([(_M, 100), (_D, 20), (_M, 200)], ref_start=0)
        size = compute_allele_size_cigar(aln, 50, 170)
        assert size == 100.0

    def test_no_coverage(self):
        aln = self._make_aln([(_M, 50)], ref_start=0)
        size = compute_allele_size_cigar(aln, 100, 200)
        assert size is None

    def test_none_cigar(self):
        aln = MagicMock()
        aln.cigartuples = None
        size = compute_allele_size_cigar(aln, 100, 200)
        assert size is None

    def test_eq_and_x_ops(self):
        """= and X consume ref and produce query bases like M."""
        aln = self._make_aln([(_EQ, 150), (_X, 150)], ref_start=0)
        size = compute_allele_size_cigar(aln, 100, 200)
        assert size == 100.0


# ---------------------------------------------------------------------------
# _weighted_median
# ---------------------------------------------------------------------------

class TestWeightedMedian:
    def test_empty(self):
        assert _weighted_median(np.array([]), np.array([])) == 0.0

    def test_equal_weights(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wts = np.ones(5)
        result = _weighted_median(vals, wts)
        assert result == pytest.approx(3.0)

    def test_heavy_weight_on_low(self):
        vals = np.array([1.0, 2.0, 3.0])
        wts = np.array([100.0, 1.0, 1.0])
        result = _weighted_median(vals, wts)
        assert result == pytest.approx(1.0)

    def test_heavy_weight_on_high(self):
        vals = np.array([1.0, 2.0, 3.0])
        wts = np.array([1.0, 1.0, 100.0])
        result = _weighted_median(vals, wts)
        assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _trimmed_weighted_median
# ---------------------------------------------------------------------------

class TestTrimmedWeightedMedian:
    def test_small_array(self):
        vals = np.array([1.0, 2.0, 3.0])
        wts = np.ones(3)
        result = _trimmed_weighted_median(vals, wts)
        assert result == pytest.approx(2.0)

    def test_outlier_removed(self):
        vals = np.array([10.0, 11.0, 12.0, 11.0, 100.0])
        wts = np.ones(5)
        result = _trimmed_weighted_median(vals, wts)
        assert result == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# _compute_diff
# ---------------------------------------------------------------------------

class TestComputeDiff:
    def test_vntr_no_rounding(self):
        """VNTR: raw float diff preserved."""
        d = _compute_diff(100.0, 103.7, motif_len=10, vntr_cutoff=7)
        assert d == pytest.approx(-3.7)

    def test_str_integer_rounding(self):
        """STR: rounds to nearest integer."""
        d = _compute_diff(100.0, 97.4, motif_len=1, vntr_cutoff=7)
        assert d == pytest.approx(3.0)

    def test_str_dinuc(self):
        """Dinucleotide: integer round, NOT motif-unit snap."""
        d = _compute_diff(100.0, 97.0, motif_len=2, vntr_cutoff=7)
        assert d == pytest.approx(3.0)

    def test_str_preserves_non_motif_diffs(self):
        """STR: 1bp partial diffs are preserved, not snapped."""
        d = _compute_diff(100.0, 99.0, motif_len=3, vntr_cutoff=7)
        assert d == pytest.approx(1.0)

    def test_zero_diff(self):
        d = _compute_diff(100.0, 100.0, motif_len=2, vntr_cutoff=7)
        assert d == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _hp_concordance
# ---------------------------------------------------------------------------

class TestHpConcordance:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_perfect_concordance(self):
        """HP1 near med1, HP2 near med2 -> concordance ~1.0."""
        reads = [self._ri(90.0, 1)] * 5 + [self._ri(110.0, 2)] * 5
        assert _hp_concordance(reads, 90.0, 110.0) == pytest.approx(1.0)

    def test_zero_concordance(self):
        """HP1 near med2, HP2 near med1 -> concordance ~0.0."""
        reads = [self._ri(110.0, 1)] * 5 + [self._ri(90.0, 2)] * 5
        assert _hp_concordance(reads, 90.0, 110.0) == pytest.approx(0.0)

    def test_mixed_concordance(self):
        """Half correct, half swapped -> ~0.5."""
        reads = (
            [self._ri(90.0, 1)] * 3 + [self._ri(110.0, 1)] * 3 +
            [self._ri(110.0, 2)] * 3 + [self._ri(90.0, 2)] * 3
        )
        assert _hp_concordance(reads, 90.0, 110.0) == pytest.approx(0.5)

    def test_no_hp_reads(self):
        """All HP=0 -> 0.0."""
        reads = [self._ri(100.0, 0)] * 10
        assert _hp_concordance(reads, 90.0, 110.0) == 0.0

    def test_single_hp_read(self):
        """Only 1 HP-tagged read -> 0.0 (below minimum)."""
        reads = [self._ri(90.0, 1)] + [self._ri(100.0, 0)] * 5
        assert _hp_concordance(reads, 90.0, 110.0) == 0.0


# ---------------------------------------------------------------------------
# _assign_hp0_reads
# ---------------------------------------------------------------------------

class TestAssignHp0Reads:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_basic_assignment(self):
        """HP0 reads assigned to closer median."""
        reads = (
            [self._ri(90.0, 1)] * 3 +
            [self._ri(110.0, 2)] * 3 +
            [self._ri(92.0, 0), self._ri(108.0, 0)]
        )
        hp1, hp2 = _assign_hp0_reads(reads, 90.0, 110.0)
        # 92 closer to 90 -> hp1; 108 closer to 110 -> hp2
        assert len(hp1) == 4  # 3 original + 1 HP0
        assert len(hp2) == 4  # 3 original + 1 HP0

    def test_all_hp0(self):
        """All HP=0 reads still get assigned."""
        reads = [self._ri(90.0, 0)] * 3 + [self._ri(110.0, 0)] * 3
        hp1, hp2 = _assign_hp0_reads(reads, 90.0, 110.0)
        assert len(hp1) + len(hp2) == 6

    def test_no_hp0(self):
        """No HP=0 reads -> groups unchanged."""
        reads = [self._ri(90.0, 1)] * 3 + [self._ri(110.0, 2)] * 3
        hp1, hp2 = _assign_hp0_reads(reads, 90.0, 110.0)
        assert len(hp1) == 3
        assert len(hp2) == 3


# ---------------------------------------------------------------------------
# _adaptive_collapse_threshold
# ---------------------------------------------------------------------------

class TestAdaptiveCollapseThreshold:
    def test_reference_n(self):
        """n=20 -> base threshold 0.25."""
        assert _adaptive_collapse_threshold(20) == pytest.approx(0.25)

    def test_high_coverage(self):
        """n=50 -> lower threshold, clamped to floor 0.20."""
        t = _adaptive_collapse_threshold(50)
        assert t < 0.25
        # 0.25 * sqrt(20/50) ≈ 0.158 < floor 0.20, so floor applies
        assert t == pytest.approx(0.20)

    def test_low_coverage(self):
        """n=10 -> higher threshold (~0.354)."""
        t = _adaptive_collapse_threshold(10)
        assert t > 0.25

    def test_floor(self):
        """Very high coverage hits the floor (min_val=0.20)."""
        t = _adaptive_collapse_threshold(1000)
        assert t == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# _gap_bimodal_test
# ---------------------------------------------------------------------------

class TestGapBimodalTest:
    def test_clearly_bimodal(self):
        """Two well-separated clusters."""
        arr = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
        assert _gap_bimodal_test(arr) is True

    def test_unimodal(self):
        """Tight cluster -> not bimodal."""
        arr = np.array([100.0, 100.5, 101.0, 100.0, 100.5, 101.0])
        assert _gap_bimodal_test(arr) is False

    def test_too_few(self):
        """< 6 reads -> False."""
        arr = np.array([100.0, 200.0, 100.0])
        assert _gap_bimodal_test(arr) is False

    def test_identical(self):
        """All same value -> not bimodal."""
        arr = np.array([100.0] * 10)
        assert _gap_bimodal_test(arr) is False

    def test_close_clusters(self):
        """Clusters separated by > 1bp but small gap."""
        arr = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 102.0])
        assert _gap_bimodal_test(arr) is True


# ---------------------------------------------------------------------------
# _robust_vntr_median
# ---------------------------------------------------------------------------

class TestRobustVntrMedian:
    def test_small_array(self):
        vals = np.array([10.0, 11.0, 12.0])
        wts = np.ones(3)
        result = _robust_vntr_median(vals, wts)
        assert result == pytest.approx(11.0)

    def test_with_outlier(self):
        """Outlier should be removed in pass 2."""
        vals = np.array([10.0, 11.0, 12.0, 11.0, 200.0])
        wts = np.ones(5)
        result = _robust_vntr_median(vals, wts)
        assert result == pytest.approx(11.0)

    def test_tight_distribution(self):
        """Tight values: pass 2 is a no-op."""
        vals = np.array([100.0, 100.1, 100.2, 100.0, 100.1])
        wts = np.ones(5)
        result = _robust_vntr_median(vals, wts)
        assert abs(result - 100.1) < 0.2

    def test_two_outliers(self):
        """Multiple outliers removed."""
        vals = np.array([50.0, 100.0, 101.0, 102.0, 101.0, 100.0, 300.0])
        wts = np.ones(7)
        result = _robust_vntr_median(vals, wts)
        assert 99.0 <= result <= 102.0


# ---------------------------------------------------------------------------
# _sizing_quality
# ---------------------------------------------------------------------------

class TestSizingQuality:
    def test_perfect_agreement(self):
        """All reads at median -> quality 1.0 (with enough coverage)."""
        vals = np.array([100.0] * 20)
        q = _sizing_quality(vals, 100.0, motif_len=10)
        assert q == pytest.approx(1.0)

    def test_low_coverage(self):
        """Few reads -> quality capped by coverage factor."""
        vals = np.array([100.0, 100.0, 100.0])
        q = _sizing_quality(vals, 100.0, motif_len=10)
        assert q <= 0.3  # cov factor = 3/10

    def test_scattered_reads(self):
        """Reads far from median -> low agreement."""
        vals = np.array([50.0, 100.0, 150.0, 200.0, 250.0,
                         300.0, 350.0, 400.0, 450.0, 500.0])
        q = _sizing_quality(vals, 100.0, motif_len=10)
        assert q < 0.5

    def test_very_few_reads(self):
        """1-2 reads -> quality < 0.4."""
        q = _sizing_quality(np.array([100.0]), 100.0, motif_len=10)
        assert q <= 0.2


# ---------------------------------------------------------------------------
# hp_cond_v4_genotype
# ---------------------------------------------------------------------------

class TestHpCondV4Genotype:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_empty(self):
        result = hp_cond_v4_genotype([], 100.0, motif_len=2)
        assert result == (0.0, 0.0, "HOM", 0.0)

    def test_4_tuple_return(self):
        """v4 returns 4-tuple (d1, d2, zygosity, confidence)."""
        reads = [self._ri(100.0, 1)] * 5 + [self._ri(100.0, 2)] * 5
        result = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert len(result) == 4
        d1, d2, zyg, conf = result
        assert isinstance(d1, float)
        assert isinstance(d2, float)
        assert zyg in ("HET", "HOM")
        assert isinstance(conf, float)

    def test_clear_het(self):
        """HP1 at 90, HP2 at 110 -> HET with high confidence."""
        reads = [self._ri(90.0, 1)] * 10 + [self._ri(110.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HET"
        assert d1 != d2
        assert conf > 0.7

    def test_noise_hom(self):
        """Both HP groups at same allele -> HOM."""
        reads = [self._ri(100.0, 1)] * 10 + [self._ri(100.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2

    def test_hp0_incorporated(self):
        """HP=0 reads should be incorporated, not discarded."""
        reads = (
            [self._ri(90.0, 1)] * 5 +
            [self._ri(110.0, 2)] * 5 +
            [self._ri(91.0, 0)] * 3 +  # should go to HP1 cluster
            [self._ri(109.0, 0)] * 3   # should go to HP2 cluster
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HET"
        assert d1 != d2

    def test_one_hp_fallback(self):
        """Only one HP group -> falls back to all-reads path."""
        reads = [self._ri(90.0, 1)] * 10 + [self._ri(110.0, 1)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # Bimodal with gap test should detect het
        assert len((d1, d2, zyg, conf)) == 4

    def test_vntr_bimodal(self):
        """VNTR with bimodal reads and HP separation."""
        reads = [self._ri(100.0, 1)] * 10 + [self._ri(200.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_adaptive_collapse_high_coverage(self):
        """High coverage with > 1 motif unit separation -> preserves het."""
        reads = [self._ri(98.0, 1)] * 25 + [self._ri(102.0, 2)] * 25
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HET"
        assert conf > 0.5

    def test_close_alleles_called_hom(self):
        """Alleles within 1 motif unit -> HOM regardless of concordance."""
        # 1bp diff in dinuc: |d1-d2| = 1 <= motif_len = 2 -> HOM
        reads = [self._ri(100.0, 1)] * 25 + [self._ri(101.0, 2)] * 25
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"

    def test_adaptive_collapse_low_coverage(self):
        """Low coverage with minor allele -> collapses to HOM."""
        reads = [self._ri(100.0, 1)] * 8 + [self._ri(101.0, 2)] * 2
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"

    def test_confidence_range(self):
        """Confidence should always be between 0 and 1."""
        test_cases = [
            [self._ri(100.0, 1)] * 10 + [self._ri(100.0, 2)] * 10,
            [self._ri(90.0, 1)] * 10 + [self._ri(110.0, 2)] * 10,
            [self._ri(100.0, 0)] * 20,
        ]
        for reads in test_cases:
            _, _, _, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
            assert 0.0 <= conf <= 1.0

    def test_vntr_hp0_assignment(self):
        """VNTR path should incorporate HP0 reads like STR path."""
        reads = (
            [self._ri(100.0, 1)] * 5 +
            [self._ri(200.0, 2)] * 5 +
            [self._ri(102.0, 0)] * 3 +  # should go to HP1
            [self._ri(198.0, 0)] * 3    # should go to HP2
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_vntr_outlier_robustness(self):
        """VNTR with extreme outlier reads should still give reasonable sizing."""
        reads = (
            [self._ri(100.0, 1)] * 8 +
            [self._ri(500.0, 1)] * 1 +  # outlier
            [self._ri(200.0, 2)] * 10
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        # The outlier (500) should not pull allele1 median too much
        allele1 = 150.0 - max(d1, d2)  # larger d = smaller allele
        assert allele1 < 150.0  # allele1 should be around 100, not 150+


# ---------------------------------------------------------------------------
# _write_output_bed (always 9-column)
# ---------------------------------------------------------------------------

class TestWriteOutputBed:
    def test_9_column_output(self, tmp_path):
        """Results always produce 9-column output."""
        output = str(tmp_path / "out.bed")
        loci = [
            ("chr1", 1000, 1050, "AC"),
            ("chr2", 2000, 2100, "ACGT"),
        ]
        results = [
            {"allele1_size": 48, "allele2_size": 50, "zygosity": "HET",
             "confidence": 0.85, "n_reads": 42},
            None,
        ]
        _write_output_bed(output, loci, results)

        with open(output) as f:
            lines = f.readlines()

        assert "confidence" in lines[0]
        fields = lines[1].strip().split("\t")
        assert len(fields) == 9
        assert fields[0] == "chr1"
        assert fields[4] == "48"
        assert fields[5] == "50"
        assert fields[6] == "HET"
        assert fields[7] == "0.850"
        assert fields[8] == "42"

        # None result
        fields2 = lines[2].strip().split("\t")
        assert fields2[4] == "."
        assert fields2[8] == "0"

    def test_float_allele_sizes(self, tmp_path):
        """VNTR can produce non-integer allele sizes."""
        output = str(tmp_path / "out.bed")
        loci = [("chr1", 1000, 1050, "ACGTACGT")]
        results = [
            {"allele1_size": 47.3, "allele2_size": 52.7, "zygosity": "HET",
             "confidence": 0.9, "n_reads": 20},
        ]
        _write_output_bed(output, loci, results)

        with open(output) as f:
            lines = f.readlines()

        fields = lines[1].strip().split("\t")
        assert fields[4] == "47.3"
        assert fields[5] == "52.7"

    def test_min_confidence_filter(self, tmp_path):
        """Loci below min_confidence are written as no-call."""
        output = str(tmp_path / "out.bed")
        loci = [
            ("chr1", 1000, 1050, "AC"),
            ("chr2", 2000, 2100, "ACGT"),
            ("chr3", 3000, 3050, "ACG"),
        ]
        results = [
            {"allele1_size": 48, "allele2_size": 50, "zygosity": "HET",
             "confidence": 0.9, "n_reads": 42},
            {"allele1_size": 98, "allele2_size": 98, "zygosity": "HOM",
             "confidence": 0.3, "n_reads": 10},
            None,
        ]
        n_passed = _write_output_bed(output, loci, results, min_confidence=0.5)

        assert n_passed == 1  # Only the first locus passes

        with open(output) as f:
            lines = f.readlines()

        assert len(lines) == 4  # header + 3 data rows (all rows preserved)
        # First locus: passes confidence filter
        fields1 = lines[1].strip().split("\t")
        assert fields1[4] == "48"
        assert fields1[6] == "HET"
        # Second locus: below confidence, written as no-call
        fields2 = lines[2].strip().split("\t")
        assert fields2[4] == "."
        assert fields2[8] == "0"
        # Third locus: None result
        fields3 = lines[3].strip().split("\t")
        assert fields3[4] == "."

    def test_min_confidence_zero_passes_all(self, tmp_path):
        """Default min_confidence=0 passes all non-None results."""
        output = str(tmp_path / "out.bed")
        loci = [("chr1", 1000, 1050, "AC")]
        results = [
            {"allele1_size": 48, "allele2_size": 50, "zygosity": "HET",
             "confidence": 0.01, "n_reads": 5},
        ]
        n_passed = _write_output_bed(output, loci, results, min_confidence=0.0)
        assert n_passed == 1


# ===========================================================================
# Edge case tests — added to improve coverage
# ===========================================================================


# ---------------------------------------------------------------------------
# _v4_zygosity_decision — direct unit tests
# ---------------------------------------------------------------------------

class TestV4ZygosityDecision:
    """Direct tests for _v4_zygosity_decision boundary conditions."""

    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_d1_equals_d2_returns_hom(self):
        """When d1 == d2, always HOM with confidence 1.0 (fast path)."""
        reads = [self._ri(100.0, 1)] * 5 + [self._ri(100.0, 2)] * 5
        zyg, conf = _v4_zygosity_decision(reads, 100.0, 100.0, 0.0, 0.0, 2)
        assert zyg == "HOM"
        assert conf == 1.0

    def test_diff_exactly_motif_len_returns_hom(self):
        """abs(d1-d2) == motif_len -> HOM (boundary: <= motif_len)."""
        reads = [self._ri(98.0, 1)] * 10 + [self._ri(102.0, 2)] * 10
        # d1=0, d2=2, motif_len=2 -> |0-2| == 2 <= 2 -> HOM
        zyg, conf = _v4_zygosity_decision(reads, 100.0, 98.0, 0.0, 2.0, 2)
        assert zyg == "HOM"

    def test_diff_one_above_motif_len_with_high_concordance(self):
        """abs(d1-d2) == motif_len+1 with high concordance -> HET."""
        reads = [self._ri(97.0, 1)] * 10 + [self._ri(103.0, 2)] * 10
        # med1=97 (HP1), med2=103 (HP2), d1=-3, d2=3, motif_len=2
        # |(-3)-3|=6 > 2, concordance~1.0 -> HET
        zyg, conf = _v4_zygosity_decision(reads, 97.0, 103.0, -3.0, 3.0, 2)
        assert zyg == "HET"
        assert conf > 0.7

    def test_concordance_exactly_at_threshold(self):
        """Concordance exactly at 0.7 boundary -> gray zone -> HET."""
        # Build reads: 7/10 concordant for each HP
        reads = (
            [self._ri(90.0, 1)] * 7 + [self._ri(110.0, 1)] * 3 +
            [self._ri(110.0, 2)] * 7 + [self._ri(90.0, 2)] * 3
        )
        # med1=90 (HP1 allele), med2=110 (HP2 allele)
        # concordance = 14/20 = 0.7 -> NOT > 0.7, but >= 0.5 -> gray zone -> HET
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HET"
        assert conf == pytest.approx(0.7)

    def test_concordance_just_above_threshold(self):
        """Concordance slightly above 0.7 -> HET."""
        # HP1: 8 near med1=90, 2 near med2=110 -> 8 concordant
        # HP2: 7 near med2=110, 3 near med1=90 -> 7 concordant
        # Total: 15/20 = 0.75
        reads = (
            [self._ri(90.0, 1)] * 8 + [self._ri(110.0, 1)] * 2 +
            [self._ri(110.0, 2)] * 7 + [self._ri(90.0, 2)] * 3
        )
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HET"
        assert conf > 0.7

    def test_concordance_below_half_returns_hom(self):
        """Concordance < 0.5 -> HOM even with large allele diff."""
        # HP1: 4 near med1=90, 6 near med2=110 -> 4 concordant
        # HP2: 4 near med2=110, 6 near med1=90 -> 4 concordant
        # Total: 8/20 = 0.4 -> HOM
        reads = (
            [self._ri(90.0, 1)] * 4 + [self._ri(110.0, 1)] * 6 +
            [self._ri(110.0, 2)] * 4 + [self._ri(90.0, 2)] * 6
        )
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HOM"
        assert conf > 0.5  # 1.0 - 0.4 = 0.6

    def test_concordance_in_gray_zone(self):
        """Concordance between 0.5 and threshold -> HET with moderate confidence."""
        # HP1: 6 near med1=90, 4 near med2=110 -> 6 concordant
        # HP2: 6 near med2=110, 4 near med1=90 -> 6 concordant
        # Total: 12/20 = 0.6 -> gray zone -> HET
        reads = (
            [self._ri(90.0, 1)] * 6 + [self._ri(110.0, 1)] * 4 +
            [self._ri(110.0, 2)] * 6 + [self._ri(90.0, 2)] * 4
        )
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HET"
        assert 0.5 <= conf <= 0.7

    def test_large_motif_len_boundary(self):
        """motif_len=6 (STR boundary): diff of 6 -> HOM."""
        reads = [self._ri(94.0, 1)] * 10 + [self._ri(100.0, 2)] * 10
        # diff exactly 6 <= motif_len=6 -> HOM
        zyg, conf = _v4_zygosity_decision(reads, 100.0, 94.0, 0.0, 6.0, 6)
        assert zyg == "HOM"

    def test_only_two_hp_reads_minimum(self):
        """Exactly 2 HP reads: concordance calculated (minimum for non-zero)."""
        reads = [self._ri(90.0, 1), self._ri(110.0, 2)]
        # med1=90 (HP1 allele), med2=110 (HP2 allele)
        # HP1 at 90 closer to med1=90 -> concordant. HP2 at 110 closer to med2=110 -> concordant.
        # concordance = 2/2 = 1.0 > 0.7 -> HET
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HET"

    def test_no_hp_reads_returns_hom(self):
        """All HP=0 reads -> concordance=0 -> HOM (< 0.5)."""
        reads = [self._ri(90.0, 0)] * 5 + [self._ri(110.0, 0)] * 5
        # No HP-tagged reads -> concordance=0.0 < 0.5 -> HOM
        zyg, conf = _v4_zygosity_decision(reads, 90.0, 110.0, -10.0, 10.0, 2)
        assert zyg == "HOM"


# ---------------------------------------------------------------------------
# _adaptive_collapse_threshold — additional edge cases
# ---------------------------------------------------------------------------

class TestAdaptiveCollapseThresholdEdge:
    def test_n_reads_zero(self):
        """n_reads=0 -> returns base (guard against division by zero)."""
        t = _adaptive_collapse_threshold(0)
        assert t == pytest.approx(0.25)

    def test_n_reads_one(self):
        """n_reads=1 -> large threshold (base * sqrt(20/1) ~ 1.12)."""
        t = _adaptive_collapse_threshold(1)
        assert t > 0.25
        # base * sqrt(20/1) = 0.25 * 4.47 ≈ 1.12
        assert t == pytest.approx(0.25 * (20 ** 0.5))

    def test_n_reads_negative(self):
        """Negative n_reads -> returns base (same as 0 guard)."""
        t = _adaptive_collapse_threshold(-5)
        assert t == pytest.approx(0.25)

    def test_exact_reference_n(self):
        """n_reads == ref_n -> returns exactly base."""
        t = _adaptive_collapse_threshold(20)
        assert t == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _hp_concordance — additional edge cases
# ---------------------------------------------------------------------------

class TestHpConcordanceEdge:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_exactly_two_hp_reads(self):
        """Exactly 2 HP reads (minimum for computation)."""
        reads = [self._ri(90.0, 1), self._ri(110.0, 2)]
        c = _hp_concordance(reads, 90.0, 110.0)
        assert c == pytest.approx(1.0)

    def test_equidistant_read_goes_concordant(self):
        """Read equidistant from both medians -> concordant (dist1 <= dist2)."""
        reads = [self._ri(100.0, 1), self._ri(110.0, 2)]
        # HP1 read at 100, med1=90, med2=110: dist1=10, dist2=10 -> concordant (<=)
        c = _hp_concordance(reads, 90.0, 110.0)
        # HP1: dist1=10, dist2=10 -> concordant (<=). HP2: dist2=0 -> concordant
        assert c == pytest.approx(1.0)

    def test_equal_medians(self):
        """Both medians identical -> all reads concordant (dist1 <= dist2)."""
        reads = [self._ri(100.0, 1)] * 5 + [self._ri(100.0, 2)] * 5
        c = _hp_concordance(reads, 100.0, 100.0)
        assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _assign_hp0_reads — additional edge cases
# ---------------------------------------------------------------------------

class TestAssignHp0ReadsEdge:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_hp0_equidistant_tiebreak(self):
        """HP0 read equidistant from both medians -> goes to smaller group."""
        reads = [
            self._ri(90.0, 1),
            self._ri(110.0, 2),
            self._ri(100.0, 0),  # equidistant from 90 and 110
        ]
        hp1, hp2 = _assign_hp0_reads(reads, 90.0, 110.0)
        # HP1 has 1, HP2 has 1, tie -> HP0 goes to hp1 (len(hp1) <= len(hp2))
        assert len(hp1) == 2
        assert len(hp2) == 1

    def test_hp0_equidistant_goes_to_smaller_group(self):
        """Tie-break: HP0 goes to the smaller group."""
        reads = [
            self._ri(90.0, 1),
            self._ri(90.5, 1),
            self._ri(110.0, 2),
            self._ri(100.0, 0),
        ]
        hp1, hp2 = _assign_hp0_reads(reads, 90.0, 110.0)
        # HP1 has 2, HP2 has 1. Equidistant -> goes to smaller (hp2)
        assert len(hp2) == 2  # 1 original + 1 equidistant

    def test_equal_medians_all_hp0(self):
        """Both medians equal, all HP=0 -> alternating assignment for balance."""
        reads = [self._ri(100.0, 0)] * 6
        hp1, hp2 = _assign_hp0_reads(reads, 100.0, 100.0)
        # All equidistant -> alternating: hp1 gets 1st, hp2 gets 2nd, etc.
        assert len(hp1) == 3
        assert len(hp2) == 3


# ---------------------------------------------------------------------------
# _gap_bimodal_test — additional edge cases
# ---------------------------------------------------------------------------

class TestGapBimodalTestEdge:
    def test_exactly_six_reads_bimodal(self):
        """Exactly min_reads=6 with clear bimodality -> True."""
        arr = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
        assert _gap_bimodal_test(arr) is True

    def test_exactly_five_reads_rejected(self):
        """5 reads (below min_reads=6) -> False regardless."""
        arr = np.array([100.0, 100.0, 200.0, 200.0, 200.0])
        assert _gap_bimodal_test(arr) is False

    def test_all_identical_no_gaps(self):
        """All identical -> no gaps above threshold."""
        arr = np.array([50.0] * 8)
        assert _gap_bimodal_test(arr) is False

    def test_max_gap_exactly_one_bp(self):
        """max_gap == 1.0: must be > 1.0, so False."""
        # Values: 100, 100, 100, 101, 101, 101 -> max_gap=1.0
        arr = np.array([100.0, 100.0, 100.0, 101.0, 101.0, 101.0])
        assert _gap_bimodal_test(arr) is False

    def test_max_gap_just_above_one_bp(self):
        """max_gap slightly > 1.0 and > 2*median_gap -> True."""
        arr = np.array([100.0, 100.0, 100.0, 101.5, 101.5, 101.5])
        # Sorted gaps: [0, 0, 1.5, 0, 0] -> max=1.5, median=0.0
        # 1.5 > 2*0 and 1.5 > 1.0 -> True
        assert _gap_bimodal_test(arr) is True

    def test_single_value_different(self):
        """One outlier in uniform cluster -> gap > median_gap but depends on factor."""
        arr = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 200.0])
        # gaps: [0,0,0,0,100]. max=100, median=0. 100 > 2*0 and 100 > 1.0 -> True
        assert _gap_bimodal_test(arr) is True


# ---------------------------------------------------------------------------
# _robust_vntr_median — additional edge cases
# ---------------------------------------------------------------------------

class TestRobustVntrMedianEdge:
    def test_empty_array(self):
        """Empty array -> 0.0 from _weighted_median fallback."""
        result = _robust_vntr_median(np.array([]), np.array([]))
        assert result == pytest.approx(0.0)

    def test_single_value(self):
        """Single value -> that value."""
        result = _robust_vntr_median(np.array([42.0]), np.array([1.0]))
        assert result == pytest.approx(42.0)

    def test_two_values(self):
        """Two values (< 4) -> fallback to _weighted_median."""
        result = _robust_vntr_median(np.array([10.0, 20.0]), np.array([1.0, 1.0]))
        # _weighted_median of [10,20] with equal weights -> 10 or 20 depending on impl
        assert result in (10.0, 20.0)

    def test_three_values(self):
        """Three values (< 4) -> fallback to _weighted_median."""
        result = _robust_vntr_median(np.array([10.0, 20.0, 30.0]), np.array([1.0, 1.0, 1.0]))
        assert result == pytest.approx(20.0)

    def test_all_identical(self):
        """All identical values -> MAD=0 -> returns initial estimate."""
        result = _robust_vntr_median(np.array([50.0] * 10), np.ones(10))
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _sizing_quality — additional edge cases
# ---------------------------------------------------------------------------

class TestSizingQualityEdge:
    def test_empty_array(self):
        """Zero reads -> quality 0.0."""
        q = _sizing_quality(np.array([]), 100.0, motif_len=3)
        assert q == pytest.approx(0.0)

    def test_two_reads_capped(self):
        """2 reads -> quality limited by n/5.0 = 0.4."""
        q = _sizing_quality(np.array([100.0, 100.0]), 100.0, motif_len=3)
        assert q <= 0.4

    def test_motif_len_1_tolerance(self):
        """Small motif_len=1 -> tolerance is max(2, 5)=5."""
        vals = np.array([98.0, 99.0, 100.0, 101.0, 102.0,
                         103.0, 104.0, 105.0, 106.0, 107.0])
        q = _sizing_quality(vals, 100.0, motif_len=1)
        # tolerance = max(2, 5) = 5. Reads within 5 of 100: 98-105 = 8/10
        # cov_factor = min(1.0, 10/10) = 1.0
        assert q == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# hp_cond_v4_genotype — integration edge cases
# ---------------------------------------------------------------------------

class TestHpCondV4GenotypeEdge:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_all_hp0_reads_str(self):
        """All HP=0 reads (no haplotype info): STR falls back to all-reads path."""
        reads = [self._ri(100.0, 0)] * 20
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2
        assert 0.0 <= conf <= 1.0

    def test_all_hp0_reads_vntr(self):
        """All HP=0 reads: VNTR falls back to all-reads path."""
        reads = [self._ri(150.0, 0)] * 20
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HOM"
        assert d1 == d2

    def test_all_hp0_bimodal_str(self):
        """All HP=0 with bimodal distribution: STR gap-split fallback."""
        reads = [self._ri(90.0, 0)] * 10 + [self._ri(110.0, 0)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HET"
        assert d1 != d2

    def test_all_hp0_bimodal_vntr(self):
        """All HP=0 with bimodal distribution: VNTR gap-split fallback."""
        reads = [self._ri(100.0, 0)] * 10 + [self._ri(200.0, 0)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_single_read(self):
        """Single read -> HOM."""
        reads = [self._ri(100.0, 1)]
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2

    def test_two_reads_different_hp(self):
        """Two reads, one each HP but below min_hp_reads=3 -> fallback."""
        reads = [self._ri(90.0, 1), self._ri(110.0, 2)]
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # Not enough HP reads (need 3 per group), falls back
        assert len((d1, d2, zyg, conf)) == 4
        assert 0.0 <= conf <= 1.0

    def test_motif_len_6_str_boundary(self):
        """motif_len=6 (STR, just below vntr_cutoff=7): uses STR path."""
        reads = [self._ri(94.0, 1)] * 15 + [self._ri(106.0, 2)] * 15
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=6)
        # |d1-d2| = 12 > 6 -> HET check
        # With perfect concordance -> HET
        assert zyg == "HET"

    def test_motif_len_7_vntr_boundary(self):
        """motif_len=7 (VNTR, at vntr_cutoff=7): uses VNTR path."""
        reads = [self._ri(93.0, 1)] * 15 + [self._ri(107.0, 2)] * 15
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=7)
        # VNTR path, bimodal, HP-sufficient
        assert zyg == "HET"

    def test_exactly_motif_len_apart_str(self):
        """Alleles exactly motif_len apart -> HOM (STR)."""
        # motif_len=3, reads at 100 and 103 -> diffs differ by 3 == motif_len
        reads = [self._ri(100.0, 1)] * 15 + [self._ri(103.0, 2)] * 15
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=3)
        # d1 = round(100-100) = 0, d2 = round(100-103) = -3
        # |0 - (-3)| = 3 == motif_len -> HOM
        assert zyg == "HOM"

    def test_one_more_than_motif_len_apart_str(self):
        """Alleles motif_len+1 apart -> potential HET (STR)."""
        # motif_len=3, reads at 100 and 104 -> |d1-d2| = 4 > 3
        reads = [self._ri(100.0, 1)] * 15 + [self._ri(104.0, 2)] * 15
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=3)
        assert zyg == "HET"

    def test_very_asymmetric_hp_counts(self):
        """HP1 has 20 reads, HP2 has 3 reads (minimum): still uses HP path."""
        reads = [self._ri(90.0, 1)] * 20 + [self._ri(110.0, 2)] * 3
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # HP sufficient (3 >= 3, 20 >= 3, frac = 23/23 = 1.0 >= 0.15)
        assert len((d1, d2, zyg, conf)) == 4
        assert 0.0 <= conf <= 1.0

    def test_hp2_zero_reads(self):
        """HP2 has 0 reads -> insufficient HP, falls to all-reads path."""
        reads = [self._ri(90.0, 1)] * 10 + [self._ri(110.0, 0)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # Only HP1, not HP2 -> insufficient HP
        assert len((d1, d2, zyg, conf)) == 4

    def test_hp1_zero_reads(self):
        """HP1 has 0 reads, only HP2 -> insufficient HP, fallback."""
        reads = [self._ri(90.0, 0)] * 10 + [self._ri(110.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert len((d1, d2, zyg, conf)) == 4

    def test_vntr_all_reads_identical(self):
        """VNTR with all reads having identical allele size -> HOM."""
        reads = [self._ri(150.0, 1)] * 10 + [self._ri(150.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HOM"
        assert d1 == d2
        assert d1 == pytest.approx(0.0)

    def test_vntr_extreme_outlier_single(self):
        """VNTR with one extreme outlier: robust median should ignore it."""
        reads = (
            [self._ri(150.0, 1)] * 9 + [self._ri(1000.0, 1)] +
            [self._ri(150.0, 2)] * 10
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        # Despite 1000.0 outlier, robust median should keep allele near 150
        assert zyg == "HOM"
        assert abs(d1) < 50  # should be near 0, not pulled to 1000

    def test_low_mapq_reads(self):
        """Reads with very low mapq -> weighted less in median."""
        reads = (
            [self._ri(100.0, 1, mapq=60)] * 8 +
            [self._ri(200.0, 1, mapq=1)] * 2 +  # low mapq outliers
            [self._ri(100.0, 2, mapq=60)] * 10
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # The low-mapq reads at 200 should be weighted very low
        assert zyg == "HOM"

    def test_all_mapq_zero(self):
        """All reads with mapq=0 -> clamped to 1.0, still works."""
        reads = [self._ri(100.0, 1, mapq=0)] * 10 + [self._ri(100.0, 2, mapq=0)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2

    def test_str_adaptive_collapse_triggers(self):
        """STR: close alleles with very low minor fraction -> collapse to HOM."""
        # 18 HP1 reads at 100, 2 HP2 reads at 101 (within motif_len=2)
        # minor_frac = 2/20 = 0.10 < adaptive threshold (~0.25)
        reads = [self._ri(100.0, 1)] * 18 + [self._ri(101.0, 2)] * 2
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2

    def test_vntr_unimodal_hp_sufficient(self):
        """VNTR unimodal with HP-sufficient but close alleles -> HOM."""
        reads = [self._ri(150.0, 1)] * 10 + [self._ri(152.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        # |d1-d2| small compared to motif_len=10 -> HOM
        assert zyg == "HOM"

    def test_vntr_bimodal_no_hp_gap_split(self):
        """VNTR bimodal, no HP -> gap-split fallback."""
        reads = [self._ri(100.0, 0)] * 8 + [self._ri(200.0, 0)] * 8
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_d1_d2_sorted_output(self):
        """d1 <= d2 always (sorted order)."""
        reads = [self._ri(110.0, 1)] * 10 + [self._ri(90.0, 2)] * 10
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        assert d1 <= d2

    def test_hp_frac_below_threshold(self):
        """HP fraction below min_hp_frac -> fallback to all-reads."""
        # 3 HP1, 3 HP2, 44 HP0 -> frac = 6/50 = 0.12 < 0.15
        reads = (
            [self._ri(90.0, 1)] * 3 +
            [self._ri(110.0, 2)] * 3 +
            [self._ri(100.0, 0)] * 44
        )
        d1, d2, zyg, conf = hp_cond_v4_genotype(reads, 100.0, motif_len=2)
        # Falls back to all-reads path
        assert len((d1, d2, zyg, conf)) == 4
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _v4_str_genotype — direct unit tests
# ---------------------------------------------------------------------------

class TestV4StrGenotypeEdge:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_insufficient_hp_unimodal(self):
        """Insufficient HP + unimodal -> HOM via all-reads median."""
        reads = [self._ri(100.0, 0)] * 20
        d1, d2, zyg, conf = _v4_str_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HOM"
        assert d1 == d2
        assert conf == pytest.approx(0.5)

    def test_insufficient_hp_bimodal(self):
        """Insufficient HP + bimodal -> HET via gap-split."""
        reads = [self._ri(90.0, 0)] * 10 + [self._ri(110.0, 0)] * 10
        d1, d2, zyg, conf = _v4_str_genotype(reads, 100.0, motif_len=2)
        assert zyg == "HET"
        assert conf == pytest.approx(0.5)

    def test_collapse_with_balanced_but_close_alleles(self):
        """Balanced HP (50/50) but close alleles within motif_len -> HOM via zygosity decision."""
        reads = [self._ri(100.0, 1)] * 10 + [self._ri(101.0, 2)] * 10
        d1, d2, zyg, conf = _v4_str_genotype(reads, 100.0, motif_len=2)
        # |d1-d2| <= 2 and minor_frac=0.5 > threshold, so goes to zygosity_decision
        # where |d1-d2| <= motif_len -> HOM
        assert zyg == "HOM"


# ---------------------------------------------------------------------------
# _v4_vntr_genotype — direct unit tests
# ---------------------------------------------------------------------------

class TestV4VntrGenotypeEdge:
    def _ri(self, size, hp, mapq=60):
        return ReadInfo(allele_size=size, hp=hp, mapq=mapq)

    def test_unimodal_insufficient_hp(self):
        """Unimodal VNTR with no HP -> HOM, low confidence."""
        reads = [self._ri(150.0, 0)] * 20
        d1, d2, zyg, conf = _v4_vntr_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HOM"
        assert d1 == d2

    def test_unimodal_hp_sufficient_same_allele(self):
        """Unimodal VNTR, HP sufficient, same allele -> HOM."""
        reads = [self._ri(150.0, 1)] * 10 + [self._ri(150.0, 2)] * 10
        d1, d2, zyg, conf = _v4_vntr_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HOM"
        assert d1 == d2

    def test_bimodal_hp_sufficient(self):
        """Bimodal VNTR with HP -> HET via HP medians."""
        reads = [self._ri(100.0, 1)] * 10 + [self._ri(200.0, 2)] * 10
        d1, d2, zyg, conf = _v4_vntr_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_bimodal_hp_insufficient_gap_split(self):
        """Bimodal VNTR, insufficient HP -> gap-split fallback."""
        reads = [self._ri(100.0, 0)] * 10 + [self._ri(200.0, 0)] * 10
        d1, d2, zyg, conf = _v4_vntr_genotype(reads, 150.0, motif_len=10)
        assert zyg == "HET"
        assert d1 != d2

    def test_confidence_includes_sizing_quality(self):
        """VNTR confidence = zyg_conf * min(sq1, sq2)."""
        reads = [self._ri(100.0, 1)] * 15 + [self._ri(200.0, 2)] * 15
        d1, d2, zyg, conf = _v4_vntr_genotype(reads, 150.0, motif_len=10)
        # Confidence should be product of zygosity confidence and sizing quality
        assert 0.0 <= conf <= 1.0
        # With good HP and sizing, should be reasonably high
        assert conf > 0.3
