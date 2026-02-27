"""Tests for HaploTR somatic instability module."""

import numpy as np
import pytest

from haplotr.genotype import ReadInfo
from unittest.mock import MagicMock, patch

from haplotr.instability import (
    _ais,
    _check_hp_tags,
    _detect_dropout,
    _ecb,
    _hii,
    _ias,
    _scr,
    _ser,
    _weighted_mad,
    _write_instability_tsv,
    compute_instability,
)


def _ri(size, hp, mapq=60):
    return ReadInfo(allele_size=size, hp=hp, mapq=mapq)


# ---------------------------------------------------------------------------
# _weighted_mad
# ---------------------------------------------------------------------------

class TestWeightedMad:
    def test_empty(self):
        assert _weighted_mad(np.array([]), np.array([]), 0.0) == 0.0

    def test_identical_values(self):
        vals = np.array([5.0, 5.0, 5.0])
        wts = np.ones(3)
        assert _weighted_mad(vals, wts, 5.0) == 0.0

    def test_symmetric(self):
        """Symmetric distribution: MAD should equal the distance from center."""
        vals = np.array([8.0, 10.0, 12.0])
        wts = np.ones(3)
        result = _weighted_mad(vals, wts, 10.0)
        assert result == pytest.approx(2.0)

    def test_asymmetric(self):
        """Asymmetric: deviations are [0, 1, 5], median deviation = 1."""
        vals = np.array([10.0, 11.0, 15.0])
        wts = np.ones(3)
        result = _weighted_mad(vals, wts, 10.0)
        assert result == pytest.approx(1.0)

    def test_weighted(self):
        """Heavy weight on close value pulls MAD down."""
        vals = np.array([10.0, 15.0])
        wts = np.array([100.0, 1.0])
        result = _weighted_mad(vals, wts, 10.0)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _hii
# ---------------------------------------------------------------------------

class TestHii:
    def test_stable_locus(self):
        """All reads at same size -> HII = 0."""
        sizes = np.array([100.0] * 10)
        wts = np.ones(10)
        assert _hii(sizes, wts, motif_len=3) == pytest.approx(0.0)

    def test_unstable_locus(self):
        """Reads spread around median -> HII > 0."""
        sizes = np.array([97.0, 100.0, 103.0, 100.0, 106.0])
        wts = np.ones(5)
        result = _hii(sizes, wts, motif_len=3)
        assert result > 0.0

    def test_motif_normalization(self):
        """Same MAD, longer motif -> lower HII."""
        sizes = np.array([97.0, 100.0, 103.0])
        wts = np.ones(3)
        hii_3 = _hii(sizes, wts, motif_len=3)
        hii_6 = _hii(sizes, wts, motif_len=6)
        assert hii_3 == pytest.approx(2 * hii_6)

    def test_single_read(self):
        """< 2 reads -> 0."""
        assert _hii(np.array([100.0]), np.ones(1), motif_len=3) == 0.0

    def test_zero_motif(self):
        assert _hii(np.array([100.0, 103.0]), np.ones(2), motif_len=0) == 0.0


# ---------------------------------------------------------------------------
# _ser
# ---------------------------------------------------------------------------

class TestSer:
    def test_no_expansion(self):
        """All reads at or below median + motif -> SER = 0."""
        sizes = np.array([100.0, 101.0, 102.0, 103.0])
        assert _ser(sizes, 100.0, motif_len=3) == pytest.approx(0.0)

    def test_all_expanded(self):
        """All reads > median + motif -> SER = 1."""
        sizes = np.array([110.0, 112.0, 115.0])
        assert _ser(sizes, 100.0, motif_len=3) == pytest.approx(1.0)

    def test_partial_expansion(self):
        """Some reads expanded: 1 of 4 > 103."""
        sizes = np.array([100.0, 101.0, 102.0, 110.0])
        assert _ser(sizes, 100.0, motif_len=3) == pytest.approx(0.25)

    def test_empty(self):
        assert _ser(np.array([]), 100.0, motif_len=3) == 0.0


# ---------------------------------------------------------------------------
# _scr
# ---------------------------------------------------------------------------

class TestScr:
    def test_no_contraction(self):
        """All reads at or above median - motif -> SCR = 0."""
        sizes = np.array([100.0, 99.0, 98.0, 97.0])
        assert _scr(sizes, 100.0, motif_len=3) == pytest.approx(0.0)

    def test_all_contracted(self):
        """All reads < median - motif -> SCR = 1."""
        sizes = np.array([90.0, 92.0, 95.0])
        assert _scr(sizes, 100.0, motif_len=3) == pytest.approx(1.0)

    def test_partial_contraction(self):
        """1 of 4 contracted."""
        sizes = np.array([100.0, 99.0, 98.0, 90.0])
        assert _scr(sizes, 100.0, motif_len=3) == pytest.approx(0.25)

    def test_empty(self):
        assert _scr(np.array([]), 100.0, motif_len=3) == 0.0


# ---------------------------------------------------------------------------
# _ecb
# ---------------------------------------------------------------------------

class TestEcb:
    def test_expansion_only(self):
        """SER=0.5, SCR=0 -> ECB=1.0."""
        assert _ecb(0.5, 0.0) == pytest.approx(1.0)

    def test_contraction_only(self):
        """SER=0, SCR=0.5 -> ECB=-1.0."""
        assert _ecb(0.0, 0.5) == pytest.approx(-1.0)

    def test_balanced(self):
        """SER=SCR -> ECB=0."""
        assert _ecb(0.3, 0.3) == pytest.approx(0.0)

    def test_no_instability(self):
        """SER=0, SCR=0 -> ECB=0."""
        assert _ecb(0.0, 0.0) == pytest.approx(0.0)

    def test_range(self):
        """ECB always in [-1, +1]."""
        for ser_val in [0.0, 0.1, 0.5, 1.0]:
            for scr_val in [0.0, 0.1, 0.5, 1.0]:
                ecb = _ecb(ser_val, scr_val)
                assert -1.0 <= ecb <= 1.0


# ---------------------------------------------------------------------------
# _ias
# ---------------------------------------------------------------------------

class TestIas:
    def test_symmetric(self):
        """Same HII -> IAS = 0."""
        assert _ias(0.5, 0.5) == pytest.approx(0.0)

    def test_one_stable(self):
        """One haplotype at 0 -> IAS = 1.0."""
        assert _ias(1.0, 0.0) == pytest.approx(1.0)
        assert _ias(0.0, 1.0) == pytest.approx(1.0)

    def test_both_zero(self):
        """Both zero -> IAS = 0."""
        assert _ias(0.0, 0.0) == pytest.approx(0.0)

    def test_range(self):
        """IAS always in [0, 1]."""
        for h1 in [0.0, 0.1, 0.5, 1.0, 2.0]:
            for h2 in [0.0, 0.1, 0.5, 1.0, 2.0]:
                ias = _ias(h1, h2)
                assert 0.0 <= ias <= 1.0


# ---------------------------------------------------------------------------
# _ais
# ---------------------------------------------------------------------------

class TestAis:
    def test_stable_locus(self):
        """Both HII=0 -> AIS ~0."""
        result = _ais(0.0, 0.0, concordance=1.0, n_total=20)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_high_instability(self):
        """High HII + good concordance + good coverage -> high AIS."""
        result = _ais(2.0, 0.5, concordance=0.9, n_total=30)
        assert result > 1.0

    def test_low_coverage_penalty(self):
        """Low coverage reduces AIS."""
        high_cov = _ais(1.0, 0.5, concordance=0.9, n_total=20)
        low_cov = _ais(1.0, 0.5, concordance=0.9, n_total=5)
        assert low_cov < high_cov

    def test_low_concordance_penalty(self):
        """Low concordance reduces AIS."""
        high_conc = _ais(1.0, 0.5, concordance=0.9, n_total=20)
        low_conc = _ais(1.0, 0.5, concordance=0.3, n_total=20)
        assert low_conc < high_conc


# ---------------------------------------------------------------------------
# compute_instability — integration tests
# ---------------------------------------------------------------------------

class TestComputeInstability:
    def test_empty_reads(self):
        assert compute_instability([], 100.0, motif_len=3) is None

    def test_stable_het(self):
        """Two clean haplotype clusters, no instability."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result is not None
        assert result["hii_h1"] == pytest.approx(0.0)
        assert result["hii_h2"] == pytest.approx(0.0)
        assert result["ser_h1"] == pytest.approx(0.0)
        assert result["scr_h1"] == pytest.approx(0.0)
        assert result["ais"] == pytest.approx(0.0, abs=1e-6)
        assert result["concordance"] > 0.9

    def test_one_haplotype_unstable(self):
        """HP=1 stable at 90, HP=2 spread around 110 -> asymmetric instability."""
        hp1_reads = [_ri(90.0, 1)] * 10
        hp2_reads = [
            _ri(105.0, 2), _ri(110.0, 2), _ri(115.0, 2),
            _ri(120.0, 2), _ri(110.0, 2), _ri(108.0, 2),
            _ri(112.0, 2), _ri(125.0, 2), _ri(110.0, 2),
            _ri(130.0, 2),
        ]
        result = compute_instability(hp1_reads + hp2_reads, 100.0, motif_len=3)
        assert result is not None
        assert result["hii_h2"] > result["hii_h1"]
        assert result["ias"] > 0.5  # asymmetric

    def test_expansion_bias(self):
        """Reads skewed toward expansion -> positive ECB."""
        hp1_reads = [_ri(100.0, 1)] * 10
        # HP2: median ~110, but with many expansions and few contractions
        hp2_reads = [
            _ri(110.0, 2), _ri(110.0, 2), _ri(110.0, 2),
            _ri(110.0, 2), _ri(110.0, 2),
            _ri(120.0, 2), _ri(125.0, 2), _ri(130.0, 2),  # expanded
            _ri(108.0, 2), _ri(109.0, 2),
        ]
        result = compute_instability(hp1_reads + hp2_reads, 100.0, motif_len=3)
        assert result is not None
        assert result["ecb_h2"] > 0.0  # expansion bias

    def test_hp0_assignment(self):
        """HP=0 reads are assigned to correct haplotype."""
        reads = (
            [_ri(90.0, 1)] * 5
            + [_ri(110.0, 2)] * 5
            + [_ri(91.0, 0)] * 3  # should join HP1
            + [_ri(109.0, 0)] * 3  # should join HP2
        )
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result is not None
        assert result["n_h1"] == 8  # 5 HP1 + 3 HP0
        assert result["n_h2"] == 8  # 5 HP2 + 3 HP0
        assert result["n_total"] == 16

    def test_insufficient_hp_gap_fallback(self):
        """Bimodal reads with no HP tags -> gap-split fallback."""
        reads = [_ri(90.0, 0)] * 5 + [_ri(110.0, 0)] * 5
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result is not None
        assert result["concordance"] == pytest.approx(0.5)
        assert result["n_h1"] + result["n_h2"] == 10

    def test_insufficient_hp_pooled_fallback(self):
        """Unimodal reads with no HP tags -> pooled fallback."""
        reads = [_ri(100.0, 0)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result is not None
        assert result["concordance"] == pytest.approx(0.0)
        assert result["n_h2"] == 0

    def test_all_metrics_present(self):
        """All 21 keys present in output."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        expected_keys = {
            "modal_h1", "modal_h2", "hii_h1", "hii_h2",
            "ser_h1", "ser_h2", "scr_h1", "scr_h2",
            "ecb_h1", "ecb_h2", "ias", "ais",
            "range_h1", "range_h2", "n_h1", "n_h2",
            "n_total", "concordance",
            "analysis_path", "unstable_haplotype", "dropout_flag",
        }
        assert set(result.keys()) == expected_keys

    def test_hii_range_nonnegative(self):
        """HII values should always be non-negative."""
        reads = [_ri(90.0 + i * 0.5, 1) for i in range(10)] + \
                [_ri(110.0 + i * 0.5, 2) for i in range(10)]
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["hii_h1"] >= 0.0
        assert result["hii_h2"] >= 0.0

    def test_ser_scr_range(self):
        """SER and SCR should be in [0, 1]."""
        reads = [_ri(90.0 + i * 3, 1) for i in range(10)] + \
                [_ri(110.0 + i * 3, 2) for i in range(10)]
        result = compute_instability(reads, 100.0, motif_len=3)
        for key in ["ser_h1", "ser_h2", "scr_h1", "scr_h2"]:
            assert 0.0 <= result[key] <= 1.0

    def test_ecb_range(self):
        """ECB should be in [-1, +1]."""
        reads = [_ri(90.0 + i * 3, 1) for i in range(10)] + \
                [_ri(110.0 + i * 3, 2) for i in range(10)]
        result = compute_instability(reads, 100.0, motif_len=3)
        assert -1.0 <= result["ecb_h1"] <= 1.0
        assert -1.0 <= result["ecb_h2"] <= 1.0


# ---------------------------------------------------------------------------
# _write_instability_tsv
# ---------------------------------------------------------------------------

class TestWriteInstabilityTsv:
    def test_header_columns(self, tmp_path):
        """Header has 25 columns."""
        output = str(tmp_path / "out.tsv")
        _write_instability_tsv(output, [], [])
        with open(output) as f:
            header = f.readline().strip()
        cols = header.split("\t")
        assert len(cols) == 25
        assert cols[0] == "#chrom"
        assert cols[-1] == "dropout_flag"
        assert cols[-2] == "unstable_haplotype"
        assert cols[-3] == "analysis_path"

    def test_data_row(self, tmp_path):
        """Data row has 25 columns with correct values."""
        output = str(tmp_path / "out.tsv")
        loci = [("chr4", 3074877, 3074933, "CAG")]
        results = [{
            "modal_h1": 56.0, "modal_h2": 56.0,
            "hii_h1": 0.0, "hii_h2": 0.0,
            "ser_h1": 0.0, "ser_h2": 0.0,
            "scr_h1": 0.0, "scr_h2": 0.0,
            "ecb_h1": 0.0, "ecb_h2": 0.0,
            "ias": 0.0, "ais": 0.0,
            "range_h1": 0.0, "range_h2": 0.0,
            "n_h1": 15, "n_h2": 13,
            "n_total": 28, "concordance": 0.95,
            "analysis_path": "hp-tagged",
            "unstable_haplotype": "none",
            "dropout_flag": False,
        }]
        _write_instability_tsv(output, loci, results)
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row
        fields = lines[1].strip().split("\t")
        assert len(fields) == 25
        assert fields[0] == "chr4"
        assert fields[1] == "3074877"
        assert fields[22] == "hp-tagged"
        assert fields[23] == "none"
        assert fields[24] == "0"

    def test_none_result(self, tmp_path):
        """None result -> '.' for metric columns, 'failed' for analysis_path."""
        output = str(tmp_path / "out.tsv")
        loci = [("chr1", 100, 200, "AC")]
        results = [None]
        _write_instability_tsv(output, loci, results)
        with open(output) as f:
            lines = f.readlines()
        fields = lines[1].strip().split("\t")
        assert len(fields) == 25
        assert fields[4] == "."  # modal_h1
        assert fields[22] == "failed"  # analysis_path
        assert fields[23] == "."  # unstable_haplotype
        assert fields[24] == "0"  # dropout_flag

    def test_min_ais_filter(self, tmp_path):
        """min_ais filters low-AIS loci."""
        output = str(tmp_path / "out.tsv")
        loci = [
            ("chr1", 100, 200, "AC"),
            ("chr1", 300, 400, "AC"),
        ]
        results = [
            {"modal_h1": 100.0, "modal_h2": 100.0,
             "hii_h1": 0.0, "hii_h2": 0.0,
             "ser_h1": 0.0, "ser_h2": 0.0,
             "scr_h1": 0.0, "scr_h2": 0.0,
             "ecb_h1": 0.0, "ecb_h2": 0.0,
             "ias": 0.0, "ais": 0.1,
             "range_h1": 0.0, "range_h2": 0.0,
             "n_h1": 10, "n_h2": 10, "n_total": 20, "concordance": 0.9,
             "analysis_path": "hp-tagged", "unstable_haplotype": "none",
             "dropout_flag": False},
            {"modal_h1": 100.0, "modal_h2": 120.0,
             "hii_h1": 0.0, "hii_h2": 2.0,
             "ser_h1": 0.0, "ser_h2": 0.5,
             "scr_h1": 0.0, "scr_h2": 0.0,
             "ecb_h1": 0.0, "ecb_h2": 1.0,
             "ias": 1.0, "ais": 1.8,
             "range_h1": 0.0, "range_h2": 10.0,
             "n_h1": 10, "n_h2": 10, "n_total": 20, "concordance": 0.9,
             "analysis_path": "hp-tagged", "unstable_haplotype": "h2",
             "dropout_flag": False},
        ]
        n = _write_instability_tsv(output, loci, results, min_ais=0.5)
        assert n == 1  # only the second locus passes
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row


# ---------------------------------------------------------------------------
# analysis_path field
# ---------------------------------------------------------------------------

class TestAnalysisPath:
    def test_hp_tagged_path(self):
        """HP-tagged reads -> analysis_path = 'hp-tagged'."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["analysis_path"] == "hp-tagged"

    def test_gap_split_path(self):
        """Bimodal reads without HP -> analysis_path = 'gap-split'."""
        reads = [_ri(90.0, 0)] * 5 + [_ri(110.0, 0)] * 5
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["analysis_path"] == "gap-split"

    def test_pooled_path(self):
        """Unimodal reads without HP -> analysis_path = 'pooled'."""
        reads = [_ri(100.0, 0)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["analysis_path"] == "pooled"


# ---------------------------------------------------------------------------
# unstable_haplotype field
# ---------------------------------------------------------------------------

class TestUnstableHaplotype:
    def test_h2_unstable(self):
        """HP=2 has higher HII -> unstable_haplotype = 'h2'."""
        hp1_reads = [_ri(90.0, 1)] * 10
        hp2_reads = [
            _ri(105.0, 2), _ri(110.0, 2), _ri(115.0, 2),
            _ri(120.0, 2), _ri(110.0, 2), _ri(108.0, 2),
            _ri(112.0, 2), _ri(125.0, 2), _ri(110.0, 2),
            _ri(130.0, 2),
        ]
        result = compute_instability(hp1_reads + hp2_reads, 100.0, motif_len=3)
        assert result["unstable_haplotype"] == "h2"

    def test_stable_het_none(self):
        """Both haplotypes equally stable -> unstable_haplotype = 'none'."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["unstable_haplotype"] == "none"

    def test_pooled_h1(self):
        """Pooled fallback -> unstable_haplotype = 'h1'."""
        reads = [_ri(100.0, 0)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["unstable_haplotype"] == "h1"


# ---------------------------------------------------------------------------
# Mononucleotide SER/SCR threshold
# ---------------------------------------------------------------------------

class TestMononucleotideThreshold:
    def test_ser_motif1_noise_immune(self):
        """motif_len=1: reads at median +2bp should NOT count as expanded.

        New threshold = median + 2 = 27. 27 > 27 is False -> SER = 0.
        """
        sizes = np.array([24.0, 25.0, 25.0, 25.0, 27.0])
        assert _ser(sizes, 25.0, motif_len=1) == pytest.approx(0.0)

    def test_scr_motif1_noise_immune(self):
        """motif_len=1: reads at median -2bp should NOT count as contracted."""
        sizes = np.array([23.0, 25.0, 25.0, 25.0, 26.0])
        assert _scr(sizes, 25.0, motif_len=1) == pytest.approx(0.0)

    def test_ser_motif1_real_expansion(self):
        """motif_len=1: reads > median+2bp are genuine expansions."""
        sizes = np.array([25.0, 25.0, 25.0, 25.0, 28.0])
        assert _ser(sizes, 25.0, motif_len=1) == pytest.approx(0.2)

    def test_ser_motif3_unchanged(self):
        """motif_len=3: behavior unchanged (threshold = median + 3bp)."""
        sizes = np.array([100.0, 101.0, 102.0, 110.0])
        assert _ser(sizes, 100.0, motif_len=3) == pytest.approx(0.25)

    def test_scr_motif3_unchanged(self):
        """motif_len=3: behavior unchanged (threshold = median - 3bp)."""
        sizes = np.array([100.0, 99.0, 98.0, 90.0])
        assert _scr(sizes, 100.0, motif_len=3) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Allele dropout detection
# ---------------------------------------------------------------------------

class TestDropoutFlag:
    def test_dropout_suspected(self):
        """Unimodal reads at large locus -> dropout_flag = True."""
        # 15 reads all at 50bp, ref_size=50bp, motif_len=3
        # size_range=0 < 3, ref_size=50 > 30 -> dropout
        reads = [_ri(50.0, 0)] * 15
        assert _detect_dropout(reads, ref_size=50.0, motif_len=3) is True

    def test_no_dropout_small_locus(self):
        """Small locus (ref_size <= 10*motif_len) -> no dropout."""
        reads = [_ri(10.0, 0)] * 15
        assert _detect_dropout(reads, ref_size=20.0, motif_len=3) is False

    def test_no_dropout_bimodal(self):
        """Bimodal reads -> size_range >= motif_len -> no dropout."""
        reads = [_ri(50.0, 0)] * 8 + [_ri(56.0, 0)] * 7
        assert _detect_dropout(reads, ref_size=50.0, motif_len=3) is False

    def test_no_dropout_few_reads(self):
        """< 10 reads -> no dropout flag."""
        reads = [_ri(50.0, 0)] * 5
        assert _detect_dropout(reads, ref_size=50.0, motif_len=3) is False

    def test_dropout_in_compute_instability(self):
        """dropout_flag propagated through compute_instability."""
        # 15 identical reads at large locus -> pooled fallback + dropout
        reads = [_ri(50.0, 0)] * 15
        result = compute_instability(reads, ref_size=50.0, motif_len=3)
        assert result is not None
        assert result["dropout_flag"] is True

    def test_no_dropout_in_compute_instability(self):
        """Normal het locus -> dropout_flag = False."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, ref_size=100.0, motif_len=3)
        assert result["dropout_flag"] is False


# ---------------------------------------------------------------------------
# _check_hp_tags
# ---------------------------------------------------------------------------

def _mock_alignment(hp=None, unmapped=False, secondary=False, duplicate=False):
    """Create a mock pysam alignment."""
    aln = MagicMock()
    aln.is_unmapped = unmapped
    aln.is_secondary = secondary
    aln.is_duplicate = duplicate
    if hp is not None:
        aln.get_tag = MagicMock(return_value=hp)
    else:
        aln.get_tag = MagicMock(side_effect=KeyError("HP"))
    return aln


class TestCheckHpTags:
    def test_no_hp_tags_raises(self):
        """BAM with zero HP tags -> ValueError."""
        mock_bam = MagicMock()
        mock_bam.__iter__ = MagicMock(
            return_value=iter([_mock_alignment(hp=None) for _ in range(20)])
        )
        mock_bam.close = MagicMock()

        with patch("pysam.AlignmentFile", return_value=mock_bam):
            with pytest.raises(ValueError, match="No HP tags found"):
                _check_hp_tags("fake.bam", sample_size=20)

    def test_some_hp_tags_passes(self):
        """BAM with >0% HP tags -> returns fraction."""
        alns = (
            [_mock_alignment(hp=1) for _ in range(6)]
            + [_mock_alignment(hp=2) for _ in range(4)]
            + [_mock_alignment(hp=None) for _ in range(10)]
        )
        mock_bam = MagicMock()
        mock_bam.__iter__ = MagicMock(return_value=iter(alns))
        mock_bam.close = MagicMock()

        with patch("pysam.AlignmentFile", return_value=mock_bam):
            frac = _check_hp_tags("fake.bam", sample_size=20)
        assert frac == pytest.approx(0.5)

    def test_all_hp_tags(self):
        """BAM where all reads have HP tags -> fraction = 1.0."""
        alns = [_mock_alignment(hp=1) for _ in range(10)]
        mock_bam = MagicMock()
        mock_bam.__iter__ = MagicMock(return_value=iter(alns))
        mock_bam.close = MagicMock()

        with patch("pysam.AlignmentFile", return_value=mock_bam):
            frac = _check_hp_tags("fake.bam", sample_size=10)
        assert frac == pytest.approx(1.0)

    def test_skips_unmapped_secondary_duplicate(self):
        """Unmapped, secondary, duplicate reads are skipped."""
        alns = [
            _mock_alignment(hp=None, unmapped=True),
            _mock_alignment(hp=None, secondary=True),
            _mock_alignment(hp=None, duplicate=True),
            _mock_alignment(hp=1),  # only mapped primary non-dup read
        ]
        mock_bam = MagicMock()
        mock_bam.__iter__ = MagicMock(return_value=iter(alns))
        mock_bam.close = MagicMock()

        with patch("pysam.AlignmentFile", return_value=mock_bam):
            frac = _check_hp_tags("fake.bam", sample_size=10)
        assert frac == pytest.approx(1.0)  # 1/1 valid read has HP

    def test_empty_bam_raises(self):
        """BAM with no valid reads -> ValueError (0/0 -> hp_frac=0.0)."""
        mock_bam = MagicMock()
        mock_bam.__iter__ = MagicMock(return_value=iter([]))
        mock_bam.close = MagicMock()

        with patch("pysam.AlignmentFile", return_value=mock_bam):
            with pytest.raises(ValueError, match="No HP tags found"):
                _check_hp_tags("fake.bam")
