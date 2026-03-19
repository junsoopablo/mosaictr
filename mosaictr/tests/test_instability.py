"""Tests for MosaicTR somatic instability module."""

import numpy as np
import pytest

from mosaictr.genotype import ReadInfo
from unittest.mock import MagicMock, patch

from mosaictr.instability import (
    _check_hp_tags,
    _detect_platform,
    _expected_noise_aad,
    _hii,
    _ias,
    _trim_outliers_mad,
    _weighted_mad,
    _write_instability_tsv,
    compute_instability,
    noise_threshold,
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
# _trim_outliers_mad
# ---------------------------------------------------------------------------

class TestTrimOutliersMad:
    def test_no_outliers(self):
        """All values close -> nothing trimmed."""
        sizes = np.array([100.0, 101.0, 99.0, 100.0, 100.0])
        weights = np.ones(5)
        trimmed_s, trimmed_w = _trim_outliers_mad(sizes, weights)
        assert len(trimmed_s) == 5

    def test_outlier_removed(self):
        """Extreme outlier in spread data -> trimmed out."""
        # Values with some spread (MAD > 0) plus one extreme outlier
        sizes = np.array([98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 250.0])
        weights = np.ones(7)
        trimmed_s, trimmed_w = _trim_outliers_mad(sizes, weights)
        assert len(trimmed_s) < 7
        assert 250.0 not in trimmed_s

    def test_too_few_reads(self):
        """< 4 reads -> no trimming."""
        sizes = np.array([100.0, 200.0, 300.0])
        weights = np.ones(3)
        trimmed_s, trimmed_w = _trim_outliers_mad(sizes, weights)
        assert len(trimmed_s) == 3

    def test_all_identical(self):
        """All identical -> MAD=0, no trimming."""
        sizes = np.array([50.0] * 10)
        weights = np.ones(10)
        trimmed_s, trimmed_w = _trim_outliers_mad(sizes, weights)
        assert len(trimmed_s) == 10


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
        """Same deviations, longer motif -> lower HII.

        Note: exact ratio depends on motif-unit weighting. With deviations
        of 6bp, both motif_len=3 and motif_len=6 yield whole-motif weights,
        so pure 2x ratio holds.
        """
        # Use 6bp deviations: whole-motif for both motif_len=3 and 6
        sizes = np.array([94.0, 100.0, 106.0])
        wts = np.ones(3)
        hii_3 = _hii(sizes, wts, motif_len=3)
        hii_6 = _hii(sizes, wts, motif_len=6)
        assert hii_3 == pytest.approx(2 * hii_6)

    def test_single_read(self):
        """< 2 reads -> 0."""
        assert _hii(np.array([100.0]), np.ones(1), motif_len=3) == 0.0

    def test_zero_motif(self):
        assert _hii(np.array([100.0, 103.0]), np.ones(2), motif_len=0) == 0.0

    def test_whole_motif_deviations_full_weight(self):
        """Deviations that are multiples of motif_len get full weight."""
        # All deviations are 3bp (= 1 motif unit for motif_len=3)
        # Median = 100, deviations = [3, 0, 3] -> AAD = 2.0 -> HII = 2/3
        sizes = np.array([97.0, 100.0, 103.0])
        wts = np.ones(3)
        hii = _hii(sizes, wts, motif_len=3)
        assert hii == pytest.approx(2.0 / 3.0)

    def test_sub_motif_deviations_downweighted(self):
        """Sub-motif deviations (likely HiFi error) are down-weighted."""
        # All deviations are 1bp (sub-motif for motif_len=3)
        # Without weighting: AAD = 2/3 -> HII = 2/9
        # With weighting (w_sub=0.1): HII much lower
        sizes = np.array([99.0, 100.0, 101.0])
        wts = np.ones(3)
        hii = _hii(sizes, wts, motif_len=3)
        # Sub-motif weighted down -> HII < unweighted value of 2/9
        assert hii < 2.0 / 9.0

    def test_mixed_motif_and_submotif(self):
        """Mixed whole-motif and sub-motif deviations -> intermediate HII."""
        # One whole-motif deviation (+3), one sub-motif (+1), median=100
        sizes = np.array([97.0, 100.0, 100.0, 100.0, 101.0])
        wts = np.ones(5)
        hii = _hii(sizes, wts, motif_len=3)
        # Should be between pure sub-motif and pure whole-motif HII
        assert hii > 0.0


class TestExpectedNoiseAad:
    def test_zero_allele(self):
        assert _expected_noise_aad(0.0) == 0.0

    def test_negative_allele(self):
        assert _expected_noise_aad(-10.0) == 0.0

    def test_positive_allele(self):
        """Noise increases with allele size."""
        noise_100 = _expected_noise_aad(100.0)
        noise_1000 = _expected_noise_aad(1000.0)
        assert noise_100 > 0.0
        assert noise_1000 > noise_100

    def test_approximate_values(self):
        """Check calibration: AAD ~ 0.003 * size^0.92."""
        # At 100bp: ~0.003 * 100^0.92 ≈ 0.003 * 69.2 ≈ 0.21
        noise = _expected_noise_aad(100.0)
        assert 0.1 < noise < 0.4

    def test_ont_higher_than_hifi(self):
        """ONT noise model produces higher values than HiFi."""
        for size in [50, 100, 500, 1000]:
            ont = _expected_noise_aad(float(size), platform="ont")
            hifi = _expected_noise_aad(float(size), platform="hifi")
            assert ont > hifi, f"ONT noise should exceed HiFi at {size}bp"

    def test_ont_approximate_ratio(self):
        """ONT noise is ~12x higher than HiFi."""
        ratio = _expected_noise_aad(100.0, "ont") / _expected_noise_aad(100.0, "hifi")
        assert 5 < ratio < 20


class TestNoiseThreshold:
    def test_hifi_global(self):
        """HiFi uses a single threshold regardless of motif length."""
        assert noise_threshold(2, "hifi") == 0.45
        assert noise_threshold(5, "hifi") == 0.45
        assert noise_threshold(10, "hifi") == 0.45

    def test_ont_motif_dependent(self):
        """ONT thresholds decrease with motif length."""
        assert noise_threshold(2, "ont") > noise_threshold(6, "ont")
        assert noise_threshold(6, "ont") > noise_threshold(10, "ont")

    def test_ont_short_motif_higher(self):
        """ONT threshold for dinucleotides is much higher than HiFi."""
        assert noise_threshold(2, "ont") > noise_threshold(2, "hifi")

    def test_ont_long_motif_converges(self):
        """ONT threshold for long motifs approaches HiFi level."""
        assert noise_threshold(15, "ont") < 1.0


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
        assert result["analysis_path"] == "gap-split"
        assert result["n_h1"] + result["n_h2"] == 10

    def test_insufficient_hp_pooled_fallback(self):
        """Unimodal reads with no HP tags -> pooled fallback."""
        reads = [_ri(100.0, 0)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result is not None
        assert result["analysis_path"] == "pooled"
        assert result["n_h2"] == 0

    def test_all_metrics_present(self):
        """All 9 keys present in output."""
        reads = [_ri(90.0, 1)] * 10 + [_ri(110.0, 2)] * 10
        result = compute_instability(reads, 100.0, motif_len=3)
        expected_keys = {
            "median_h1", "median_h2", "hii_h1", "hii_h2",
            "ias", "n_h1", "n_h2",
            "n_total", "analysis_path",
        }
        assert set(result.keys()) == expected_keys

    def test_hii_range_nonnegative(self):
        """HII values should always be non-negative."""
        reads = [_ri(90.0 + i * 0.5, 1) for i in range(10)] + \
                [_ri(110.0 + i * 0.5, 2) for i in range(10)]
        result = compute_instability(reads, 100.0, motif_len=3)
        assert result["hii_h1"] >= 0.0
        assert result["hii_h2"] >= 0.0



# ---------------------------------------------------------------------------
# _write_instability_tsv
# ---------------------------------------------------------------------------

class TestWriteInstabilityTsv:
    def test_header_columns(self, tmp_path):
        """Header has 13 columns."""
        output = str(tmp_path / "out.tsv")
        _write_instability_tsv(output, [], [])
        with open(output) as f:
            header = f.readline().strip()
        cols = header.split("\t")
        assert len(cols) == 13
        assert cols[0] == "#chrom"
        assert cols[-1] == "analysis_path"

    def test_data_row(self, tmp_path):
        """Data row has 13 columns with correct values."""
        output = str(tmp_path / "out.tsv")
        loci = [("chr4", 3074877, 3074933, "CAG")]
        results = [{
            "median_h1": 56.0, "median_h2": 56.0,
            "hii_h1": 0.0, "hii_h2": 0.0,
            "ias": 0.0,
            "n_h1": 15, "n_h2": 13,
            "n_total": 28,
            "analysis_path": "hp-tagged",
        }]
        _write_instability_tsv(output, loci, results)
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row
        fields = lines[1].strip().split("\t")
        assert len(fields) == 13
        assert fields[0] == "chr4"
        assert fields[1] == "3074877"
        assert fields[12] == "hp-tagged"

    def test_none_result(self, tmp_path):
        """None result -> '.' for metric columns, 'failed' for analysis_path."""
        output = str(tmp_path / "out.tsv")
        loci = [("chr1", 100, 200, "AC")]
        results = [None]
        _write_instability_tsv(output, loci, results)
        with open(output) as f:
            lines = f.readlines()
        fields = lines[1].strip().split("\t")
        assert len(fields) == 13
        assert fields[4] == "."  # median_h1
        assert fields[12] == "failed"  # analysis_path

    def test_min_hii_filter(self, tmp_path):
        """min_hii filters low-HII loci."""
        output = str(tmp_path / "out.tsv")
        loci = [
            ("chr1", 100, 200, "AC"),
            ("chr1", 300, 400, "AC"),
        ]
        results = [
            {"median_h1": 100.0, "median_h2": 100.0,
             "hii_h1": 0.0, "hii_h2": 0.0,
             "ias": 0.0,
             "n_h1": 10, "n_h2": 10, "n_total": 20,
             "analysis_path": "hp-tagged"},
            {"median_h1": 100.0, "median_h2": 120.0,
             "hii_h1": 0.0, "hii_h2": 2.0,
             "ias": 1.0,
             "n_h1": 10, "n_h2": 10, "n_total": 20,
             "analysis_path": "hp-tagged"},
        ]
        n = _write_instability_tsv(output, loci, results, min_hii=0.5)
        assert n == 1  # only the second locus passes
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row

    def test_nan_inf_values_produce_dots(self, tmp_path):
        """NaN/Inf in metric values should produce '.' not crash."""
        output = str(tmp_path / "out.tsv")
        loci = [("chr1", 100, 200, "AC")]
        results = [{
            "median_h1": float("nan"), "median_h2": float("inf"),
            "hii_h1": float("nan"), "hii_h2": 0.0,
            "ias": float("nan"),
            "n_h1": 10, "n_h2": 10,
            "n_total": 20,
            "analysis_path": "hp-tagged",
        }]
        n = _write_instability_tsv(output, loci, results)
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        row = data[0].lower()
        assert "nan" not in row
        assert "inf" not in row


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
