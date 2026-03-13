"""Tests for mosaictr.visualization module (smoke tests)."""

from __future__ import annotations

import os

import pytest

from mosaictr.genotype import ReadInfo


# ---------------------------------------------------------------------------
# waterfall_plot
# ---------------------------------------------------------------------------

class TestWaterfallPlot:
    def test_empty_reads_raises(self, tmp_path):
        from mosaictr.visualization import waterfall_plot
        with pytest.raises(ValueError, match="empty"):
            waterfall_plot([], 100.0, 3, str(tmp_path / "out.png"))

    def test_normal_execution(self, tmp_path):
        from mosaictr.visualization import waterfall_plot
        reads = [
            ReadInfo(95.0, 1, 60),
            ReadInfo(100.0, 1, 55),
            ReadInfo(105.0, 2, 60),
            ReadInfo(110.0, 2, 50),
            ReadInfo(102.0, 0, 40),
        ]
        out = str(tmp_path / "waterfall.png")
        result = waterfall_plot(reads, 100.0, 3, out)
        assert result == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# allele_histogram
# ---------------------------------------------------------------------------

    def test_show_units_mode(self, tmp_path):
        from mosaictr.visualization import waterfall_plot
        reads = [
            ReadInfo(90.0, 1, 60),
            ReadInfo(93.0, 1, 55),
            ReadInfo(96.0, 2, 60),
        ]
        out = str(tmp_path / "waterfall_units.png")
        result = waterfall_plot(reads, 90.0, 3, out, show_units=True)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_single_read(self, tmp_path):
        """Single read should not crash (edge case for size_range=0)."""
        from mosaictr.visualization import waterfall_plot
        reads = [ReadInfo(100.0, 1, 60)]
        out = str(tmp_path / "single.png")
        waterfall_plot(reads, 100.0, 3, out)
        assert os.path.isfile(out)

    def test_all_hp0_reads(self, tmp_path):
        """Only HP=0 (unphased) reads should work."""
        from mosaictr.visualization import waterfall_plot
        reads = [ReadInfo(100.0, 0, 60), ReadInfo(105.0, 0, 55)]
        out = str(tmp_path / "hp0.png")
        waterfall_plot(reads, 100.0, 3, out)
        assert os.path.isfile(out)


class TestAlleleHistogram:
    def test_empty_reads_raises(self, tmp_path):
        from mosaictr.visualization import allele_histogram
        with pytest.raises(ValueError, match="empty"):
            allele_histogram([], 100.0, 3, str(tmp_path / "out.png"))

    def test_normal_execution(self, tmp_path):
        from mosaictr.visualization import allele_histogram
        reads = [
            ReadInfo(95.0, 1, 60),
            ReadInfo(100.0, 1, 55),
            ReadInfo(105.0, 2, 60),
            ReadInfo(110.0, 2, 50),
        ]
        out = str(tmp_path / "hist.png")
        result = allele_histogram(reads, 100.0, 3, out)
        assert result == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_show_units_mode(self, tmp_path):
        from mosaictr.visualization import allele_histogram
        reads = [
            ReadInfo(90.0, 1, 60),
            ReadInfo(96.0, 2, 60),
        ]
        out = str(tmp_path / "hist_units.png")
        allele_histogram(reads, 90.0, 3, out, show_units=True)
        assert os.path.isfile(out)

    def test_vntr_auto_binning(self, tmp_path):
        """VNTR (motif_len > 6) should use auto bin width."""
        from mosaictr.visualization import allele_histogram
        reads = [
            ReadInfo(500.0, 1, 60),
            ReadInfo(520.0, 1, 55),
            ReadInfo(540.0, 2, 60),
        ]
        out = str(tmp_path / "vntr_hist.png")
        allele_histogram(reads, 500.0, 10, out)
        assert os.path.isfile(out)

    def test_with_hii_annotation(self, tmp_path):
        """HII values should be annotated without crash."""
        from mosaictr.visualization import allele_histogram
        reads = [
            ReadInfo(95.0, 1, 60),
            ReadInfo(105.0, 2, 60),
        ]
        out = str(tmp_path / "hii.png")
        allele_histogram(reads, 100.0, 3, out, hii_h1=0.1, hii_h2=0.5)
        assert os.path.isfile(out)


# ---------------------------------------------------------------------------
# instability_summary_plot
# ---------------------------------------------------------------------------

class TestInstabilitySummaryPlot:
    def test_empty_reads_raises(self, tmp_path):
        from mosaictr.visualization import instability_summary_plot
        with pytest.raises(ValueError, match="empty"):
            instability_summary_plot([], 100.0, 3, {}, str(tmp_path / "out.png"))

    def test_normal_execution(self, tmp_path):
        from mosaictr.visualization import instability_summary_plot
        reads = [
            ReadInfo(95.0, 1, 60),
            ReadInfo(100.0, 1, 55),
            ReadInfo(105.0, 2, 60),
            ReadInfo(110.0, 2, 50),
        ]
        inst_result = {
            "hii_h1": 0.1, "hii_h2": 0.5,
            "ias": 0.3,
            "analysis_path": "hp-tagged",
        }
        out = str(tmp_path / "summary.png")
        result = instability_summary_plot(reads, 100.0, 3, inst_result, out)
        assert result == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0
