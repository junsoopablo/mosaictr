"""Tests for mosaictr.compare module."""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from mosaictr.compare import (
    _max_hii,
    _safe_float,
    build_matrix,
    compare_tissues,
    format_compare_summary,
    format_matrix_summary,
    load_instability_tsv,
    write_compare_tsv,
    write_matrix_tsv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(hii_h1=0.0, hii_h2=0.0, **kwargs):
    """Create a minimal instability row dict."""
    row = {
        "median_h1": 100.0, "median_h2": 100.0,
        "hii_h1": hii_h1, "hii_h2": hii_h2,
        "ser_h1": 0.0, "ser_h2": 0.0,
        "scr_h1": 0.0, "scr_h2": 0.0,
        "ecb_h1": 0.0, "ecb_h2": 0.0,
        "ias": 0.0, "ais": 0.0,
        "range_h1": 0.0, "range_h2": 0.0,
        "n_h1": 10, "n_h2": 10, "n_total": 20,
        "concordance": 0.9,
        "analysis_path": "hp-tagged",
        "unstable_haplotype": "none",
        "dropout_flag": False,
        "n_trimmed_h1": 0, "n_trimmed_h2": 0,
    }
    row.update(kwargs)
    return row


LOCUS_A = ("chr1", 1000, 1010, "CAG")
LOCUS_B = ("chr1", 2000, 2010, "AT")
LOCUS_C = ("chr2", 3000, 3020, "AAGGG")


# ---------------------------------------------------------------------------
# _safe_float / _max_hii
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(1.5) == 1.5

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf")) == 0.0


class TestMaxHii:
    def test_normal(self):
        row = _make_row(hii_h1=1.0, hii_h2=2.0)
        assert _max_hii(row) == 2.0

    def test_nan_h1(self):
        row = _make_row(hii_h1=float("nan"), hii_h2=0.5)
        assert _max_hii(row) == 0.5

    def test_both_nan(self):
        row = _make_row(hii_h1=float("nan"), hii_h2=float("nan"))
        assert _max_hii(row) == 0.0


# ---------------------------------------------------------------------------
# load_instability_tsv
# ---------------------------------------------------------------------------

class TestLoadInstabilityTsv:
    def test_basic_load(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        header = (
            "#chrom\tstart\tend\tmotif\t"
            "median_h1\tmedian_h2\thii_h1\thii_h2\t"
            "ser_h1\tser_h2\tscr_h1\tscr_h2\t"
            "ecb_h1\tecb_h2\tias\tais\t"
            "range_h1\trange_h2\tn_h1\tn_h2\tn_total\tconcordance\t"
            "analysis_path\tunstable_haplotype\tdropout_flag\t"
            "n_trimmed_h1\tn_trimmed_h2\n"
        )
        row = (
            "chr1\t1000\t1010\tCAG\t"
            "100\t100\t0.5\t0.3\t"
            "0.1\t0.1\t0.05\t0.05\t"
            "0.5\t0.5\t0.2\t0.1\t"
            "10\t10\t15\t15\t30\t0.9\t"
            "hp-tagged\tnone\t0\t"
            "1\t2\n"
        )
        tsv.write_text(header + row)
        data = load_instability_tsv(str(tsv))
        assert len(data) == 1
        key = ("chr1", 1000, 1010, "CAG")
        assert key in data
        assert data[key]["hii_h1"] == 0.5
        assert data[key]["n_total"] == 30

    def test_dot_values(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        header = (
            "#chrom\tstart\tend\tmotif\t"
            "median_h1\tmedian_h2\thii_h1\thii_h2\t"
            "ser_h1\tser_h2\tscr_h1\tscr_h2\t"
            "ecb_h1\tecb_h2\tias\tais\t"
            "range_h1\trange_h2\tn_h1\tn_h2\tn_total\tconcordance\t"
            "analysis_path\tunstable_haplotype\tdropout_flag\t"
            "n_trimmed_h1\tn_trimmed_h2\n"
        )
        row = (
            "chr1\t1000\t1010\tCAG\t"
            ".\t.\t.\t.\t"
            ".\t.\t.\t.\t"
            ".\t.\t.\t.\t"
            ".\t.\t0\t0\t0\t0\t"
            "failed\t.\t0\t"
            "0\t0\n"
        )
        tsv.write_text(header + row)
        data = load_instability_tsv(str(tsv))
        key = ("chr1", 1000, 1010, "CAG")
        assert math.isnan(data[key]["hii_h1"])


# ---------------------------------------------------------------------------
# compare_tissues
# ---------------------------------------------------------------------------

class TestCompareTissues:
    def test_tissue_specific(self):
        baseline = {LOCUS_A: _make_row(hii_h1=0.0, hii_h2=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=2.0, hii_h2=0.1)}
        results = compare_tissues(baseline, target, noise_threshold=0.45, min_delta=0.5)
        assert len(results) == 1
        assert results[0]["category"] == "tissue_specific"
        assert results[0]["delta_max_hii"] == 2.0

    def test_both_unstable(self):
        baseline = {LOCUS_A: _make_row(hii_h1=1.0, hii_h2=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=3.0, hii_h2=0.0)}
        results = compare_tissues(baseline, target, noise_threshold=0.45, min_delta=0.5)
        assert len(results) == 1
        assert results[0]["category"] == "both_unstable"
        assert results[0]["delta_max_hii"] == 2.0

    def test_stable(self):
        baseline = {LOCUS_A: _make_row(hii_h1=0.0, hii_h2=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=0.1, hii_h2=0.0)}
        results = compare_tissues(baseline, target, noise_threshold=0.45, min_delta=0.5)
        assert len(results) == 0  # filtered by min_delta

    def test_baseline_only(self):
        baseline = {LOCUS_A: _make_row(hii_h1=2.0, hii_h2=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=0.1, hii_h2=0.0)}
        results = compare_tissues(baseline, target, noise_threshold=0.45, min_delta=0.0)
        bl_only = [r for r in results if r["category"] == "baseline_only"]
        assert len(bl_only) == 1

    def test_non_overlapping_loci(self):
        baseline = {LOCUS_A: _make_row()}
        target = {LOCUS_B: _make_row()}
        results = compare_tissues(baseline, target, min_delta=0.0)
        assert len(results) == 0

    def test_sorted_by_delta(self):
        baseline = {
            LOCUS_A: _make_row(hii_h1=0.0),
            LOCUS_B: _make_row(hii_h1=0.0),
        }
        target = {
            LOCUS_A: _make_row(hii_h1=1.0),
            LOCUS_B: _make_row(hii_h1=5.0),
        }
        results = compare_tissues(baseline, target, min_delta=0.5)
        assert results[0]["delta_max_hii"] > results[1]["delta_max_hii"]

    def test_fold_change(self):
        baseline = {LOCUS_A: _make_row(hii_h1=0.5)}
        target = {LOCUS_A: _make_row(hii_h1=2.5)}
        results = compare_tissues(baseline, target, min_delta=0.0)
        assert results[0]["fold_change"] == pytest.approx(5.0)

    def test_fold_change_zero_baseline(self):
        baseline = {LOCUS_A: _make_row(hii_h1=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=1.0)}
        results = compare_tissues(baseline, target, min_delta=0.0)
        # fold_change = 1.0 / 0.01 = 100
        assert results[0]["fold_change"] == pytest.approx(100.0)

    def test_nan_hii_treated_as_zero(self):
        baseline = {LOCUS_A: _make_row(hii_h1=float("nan"), hii_h2=float("nan"))}
        target = {LOCUS_A: _make_row(hii_h1=1.0)}
        results = compare_tissues(baseline, target, min_delta=0.0)
        assert results[0]["baseline_max_hii"] == 0.0
        assert results[0]["target_max_hii"] == 1.0


# ---------------------------------------------------------------------------
# write_compare_tsv
# ---------------------------------------------------------------------------

class TestWriteCompareTsv:
    def test_roundtrip(self, tmp_path):
        baseline = {LOCUS_A: _make_row(hii_h1=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=2.0)}
        results = compare_tissues(baseline, target, min_delta=0.0)

        out = tmp_path / "compare.tsv"
        n = write_compare_tsv(str(out), results)
        assert n == 1

        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert len(lines) == 2
        cols = lines[1].split("\t")
        assert cols[0] == "chr1"


# ---------------------------------------------------------------------------
# format_compare_summary
# ---------------------------------------------------------------------------

class TestFormatCompareSummary:
    def test_basic(self):
        baseline = {LOCUS_A: _make_row(hii_h1=0.0)}
        target = {LOCUS_A: _make_row(hii_h1=2.0)}
        results = compare_tissues(baseline, target, min_delta=0.0)
        summary = format_compare_summary(results, 1, "blood", "colon")
        assert "blood vs colon" in summary
        assert "tissue_specific" in summary


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------

class TestBuildMatrix:
    def test_basic(self):
        samples = {
            "blood": {LOCUS_A: _make_row(hii_h1=0.1), LOCUS_B: _make_row(hii_h1=0.2)},
            "colon": {LOCUS_A: _make_row(hii_h1=2.0), LOCUS_B: _make_row(hii_h1=0.1)},
        }
        loci, names, matrix, stats = build_matrix(samples)
        assert len(loci) == 2
        assert names == ["blood", "colon"]
        assert len(matrix) == 2

    def test_common_loci_only(self):
        samples = {
            "blood": {LOCUS_A: _make_row(), LOCUS_B: _make_row()},
            "colon": {LOCUS_A: _make_row(), LOCUS_C: _make_row()},
        }
        loci, names, matrix, stats = build_matrix(samples)
        assert len(loci) == 1
        assert loci[0] == LOCUS_A

    def test_categories(self):
        samples = {
            "blood": {
                LOCUS_A: _make_row(hii_h1=0.0),  # stable in blood
                LOCUS_B: _make_row(hii_h1=1.0),  # unstable in blood
                LOCUS_C: _make_row(hii_h1=0.0),  # stable in blood
            },
            "colon": {
                LOCUS_A: _make_row(hii_h1=0.0),  # stable in colon → stable
                LOCUS_B: _make_row(hii_h1=2.0),  # unstable in colon → constitutive
                LOCUS_C: _make_row(hii_h1=3.0),  # unstable in colon → tissue_variable
            },
        }
        loci, names, matrix, stats = build_matrix(samples, noise_threshold=0.45)
        cat_map = {loci[i]: stats[i]["category"] for i in range(len(loci))}
        assert cat_map[LOCUS_A] == "stable"
        assert cat_map[LOCUS_B] == "constitutive"
        assert cat_map[LOCUS_C] == "tissue_variable"

    def test_empty_samples(self):
        loci, names, matrix, stats = build_matrix({})
        assert loci == []

    def test_tissue_max(self):
        samples = {
            "blood": {LOCUS_A: _make_row(hii_h1=0.1)},
            "colon": {LOCUS_A: _make_row(hii_h1=5.0)},
            "brain": {LOCUS_A: _make_row(hii_h1=2.0)},
        }
        loci, names, matrix, stats = build_matrix(samples)
        assert stats[0]["tissue_max"] == "colon"
        assert stats[0]["max_hii"] == 5.0

    def test_sd_calculation(self):
        samples = {
            "s1": {LOCUS_A: _make_row(hii_h1=1.0)},
            "s2": {LOCUS_A: _make_row(hii_h1=3.0)},
        }
        loci, names, matrix, stats = build_matrix(samples)
        # mean=2, sd=sqrt((1+1)/2)=1
        assert stats[0]["mean_hii"] == pytest.approx(2.0)
        assert stats[0]["sd_hii"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# write_matrix_tsv
# ---------------------------------------------------------------------------

class TestWriteMatrixTsv:
    def test_roundtrip(self, tmp_path):
        samples = {
            "blood": {LOCUS_A: _make_row(hii_h1=0.1)},
            "colon": {LOCUS_A: _make_row(hii_h1=2.0)},
        }
        loci, names, matrix, stats = build_matrix(samples)
        out = tmp_path / "matrix.tsv"
        n = write_matrix_tsv(str(out), loci, names, matrix, stats)
        assert n == 1

        lines = out.read_text().strip().split("\n")
        assert "hii_blood" in lines[0]
        assert "hii_colon" in lines[0]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# format_matrix_summary
# ---------------------------------------------------------------------------

class TestFormatMatrixSummary:
    def test_basic(self):
        samples = {
            "blood": {LOCUS_A: _make_row(hii_h1=0.0), LOCUS_B: _make_row(hii_h1=1.0)},
            "colon": {LOCUS_A: _make_row(hii_h1=3.0), LOCUS_B: _make_row(hii_h1=1.0)},
        }
        loci, names, matrix, stats = build_matrix(samples)
        summary = format_matrix_summary(loci, names, stats)
        assert "Multi-Tissue" in summary
        assert "stable" in summary
