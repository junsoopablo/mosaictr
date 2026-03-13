"""Tests for mosaictr.utils module."""

from __future__ import annotations

import pytest

from mosaictr.utils import (
    Tier1Locus,
    chrom_split,
    load_adotto_catalog,
    load_loci_bed,
    load_tier1_bed,
    match_tier1_to_catalog,
)


# ---------------------------------------------------------------------------
# load_loci_bed
# ---------------------------------------------------------------------------

class TestLoadLociBed:
    def test_normal_loading(self, tmp_path):
        bed = tmp_path / "loci.bed"
        bed.write_text("chr1\t100\t200\tAC\nchr1\t300\t400\tCAG\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 2
        assert loci[0] == ("chr1", 100, 200, "AC")
        assert loci[1] == ("chr1", 300, 400, "CAG")

    def test_empty_file(self, tmp_path):
        bed = tmp_path / "empty.bed"
        bed.write_text("")
        assert load_loci_bed(str(bed)) == []

    def test_skips_short_lines(self, tmp_path):
        bed = tmp_path / "short.bed"
        bed.write_text("chr1\t100\t200\nfoo\nchr1\t300\t400\tCAG\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 1

    def test_skips_comments(self, tmp_path):
        bed = tmp_path / "comments.bed"
        bed.write_text("#header\nchr1\t100\t200\tAC\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 1

    def test_chroms_filter(self, tmp_path):
        bed = tmp_path / "multi.bed"
        bed.write_text("chr1\t100\t200\tAC\nchr2\t100\t200\tAC\nchr3\t100\t200\tAC\n")
        loci = load_loci_bed(str(bed), chroms={"chr1", "chr3"})
        assert len(loci) == 2
        assert all(c in ("chr1", "chr3") for c, *_ in loci)

    def test_skips_invalid_coordinates(self, tmp_path):
        bed = tmp_path / "invalid.bed"
        bed.write_text("chr1\tXXX\t200\tAC\nchr1\t100\t200\tAC\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 1

    def test_skips_start_equals_end(self, tmp_path):
        """start >= end should be filtered out."""
        bed = tmp_path / "degenerate.bed"
        bed.write_text("chr1\t200\t200\tAC\nchr1\t300\t200\tAC\nchr1\t100\t200\tAC\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 1
        assert loci[0] == ("chr1", 100, 200, "AC")

    def test_skips_empty_motif(self, tmp_path):
        """Empty motif field should be filtered."""
        bed = tmp_path / "emptymotif.bed"
        bed.write_text("chr1\t100\t200\t\nchr1\t300\t400\tCAG\n")
        loci = load_loci_bed(str(bed))
        assert len(loci) == 1


# ---------------------------------------------------------------------------
# load_tier1_bed
# ---------------------------------------------------------------------------

class TestTier1LocusProperties:
    def test_is_variant_tp(self):
        locus = Tier1Locus("chr1", 100, 200, "Tier1", "TP_TP_TP", 3, 1.0, 5, -3)
        assert locus.is_variant is True

    def test_is_variant_tn(self):
        locus = Tier1Locus("chr1", 100, 200, "Tier1", "TN_TN_TN", 3, 1.0, 0, 0)
        assert locus.is_variant is False

    def test_is_het_different_diffs(self):
        locus = Tier1Locus("chr1", 100, 200, "Tier1", "TP_TP_TP", 3, 1.0, 5, -3)
        assert locus.is_het is True

    def test_is_het_same_nonzero(self):
        """Same hap diffs (homozygous alt) -> not HET."""
        locus = Tier1Locus("chr1", 100, 200, "Tier1", "TP_TP_TP", 3, 1.0, 5, 5)
        assert locus.is_het is False

    def test_is_het_both_zero(self):
        """Both zero (homozygous ref) -> not HET."""
        locus = Tier1Locus("chr1", 100, 200, "Tier1", "TN_TN_TN", 3, 1.0, 0, 0)
        assert locus.is_het is False


class TestLoadTier1Bed:
    def test_normal_loading(self, tmp_path):
        bed = tmp_path / "tier1.bed"
        bed.write_text(
            "chr1\t100\t200\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
            "chr1\t300\t400\tTier1\tTN_TN_TN\t2\t0.5\t0\t0\n"
        )
        loci = load_tier1_bed(str(bed))
        assert len(loci) == 2
        assert loci[0].chrom == "chr1"
        assert loci[0].hap1_diff_bp == 5
        assert loci[0].is_variant is True
        assert loci[1].is_variant is False

    def test_chroms_filter(self, tmp_path):
        bed = tmp_path / "tier1.bed"
        bed.write_text(
            "chr1\t100\t200\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
            "chr2\t100\t200\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
        )
        loci = load_tier1_bed(str(bed), chroms={"chr1"})
        assert len(loci) == 1
        assert loci[0].chrom == "chr1"

    def test_skips_short_lines(self, tmp_path):
        bed = tmp_path / "tier1_short.bed"
        bed.write_text(
            "chr1\t100\t200\n"
            "chr1\t300\t400\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
        )
        loci = load_tier1_bed(str(bed))
        assert len(loci) == 1

    def test_skips_invalid_values(self, tmp_path):
        bed = tmp_path / "tier1_bad.bed"
        bed.write_text(
            "chr1\tXXX\t200\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
            "chr1\t100\t200\tTier1\tTP_TP_TP\t3\t1.0\t5\t-3\n"
        )
        loci = load_tier1_bed(str(bed))
        assert len(loci) == 1


# ---------------------------------------------------------------------------
# load_adotto_catalog
# ---------------------------------------------------------------------------

class TestLoadAdottoCatalog:
    def test_normal_loading(self, tmp_path):
        bed = tmp_path / "catalog.bed"
        bed.write_text("chr1\t100\t200\tAC\nchr2\t300\t400\tCAG\n")
        catalog = load_adotto_catalog(str(bed))
        assert len(catalog) == 2
        assert catalog[("chr1", 100, 200)] == "AC"

    def test_chroms_filter(self, tmp_path):
        bed = tmp_path / "catalog.bed"
        bed.write_text("chr1\t100\t200\tAC\nchr2\t300\t400\tCAG\n")
        catalog = load_adotto_catalog(str(bed), chroms={"chr1"})
        assert len(catalog) == 1
        assert ("chr2", 300, 400) not in catalog


# ---------------------------------------------------------------------------
# match_tier1_to_catalog
# ---------------------------------------------------------------------------

class TestMatchTier1ToCatalog:
    def _make_locus(self, chrom="chr1", start=100, end=200):
        return Tier1Locus(
            chrom=chrom, start=start, end=end,
            tier="Tier1", tp_status="TP_TP_TP",
            col6=3, col7=1.0, hap1_diff_bp=5, hap2_diff_bp=-3,
        )

    def test_exact_match(self):
        loci = [self._make_locus()]
        catalog = {("chr1", 100, 200): "AC"}
        matched = match_tier1_to_catalog(loci, catalog, tolerance=0)
        assert len(matched) == 1
        assert matched[0][1] == "AC"

    def test_no_match(self):
        loci = [self._make_locus()]
        catalog = {("chr1", 500, 600): "AC"}
        matched = match_tier1_to_catalog(loci, catalog, tolerance=0)
        assert len(matched) == 0

    def test_approximate_match(self):
        loci = [self._make_locus(start=105, end=205)]
        catalog = {("chr1", 100, 200): "AC"}
        matched = match_tier1_to_catalog(loci, catalog, tolerance=10)
        assert len(matched) == 1


# ---------------------------------------------------------------------------
# chrom_split
# ---------------------------------------------------------------------------

class TestChromSplit:
    def test_train(self):
        assert chrom_split("chr1") == "train"
        assert chrom_split("chr18") == "train"

    def test_val(self):
        assert chrom_split("chr19") == "val"
        assert chrom_split("chr20") == "val"

    def test_test(self):
        assert chrom_split("chr21") == "test"
        assert chrom_split("chrX") == "test"

    def test_other(self):
        assert chrom_split("chrY") == "other"
        assert chrom_split("chrM") == "other"
