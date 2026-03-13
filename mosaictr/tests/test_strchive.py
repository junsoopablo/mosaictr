"""Tests for mosaictr.strchive module."""

from __future__ import annotations

import pytest

from mosaictr.strchive import (
    PATHOGENIC_CATALOG,
    annotate_locus,
    annotate_results,
    classify_allele,
    get_pathogenic_bed,
    get_pathogenic_loci,
    _classify_status,
)


# ---------------------------------------------------------------------------
# get_pathogenic_loci / get_pathogenic_bed
# ---------------------------------------------------------------------------

class TestGetPathogenicLoci:
    def test_returns_nonempty_list(self):
        loci = get_pathogenic_loci()
        assert len(loci) > 0

    def test_required_keys(self):
        required = {"gene", "disease", "chrom", "start", "end", "motif",
                     "normal_max", "pathogenic_min", "inheritance"}
        for entry in get_pathogenic_loci():
            assert required.issubset(entry.keys()), f"Missing keys in {entry['gene']}"

    def test_returns_defensive_copy(self):
        loci1 = get_pathogenic_loci()
        loci1[0]["gene"] = "MODIFIED"
        loci2 = get_pathogenic_loci()
        assert loci2[0]["gene"] != "MODIFIED"


class TestGetPathogenicBed:
    def test_returns_tuples(self):
        bed = get_pathogenic_bed()
        assert len(bed) == len(PATHOGENIC_CATALOG)
        for chrom, start, end, motif in bed:
            assert isinstance(chrom, str)
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(motif, str)
            assert start < end


# ---------------------------------------------------------------------------
# annotate_locus
# ---------------------------------------------------------------------------

class TestAnnotateLocus:
    def test_exact_match_htt(self):
        """HTT locus should match with exact coordinates."""
        result = annotate_locus("chr4", 3074876, 3074933, "CAG")
        assert result is not None
        assert result["gene"] == "HTT"

    def test_tolerance_match(self):
        """Match within tolerance window."""
        result = annotate_locus("chr4", 3074876 + 30, 3074933 + 30, "CAG", tolerance=50)
        assert result is not None
        assert result["gene"] == "HTT"

    def test_no_match_wrong_chrom(self):
        result = annotate_locus("chr1", 3074876, 3074933, "CAG")
        assert result is None

    def test_no_match_far_coordinates(self):
        result = annotate_locus("chr4", 1000000, 1000100, "CAG")
        assert result is None

    def test_motif_mismatch(self):
        """Exact coordinates but wrong motif should not match."""
        result = annotate_locus("chr4", 3074876, 3074933, "CTG")
        assert result is None

    def test_no_motif_filter(self):
        """Without motif filter, coordinates alone should match."""
        result = annotate_locus("chr4", 3074876, 3074933)
        assert result is not None
        assert result["gene"] == "HTT"


# ---------------------------------------------------------------------------
# classify_allele
# ---------------------------------------------------------------------------

class TestClassifyAllele:
    def test_normal(self):
        # HTT: normal_max=35, motif=CAG(3bp). 35*3=105bp is normal
        assert classify_allele(105.0, 57.0, 3, 35, 40) == "normal"

    def test_pathogenic(self):
        # HTT: pathogenic_min=40, 40*3=120bp is pathogenic
        assert classify_allele(120.0, 57.0, 3, 35, 40) == "pathogenic"

    def test_intermediate(self):
        # Between normal_max and pathogenic_min: 36-39 repeats
        assert classify_allele(111.0, 57.0, 3, 35, 40) == "intermediate"

    def test_boundary_normal_max(self):
        # Exactly at normal_max boundary
        assert classify_allele(35 * 3.0, 57.0, 3, 35, 40) == "normal"

    def test_boundary_pathogenic_min(self):
        # Exactly at pathogenic_min boundary
        assert classify_allele(40 * 3.0, 57.0, 3, 35, 40) == "pathogenic"

    def test_zero_motif_len(self):
        assert classify_allele(100.0, 57.0, 0, 35, 40) == "normal"


# ---------------------------------------------------------------------------
# _classify_status
# ---------------------------------------------------------------------------

class TestClassifyStatus:
    def test_both_normal(self):
        assert _classify_status(90.0, 90.0, 57.0, 3, 35, 40) == "normal"

    def test_one_pathogenic(self):
        assert _classify_status(90.0, 120.0, 57.0, 3, 35, 40) == "pathogenic"

    def test_one_intermediate(self):
        assert _classify_status(90.0, 111.0, 57.0, 3, 35, 40) == "intermediate"

    def test_pathogenic_overrides_intermediate(self):
        assert _classify_status(111.0, 120.0, 57.0, 3, 35, 40) == "pathogenic"


# ---------------------------------------------------------------------------
# annotate_results
# ---------------------------------------------------------------------------

class TestAnnotateResults:
    def test_matching_locus_annotated(self):
        loci = [("chr4", 3074876, 3074933, "CAG")]
        results = [{"allele1_size": 90.0, "allele2_size": 90.0, "zygosity": "HOM"}]
        annotated = annotate_results(loci, results)
        assert len(annotated) == 1
        assert "pathogenic_annotation" in annotated[0]
        assert annotated[0]["pathogenic_annotation"]["gene"] == "HTT"
        assert annotated[0]["pathogenic_annotation"]["status"] == "normal"

    def test_none_result_still_annotates(self):
        loci = [("chr4", 3074876, 3074933, "CAG")]
        results = [None]
        annotated = annotate_results(loci, results)
        assert "pathogenic_annotation" in annotated[0]
        assert annotated[0]["pathogenic_annotation"]["allele1_status"] is None

    def test_non_pathogenic_locus_not_annotated(self):
        loci = [("chr1", 1000, 2000, "AC")]
        results = [{"allele1_size": 1000.0, "allele2_size": 1000.0}]
        annotated = annotate_results(loci, results)
        assert "pathogenic_annotation" not in annotated[0]

    def test_length_mismatch_raises(self):
        loci = [("chr1", 100, 200, "AC")]
        results = [None, None]
        with pytest.raises(ValueError, match="same length"):
            annotate_results(loci, results)
