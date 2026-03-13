"""Tests for mosaictr.vcf_output module."""

from __future__ import annotations

import pytest

from mosaictr.vcf_output import (
    _collect_contigs,
    _encode_gt,
    _fmt_float,
    _fmt_size,
    write_genotype_vcf,
    write_instability_vcf,
)


# ---------------------------------------------------------------------------
# Helper formatting
# ---------------------------------------------------------------------------

class TestFmtFloat:
    def test_integer_value(self):
        assert _fmt_float(3.0) == "3"

    def test_fractional_value(self):
        assert _fmt_float(0.1234) == "0.1234"

    def test_zero(self):
        assert _fmt_float(0.0) == "0"

    def test_negative_integer(self):
        assert _fmt_float(-5.0) == "-5"

    def test_custom_decimals(self):
        assert _fmt_float(0.123456, decimals=2) == "0.12"

    def test_nan(self):
        assert _fmt_float(float("nan")) == "."

    def test_inf(self):
        assert _fmt_float(float("inf")) == "."

    def test_neg_inf(self):
        assert _fmt_float(float("-inf")) == "."


class TestFmtSize:
    def test_integer_size(self):
        assert _fmt_size(100.0) == "100"

    def test_fractional_size(self):
        assert _fmt_size(100.5) == "100.5"

    def test_nan(self):
        assert _fmt_size(float("nan")) == "."

    def test_inf(self):
        assert _fmt_size(float("inf")) == "."


class TestCollectContigs:
    def test_preserves_order(self):
        loci = [("chr2", 0, 10, "AC"), ("chr1", 0, 10, "AC"), ("chr2", 20, 30, "AC")]
        assert _collect_contigs(loci) == ["chr2", "chr1"]

    def test_empty_input(self):
        assert _collect_contigs([]) == []


class TestEncodeGt:
    def test_het(self):
        assert _encode_gt(90.0, 110.0, 100, "HET") == "0/1"

    def test_hom_ref(self):
        assert _encode_gt(100.0, 100.0, 100, "HOM") == "0/0"

    def test_hom_alt(self):
        assert _encode_gt(120.0, 120.0, 100, "HOM") == "1/1"

    def test_nan_allele1(self):
        assert _encode_gt(float("nan"), 100.0, 100, "HOM") == "./."

    def test_nan_both_alleles(self):
        assert _encode_gt(float("nan"), float("nan"), 100, "HET") == "./."


# ---------------------------------------------------------------------------
# Genotype VCF
# ---------------------------------------------------------------------------

class TestWriteGenotypeVcf:
    def test_basic_output(self, tmp_path):
        """Write a simple genotype VCF and verify structure."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [{
            "allele1_size": 100.0,
            "allele2_size": 110.0,
            "zygosity": "HET",
            "confidence": 0.95,
            "n_reads": 30,
        }]
        n = write_genotype_vcf(output, loci, results, "SAMPLE1")
        assert n == 1

        with open(output) as f:
            lines = f.readlines()

        # Check header
        assert lines[0].startswith("##fileformat=VCFv4.2")
        header_line = [l for l in lines if l.startswith("#CHROM")][0]
        assert "SAMPLE1" in header_line

        # Check data line
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 1
        fields = data_lines[0].strip().split("\t")
        assert fields[0] == "chr1"
        assert fields[1] == "101"  # 0-based to 1-based
        assert fields[4] == "<TR>"
        assert "GT:AL:DP:CONF" in fields[8]
        sample = fields[9]
        assert sample.startswith("0/1:")

    def test_none_result(self, tmp_path):
        """None results should be written as no-call."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [None]
        n = write_genotype_vcf(output, loci, results, "SAMPLE1")
        assert n == 0

        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        assert len(data) == 1
        assert ".:" in data[0]  # missing GT

    def test_multiple_loci(self, tmp_path):
        """Multiple loci should all be written."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC"), ("chr1", 300, 400, "AT")]
        results = [
            {"allele1_size": 100.0, "allele2_size": 100.0, "zygosity": "HOM",
             "confidence": 0.99, "n_reads": 40},
            {"allele1_size": 105.0, "allele2_size": 95.0, "zygosity": "HET",
             "confidence": 0.85, "n_reads": 25},
        ]
        n = write_genotype_vcf(output, loci, results, "S1")
        assert n == 2

    def test_nan_confidence_produces_dot(self, tmp_path):
        """NaN confidence should produce '.' not 'nan' in VCF."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [{
            "allele1_size": 100.0,
            "allele2_size": 110.0,
            "zygosity": "HET",
            "confidence": float("nan"),
            "n_reads": 30,
        }]
        n = write_genotype_vcf(output, loci, results, "S1")
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        sample_fields = data[0].strip().split("\t")[9].split(":")
        conf_field = sample_fields[3]  # GT:AL:DP:CONF
        assert conf_field == "."
        assert "nan" not in data[0].lower()

    def test_nan_allele_sizes(self, tmp_path):
        """NaN allele sizes produce '.' in AL and './.' in GT."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [{
            "allele1_size": float("nan"),
            "allele2_size": float("nan"),
            "zygosity": "HOM",
            "confidence": 0.5,
            "n_reads": 10,
        }]
        n = write_genotype_vcf(output, loci, results, "S1")
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        sample = data[0].strip().split("\t")[9]
        assert sample.startswith("./.")
        assert "nan" not in data[0].lower()

    def test_missing_confidence_key(self, tmp_path):
        """Missing 'confidence' key falls back to 0.0."""
        output = str(tmp_path / "geno.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [{
            "allele1_size": 100.0,
            "allele2_size": 100.0,
            "zygosity": "HOM",
            "n_reads": 20,
        }]
        n = write_genotype_vcf(output, loci, results, "S1")
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        sample_fields = data[0].strip().split("\t")[9].split(":")
        assert sample_fields[3] == "0"  # 0.0 formatted as int


# ---------------------------------------------------------------------------
# Instability VCF
# ---------------------------------------------------------------------------

class TestWriteInstabilityVcf:
    def _make_result(self, **overrides):
        base = {
            "median_h1": 100.0, "median_h2": 110.0,
            "hii_h1": 0.0, "hii_h2": 0.5,
            "ias": 1.0,
            "range_h1": 0.0, "range_h2": 5.0,
            "n_h1": 15, "n_h2": 12,
            "n_total": 27, "concordance": 0.95,
            "analysis_path": "hp-tagged",
            "unstable_haplotype": "h2",
            "dropout_flag": False,
        }
        base.update(overrides)
        return base

    def test_basic_output(self, tmp_path):
        """Write a simple instability VCF and verify structure."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr4", 3074876, 3074933, "CAG")]
        results = [self._make_result()]
        n = write_instability_vcf(output, loci, results, "HG002")
        assert n == 1

        with open(output) as f:
            lines = f.readlines()

        assert lines[0].startswith("##fileformat=VCFv4.2")
        header_line = [l for l in lines if l.startswith("#CHROM")][0]
        assert "HG002" in header_line

        data = [l for l in lines if not l.startswith("#")]
        assert len(data) == 1
        fields = data[0].strip().split("\t")
        assert fields[0] == "chr4"
        assert fields[1] == "3074877"  # 1-based
        assert "ANALYSIS_PATH=hp-tagged" in fields[7]

    def test_none_result(self, tmp_path):
        """None results handled gracefully."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [None]
        n = write_instability_vcf(output, loci, results, "S1")
        assert n == 0

    def test_dropout_flag(self, tmp_path):
        """Dropout flag encoded as 0/1 in VCF."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [self._make_result(dropout_flag=True)]
        write_instability_vcf(output, loci, results, "S1")

        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        fields = data[0].strip().split("\t")
        sample_fields = fields[9].split(":")
        # DROPOUT is the last FORMAT field
        assert sample_fields[-1] == "1"

    def test_format_keys_present(self, tmp_path):
        """All FORMAT keys should be defined in header."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [self._make_result()]
        write_instability_vcf(output, loci, results, "S1")

        with open(output) as f:
            content = f.read()

        for key in ["HII", "IAS", "DP", "CONC", "APATH", "MEDIAN", "DROPOUT"]:
            assert f"##FORMAT=<ID={key}" in content

    def test_nan_metrics_produce_dots(self, tmp_path):
        """NaN in instability metrics should produce '.' not 'nan'."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr1", 100, 200, "AC")]
        results = [self._make_result(
            hii_h1=float("nan"), hii_h2=float("inf"),
            ias=float("nan"), concordance=float("nan"),
        )]
        n = write_instability_vcf(output, loci, results, "S1")
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        assert "nan" not in data[0].lower()
        assert "inf" not in data[0].lower()

    def test_missing_median_keys(self, tmp_path):
        """Missing median_h1/h2 should fall back to 0.0."""
        output = str(tmp_path / "inst.vcf")
        loci = [("chr1", 100, 200, "AC")]
        result = self._make_result()
        del result["median_h1"]
        del result["median_h2"]
        n = write_instability_vcf(output, loci, [result], "S1")
        assert n == 1
        with open(output) as f:
            data = [l for l in f if not l.startswith("#")]
        # Should not crash, and produce valid output
        assert len(data) == 1
