"""VCF 4.2 output writers for MosaicTR genotyping and instability results.

Provides two main functions:
  - write_genotype_vcf: Converts genotyping BED results to VCF format.
  - write_instability_vcf: Converts instability TSV results to VCF format.

Uses only the Python standard library (no pysam dependency).
Does not import from mosaictr.* -- importable standalone.
"""

from __future__ import annotations

import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vcf_filedate() -> str:
    """Return today's date in VCF fileDate format (YYYYMMDD)."""
    return datetime.date.today().strftime("%Y%m%d")


def _collect_contigs(loci: list[tuple[str, int, int, str]]) -> list[str]:
    """Return an ordered list of unique chromosome names from *loci*.

    Preserves first-seen order so the VCF contig lines follow the input
    order (usually karyotypic).
    """
    seen: set[str] = set()
    contigs: list[str] = []
    for chrom, *_ in loci:
        if chrom not in seen:
            seen.add(chrom)
            contigs.append(chrom)
    return contigs


def _fmt_float(val: float, decimals: int = 4) -> str:
    """Format a float for VCF fields.

    Integers are printed without a decimal point; other values are rounded
    to *decimals* places.  NaN and Inf are returned as "." (VCF missing).
    """
    import math
    if math.isnan(val) or math.isinf(val):
        return "."
    if val == int(val) and abs(val) < 1e12:
        return str(int(val))
    return f"{val:.{decimals}f}"


def _fmt_size(size: float) -> str:
    """Format an allele size (integer when possible, one decimal otherwise).

    NaN and Inf are returned as "." (VCF missing).
    """
    import math
    if math.isnan(size) or math.isinf(size):
        return "."
    if size == int(size):
        return str(int(size))
    return f"{size:.1f}"


# ---------------------------------------------------------------------------
# Genotype VCF
# ---------------------------------------------------------------------------

_GENOTYPE_VCF_INFO_FIELDS = [
    '##INFO=<ID=MOTIF,Number=1,Type=String,Description="Tandem repeat motif sequence">',
    '##INFO=<ID=REF_SIZE,Number=1,Type=Integer,Description="Reference allele size in bp">',
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the tandem repeat region">',
]

_GENOTYPE_VCF_FORMAT_FIELDS = [
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=AL,Number=.,Type=Float,Description="Allele lengths in bp (comma-separated)">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Total read depth at locus">',
    '##FORMAT=<ID=CONF,Number=1,Type=Float,Description="Genotype confidence score (0-1)">',
]

_GENOTYPE_VCF_ALT = [
    '##ALT=<ID=TR,Description="Tandem repeat allele">',
]


def _encode_gt(
    allele1_size: float,
    allele2_size: float,
    ref_size: int,
    zygosity: str,
) -> str:
    """Encode VCF GT field from MosaicTR genotype result.

    Rules:
      - NaN allele sizes                          -> ./.
      - HOM with both alleles equal to ref_size   -> 0/0
      - HOM with alleles different from ref_size  -> 1/1
      - HET                                       -> 0/1
    """
    import math
    if math.isnan(allele1_size) or math.isnan(allele2_size):
        return "./."
    if zygosity == "HET":
        return "0/1"
    # HOM -- check if alleles match reference size
    # Use a small tolerance (0.5 bp) for float comparison
    a1_is_ref = abs(allele1_size - ref_size) < 0.5
    a2_is_ref = abs(allele2_size - ref_size) < 0.5
    if a1_is_ref and a2_is_ref:
        return "0/0"
    return "1/1"


def write_genotype_vcf(
    output_path: str,
    loci: list[tuple[str, int, int, str]],
    results: list[Optional[dict]],
    sample_name: str,
    ref_path: Optional[str] = None,
) -> int:
    """Write genotyping results as a VCF 4.2 file.

    Args:
        output_path: Path to the output VCF file.
        loci: List of (chrom, start, end, motif) tuples defining each TR
            locus.  *start* and *end* are 0-based half-open BED coordinates.
        results: Parallel list of genotype result dicts (or ``None`` for
            loci that failed genotyping).  Each dict has keys:
            ``allele1_size``, ``allele2_size``, ``zygosity``, ``n_reads``,
            ``confidence``.
        sample_name: Sample identifier used in the VCF column header.
        ref_path: Optional path to the reference FASTA.  When provided it
            is recorded in a ``##reference`` header line and used to fetch
            the REF base at each position.  If ``None``, "N" is used as the
            REF base.

    Returns:
        Number of loci with a genotype call written (i.e. non-None results).
    """
    # Optionally open reference to fetch REF bases
    ref_fasta = None
    if ref_path is not None:
        try:
            import pysam  # type: ignore
            ref_fasta = pysam.FastaFile(ref_path)
        except (ImportError, Exception):
            # Fall back to "N" if pysam is unavailable or file cannot be opened
            ref_fasta = None

    contigs = _collect_contigs(loci)
    n_called = 0

    with open(output_path, "w") as f:
        # --- Meta-information lines ---
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##fileDate={_vcf_filedate()}\n")
        f.write("##source=MosaicTR\n")
        if ref_path is not None:
            f.write(f"##reference={ref_path}\n")
        for chrom in contigs:
            f.write(f"##contig=<ID={chrom}>\n")
        for line in _GENOTYPE_VCF_ALT:
            f.write(line + "\n")
        for line in _GENOTYPE_VCF_INFO_FIELDS:
            f.write(line + "\n")
        for line in _GENOTYPE_VCF_FORMAT_FIELDS:
            f.write(line + "\n")

        # --- Header line ---
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n")

        # --- Data lines ---
        for locus, result in zip(loci, results):
            chrom, start, end, motif = locus
            # VCF uses 1-based positions; BED start is 0-based
            pos = start + 1
            ref_size = end - start

            # Fetch REF base
            ref_base = "N"
            if ref_fasta is not None:
                try:
                    ref_base = ref_fasta.fetch(chrom, start, start + 1).upper()
                    if not ref_base:
                        ref_base = "N"
                except Exception:
                    ref_base = "N"

            if result is None:
                # No-call: write with missing fields
                info = f"MOTIF={motif};REF_SIZE={ref_size};END={end}"
                sample_data = ".:.,.:0:."
                f.write(
                    f"{chrom}\t{pos}\t.\t{ref_base}\t<TR>\t.\t.\t{info}\t"
                    f"GT:AL:DP:CONF\t{sample_data}\n"
                )
                continue

            n_called += 1

            a1 = result["allele1_size"]
            a2 = result["allele2_size"]
            zygosity = result["zygosity"]
            n_reads = result["n_reads"]
            confidence = result.get("confidence", 0.0)

            gt = _encode_gt(a1, a2, ref_size, zygosity)

            info = f"MOTIF={motif};REF_SIZE={ref_size};END={end}"
            al_str = f"{_fmt_size(a1)},{_fmt_size(a2)}"
            sample_data = f"{gt}:{al_str}:{n_reads}:{_fmt_float(confidence, decimals=3)}"

            f.write(
                f"{chrom}\t{pos}\t.\t{ref_base}\t<TR>\t.\t.\t{info}\t"
                f"GT:AL:DP:CONF\t{sample_data}\n"
            )

    if ref_fasta is not None:
        ref_fasta.close()

    return n_called


# ---------------------------------------------------------------------------
# Instability VCF
# ---------------------------------------------------------------------------

_INSTABILITY_VCF_INFO_FIELDS = [
    '##INFO=<ID=MOTIF,Number=1,Type=String,Description="Tandem repeat motif sequence">',
    '##INFO=<ID=REF_SIZE,Number=1,Type=Integer,Description="Reference allele size in bp">',
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the tandem repeat region">',
    '##INFO=<ID=ANALYSIS_PATH,Number=1,Type=String,Description="Analysis path used: hp-tagged, gap-split, or pooled">',
]

_INSTABILITY_VCF_FORMAT_FIELDS = [
    '##FORMAT=<ID=HII,Number=2,Type=Float,Description="Haplotype Instability Index per haplotype (h1,h2)">',
    '##FORMAT=<ID=IAS,Number=1,Type=Float,Description="Instability Asymmetry Score">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Total read depth at locus">',
    '##FORMAT=<ID=APATH,Number=1,Type=String,Description="Analysis path used: hp-tagged, gap-split, or pooled">',
    '##FORMAT=<ID=MEDIAN,Number=2,Type=Float,Description="Per-haplotype median allele sizes (h1,h2)">',
]


def write_instability_vcf(
    output_path: str,
    loci: list[tuple[str, int, int, str]],
    results: list[Optional[dict]],
    sample_name: str,
    ref_path: Optional[str] = None,
) -> int:
    """Write instability analysis results as a VCF 4.2 file.

    Args:
        output_path: Path to the output VCF file.
        loci: List of (chrom, start, end, motif) tuples defining each TR
            locus.  *start* and *end* are 0-based half-open BED coordinates.
        results: Parallel list of instability result dicts (or ``None`` for
            loci that failed analysis).  Each dict has keys:
            ``median_h1``, ``median_h2``,
            ``hii_h1``, ``hii_h2``, ``ias``,
            ``n_total``, ``analysis_path``.
        sample_name: Sample identifier used in the VCF column header.
        ref_path: Optional path to the reference FASTA.  When provided it
            is recorded in a ``##reference`` header line and used to fetch
            the REF base at each position.  If ``None``, "N" is used.

    Returns:
        Number of loci with instability data written (i.e. non-None results).
    """
    ref_fasta = None
    if ref_path is not None:
        try:
            import pysam  # type: ignore
            ref_fasta = pysam.FastaFile(ref_path)
        except (ImportError, Exception):
            ref_fasta = None

    contigs = _collect_contigs(loci)
    n_written = 0

    fmt_keys = "HII:IAS:DP:APATH:MEDIAN"

    with open(output_path, "w") as f:
        # --- Meta-information lines ---
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##fileDate={_vcf_filedate()}\n")
        f.write("##source=MosaicTR\n")
        if ref_path is not None:
            f.write(f"##reference={ref_path}\n")
        for chrom in contigs:
            f.write(f"##contig=<ID={chrom}>\n")
        for line in _INSTABILITY_VCF_INFO_FIELDS:
            f.write(line + "\n")
        for line in _INSTABILITY_VCF_FORMAT_FIELDS:
            f.write(line + "\n")

        # --- Header line ---
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n")

        # --- Data lines ---
        for locus, result in zip(loci, results):
            chrom, start, end, motif = locus
            pos = start + 1
            ref_size = end - start

            # REF base
            ref_base = "N"
            if ref_fasta is not None:
                try:
                    ref_base = ref_fasta.fetch(chrom, start, start + 1).upper()
                    if not ref_base:
                        ref_base = "N"
                except Exception:
                    ref_base = "N"

            if result is None:
                info = f"MOTIF={motif};REF_SIZE={ref_size};END={end}"
                missing = ".,.:.:0:.:.,."
                f.write(
                    f"{chrom}\t{pos}\t.\t{ref_base}\t.\t.\t.\t{info}\t"
                    f"{fmt_keys}\t{missing}\n"
                )
                continue

            n_written += 1

            med1 = result.get("median_h1", 0.0)
            med2 = result.get("median_h2", 0.0)

            hii_h1 = result["hii_h1"]
            hii_h2 = result["hii_h2"]
            ias = result["ias"]
            n_total = result["n_total"]
            analysis_path = result["analysis_path"]

            # INFO
            info = (
                f"MOTIF={motif};REF_SIZE={ref_size};END={end};"
                f"ANALYSIS_PATH={analysis_path}"
            )

            # FORMAT sample values
            hii_str = f"{_fmt_float(hii_h1)},{_fmt_float(hii_h2)}"
            ias_str = _fmt_float(ias)
            median_str = f"{_fmt_float(med1)},{_fmt_float(med2)}"

            sample_data = (
                f"{hii_str}:{ias_str}:{n_total}:"
                f"{analysis_path}:{median_str}"
            )

            f.write(
                f"{chrom}\t{pos}\t.\t{ref_base}\t.\t.\t.\t{info}\t"
                f"{fmt_keys}\t{sample_data}\n"
            )

    if ref_fasta is not None:
        ref_fasta.close()

    return n_written
