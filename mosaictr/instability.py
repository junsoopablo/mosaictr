"""MosaicTR somatic instability module.

Haplotype-resolved somatic TR instability metrics from HP-tagged long-read BAMs.
Provides per-haplotype instability decomposition that no existing tool offers:

Metrics:
- HII (Haplotype Instability Index): motif-normalized dispersion per haplotype
- IAS (Instability Asymmetry Score): inter-haplotype instability difference
"""

from __future__ import annotations

import gc
import logging
import math
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from .genotype import (
    ReadInfo,
    _assign_hp0_reads,
    _gap_bimodal_test,
    _group_loci_by_chrom,
    _weighted_median,
    extract_reads_enhanced,
)
from .utils import load_loci_bed

logger = logging.getLogger(__name__)

# Small constant to avoid division by zero
_EPS = 1e-9


# ---------------------------------------------------------------------------
# HP tag pre-flight check
# ---------------------------------------------------------------------------

def _check_hp_tags(bam_path: str, sample_size: int = 1000) -> float:
    """Sample reads from BAM and return HP tag fraction.

    Raises ValueError if no HP tags found.
    """
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    n_hp, n_total = 0, 0
    try:
        for aln in bam:
            if n_total >= sample_size:
                break
            if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
                continue
            n_total += 1
            try:
                hp = aln.get_tag("HP")
                if hp in (1, 2):
                    n_hp += 1
            except KeyError:
                pass
    finally:
        bam.close()

    hp_frac = n_hp / n_total if n_total > 0 else 0.0
    if hp_frac == 0.0:
        raise ValueError(
            f"No HP tags found in {n_total} sampled reads. "
            "The instability module requires an HP-tagged BAM. "
            "Tag with: whatshap haplotag / hiphase / longphase haplotag"
        )
    return hp_frac


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------

def _weighted_mad(
    values: np.ndarray, weights: np.ndarray, center: float,
) -> float:
    """Weighted median absolute deviation around a center value.

    Args:
        values: Array of observations.
        weights: Array of weights (e.g., MapQ).
        center: Center value (typically weighted median).

    Returns:
        Weighted MAD (>= 0).
    """
    if len(values) == 0:
        return 0.0
    deviations = np.abs(values - center)
    return _weighted_median(deviations, weights)


def _trim_outliers_mad(
    sizes: np.ndarray, weights: np.ndarray, mad_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove outlier reads using MAD-based trimming.

    Removes reads further than mad_factor * MAD from the weighted median.
    This handles misphased reads that corrupt per-haplotype metrics
    (e.g., 10 normal reads mixed into the expanded haplotype cluster).

    Returns trimmed (sizes, weights) arrays. If trimming would leave
    fewer than 3 reads, returns the original arrays unchanged.
    """
    if len(sizes) < 4:
        return sizes, weights
    center = _weighted_median(sizes, weights)
    mad = _weighted_mad(sizes, weights, center)
    if mad < _EPS:
        return sizes, weights
    threshold = mad_factor * mad
    mask = np.abs(sizes - center) <= threshold
    if mask.sum() < 3:
        return sizes, weights
    return sizes[mask], weights[mask]


# ---------------------------------------------------------------------------
# Per-haplotype metrics
# ---------------------------------------------------------------------------

def _hii(sizes: np.ndarray, weights: np.ndarray, motif_len: int) -> float:
    """Haplotype Instability Index: weighted MAD / motif_len.

    Motif-normalized dispersion. Stable locus -> ~0, somatic mosaicism -> >0.5.
    Robust to outliers via MAD (not variance).
    """
    if len(sizes) < 2 or motif_len < 1:
        return 0.0
    center = _weighted_median(sizes, weights)
    mad = _weighted_mad(sizes, weights, center)
    return mad / motif_len


# ---------------------------------------------------------------------------
# Inter-haplotype metrics
# ---------------------------------------------------------------------------

def _ias(hii_1: float, hii_2: float) -> float:
    """Instability Asymmetry Score: |HII_1 - HII_2| / max(HII_1, HII_2, eps).

    Range [0, 1]. High IAS -> one allele unstable, other stable.
    Typical of expansion diseases where only the pathogenic allele is unstable.
    """
    max_hii = max(hii_1, hii_2)
    if max_hii < _EPS:
        return 0.0
    return abs(hii_1 - hii_2) / max_hii


# ---------------------------------------------------------------------------
# Per-locus orchestrator
# ---------------------------------------------------------------------------

def compute_instability(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
) -> Optional[dict]:
    """Compute all instability metrics for a single locus.

    Separates reads by HP tag, assigns HP=0 reads to closest haplotype,
    then computes per-haplotype and inter-haplotype metrics.

    Args:
        reads_info: List of ReadInfo from extract_reads_enhanced().
        ref_size: Reference allele size (end - start).
        motif_len: Repeat motif length in bp.
        min_hp_reads: Minimum HP-tagged reads per haplotype.
        min_hp_frac: Minimum fraction of HP-tagged reads.

    Returns:
        Dict with all metrics, or None if insufficient reads.
    """
    if not reads_info or motif_len < 1:
        return None

    n_total = len(reads_info)

    hp1_reads = [r for r in reads_info if r.hp == 1]
    hp2_reads = [r for r in reads_info if r.hp == 2]
    n_hp1, n_hp2 = len(hp1_reads), len(hp2_reads)
    hp_frac = (n_hp1 + n_hp2) / n_total if n_total > 0 else 0

    hp_sufficient = (
        n_hp1 >= min_hp_reads
        and n_hp2 >= min_hp_reads
        and hp_frac >= min_hp_frac
    )

    if not hp_sufficient:
        # Fallback: try gap-based split if bimodal
        all_sizes = np.array([r.allele_size for r in reads_info])
        if _gap_bimodal_test(all_sizes):
            return _instability_from_gap_split(reads_info, ref_size, motif_len)
        # Cannot separate haplotypes — report pooled metrics only
        return _instability_pooled_fallback(reads_info, ref_size, motif_len)

    # Initial HP medians
    hp1_sizes = np.array([r.allele_size for r in hp1_reads])
    hp1_w = np.maximum(np.array([r.mapq for r in hp1_reads], dtype=float), 1.0)
    hp2_sizes = np.array([r.allele_size for r in hp2_reads])
    hp2_w = np.maximum(np.array([r.mapq for r in hp2_reads], dtype=float), 1.0)

    med1 = _weighted_median(hp1_sizes, hp1_w)
    med2 = _weighted_median(hp2_sizes, hp2_w)

    # Assign HP=0 reads and recompute
    aug_hp1, aug_hp2 = _assign_hp0_reads(reads_info, med1, med2)

    aug1_sizes = np.array([r.allele_size for r in aug_hp1])
    aug1_w = np.maximum(np.array([r.mapq for r in aug_hp1], dtype=float), 1.0)
    aug2_sizes = np.array([r.allele_size for r in aug_hp2])
    aug2_w = np.maximum(np.array([r.mapq for r in aug_hp2], dtype=float), 1.0)

    # Trim outliers (handles misphased reads corrupting haplotype clusters)
    trim1_sizes, trim1_w = _trim_outliers_mad(aug1_sizes, aug1_w)
    trim2_sizes, trim2_w = _trim_outliers_mad(aug2_sizes, aug2_w)

    med1 = _weighted_median(trim1_sizes, trim1_w)
    med2 = _weighted_median(trim2_sizes, trim2_w)

    # Per-haplotype metrics (computed on trimmed reads)
    hii_h1 = _hii(trim1_sizes, trim1_w, motif_len)
    hii_h2 = _hii(trim2_sizes, trim2_w, motif_len)

    # Inter-haplotype metrics
    ias_val = _ias(hii_h1, hii_h2)

    return {
        "median_h1": med1,
        "median_h2": med2,
        "hii_h1": hii_h1,
        "hii_h2": hii_h2,
        "ias": ias_val,
        "n_h1": len(aug_hp1),
        "n_h2": len(aug_hp2),
        "n_total": n_total,
        "analysis_path": "hp-tagged",
    }


def _instability_from_gap_split(
    reads_info: list[ReadInfo], ref_size: float, motif_len: int,
    min_cluster_size: int = 3,
) -> dict:
    """Compute instability metrics using gap-based bimodal split (no HP tags).

    Used when HP tags are insufficient but reads show bimodal distribution.
    Falls back to pooled if either cluster has fewer than min_cluster_size reads.
    """
    all_sizes = np.array([r.allele_size for r in reads_info])
    all_w = np.maximum(np.array([r.mapq for r in reads_info], dtype=float), 1.0)

    sorted_idx = np.argsort(all_sizes)
    sorted_sizes = all_sizes[sorted_idx]
    gaps = np.diff(sorted_sizes)
    split_pos = int(np.argmax(gaps)) + 1

    lo_idx = sorted_idx[:split_pos]
    hi_idx = sorted_idx[split_pos:]

    # Require minimum cluster size to avoid false splits from single outliers
    if len(lo_idx) < min_cluster_size or len(hi_idx) < min_cluster_size:
        return _instability_pooled_fallback(reads_info, ref_size, motif_len)

    lo_sizes = all_sizes[lo_idx]
    lo_w = all_w[lo_idx]
    hi_sizes = all_sizes[hi_idx]
    hi_w = all_w[hi_idx]

    # Trim outliers within each cluster
    trim_lo_sizes, trim_lo_w = _trim_outliers_mad(lo_sizes, lo_w)
    trim_hi_sizes, trim_hi_w = _trim_outliers_mad(hi_sizes, hi_w)

    med1 = _weighted_median(trim_lo_sizes, trim_lo_w)
    med2 = _weighted_median(trim_hi_sizes, trim_hi_w)

    hii_h1 = _hii(trim_lo_sizes, trim_lo_w, motif_len)
    hii_h2 = _hii(trim_hi_sizes, trim_hi_w, motif_len)

    n_total = len(reads_info)

    return {
        "median_h1": med1,
        "median_h2": med2,
        "hii_h1": hii_h1,
        "hii_h2": hii_h2,
        "ias": _ias(hii_h1, hii_h2),
        "n_h1": len(lo_idx),
        "n_h2": len(hi_idx),
        "n_total": n_total,
        "analysis_path": "gap-split",
    }


def _instability_pooled_fallback(
    reads_info: list[ReadInfo], ref_size: float, motif_len: int,
) -> dict:
    """Compute instability metrics from pooled reads (no haplotype separation).

    Used when reads are unimodal and HP tags are insufficient.
    Reports h1 metrics only; h2 metrics are set to 0.
    """
    all_sizes = np.array([r.allele_size for r in reads_info])
    all_w = np.maximum(np.array([r.mapq for r in reads_info], dtype=float), 1.0)

    # Trim outliers
    trim_sizes, trim_w = _trim_outliers_mad(all_sizes, all_w)

    med = _weighted_median(trim_sizes, trim_w)
    n_total = len(reads_info)

    hii_val = _hii(trim_sizes, trim_w, motif_len)

    return {
        "median_h1": med,
        "median_h2": med,
        "hii_h1": hii_val,
        "hii_h2": 0.0,
        "ias": 0.0,
        "n_h1": n_total,
        "n_h2": 0,
        "n_total": n_total,
        "analysis_path": "pooled",
    }


# ---------------------------------------------------------------------------
# Chunk worker
# ---------------------------------------------------------------------------

def _instability_chunk(
    bam_path: str,
    loci_chunk: list[tuple[str, int, int, str]],
    min_mapq: int = 5,
    min_flank: int = 50,
    max_reads: int = 200,
    ref_path: Optional[str] = None,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    min_reads: int = 0,
) -> list[Optional[dict]]:
    """Compute instability for a chunk of loci (runs in a worker process)."""
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(ref_path) if ref_path else None
    results = []

    try:
        for chrom, start, end, motif in loci_chunk:
            ref_size = end - start
            motif_len = len(motif)

            reads = extract_reads_enhanced(
                bam, chrom, start, end,
                min_mapq=min_mapq, min_flank=min_flank, max_reads=max_reads,
                ref_fasta=ref_fasta, motif_len=motif_len,
            )
            if not reads or len(reads) < min_reads:
                results.append(None)
                continue

            result = compute_instability(
                reads, ref_size, motif_len,
                min_hp_reads=min_hp_reads,
                min_hp_frac=min_hp_frac,
            )
            results.append(result)
    finally:
        bam.close()
        if ref_fasta:
            ref_fasta.close()
    return results


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    "#chrom\tstart\tend\tmotif\t"
    "median_h1\tmedian_h2\t"
    "hii_h1\thii_h2\t"
    "ias\t"
    "n_h1\tn_h2\tn_total\t"
    "analysis_path\n"
)


def _write_instability_tsv(
    output_path: str,
    loci: list[tuple[str, int, int, str]],
    results: list[Optional[dict]],
    min_hii: float = 0.0,
) -> int:
    """Write instability results to 13-column TSV.

    Args:
        output_path: Output file path.
        loci: List of (chrom, start, end, motif) tuples.
        results: List of result dicts (or None for failed loci).
        min_hii: Minimum max(HII) threshold for output filtering.

    Returns:
        Number of loci written.
    """
    n_written = 0

    def _fmt(val):
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return "."
            if val == int(val) and abs(val) < 1e12:
                return str(int(val))
            return f"{val:.4f}"
        return str(val)

    with open(output_path, "w") as f:
        f.write(_TSV_HEADER)
        for locus, result in zip(loci, results):
            chrom, start, end, motif = locus
            if result is None:
                if min_hii <= 0:
                    f.write(
                        f"{chrom}\t{start}\t{end}\t{motif}\t"
                        ".\t.\t.\t.\t.\t0\t0\t0\tfailed\n"
                    )
                    n_written += 1
                continue

            max_hii = max(result.get("hii_h1", 0.0), result.get("hii_h2", 0.0))
            if max_hii < min_hii:
                continue

            f.write(
                f"{chrom}\t{start}\t{end}\t{motif}\t"
                f"{_fmt(result['median_h1'])}\t{_fmt(result['median_h2'])}\t"
                f"{_fmt(result['hii_h1'])}\t{_fmt(result['hii_h2'])}\t"
                f"{_fmt(result['ias'])}\t"
                f"{result['n_h1']}\t{result['n_h2']}\t"
                f"{result['n_total']}\t"
                f"{result['analysis_path']}\n"
            )
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_instability(
    bam_path: str,
    loci_bed_path: str,
    output_path: str,
    nprocs: int = 1,
    chunk_size: int = 500,
    min_mapq: int = 5,
    min_flank: int = 50,
    max_reads: int = 200,
    ref_path: Optional[str] = None,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    min_instability: float = 0.0,
    min_reads: int = 0,
    skip_hp_check: bool = False,
) -> str:
    """Run MosaicTR somatic instability analysis on a set of TR loci.

    Pipeline: BAM + loci BED -> per-chromosome instability -> TSV output

    Args:
        bam_path: Input BAM file (with HP tags for best results).
        loci_bed_path: 4-column BED with loci to analyze.
        output_path: Output TSV path.
        nprocs: Number of parallel worker processes.
        chunk_size: Loci per worker chunk.
        min_mapq: Minimum mapping quality.
        min_flank: Minimum flanking bases.
        max_reads: Maximum reads per locus.
        ref_path: Reference FASTA for parasail realignment (optional).
        min_hp_reads: Minimum HP-tagged reads per haplotype.
        min_hp_frac: Minimum fraction of HP-tagged reads.
        min_instability: Minimum HII threshold for output (0 = no filter).
        min_reads: Minimum total reads per locus (0 = no filter).
        skip_hp_check: Skip HP tag pre-flight check (for non-HP BAMs).

    Returns:
        Path to output TSV file.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if ref_path:
        logger.info("Parasail realignment enabled (ref: %s)", ref_path)

    loci = load_loci_bed(loci_bed_path)
    logger.info("Loaded %d loci from %s", len(loci), loci_bed_path)

    if not skip_hp_check:
        hp_frac = _check_hp_tags(bam_path)
        logger.info("HP tag check: %.1f%% of sampled reads have HP tags", hp_frac * 100)
    else:
        logger.warning(
            "HP tag check skipped. Loci will use gap-split/pooled fallback "
            "if reads lack HP tags."
        )

    chrom_groups = _group_loci_by_chrom(loci)
    logger.info(
        "Split into %d chromosomes: %s",
        len(chrom_groups),
        ", ".join(f"{c}({len(l)})" for c, l in chrom_groups),
    )

    chunk_kwargs = dict(
        min_mapq=min_mapq,
        min_flank=min_flank,
        max_reads=max_reads,
        ref_path=ref_path,
        min_hp_reads=min_hp_reads,
        min_hp_frac=min_hp_frac,
        min_reads=min_reads,
    )

    t0 = time.time()
    all_results: list[Optional[dict]] = []
    all_loci: list[tuple[str, int, int, str]] = []

    for chrom, chrom_loci in chrom_groups:
        chrom_t0 = time.time()
        chunks = [
            chrom_loci[i : i + chunk_size]
            for i in range(0, len(chrom_loci), chunk_size)
        ]

        if nprocs <= 1:
            for chunk in tqdm(chunks, desc=f"Instability {chrom}"):
                chunk_results = _instability_chunk(bam_path, chunk, **chunk_kwargs)
                all_results.extend(chunk_results)
                all_loci.extend(chunk)
        else:
            max_pending = nprocs * 2
            with ProcessPoolExecutor(max_workers=nprocs) as executor:
                pbar = tqdm(total=len(chunks), desc=f"Instability {chrom}")
                pending: dict = {}
                chunk_iter = iter(enumerate(chunks))

                def _submit_next():
                    try:
                        i, chunk = next(chunk_iter)
                        fut = executor.submit(
                            _instability_chunk, bam_path, chunk, **chunk_kwargs,
                        )
                        pending[fut] = (i, chunk)
                    except StopIteration:
                        pass

                for _ in range(min(max_pending, len(chunks))):
                    _submit_next()

                while pending:
                    done = next(iter(as_completed(pending)))
                    i, chunk = pending.pop(done)
                    try:
                        chunk_results = done.result()
                    except Exception as e:
                        logger.error("Chunk %d on %s failed: %s", i, chrom, e)
                        chunk_results = [None] * len(chunk)
                    all_results.extend(chunk_results)
                    all_loci.extend(chunk)
                    pbar.update(1)
                    _submit_next()

                pbar.close()

        gc.collect()
        chrom_elapsed = time.time() - chrom_t0
        chrom_analyzed = sum(
            1
            for r in all_results[len(all_results) - len(chrom_loci) :]
            if r is not None
        )
        logger.info(
            "%s: %d/%d loci in %.1fs",
            chrom, chrom_analyzed, len(chrom_loci), chrom_elapsed,
        )

    elapsed = time.time() - t0
    analyzed = sum(1 for r in all_results if r is not None)
    logger.info(
        "Analyzed %d/%d loci in %.1fs (%.0f loci/sec)",
        analyzed,
        len(all_loci),
        elapsed,
        len(all_loci) / elapsed if elapsed > 0 else 0,
    )

    # --- Analysis path summary + HP tag warning ---
    path_counts = {"hp-tagged": 0, "gap-split": 0, "pooled": 0}
    for r in all_results:
        if r is not None:
            path_counts[r.get("analysis_path", "unknown")] = \
                path_counts.get(r.get("analysis_path", "unknown"), 0) + 1

    n_analyzed = sum(path_counts.values())
    if n_analyzed > 0:
        hp_pct = path_counts["hp-tagged"] / n_analyzed * 100
        logger.info(
            "Analysis paths: hp-tagged=%d (%.0f%%), gap-split=%d, pooled=%d",
            path_counts["hp-tagged"], hp_pct,
            path_counts["gap-split"], path_counts["pooled"],
        )
        if hp_pct < 50:
            logger.warning(
                "Only %.0f%% of loci used HP-tagged analysis. "
                "For best per-haplotype instability results, provide an HP-tagged BAM. "
                "Use: whatshap haplotag / hiphase / longphase haplotag",
                hp_pct,
            )

    n_written = _write_instability_tsv(
        output_path, all_loci, all_results, min_hii=min_instability,
    )
    logger.info(
        "Output written to %s (%d loci, min_hii=%.2f)",
        output_path, n_written, min_instability,
    )
    return output_path
