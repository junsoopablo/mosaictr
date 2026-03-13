"""MosaicTR genotyping pipeline.

Haplotype-aware tandem repeat genotyping from long-read sequencing.
Uses HP (haplotype) tags from phased BAM files to separate alleles,
with concordance-based zygosity, gap-based bimodality detection,
and robust weighted medians for VNTR sizing.

Key features:
- Concordance-based HET/HOM decision (|d1-d2| <= motif_len -> HOM).
- HP=0 read assignment to nearest haplotype cluster.
- Adaptive collapse threshold for close alleles.
- Gap-based bimodality test for VNTR.
- Two-pass MAD trimming for robust VNTR sizing.
- Optional parasail local realignment with motif-aware gap penalties.
"""

from __future__ import annotations

import gc
import logging
import math
import re
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pysam
from tqdm import tqdm

from .utils import load_loci_bed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ReadInfo = namedtuple("ReadInfo", ["allele_size", "hp", "mapq"])


# ---------------------------------------------------------------------------
# CIGAR-based allele size computation
# ---------------------------------------------------------------------------

def compute_allele_size_cigar(aln, locus_start: int, locus_end: int) -> Optional[float]:
    """Compute allele size (query bases within locus) from CIGAR.

    Walks through the CIGAR string and counts query bases that fall
    within [locus_start, locus_end) on the reference.
    """
    if aln.cigartuples is None:
        return None
    ref_pos = aln.reference_start
    qbases = 0
    for op, length in aln.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            ov_start = max(ref_pos, locus_start)
            ov_end = min(ref_pos + length, locus_end)
            if ov_start < ov_end:
                qbases += (ov_end - ov_start)
            ref_pos += length
        elif op == 1:  # I
            if locus_start <= ref_pos <= locus_end:
                qbases += length
        elif op in (2, 3):  # D, N
            ref_pos += length
    return float(qbases) if qbases > 0 else None


def _extract_query_at_locus(
    aln, locus_start: int, locus_end: int, flank: int = 100,
) -> Optional[str]:
    """Extract the query subsequence aligning to the locus region with flanking.

    Used for parasail realignment. Returns query bases corresponding to
    [locus_start - flank, locus_end + flank) on the reference.
    """
    if aln.cigartuples is None or aln.query_sequence is None:
        return None

    ext_start = max(locus_start - flank, aln.reference_start)
    ext_end = locus_end + flank

    ref_pos = aln.reference_start
    query_pos = 0
    query_start = None
    query_end = None

    for op, length in aln.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            ov_start = max(ref_pos, ext_start)
            ov_end = min(ref_pos + length, ext_end)
            if ov_start < ov_end:
                q_off_start = ov_start - ref_pos
                q_off_end = ov_end - ref_pos
                if query_start is None:
                    query_start = query_pos + q_off_start
                query_end = query_pos + q_off_end
            ref_pos += length
            query_pos += length
        elif op == 1:  # I
            if ext_start <= ref_pos <= ext_end:
                if query_start is None:
                    query_start = query_pos
                query_end = query_pos + length
            query_pos += length
        elif op in (2, 3):  # D, N
            ref_pos += length
        elif op == 4:  # S (soft clip)
            query_pos += length

    if query_start is not None and query_end is not None:
        seq = aln.query_sequence[query_start:query_end]
        return seq if len(seq) > 0 else None
    return None


# ---------------------------------------------------------------------------
# Parasail local realignment (optional)
# ---------------------------------------------------------------------------

def _get_motif_gap_penalties(motif_len: int) -> tuple[int, int]:
    """Gap penalties tuned by motif length.

    Shorter motifs get smaller gap penalties since indels of motif-length
    are biologically expected.
    """
    if motif_len == 1:
        return 3, 1
    elif motif_len == 2:
        return 4, 2
    elif motif_len <= 6:
        return 5, 2
    else:
        return 6, 3


def _realign_allele_size(
    aln,
    locus_start: int,
    locus_end: int,
    ref_fasta: pysam.FastaFile,
    motif_len: int,
    flank: int = 100,
) -> Optional[float]:
    """Re-estimate allele size using parasail semi-global alignment.

    Extracts query and reference around the locus, realigns with
    motif-aware gap penalties, and counts query bases within the locus
    from the new CIGAR.
    """
    try:
        import parasail
    except ImportError:
        return None

    query_seq = _extract_query_at_locus(aln, locus_start, locus_end, flank=flank)
    if not query_seq or len(query_seq) < 10:
        return None

    chrom = aln.reference_name
    ref_start = max(0, locus_start - flank)
    ref_end = locus_end + flank
    try:
        ref_seq = ref_fasta.fetch(chrom, ref_start, ref_end)
    except (ValueError, KeyError):
        return None
    if not ref_seq or len(ref_seq) < 10:
        return None

    gap_open, gap_extend = _get_motif_gap_penalties(motif_len)
    try:
        result = parasail.sg_dx_trace_striped_16(
            query_seq, ref_seq, gap_open, gap_extend, parasail.dnafull
        )
    except Exception:
        return None

    # Parse CIGAR string from parasail result
    try:
        cigar_str = result.cigar.decode.decode()  # bytes -> str
        ref_pos = ref_start + result.cigar.beg_ref
    except Exception:
        return None

    ops = re.findall(r"(\d+)([MIDNSHP=X])", cigar_str)
    qbases = 0
    for length_str, op in ops:
        length = int(length_str)
        if op in ("M", "=", "X"):
            ov_start = max(ref_pos, locus_start)
            ov_end = min(ref_pos + length, locus_end)
            if ov_start < ov_end:
                qbases += (ov_end - ov_start)
            ref_pos += length
        elif op == "I":
            if locus_start <= ref_pos <= locus_end:
                qbases += length
        elif op == "D":
            ref_pos += length

    return float(qbases) if qbases > 0 else None


# ---------------------------------------------------------------------------
# Read extraction from BAM
# ---------------------------------------------------------------------------

def extract_reads_enhanced(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    min_mapq: int = 5,
    min_flank: int = 50,
    max_reads: int = 200,
    ref_fasta: Optional[pysam.FastaFile] = None,
    motif_len: int = 1,
) -> list[ReadInfo]:
    """Extract allele sizes, HP tags, and MapQ from BAM reads.

    Enhanced version that returns ReadInfo with mapq for weighted median.
    Optionally uses parasail realignment if ref_fasta is provided.
    """
    reads = []
    seen: set[str] = set()
    try:
        fetched = bam.fetch(chrom, max(0, start - min_flank), end + min_flank)
    except ValueError:
        return reads
    for aln in fetched:
        if len(reads) >= max_reads:
            break
        if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
            continue
        if aln.mapping_quality < min_mapq:
            continue
        if aln.query_name in seen:
            continue
        seen.add(aln.query_name)
        if aln.reference_start is None or aln.reference_end is None:
            continue
        if aln.reference_start > start - min_flank or aln.reference_end < end + min_flank:
            continue

        allele_size = None
        if ref_fasta is not None and motif_len < 7:
            # Parasail realignment only for STRs; net-negative for VNTRs
            allele_size = _realign_allele_size(
                aln, start, end, ref_fasta, motif_len,
            )
        if allele_size is None:
            allele_size = compute_allele_size_cigar(aln, start, end)
        if allele_size is None:
            continue

        try:
            hp = aln.get_tag("HP")
        except KeyError:
            hp = 0
        reads.append(ReadInfo(
            allele_size=float(allele_size),
            hp=int(hp),
            mapq=int(aln.mapping_quality),
        ))
    return reads


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median."""
    if len(values) == 0:
        return 0.0
    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sw = weights[sorted_idx]
    cumw = np.cumsum(sw)
    cutoff = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, cutoff)
    return float(sv[min(idx, len(sv) - 1)])


def _trimmed_weighted_median(
    values: np.ndarray, weights: np.ndarray, iqr_factor: float = 1.5,
) -> float:
    """IQR-trimmed weighted median: remove outliers then compute weighted median."""
    if len(values) < 4:
        return _weighted_median(values, weights)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    mask = (values >= q1 - iqr_factor * iqr) & (values <= q3 + iqr_factor * iqr)
    if mask.sum() == 0:
        return _weighted_median(values, weights)
    return _weighted_median(values[mask], weights[mask])


def _compute_diff(
    ref_size: float,
    allele_size: float,
    motif_len: int,
    vntr_cutoff: int = 7,
) -> float:
    """Compute allele diff with conditional rounding.

    STR (motif < vntr_cutoff): integer rounding (same as v2).
    VNTR (motif >= vntr_cutoff): no rounding (keep float precision).
    """
    raw_diff = ref_size - allele_size
    if motif_len >= vntr_cutoff:
        return float(raw_diff)
    return float(round(raw_diff))


# ---------------------------------------------------------------------------
# v4 helper functions
# ---------------------------------------------------------------------------

def _hp_concordance(
    reads_info: list[ReadInfo], med1: float, med2: float,
) -> float:
    """Fraction of HP-tagged reads closer to their own haplotype median.

    HP=1 reads should be closer to med1, HP=2 reads to med2.
    Returns 0.0 if fewer than 2 HP-tagged reads.
    """
    hp_reads = [r for r in reads_info if r.hp in (1, 2)]
    if len(hp_reads) < 2:
        return 0.0
    concordant = 0
    for r in hp_reads:
        dist1 = abs(r.allele_size - med1)
        dist2 = abs(r.allele_size - med2)
        if r.hp == 1 and dist1 <= dist2:
            concordant += 1
        elif r.hp == 2 and dist2 <= dist1:
            concordant += 1
    return concordant / len(hp_reads)


def _assign_hp0_reads(
    reads_info: list[ReadInfo], med1: float, med2: float,
    n_iterations: int = 2,
) -> tuple[list[ReadInfo], list[ReadInfo]]:
    """Assign HP=0 reads to the closer haplotype cluster with EM refinement.

    Iteratively assigns HP=0 reads to nearest cluster, then recomputes
    cluster medians to refine assignments. Converges in 2-3 iterations.

    Returns augmented (hp1_reads, hp2_reads) including assigned HP=0 reads.
    Ties go to the smaller group for balance.
    """
    hp1_tagged = [r for r in reads_info if r.hp == 1]
    hp2_tagged = [r for r in reads_info if r.hp == 2]
    hp0 = [r for r in reads_info if r.hp == 0]

    if not hp0:
        return list(hp1_tagged), list(hp2_tagged)

    cur_med1, cur_med2 = med1, med2

    for _iter in range(max(1, n_iterations)):
        hp1 = list(hp1_tagged)
        hp2 = list(hp2_tagged)
        for r in hp0:
            dist1 = abs(r.allele_size - cur_med1)
            dist2 = abs(r.allele_size - cur_med2)
            if dist1 < dist2:
                hp1.append(r)
            elif dist2 < dist1:
                hp2.append(r)
            else:
                if len(hp1) <= len(hp2):
                    hp1.append(r)
                else:
                    hp2.append(r)

        # Recompute medians for next iteration
        if _iter < n_iterations - 1 and hp1 and hp2:
            s1 = np.array([r.allele_size for r in hp1])
            w1 = np.maximum(np.array([r.mapq for r in hp1], dtype=float), 1.0)
            s2 = np.array([r.allele_size for r in hp2])
            w2 = np.maximum(np.array([r.mapq for r in hp2], dtype=float), 1.0)
            new_med1 = _weighted_median(s1, w1)
            new_med2 = _weighted_median(s2, w2)
            # Early exit if converged
            if abs(new_med1 - cur_med1) < 0.1 and abs(new_med2 - cur_med2) < 0.1:
                break
            cur_med1, cur_med2 = new_med1, new_med2

    return hp1, hp2


def _adaptive_collapse_threshold(
    n_reads: int,
    base: float = 0.25,
    min_val: float = 0.20,
    ref_n: int = 20,
) -> float:
    """Coverage-adaptive collapse threshold: max(min_val, base * sqrt(ref_n / n)).

    n=20 → 0.25, n=50 → ~0.20, n=10 → ~0.35.
    Higher min_val (0.20) preserves true close alleles at high coverage.
    """
    if n_reads <= 0:
        return base
    return max(min_val, base * math.sqrt(ref_n / n_reads))


def _gap_bimodal_test(
    all_sizes: np.ndarray, min_reads: int = 6, gap_factor: float = 2.0,
) -> bool:
    """Gap-based bimodality test replacing _is_bimodal for v4.

    Returns True if max consecutive gap > gap_factor * median_gap AND > 1.0 bp.
    """
    if len(all_sizes) < min_reads:
        return False
    sorted_sizes = np.sort(all_sizes)
    gaps = np.diff(sorted_sizes)
    if len(gaps) == 0:
        return False
    max_gap = float(np.max(gaps))
    median_gap = float(np.median(gaps))
    return max_gap > gap_factor * median_gap and max_gap > 1.0


def _robust_vntr_median(
    values: np.ndarray, weights: np.ndarray, mad_factor: float = 3.0,
) -> float:
    """Two-pass robust weighted median for VNTRs.

    Pass 1: IQR-trimmed weighted median (standard).
    Pass 2: MAD-based trimming around pass-1 estimate to remove
    remaining outlier reads that distort VNTR sizing.
    """
    if len(values) < 4:
        return _weighted_median(values, weights)

    # Pass 1: standard IQR trimming
    initial = _trimmed_weighted_median(values, weights)

    # Pass 2: MAD trimming around initial estimate
    deviations = np.abs(values - initial)
    mad = float(np.median(deviations))
    if mad < 0.5:
        return initial

    threshold = mad_factor * mad
    mask = deviations <= threshold
    n_kept = int(mask.sum())
    if n_kept < 3:
        return initial

    return _weighted_median(values[mask], weights[mask])


def _sizing_quality(
    values: np.ndarray, estimate: float, motif_len: int,
) -> float:
    """Sizing quality score (0-1) based on read agreement around estimate.

    Measures what fraction of reads are within 2*motif_len of the median,
    scaled by a coverage factor. Low quality indicates unreliable sizing.
    """
    n = len(values)
    if n < 3:
        return max(0.0, n / 5.0)

    tol = max(2.0 * motif_len, 5.0)
    close = float(np.sum(np.abs(values - estimate) <= tol))
    agreement = close / n

    cov_factor = min(1.0, n / 10.0)
    return agreement * cov_factor


# ---------------------------------------------------------------------------
# Genotyping algorithms
# ---------------------------------------------------------------------------

def _v4_zygosity_decision(
    reads_info: list[ReadInfo],
    med1: float,
    med2: float,
    d1: float,
    d2: float,
    motif_len: int,
    concordance_threshold: float = 0.7,
) -> tuple[str, float]:
    """Integrated HET/HOM decision with confidence score.

    Allele diff must exceed 1 motif unit to be considered HET,
    matching the GIAB truth definition: |d1 - d2| > motif_len.
    Concordance acts as a filter: low concordance downgrades potential HET to HOM.

    Returns (zygosity, confidence) where confidence is 0.0-1.0.
    """
    if d1 == d2:
        return "HOM", 1.0

    concordance = _hp_concordance(reads_info, med1, med2)

    # Close alleles (diff <= 1 motif unit) → always HOM
    if abs(d1 - d2) <= motif_len:
        return "HOM", 1.0 - concordance

    # Significant allele diff + high concordance → HET
    if concordance > concordance_threshold:
        return "HET", concordance

    # Low concordance → HOM despite large diff (noise)
    if concordance < 0.5:
        return "HOM", 1.0 - concordance

    # Gray zone (0.5 - threshold): significant diff → HET with moderate confidence
    return "HET", concordance


def _v4_str_genotype(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    vntr_cutoff: int = 7,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    concordance_threshold: float = 0.7,
) -> tuple[float, float, str, float]:
    """v4 STR genotyping: HP weighted median + HP=0 assignment + adaptive collapse + concordance."""
    all_sizes = np.array([r.allele_size for r in reads_info])
    all_w = np.maximum(np.array([r.mapq for r in reads_info], dtype=float), 1.0)

    hp1_reads = [r for r in reads_info if r.hp == 1]
    hp2_reads = [r for r in reads_info if r.hp == 2]
    n_hp1, n_hp2 = len(hp1_reads), len(hp2_reads)
    n_total = len(reads_info)
    hp_frac = (n_hp1 + n_hp2) / n_total if n_total > 0 else 0

    # HP sufficiency check
    if n_hp1 >= min_hp_reads and n_hp2 >= min_hp_reads and hp_frac >= min_hp_frac:
        hp1_sizes = np.array([r.allele_size for r in hp1_reads])
        hp1_w = np.maximum(np.array([r.mapq for r in hp1_reads], dtype=float), 1.0)
        hp2_sizes = np.array([r.allele_size for r in hp2_reads])
        hp2_w = np.maximum(np.array([r.mapq for r in hp2_reads], dtype=float), 1.0)

        med1 = _weighted_median(hp1_sizes, hp1_w)
        med2 = _weighted_median(hp2_sizes, hp2_w)

        # Assign HP=0 reads and recompute medians
        aug_hp1, aug_hp2 = _assign_hp0_reads(reads_info, med1, med2)
        aug1_sizes = np.array([r.allele_size for r in aug_hp1])
        aug1_w = np.maximum(np.array([r.mapq for r in aug_hp1], dtype=float), 1.0)
        aug2_sizes = np.array([r.allele_size for r in aug_hp2])
        aug2_w = np.maximum(np.array([r.mapq for r in aug_hp2], dtype=float), 1.0)

        med1 = _weighted_median(aug1_sizes, aug1_w)
        med2 = _weighted_median(aug2_sizes, aug2_w)

        d1 = _compute_diff(ref_size, med1, motif_len, vntr_cutoff)
        d2 = _compute_diff(ref_size, med2, motif_len, vntr_cutoff)

        # Adaptive collapse for close alleles
        if d1 != d2 and abs(d1 - d2) <= motif_len:
            thresh = _adaptive_collapse_threshold(n_total)
            minor_count = min(len(aug_hp1), len(aug_hp2))
            minor_frac = minor_count / n_total if n_total > 0 else 0
            if minor_frac < thresh:
                med_all = _weighted_median(all_sizes, all_w)
                d = _compute_diff(ref_size, med_all, motif_len, vntr_cutoff)
                return float(d), float(d), "HOM", 1.0 - minor_frac

        d1, d2 = sorted([float(d1), float(d2)])
        zyg, conf = _v4_zygosity_decision(
            reads_info, med1, med2, d1, d2, motif_len, concordance_threshold,
        )
        return d1, d2, zyg, conf

    # Insufficient HP: ALL reads fallback
    if _gap_bimodal_test(all_sizes):
        sorted_idx = np.argsort(all_sizes)
        sorted_sizes = all_sizes[sorted_idx]
        gaps = np.diff(sorted_sizes)
        split_pos = int(np.argmax(gaps)) + 1
        lo_idx = sorted_idx[:split_pos]
        hi_idx = sorted_idx[split_pos:]
        med_lo = _weighted_median(all_sizes[lo_idx], all_w[lo_idx])
        med_hi = _weighted_median(all_sizes[hi_idx], all_w[hi_idx])
        d1 = _compute_diff(ref_size, med_lo, motif_len, vntr_cutoff)
        d2 = _compute_diff(ref_size, med_hi, motif_len, vntr_cutoff)
        d1, d2 = sorted([float(d1), float(d2)])
        return d1, d2, "HET" if d1 != d2 else "HOM", 0.5

    med = _weighted_median(all_sizes, all_w)
    d = _compute_diff(ref_size, med, motif_len, vntr_cutoff)
    return float(d), float(d), "HOM", 0.5


def _v4_vntr_genotype(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    vntr_cutoff: int = 7,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    concordance_threshold: float = 0.7,
) -> tuple[float, float, str, float]:
    """v4 VNTR genotyping: gap bimodality + robust median + HP0 assignment + concordance.

    Improvements over previous version:
    - Two-pass MAD trimming (via _robust_vntr_median) for outlier robustness.
    - HP=0 read assignment to nearest haplotype cluster (like STR path).
    - Sizing quality incorporated into confidence score.
    """
    all_sizes = np.array([r.allele_size for r in reads_info])
    all_w = np.maximum(np.array([r.mapq for r in reads_info], dtype=float), 1.0)

    hp1_reads = [r for r in reads_info if r.hp == 1]
    hp2_reads = [r for r in reads_info if r.hp == 2]
    n_hp1, n_hp2 = len(hp1_reads), len(hp2_reads)
    n_total = len(reads_info)
    hp_frac = (n_hp1 + n_hp2) / n_total if n_total > 0 else 0
    hp_sufficient = (
        n_hp1 >= min_hp_reads and n_hp2 >= min_hp_reads and hp_frac >= min_hp_frac
    )

    # Step 1: gap bimodal test
    if _gap_bimodal_test(all_sizes):
        if hp_sufficient:
            hp1_sizes = np.array([r.allele_size for r in hp1_reads])
            hp1_w = np.maximum(np.array([r.mapq for r in hp1_reads], dtype=float), 1.0)
            hp2_sizes = np.array([r.allele_size for r in hp2_reads])
            hp2_w = np.maximum(np.array([r.mapq for r in hp2_reads], dtype=float), 1.0)

            med1 = _robust_vntr_median(hp1_sizes, hp1_w)
            med2 = _robust_vntr_median(hp2_sizes, hp2_w)

            # HP0 assignment and recompute
            aug_hp1, aug_hp2 = _assign_hp0_reads(reads_info, med1, med2)
            aug1_sizes = np.array([r.allele_size for r in aug_hp1])
            aug1_w = np.maximum(np.array([r.mapq for r in aug_hp1], dtype=float), 1.0)
            aug2_sizes = np.array([r.allele_size for r in aug_hp2])
            aug2_w = np.maximum(np.array([r.mapq for r in aug_hp2], dtype=float), 1.0)

            med1 = _robust_vntr_median(aug1_sizes, aug1_w)
            med2 = _robust_vntr_median(aug2_sizes, aug2_w)

            d1 = _compute_diff(ref_size, med1, motif_len, vntr_cutoff)
            d2 = _compute_diff(ref_size, med2, motif_len, vntr_cutoff)
            d1, d2 = sorted([float(d1), float(d2)])

            zyg, zyg_conf = _v4_zygosity_decision(
                reads_info, med1, med2, d1, d2, motif_len, concordance_threshold,
            )
            sq1 = _sizing_quality(aug1_sizes, med1, motif_len)
            sq2 = _sizing_quality(aug2_sizes, med2, motif_len)
            conf = zyg_conf * min(sq1, sq2)
            return d1, d2, zyg, conf
        else:
            # Split at largest gap
            sorted_idx = np.argsort(all_sizes)
            sorted_sizes = all_sizes[sorted_idx]
            gaps = np.diff(sorted_sizes)
            split_pos = int(np.argmax(gaps)) + 1
            lo_idx = sorted_idx[:split_pos]
            hi_idx = sorted_idx[split_pos:]
            med_lo = _robust_vntr_median(all_sizes[lo_idx], all_w[lo_idx])
            med_hi = _robust_vntr_median(all_sizes[hi_idx], all_w[hi_idx])
            d1 = _compute_diff(ref_size, med_lo, motif_len, vntr_cutoff)
            d2 = _compute_diff(ref_size, med_hi, motif_len, vntr_cutoff)
            d1, d2 = sorted([float(d1), float(d2)])
            sq1 = _sizing_quality(all_sizes[lo_idx], med_lo, motif_len)
            sq2 = _sizing_quality(all_sizes[hi_idx], med_hi, motif_len)
            conf = 0.5 * min(sq1, sq2)
            return d1, d2, "HET" if d1 != d2 else "HOM", conf

    # Step 2: unimodal
    if hp_sufficient:
        hp1_sizes = np.array([r.allele_size for r in hp1_reads])
        hp1_w = np.maximum(np.array([r.mapq for r in hp1_reads], dtype=float), 1.0)
        hp2_sizes = np.array([r.allele_size for r in hp2_reads])
        hp2_w = np.maximum(np.array([r.mapq for r in hp2_reads], dtype=float), 1.0)

        med1 = _robust_vntr_median(hp1_sizes, hp1_w)
        med2 = _robust_vntr_median(hp2_sizes, hp2_w)

        # HP0 assignment and recompute
        aug_hp1, aug_hp2 = _assign_hp0_reads(reads_info, med1, med2)
        aug1_sizes = np.array([r.allele_size for r in aug_hp1])
        aug1_w = np.maximum(np.array([r.mapq for r in aug_hp1], dtype=float), 1.0)
        aug2_sizes = np.array([r.allele_size for r in aug_hp2])
        aug2_w = np.maximum(np.array([r.mapq for r in aug_hp2], dtype=float), 1.0)

        med1 = _robust_vntr_median(aug1_sizes, aug1_w)
        med2 = _robust_vntr_median(aug2_sizes, aug2_w)

        d1 = _compute_diff(ref_size, med1, motif_len, vntr_cutoff)
        d2 = _compute_diff(ref_size, med2, motif_len, vntr_cutoff)
        d1, d2 = sorted([float(d1), float(d2)])

        zyg, zyg_conf = _v4_zygosity_decision(
            reads_info, med1, med2, d1, d2, motif_len, concordance_threshold,
        )
        sq1 = _sizing_quality(aug1_sizes, med1, motif_len)
        sq2 = _sizing_quality(aug2_sizes, med2, motif_len)
        conf = zyg_conf * min(sq1, sq2)
        return d1, d2, zyg, conf

    # Unimodal + insufficient HP: all reads -> HOM
    med = _robust_vntr_median(all_sizes, all_w)
    d = _compute_diff(ref_size, med, motif_len, vntr_cutoff)
    sq = _sizing_quality(all_sizes, med, motif_len)
    return float(d), float(d), "HOM", 0.5 * sq


def hp_cond_v4_genotype(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    vntr_motif_cutoff: int = 7,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    concordance_threshold: float = 0.7,
) -> tuple[float, float, str, float]:
    """HPMedian v4: zygosity-focused genotyping with concordance-based HET/HOM.

    Returns (d1, d2, zygosity, confidence) -- 4-tuple unlike v2/v3's 2-tuple.
    """
    if not reads_info:
        return 0.0, 0.0, "HOM", 0.0

    if motif_len < vntr_motif_cutoff:
        return _v4_str_genotype(
            reads_info, ref_size, motif_len,
            vntr_cutoff=vntr_motif_cutoff,
            min_hp_reads=min_hp_reads,
            min_hp_frac=min_hp_frac,
            concordance_threshold=concordance_threshold,
        )

    return _v4_vntr_genotype(
        reads_info, ref_size, motif_len,
        vntr_cutoff=vntr_motif_cutoff,
        min_hp_reads=min_hp_reads,
        min_hp_frac=min_hp_frac,
        concordance_threshold=concordance_threshold,
    )


# ---------------------------------------------------------------------------
# Chunk-based parallel genotyping worker
# ---------------------------------------------------------------------------

def _genotype_chunk(
    bam_path: str,
    loci_chunk: list[tuple[str, int, int, str]],
    min_mapq: int = 5,
    min_flank: int = 50,
    max_reads: int = 200,
    vntr_motif_cutoff: int = 7,
    ref_path: Optional[str] = None,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    concordance_threshold: float = 0.7,
) -> list[Optional[dict]]:
    """Genotype a chunk of loci (runs in a worker process).

    Returns:
        List of result dicts (or None for failed loci), one per locus.
    """
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
            if not reads:
                results.append(None)
                continue
            d1, d2, zygosity, confidence = hp_cond_v4_genotype(
                reads, ref_size, motif_len,
                vntr_motif_cutoff=vntr_motif_cutoff,
                min_hp_reads=min_hp_reads,
                min_hp_frac=min_hp_frac,
                concordance_threshold=concordance_threshold,
            )
            allele1_size = ref_size - d1
            allele2_size = ref_size - d2
            results.append({
                "allele1_size": allele1_size,
                "allele2_size": allele2_size,
                "zygosity": zygosity,
                "n_reads": len(reads),
                "confidence": confidence,
            })
    finally:
        bam.close()
        if ref_fasta:
            ref_fasta.close()
    return results


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _write_output_bed(
    output_path: str,
    loci: list[tuple[str, int, int, str]],
    results: list[Optional[dict]],
    min_confidence: float = 0.0,
) -> int:
    """Write genotyping results to 9-column BED format.

    Columns: chrom start end motif allele1 allele2 zygosity confidence n_reads

    Args:
        output_path: Output file path.
        loci: List of (chrom, start, end, motif) tuples.
        results: List of result dicts (or None for failed loci).
        min_confidence: Minimum confidence threshold. Loci below this
            are written as no-call (all fields ".") to preserve row count.

    Returns:
        Number of loci that passed the confidence filter.
    """
    def _fmt(size):
        import math
        if math.isnan(size) or math.isinf(size):
            return "."
        if size == int(size):
            return str(int(size))
        return f"{size:.1f}"

    n_passed = 0
    with open(output_path, "w") as f:
        f.write("#chrom\tstart\tend\tmotif\tallele1_size\tallele2_size\t"
                "zygosity\tconfidence\tn_reads\n")
        for locus, result in zip(loci, results):
            chrom, start, end, motif = locus
            if result is None:
                f.write(f"{chrom}\t{start}\t{end}\t{motif}\t.\t.\t.\t.\t0\n")
                continue
            conf = result.get("confidence", 0.0)
            if math.isnan(conf) or math.isinf(conf) or conf < min_confidence:
                f.write(f"{chrom}\t{start}\t{end}\t{motif}\t.\t.\t.\t.\t0\n")
                continue
            n_passed += 1
            conf_str = f"{conf:.3f}"
            f.write(f"{chrom}\t{start}\t{end}\t{motif}\t"
                    f"{_fmt(result['allele1_size'])}\t"
                    f"{_fmt(result['allele2_size'])}\t"
                    f"{result['zygosity']}\t{conf_str}\t{result['n_reads']}\n")
    return n_passed


# ---------------------------------------------------------------------------
# Main genotyping pipeline
# ---------------------------------------------------------------------------

def _group_loci_by_chrom(
    loci: list[tuple[str, int, int, str]],
) -> list[tuple[str, list[tuple[str, int, int, str]]]]:
    """Group loci by chromosome, preserving order within each chromosome.

    Returns list of (chrom, loci_list) sorted by chromosome order.
    """
    from collections import OrderedDict
    from .utils import CHROM_ORDER

    groups: dict[str, list[tuple[str, int, int, str]]] = OrderedDict()
    for locus in loci:
        chrom = locus[0]
        if chrom not in groups:
            groups[chrom] = []
        groups[chrom].append(locus)

    # Sort by CHROM_ORDER (unknown chroms go last)
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: CHROM_ORDER.get(x[0], 999),
    )
    return sorted_groups


def genotype(
    bam_path: str,
    loci_bed_path: str,
    output_path: str,
    nprocs: int = 8,
    chunk_size: int = 500,
    min_mapq: int = 5,
    min_flank: int = 50,
    max_reads: int = 200,
    vntr_motif_cutoff: int = 7,
    ref_path: Optional[str] = None,
    min_hp_reads: int = 3,
    min_hp_frac: float = 0.15,
    concordance_threshold: float = 0.7,
    min_confidence: float = 0.0,
) -> str:
    """Run MosaicTR genotyping on a set of TR loci.

    Pipeline: BAM + loci BED -> per-chromosome genotyping -> BED output

    Processes loci chromosome-by-chromosome with fresh BAM handles to
    avoid segfaults that occur when processing many loci at once.

    Args:
        bam_path: Input BAM file (with HP tags for best results).
        loci_bed_path: 4-column BED with loci to genotype.
        output_path: Output BED path.
        nprocs: Number of parallel worker processes.
        chunk_size: Loci per worker chunk.
        min_mapq: Minimum mapping quality.
        min_flank: Minimum flanking bases.
        max_reads: Maximum reads per locus.
        vntr_motif_cutoff: Motif length cutoff for STR vs VNTR.
        ref_path: Reference FASTA for parasail realignment (optional).
        min_hp_reads: Minimum HP-tagged reads per haplotype.
        min_hp_frac: Minimum fraction of HP-tagged reads.
        concordance_threshold: HP concordance threshold for HET call.
        min_confidence: Minimum confidence to report a call (default: 0, no filter).

    Returns:
        Path to output BED file.
    """
    if ref_path:
        logger.info("Parasail realignment enabled (ref: %s)", ref_path)

    loci = load_loci_bed(loci_bed_path)
    logger.info("Loaded %d loci from %s", len(loci), loci_bed_path)

    chrom_groups = _group_loci_by_chrom(loci)
    logger.info("Split into %d chromosomes: %s",
                len(chrom_groups),
                ", ".join(f"{c}({len(l)})" for c, l in chrom_groups))

    chunk_kwargs = dict(
        min_mapq=min_mapq, min_flank=min_flank, max_reads=max_reads,
        vntr_motif_cutoff=vntr_motif_cutoff,
        ref_path=ref_path,
        min_hp_reads=min_hp_reads,
        min_hp_frac=min_hp_frac,
        concordance_threshold=concordance_threshold,
    )

    t0 = time.time()
    all_results: list[Optional[dict]] = []
    all_loci: list[tuple[str, int, int, str]] = []

    for chrom, chrom_loci in chrom_groups:
        chrom_t0 = time.time()
        chunks = [chrom_loci[i:i + chunk_size]
                  for i in range(0, len(chrom_loci), chunk_size)]

        if nprocs <= 1:
            for chunk in tqdm(chunks, desc=f"Genotyping {chrom}"):
                chunk_results = _genotype_chunk(bam_path, chunk, **chunk_kwargs)
                all_results.extend(chunk_results)
                all_loci.extend(chunk)
        else:
            max_pending = nprocs * 2
            with ProcessPoolExecutor(max_workers=nprocs) as executor:
                pbar = tqdm(total=len(chunks), desc=f"Genotyping {chrom}")
                pending: dict = {}
                chunk_iter = iter(enumerate(chunks))

                def _submit_next():
                    try:
                        i, chunk = next(chunk_iter)
                        fut = executor.submit(
                            _genotype_chunk, bam_path, chunk, **chunk_kwargs,
                        )
                        pending[fut] = (i, chunk)
                    except StopIteration:
                        pass

                # Seed initial batch
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
        chrom_genotyped = sum(
            1 for r in all_results[len(all_results) - len(chrom_loci):]
            if r is not None
        )
        logger.info("%s: %d/%d loci in %.1fs",
                    chrom, chrom_genotyped, len(chrom_loci), chrom_elapsed)

    elapsed = time.time() - t0
    genotyped = sum(1 for r in all_results if r is not None)
    logger.info("Genotyped %d/%d loci in %.1fs (%.0f loci/sec)",
                genotyped, len(all_loci), elapsed,
                len(all_loci) / elapsed if elapsed > 0 else 0)

    n_passed = _write_output_bed(
        output_path, all_loci, all_results, min_confidence=min_confidence,
    )
    if min_confidence > 0:
        logger.info(
            "Output written to %s (%d/%d passed min_confidence=%.2f)",
            output_path, n_passed, genotyped, min_confidence,
        )
    else:
        logger.info("Output written to %s", output_path)
    return output_path
