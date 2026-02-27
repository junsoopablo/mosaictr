#!/usr/bin/env python3
"""Head-to-head comparison of TR genotyping tools on the same test set.

Compares: LongTR, TRGT, DeepTR Hybrid (oracle + split_median)
Ground truth: GIAB Tier1 allele diffs
Test set: chr21, chr22, chrX

Comparison uses allele SIZE DIFFERENCES (expansion/contraction from reference).
This is reference-frame-invariant: the same physical repeat expansion produces
the same diff regardless of locus boundary definition.

TRGT fix: VCF REF includes an anchor base, so ref_size = len(REF) - 1.
"""

import argparse
import bisect
import gzip
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pysam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALLELE_SIZE_IDX = 0
REF_SIZE_IDX = 0
MOTIF_LEN_IDX = 1
TEST_CHROMS = {"chr21", "chr22", "chrX"}
MATCH_TOLERANCE = 15


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(true_diffs, pred_diffs, motif_lens):
    """Compute metrics on allele diffs (ref_size - allele_size)."""
    true = np.array(true_diffs, dtype=np.float64)
    pred = np.array(pred_diffs, dtype=np.float64)
    ml = np.array(motif_lens, dtype=np.float64).reshape(-1, 1)
    n = len(true)
    if n == 0:
        return {}

    errs = np.abs(true - pred)
    flat_errs = errs.flatten()
    motif_errs = errs / np.maximum(ml, 1.0)

    true_zyg = (np.abs(true[:, 1] - true[:, 0]) > ml.flatten()).astype(int)
    pred_zyg = (np.abs(pred[:, 1] - pred[:, 0]) > ml.flatten()).astype(int)

    flat_true = true.flatten()
    flat_pred = pred.flatten()
    ss_res = np.sum((flat_true - flat_pred) ** 2)
    ss_tot = np.sum((flat_true - np.mean(flat_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "n": n,
        "exact": np.all(errs < 0.5, axis=1).mean(),
        "w1bp": (flat_errs <= 1.0).mean(),
        "w1motif": (motif_errs.flatten() <= 1.0).mean(),
        "w5bp": (flat_errs <= 5.0).mean(),
        "mae": flat_errs.mean(),
        "median_ae": np.median(flat_errs),
        "r2": r2,
        "zyg_acc": (pred_zyg == true_zyg).mean(),
        "geno_conc": np.all(motif_errs <= 1.0, axis=1).mean(),
    }


# ── Parsers (return DIFFS: ref_size - allele_size) ──────────────────────

def parse_longtr_vcf(vcf_path, chroms):
    """Parse LongTR VCF → allele diffs.

    BPDIFFS = allele_expansion from ref. diff = -BPDIFFS.
    Returns: dict[(chrom, start)] -> (diff1, diff2, ref_size) sorted by diff.
    """
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open

    t0 = time.time()
    n_parsed = 0
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chrom not in chroms:
                continue

            info = fields[7]
            start = end = None
            bpdiffs_str = None
            for kv in info.split(";"):
                if kv.startswith("START="):
                    start = int(kv[6:])
                elif kv.startswith("END="):
                    end = int(kv[4:])
                elif kv.startswith("BPDIFFS="):
                    bpdiffs_str = kv[8:]

            if start is None or end is None:
                continue
            ref_size = end - start

            if bpdiffs_str is None or bpdiffs_str == ".":
                results[(chrom, start)] = (0.0, 0.0, ref_size)
                n_parsed += 1
                continue

            parts = bpdiffs_str.split(",")
            fmt_keys = fields[8].split(":")
            fmt_vals = fields[9].split(":")
            gt_idx = fmt_keys.index("GT")
            gt_str = fmt_vals[gt_idx]

            sep = "|" if "|" in gt_str else "/"
            a1, a2 = int(gt_str.split(sep)[0]), int(gt_str.split(sep)[1])

            all_bpdiffs = [0] + [int(x) for x in parts]
            if a1 < len(all_bpdiffs) and a2 < len(all_bpdiffs):
                d1 = -all_bpdiffs[a1]
                d2 = -all_bpdiffs[a2]
            else:
                d1 = d2 = 0.0

            d1, d2 = sorted([d1, d2])
            results[(chrom, start)] = (d1, d2, ref_size)
            n_parsed += 1

    logger.info(f"LongTR: parsed {n_parsed:,} loci in {time.time()-t0:.1f}s")
    return results


def parse_trgt_vcf(vcf_path, chroms):
    """Parse TRGT VCF → allele diffs.

    TRGT VCF REF includes a 1bp anchor base, so ref_size = len(REF) - 1.
    AL = allele lengths (repeat portion only, no anchor).
    diff = ref_size - AL.

    Returns: dict[(chrom, start_0based)] -> (diff1, diff2, ref_size) sorted.
    """
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open

    t0 = time.time()
    n_parsed = 0
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chrom not in chroms:
                continue

            pos = int(fields[1])  # 1-based VCF POS (anchor base position)

            # ref_size = len(REF) - 1 to remove anchor base
            ref_size = len(fields[3]) - 1

            fmt_keys = fields[8].split(":")
            fmt_vals = fields[9].split(":")
            try:
                al_idx = fmt_keys.index("AL")
                al_str = fmt_vals[al_idx]
            except (ValueError, IndexError):
                continue

            if al_str == ".":
                continue
            al_parts = al_str.split(",")
            if len(al_parts) < 2:
                continue
            try:
                a1_len = int(al_parts[0])
                a2_len = int(al_parts[1])
            except ValueError:
                continue

            d1 = ref_size - a1_len
            d2 = ref_size - a2_len
            d1, d2 = sorted([d1, d2])

            # 0-based start = POS (VCF 1-based anchor) → actual repeat starts at POS
            start_0based = pos
            results[(chrom, start_0based)] = (d1, d2, ref_size)
            n_parsed += 1

    logger.info(f"TRGT: parsed {n_parsed:,} loci in {time.time()-t0:.1f}s")
    return results


# ── Oracle split-median ──────────────────────────────────────────────────

def split_median_genotype(allele_sizes, ref_size):
    """Sort reads, split at midpoint, return diffs."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = ref_size - sizes[0]
        return d, d
    mid = n // 2
    a1 = np.median(sizes[:mid])
    a2 = np.median(sizes[mid:])
    d1 = ref_size - a1
    d2 = ref_size - a2
    return sorted([d1, d2])


def split_median_round_genotype(allele_sizes, ref_size):
    """Split-median with integer rounding (no gate version)."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    a1 = np.median(sizes[:mid])
    a2 = np.median(sizes[mid:])
    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def mode_round_genotype(allele_sizes, ref_size):
    """Split using MODE (most frequent value) + integer rounding."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    mid = n // 2
    half1 = sizes[:mid]
    half2 = sizes[mid:]

    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    a1 = get_mode(half1)
    a2 = get_mode(half2)
    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def prop_mode_genotype(allele_sizes, ref_size, collapse_threshold=0.30):
    """Proportion-aware mode: midpoint split + mode, collapse to hom when
    half-modes differ by 1bp and minor allele has < threshold fraction of reads.

    This correctly identifies hom loci (where noise causes 1bp mode split)
    while preserving close het calls (where both alleles have read support).
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    mid = n // 2

    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        # Check read support for each mode
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < collapse_threshold:
            # Noise-induced split: collapse to single mode
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
        else:
            # True close het
            d1 = round(ref_size - m1)
            d2 = round(ref_size - m2)
            return sorted([d1, d2])
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


# ── HP-aware genotyping ──────────────────────────────────────────────────

MIN_FLANK = 50
MIN_MAPQ = 5
MAX_READS_HP = 200


def compute_allele_size_cigar(aln, locus_start, locus_end):
    """Compute allele size (query bases within locus) from CIGAR."""
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
    return qbases if qbases > 0 else None


def extract_reads_with_hp(bam, chrom, start, end):
    """Extract allele sizes and HP tags from BAM reads at a locus."""
    reads = []
    seen = set()
    try:
        fetched = bam.fetch(chrom, max(0, start - MIN_FLANK), end + MIN_FLANK)
    except ValueError:
        return reads
    for aln in fetched:
        if len(reads) >= MAX_READS_HP:
            break
        if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
            continue
        if aln.mapping_quality < MIN_MAPQ:
            continue
        if aln.query_name in seen:
            continue
        seen.add(aln.query_name)
        if aln.reference_start is None or aln.reference_end is None:
            continue
        if aln.reference_start > start - MIN_FLANK or aln.reference_end < end + MIN_FLANK:
            continue
        allele_size = compute_allele_size_cigar(aln, start, end)
        if allele_size is None:
            continue
        try:
            hp = aln.get_tag('HP')
        except KeyError:
            hp = 0
        reads.append((float(allele_size), int(hp)))
    return reads


def hp_median_genotype(sizes_hp, ref_size):
    """HP separation + median within each group."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - np.median(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - np.median(hp1))
        return d, d
    d1 = round(ref_size - np.median(hp1))
    d2 = round(ref_size - np.median(hp2))
    return sorted([d1, d2])


def hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25):
    """HP median + proportion-based hom collapse."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(hp1) == 0 and len(hp2) == 0:
        return prop_mode_genotype(all_sizes, ref_size, threshold)
    if len(hp1) == 0:
        d = round(ref_size - np.median(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - np.median(hp1))
        return d, d
    med1 = np.median(hp1)
    med2 = np.median(hp2)
    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)
    if d1 == d2:
        return d1, d2
    elif abs(d1 - d2) == 1:
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(int_sizes == int(round(med1)))
        c2 = np.sum(int_sizes == int(round(med2)))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < threshold:
            d = round(ref_size - np.median(all_sizes))
            return d, d
    return sorted([d1, d2])


def hp_cond_genotype(sizes_hp, ref_size, motif_len):
    """Conditional: hp_median for VNTR (motif>=7), hp_median_prop for dinuc/STR."""
    if motif_len >= 7:
        return hp_median_genotype(sizes_hp, ref_size)
    else:
        return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)


def _trimmed_median(arr, iqr_factor=1.5):
    """IQR-based trimmed median."""
    if len(arr) < 4:
        return np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    trimmed = arr[(arr >= q1 - iqr_factor * iqr) & (arr <= q3 + iqr_factor * iqr)]
    return np.median(trimmed) if len(trimmed) > 0 else np.median(arr)


def _is_bimodal(all_sizes, threshold=0.05):
    """Check if read size distribution is bimodal (IQR > threshold * median)."""
    if len(all_sizes) < 6:
        return False
    iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
    median_size = np.median(all_sizes)
    return median_size > 0 and iqr / median_size > threshold


def hp_cond_v2_genotype(sizes_hp, ref_size, motif_len):
    """HPMedian v2: bimodal-first + trimmed median for VNTR, hp_median_prop for STR/dinuc.

    For VNTRs:
    1. Check bimodality on raw reads (IQR > 5% of median)
    2. If bimodal AND HP says homozygous → split_median (captures het without HP)
    3. If not bimodal → trimmed HP median (1.5*IQR outlier removal)

    For dinuc/STR: same as v1 (hp_median_prop with 25% collapse threshold)
    """
    if motif_len < 7:
        return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)

    # VNTR path
    if not sizes_hp:
        return 0.0, 0.0

    all_sizes = np.array([s for s, _ in sizes_hp])

    # Step 1: bimodal check on raw reads
    if _is_bimodal(all_sizes, threshold=0.05):
        # Check HP median first — if HP gives het, trust it
        hp_res = hp_median_genotype(sizes_hp, ref_size)
        if hp_res[0] != hp_res[1]:
            return hp_res
        # HP says hom but reads are bimodal → split_median
        sizes_sorted = np.sort(all_sizes)
        n = len(sizes_sorted)
        mid = n // 2
        d1 = round(ref_size - np.median(sizes_sorted[:mid]))
        d2 = round(ref_size - np.median(sizes_sorted[mid:]))
        return sorted([d1, d2])

    # Step 2: not bimodal → trimmed HP median
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - _trimmed_median(hp2, 1.5))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - _trimmed_median(hp1, 1.5))
        return d, d

    d1 = round(ref_size - _trimmed_median(hp1, 1.5))
    d2 = round(ref_size - _trimmed_median(hp2, 1.5))
    return sorted([d1, d2])


def weighted_median(values, weights):
    """Compute weighted median."""
    if len(values) == 0:
        return 0.0
    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sw = weights[sorted_idx]
    cumw = np.cumsum(sw)
    cutoff = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, cutoff)
    return float(sv[min(idx, len(sv)-1)])


def hp_wmedian_genotype(sizes, mapqs, hps, ref_size, motif_len):
    """HP separation + mapq-weighted median.

    Uses mapq as weight: higher mapq = better alignment = more accurate allele size.
    Conditional collapse: VNTR no collapse, dinuc/STR collapse with prop threshold.
    """
    if len(sizes) == 0:
        return 0.0, 0.0

    # Separate by HP
    hp1_idx = hps == 1
    hp2_idx = hps == 2
    all_w = np.maximum(mapqs, 1.0)  # min weight 1

    def wmed(mask):
        s = sizes[mask]
        w = all_w[mask]
        if len(s) == 0:
            return None
        return weighted_median(s, w)

    m1 = wmed(hp1_idx)
    m2 = wmed(hp2_idx)

    # Fallback if no HP tags
    if m1 is None and m2 is None:
        m = weighted_median(sizes, all_w)
        d = round(ref_size - m)
        return d, d
    if m1 is None:
        d = round(ref_size - m2)
        return d, d
    if m2 is None:
        d = round(ref_size - m1)
        return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)

    # Collapse logic for dinuc/STR (motif < 7)
    if motif_len < 7 and d1 != d2 and abs(d1 - d2) == 1:
        int_sizes = np.round(sizes).astype(int)
        s1 = int(round(m1))
        s2 = int(round(m2))
        c1 = np.sum(int_sizes == s1)
        c2 = np.sum(int_sizes == s2)
        minor_frac = min(c1, c2) / len(sizes)
        if minor_frac < 0.25:
            m_all = weighted_median(sizes, all_w)
            d = round(ref_size - m_all)
            return d, d

    return sorted([d1, d2])


# ── Matching ─────────────────────────────────────────────────────────────

def match_loci(h5_keys, h5_ref_sizes, tool_dict, tolerance=MATCH_TOLERANCE):
    """Match H5 test loci to tool results.

    Also requires ref_size within 50% to avoid catastrophic mismatches.
    """
    by_chrom = defaultdict(list)
    for (chrom, start), val in tool_dict.items():
        by_chrom[chrom].append((start, val))
    for chrom in by_chrom:
        by_chrom[chrom].sort()

    matched_idx = []
    matched_vals = []
    unmatched = 0

    for i, (chrom, start, end) in enumerate(h5_keys):
        candidates = by_chrom.get(chrom, [])
        if not candidates:
            unmatched += 1
            continue

        h5_ref = h5_ref_sizes[i]
        pos = bisect.bisect_left(candidates, (start,))

        best_dist = tolerance + 1
        best_val = None
        for j in range(max(0, pos - 2), min(len(candidates), pos + 3)):
            dist = abs(candidates[j][0] - start)
            tool_ref = candidates[j][1][2]  # ref_size from tool
            # Also check ref_size similarity to avoid catastrophic mismatches
            if h5_ref > 0 and tool_ref > 0:
                ratio = tool_ref / h5_ref
                if ratio < 0.5 or ratio > 2.0:
                    continue  # Skip: ref sizes too different
            if dist < best_dist:
                best_dist = dist
                best_val = candidates[j][1]

        if best_dist <= tolerance and best_val is not None:
            matched_idx.append(i)
            matched_vals.append(best_val)
        else:
            unmatched += 1

    return matched_idx, matched_vals, unmatched


# ── Report ───────────────────────────────────────────────────────────────

HEADER = "{:<35} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
    "Tool", "Exact", "≤1bp", "≤1mu", "≤5bp", "MAE", "ZygAcc", "GenConc")
SEP = "-" * len(HEADER)


def fmt_row(name, m):
    return "{:<35} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.2f} {:>6.1%} {:>6.1%}".format(
        name, m['exact'], m['w1bp'], m['w1motif'], m['w5bp'],
        m['mae'], m['zyg_acc'], m['geno_conc'])


def main():
    parser = argparse.ArgumentParser(description="Head-to-head tool comparison")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--longtr-vcf", required=True)
    parser.add_argument("--trgt-vcf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bam", default=None, help="BAM file with HP tags for HPMedian")
    parser.add_argument("--hp-cache", default=None, help="Cached HP reads (pickle)")
    parser.add_argument("--tolerance", type=int, default=MATCH_TOLERANCE)
    args = parser.parse_args()

    # ── Load H5 ──────────────────────────────────────────────────────
    logger.info(f"Loading HDF5: {args.h5}")
    h5 = h5py.File(args.h5, "r")
    all_chroms = [c.decode() for c in h5["chroms"][:]]
    all_starts = h5["starts"][:]
    all_ends = h5["ends"][:]
    all_labels = h5["labels"][:]
    all_locus_features = h5["locus_features"][:]
    all_read_features = h5["read_features"][:]
    all_read_offsets = h5["read_offsets"][:]
    all_read_counts = h5["read_counts"][:]
    tp_statuses = [s.decode() for s in h5["tp_statuses"][:]]

    test_indices = [i for i, c in enumerate(all_chroms) if c in TEST_CHROMS]
    logger.info(f"Test set: {len(test_indices):,} loci")

    h5_keys = []
    true_diffs = []
    motif_lens = []
    is_tp = []
    ref_sizes = []

    for idx in test_indices:
        chrom = all_chroms[idx]
        start = int(all_starts[idx])
        end = int(all_ends[idx])
        h5_keys.append((chrom, start, end))

        h1, h2, ml = all_labels[idx]
        d1, d2 = sorted([h1, h2])
        true_diffs.append([d1, d2])
        motif_lens.append(ml)
        ref_sizes.append(all_locus_features[idx, REF_SIZE_IDX])
        is_tp.append("TP" in tp_statuses[idx])

    true_diffs = np.array(true_diffs)
    motif_lens = np.array(motif_lens)
    is_tp = np.array(is_tp)
    ref_sizes = np.array(ref_sizes)

    n_tp = is_tp.sum()
    n_tn = (~is_tp).sum()
    logger.info(f"Ground truth: {n_tp:,} TP, {n_tn:,} TN")

    # ── Oracle split-median (gated: TP only) ────────────────────────
    logger.info("Computing oracle split-median (gated)...")
    hybrid_diffs = np.zeros_like(true_diffs)
    t0 = time.time()
    for i, idx in enumerate(test_indices):
        if is_tp[i]:
            offset = all_read_offsets[idx]
            count = all_read_counts[idx]
            reads = all_read_features[offset:offset + count]
            allele_sizes = reads[:, ALLELE_SIZE_IDX]
            ref_size = all_locus_features[idx, REF_SIZE_IDX]
            d1, d2 = split_median_genotype(allele_sizes, ref_size)
            hybrid_diffs[i] = [d1, d2]
    logger.info(f"Oracle split-median (gated): {time.time()-t0:.1f}s")

    # ── No-gate mode-round (ALL loci) ───────────────────────────
    logger.info("Computing no-gate mode-round...")
    nogate_diffs = np.zeros_like(true_diffs)
    t0 = time.time()
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        d1, d2 = mode_round_genotype(allele_sizes, ref_size)
        nogate_diffs[i] = [d1, d2]
    logger.info(f"No-gate mode-round: {time.time()-t0:.1f}s")

    # ── No-gate prop-mode (ALL loci) ──────────────────────────
    logger.info("Computing no-gate prop-mode (30% collapse)...")
    propmode_diffs = np.zeros_like(true_diffs)
    t0 = time.time()
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        d1, d2 = prop_mode_genotype(allele_sizes, ref_size)
        propmode_diffs[i] = [d1, d2]
    logger.info(f"No-gate prop-mode: {time.time()-t0:.1f}s")

    h5.close()

    # ── HPMedian (haplotag-aware conditional median) ────────────────
    hpmed_diffs = np.zeros_like(true_diffs)
    if args.bam or args.hp_cache:
        cache_path = Path(args.hp_cache) if args.hp_cache else Path(args.output).with_suffix(".hp_cache.pkl")
        if cache_path.exists():
            logger.info(f"Loading HP cache from {cache_path}")
            with open(cache_path, "rb") as f:
                all_reads_hp = pickle.load(f)["all_reads_hp"]
        else:
            logger.info(f"Extracting HP reads from BAM: {args.bam}")
            bam = pysam.AlignmentFile(args.bam, "rb")
            all_reads_hp = []
            for i, (chrom, start, end) in enumerate(h5_keys):
                if i % 5000 == 0 and i > 0:
                    logger.info(f"  HP extraction: {i:,}/{len(h5_keys):,}")
                all_reads_hp.append(extract_reads_with_hp(bam, chrom, start, end))
            bam.close()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"all_reads_hp": all_reads_hp}, f)
            logger.info(f"Saved HP cache to {cache_path}")

        logger.info("Computing HPMedian (hp_cond) genotypes...")
        t0 = time.time()
        for i in range(len(h5_keys)):
            reads_hp = all_reads_hp[i]
            ref = ref_sizes[i]
            ml = motif_lens[i]
            if reads_hp:
                hpmed_diffs[i] = hp_cond_genotype(reads_hp, ref, ml)
            else:
                hpmed_diffs[i] = propmode_diffs[i]
        logger.info(f"HPMedian: {time.time()-t0:.1f}s")

        # HPMedian_H5: Use HDF5 allele sizes (more accurate) with BAM HP tags
        logger.info("Computing HPMedian_H5 (HDF5 sizes + BAM HP tags)...")
        t0 = time.time()
        hpmed_h5_diffs = np.zeros_like(true_diffs)
        read_count_matches = 0
        for i, idx in enumerate(test_indices):
            h5_offset = all_read_offsets[idx]
            h5_count = all_read_counts[idx]
            h5_allele_sizes = all_read_features[h5_offset:h5_offset+h5_count, ALLELE_SIZE_IDX]
            bam_reads = all_reads_hp[i]
            ref = ref_sizes[i]
            ml = motif_lens[i]

            if h5_count == len(bam_reads) and h5_count > 0:
                # Read counts match — pair HDF5 sizes with BAM HP tags
                read_count_matches += 1
                sizes_hp = [(float(h5_allele_sizes[j]), bam_reads[j][1])
                            for j in range(h5_count)]
                hpmed_h5_diffs[i] = hp_cond_genotype(sizes_hp, ref, ml)
            elif bam_reads:
                # Fallback to BAM sizes
                hpmed_h5_diffs[i] = hp_cond_genotype(bam_reads, ref, ml)
            else:
                hpmed_h5_diffs[i] = propmode_diffs[i]
        logger.info(f"HPMedian_H5: {time.time()-t0:.1f}s, "
                     f"read count matches: {read_count_matches}/{len(test_indices)}")

        # HPMedian_v2: bimodal-first + trimmed median for VNTRs
        logger.info("Computing HPMedian_v2 (bimodal_first + trimmed)...")
        t0 = time.time()
        hpmed_v2_diffs = np.zeros_like(true_diffs)
        for i in range(len(h5_keys)):
            reads_hp = all_reads_hp[i]
            ref = ref_sizes[i]
            ml = motif_lens[i]
            if reads_hp:
                hpmed_v2_diffs[i] = hp_cond_v2_genotype(reads_hp, ref, ml)
            else:
                hpmed_v2_diffs[i] = propmode_diffs[i]
        logger.info(f"HPMedian_v2: {time.time()-t0:.1f}s")
    else:
        logger.info("No --bam/--hp-cache provided, HPMedian = PropMode fallback")
        hpmed_diffs = propmode_diffs.copy()
        hpmed_h5_diffs = propmode_diffs.copy()
        hpmed_v2_diffs = propmode_diffs.copy()

    # ── Parse tools ──────────────────────────────────────────────────
    longtr = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)
    trgt = parse_trgt_vcf(args.trgt_vcf, TEST_CHROMS)
    tools = {"LongTR": longtr, "TRGT": trgt}

    # ── Match loci ───────────────────────────────────────────────────
    tool_pred_diffs = {}
    tool_matched_mask = {}

    for name, tool_dict in tools.items():
        logger.info(f"Matching {name}...")
        matched_idx, matched_vals, unmatched = match_loci(
            h5_keys, ref_sizes, tool_dict, tolerance=args.tolerance
        )
        n_matched = len(matched_idx)
        logger.info(f"  {name}: {n_matched:,} matched, {unmatched:,} unmatched")

        mask = np.zeros(len(h5_keys), dtype=bool)
        pred = np.zeros((len(h5_keys), 2), dtype=np.float64)
        for j, val in zip(matched_idx, matched_vals):
            mask[j] = True
            pred[j] = [val[0], val[1]]

        tool_pred_diffs[name] = pred
        tool_matched_mask[name] = mask

    # ── Compute all results ──────────────────────────────────────────
    logger.info("Computing metrics...")

    common_mask = np.ones(len(h5_keys), dtype=bool)
    for name in tools:
        common_mask &= tool_matched_mask[name]
    n_common = common_mask.sum()
    logger.info(f"Common loci: {n_common:,}")

    longtr_mask = tool_matched_mask["LongTR"]
    n_longtr = longtr_mask.sum()

    R = {}

    # Per-tool own set
    for name in tools:
        m = tool_matched_mask[name]
        R[f"{name} (own)"] = compute_metrics(
            true_diffs[m], tool_pred_diffs[name][m], motif_lens[m])

    # Hybrid full (gated)
    R["Hybrid (full)"] = compute_metrics(true_diffs, hybrid_diffs, motif_lens)
    # No-gate full
    R["NoGate (full)"] = compute_metrics(true_diffs, nogate_diffs, motif_lens)
    # PropMode full
    R["PropMode (full)"] = compute_metrics(true_diffs, propmode_diffs, motif_lens)
    # HPMedian full
    R["HPMedian (full)"] = compute_metrics(true_diffs, hpmed_diffs, motif_lens)
    R["HPMedian_H5 (full)"] = compute_metrics(true_diffs, hpmed_h5_diffs, motif_lens)
    R["HPMedian_v2 (full)"] = compute_metrics(true_diffs, hpmed_v2_diffs, motif_lens)

    # Common set
    for name in tools:
        R[f"{name} (common)"] = compute_metrics(
            true_diffs[common_mask], tool_pred_diffs[name][common_mask],
            motif_lens[common_mask])
    R["Hybrid (common)"] = compute_metrics(
        true_diffs[common_mask], hybrid_diffs[common_mask],
        motif_lens[common_mask])
    R["NoGate (common)"] = compute_metrics(
        true_diffs[common_mask], nogate_diffs[common_mask],
        motif_lens[common_mask])
    R["PropMode (common)"] = compute_metrics(
        true_diffs[common_mask], propmode_diffs[common_mask],
        motif_lens[common_mask])
    R["HPMedian (common)"] = compute_metrics(
        true_diffs[common_mask], hpmed_diffs[common_mask],
        motif_lens[common_mask])
    R["HPMedian_v2 (common)"] = compute_metrics(
        true_diffs[common_mask], hpmed_v2_diffs[common_mask],
        motif_lens[common_mask])

    # LongTR set
    R["LongTR (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], tool_pred_diffs["LongTR"][longtr_mask],
        motif_lens[longtr_mask])
    R["Hybrid (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], hybrid_diffs[longtr_mask],
        motif_lens[longtr_mask])
    R["NoGate (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], nogate_diffs[longtr_mask],
        motif_lens[longtr_mask])
    R["PropMode (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], propmode_diffs[longtr_mask],
        motif_lens[longtr_mask])
    R["HPMedian (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], hpmed_diffs[longtr_mask],
        motif_lens[longtr_mask])
    R["HPMedian_H5 (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], hpmed_h5_diffs[longtr_mask],
        motif_lens[longtr_mask])
    R["HPMedian_v2 (ltr_set)"] = compute_metrics(
        true_diffs[longtr_mask], hpmed_v2_diffs[longtr_mask],
        motif_lens[longtr_mask])

    # Stratified: TP / TN
    for sub, sub_mask in [("TP", is_tp), ("TN", ~is_tp)]:
        for base_name, base_mask in [("common", common_mask), ("ltr_set", longtr_mask)]:
            sm = base_mask & sub_mask
            if sm.sum() == 0:
                continue
            for name in tools:
                if base_name == "ltr_set" and name == "TRGT":
                    continue
                R[f"{name} ({sub}, {base_name})"] = compute_metrics(
                    true_diffs[sm], tool_pred_diffs[name][sm], motif_lens[sm])
            R[f"Hybrid ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], hybrid_diffs[sm], motif_lens[sm])
            R[f"NoGate ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], nogate_diffs[sm], motif_lens[sm])
            R[f"PropMode ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], propmode_diffs[sm], motif_lens[sm])
            R[f"HPMedian ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], hpmed_diffs[sm], motif_lens[sm])
            R[f"HPMedian_H5 ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], hpmed_h5_diffs[sm], motif_lens[sm])
            R[f"HPMedian_v2 ({sub}, {base_name})"] = compute_metrics(
                true_diffs[sm], hpmed_v2_diffs[sm], motif_lens[sm])

    # Stratified: motif
    motif_groups = {
        "dinucleotide": motif_lens == 2,
        "STR (3-6bp)": (motif_lens >= 3) & (motif_lens <= 6),
        "VNTR (7+bp)": motif_lens >= 7,
    }
    for gn, gm_base in motif_groups.items():
        gm = common_mask & gm_base
        if gm.sum() == 0:
            continue
        for name in tools:
            R[f"{name} ({gn})"] = compute_metrics(
                true_diffs[gm], tool_pred_diffs[name][gm], motif_lens[gm])
        R[f"Hybrid ({gn})"] = compute_metrics(
            true_diffs[gm], hybrid_diffs[gm], motif_lens[gm])
        R[f"NoGate ({gn})"] = compute_metrics(
            true_diffs[gm], nogate_diffs[gm], motif_lens[gm])
        R[f"PropMode ({gn})"] = compute_metrics(
            true_diffs[gm], propmode_diffs[gm], motif_lens[gm])
        R[f"HPMedian ({gn})"] = compute_metrics(
            true_diffs[gm], hpmed_diffs[gm], motif_lens[gm])
        R[f"HPMedian_v2 ({gn})"] = compute_metrics(
            true_diffs[gm], hpmed_v2_diffs[gm], motif_lens[gm])

    # Stratified: length
    len_groups = {
        "<100bp": ref_sizes < 100,
        "100-500bp": (ref_sizes >= 100) & (ref_sizes < 500),
        "500-1000bp": (ref_sizes >= 500) & (ref_sizes < 1000),
        ">1000bp": ref_sizes >= 1000,
    }
    for gn, gm_base in len_groups.items():
        gm = common_mask & gm_base
        if gm.sum() == 0:
            continue
        for name in tools:
            R[f"{name} (len {gn})"] = compute_metrics(
                true_diffs[gm], tool_pred_diffs[name][gm], motif_lens[gm])
        R[f"Hybrid (len {gn})"] = compute_metrics(
            true_diffs[gm], hybrid_diffs[gm], motif_lens[gm])
        R[f"NoGate (len {gn})"] = compute_metrics(
            true_diffs[gm], nogate_diffs[gm], motif_lens[gm])
        R[f"PropMode (len {gn})"] = compute_metrics(
            true_diffs[gm], propmode_diffs[gm], motif_lens[gm])
        R[f"HPMedian (len {gn})"] = compute_metrics(
            true_diffs[gm], hpmed_diffs[gm], motif_lens[gm])
        R[f"HPMedian_v2 (len {gn})"] = compute_metrics(
            true_diffs[gm], hpmed_v2_diffs[gm], motif_lens[gm])

    # ── Write report ─────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tp_common = (common_mask & is_tp).sum()
    tn_common = (common_mask & ~is_tp).sum()
    tp_ltr = (longtr_mask & is_tp).sum()
    tn_ltr = (longtr_mask & ~is_tp).sum()

    L = []
    def w(s=""):
        L.append(s)

    def section(title, keys, n):
        w("=" * 70)
        w(f"  {title} (n={n:,})")
        w("=" * 70)
        w()
        w(HEADER)
        w(SEP)
        for k, label in keys:
            m = R.get(k, {})
            if m:
                w(fmt_row(label, m))
        w()

    w("=" * 70)
    w("  HEAD-TO-HEAD: TR GENOTYPING TOOLS")
    w(f"  Test set: chr21, chr22, chrX (n={len(h5_keys):,})")
    w(f"  Ground truth: GIAB Tier1 allele diffs")
    w(f"  Comparison: allele size diffs (ref-invariant)")
    w("=" * 70)
    w()
    w(f"  LongTR matched:  {n_longtr:,} / {len(h5_keys):,} ({n_longtr/len(h5_keys):.1%})")
    w(f"  TRGT matched:    {tool_matched_mask['TRGT'].sum():,} / {len(h5_keys):,} ({tool_matched_mask['TRGT'].sum()/len(h5_keys):.1%})")
    w(f"  Common (all):    {n_common:,} / {len(h5_keys):,} ({n_common/len(h5_keys):.1%})")
    w()

    section("OVERALL — Common loci", [
        ("LongTR (common)", "LongTR"),
        ("TRGT (common)", "TRGT"),
        ("Hybrid (common)", "Hybrid (gate+split_median)"),
        ("NoGate (common)", "NoGate (mode_round)"),
        ("PropMode (common)", "PropMode (prop_30pct)"),
        ("HPMedian (common)", "HPMedian (hp_cond)"),
        ("HPMedian_v2 (common)", "HPMedian_v2 (bimodal+trimmed)"),
    ], n_common)

    section("VARIANT (TP) — Common", [
        ("LongTR (TP, common)", "LongTR"),
        ("TRGT (TP, common)", "TRGT"),
        ("Hybrid (TP, common)", "Hybrid"),
        ("NoGate (TP, common)", "NoGate"),
        ("PropMode (TP, common)", "PropMode"),
        ("HPMedian (TP, common)", "HPMedian"),
        ("HPMedian_v2 (TP, common)", "HPMedian_v2"),
    ], tp_common)

    section("REFERENCE (TN) — Common", [
        ("LongTR (TN, common)", "LongTR"),
        ("TRGT (TN, common)", "TRGT"),
        ("Hybrid (TN, common)", "Hybrid"),
        ("NoGate (TN, common)", "NoGate"),
        ("PropMode (TN, common)", "PropMode"),
        ("HPMedian (TN, common)", "HPMedian"),
        ("HPMedian_v2 (TN, common)", "HPMedian_v2"),
    ], tn_common)

    section("OVERALL — LongTR set (larger)", [
        ("LongTR (ltr_set)", "LongTR"),
        ("Hybrid (ltr_set)", "Hybrid"),
        ("NoGate (ltr_set)", "NoGate"),
        ("PropMode (ltr_set)", "PropMode"),
        ("HPMedian (ltr_set)", "HPMedian (hp_cond)"),
        ("HPMedian_v2 (ltr_set)", "HPMedian_v2 (bimodal+trimmed)"),
    ], n_longtr)

    section("VARIANT (TP) — LongTR set", [
        ("LongTR (TP, ltr_set)", "LongTR"),
        ("Hybrid (TP, ltr_set)", "Hybrid"),
        ("NoGate (TP, ltr_set)", "NoGate"),
        ("PropMode (TP, ltr_set)", "PropMode"),
        ("HPMedian (TP, ltr_set)", "HPMedian (hp_cond)"),
        ("HPMedian_v2 (TP, ltr_set)", "HPMedian_v2 (bimodal+trimmed)"),
    ], tp_ltr)

    section("REFERENCE (TN) — LongTR set", [
        ("LongTR (TN, ltr_set)", "LongTR"),
        ("Hybrid (TN, ltr_set)", "Hybrid"),
        ("NoGate (TN, ltr_set)", "NoGate"),
        ("PropMode (TN, ltr_set)", "PropMode"),
        ("HPMedian (TN, ltr_set)", "HPMedian (hp_cond)"),
        ("HPMedian_v2 (TN, ltr_set)", "HPMedian_v2 (bimodal+trimmed)"),
    ], tn_ltr)

    for gn in motif_groups:
        gm = common_mask & motif_groups[gn]
        if gm.sum() == 0:
            continue
        section(f"MOTIF: {gn} — Common", [
            (f"LongTR ({gn})", "LongTR"),
            (f"TRGT ({gn})", "TRGT"),
            (f"Hybrid ({gn})", "Hybrid"),
            (f"NoGate ({gn})", "NoGate"),
            (f"PropMode ({gn})", "PropMode"),
            (f"HPMedian ({gn})", "HPMedian"),
            (f"HPMedian_v2 ({gn})", "HPMedian_v2"),
        ], gm.sum())

    for gn in len_groups:
        gm = common_mask & len_groups[gn]
        if gm.sum() == 0:
            continue
        section(f"REPEAT LENGTH: {gn} — Common", [
            (f"LongTR (len {gn})", "LongTR"),
            (f"TRGT (len {gn})", "TRGT"),
            (f"Hybrid (len {gn})", "Hybrid"),
            (f"NoGate (len {gn})", "NoGate"),
            (f"PropMode (len {gn})", "PropMode"),
            (f"HPMedian (len {gn})", "HPMedian"),
            (f"HPMedian_v2 (len {gn})", "HPMedian_v2"),
        ], gm.sum())

    section("FULL test set", [
        ("Hybrid (full)", "Hybrid (gate+split_median)"),
        ("NoGate (full)", "NoGate (mode_round)"),
        ("PropMode (full)", "PropMode (prop_30pct)"),
        ("HPMedian (full)", "HPMedian (hp_cond)"),
        ("HPMedian_v2 (full)", "HPMedian_v2 (bimodal+trimmed)"),
    ], len(h5_keys))

    report = "\n".join(L)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {output_path}")
    print(report)


if __name__ == "__main__":
    main()
