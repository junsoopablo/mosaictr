#!/usr/bin/env python3
"""TP gap analysis v3: Refined adaptive strategies.

From v2 findings:
- adaptive_mode (collapse when |m1-m2|<=1): Hom 88.8% (+11.6%p), Het 83.6% (-1.7%p)
- Need: keep hom gains, recover het losses

Key strategies:
1. Strict collapse (|m1-m2|==0): only collapse when modes exactly equal
2. Proportion-based het detection: check read proportion supporting each mode
3. Hybrid: use adaptive collapse + refined het split
"""

import argparse
import bisect
import gzip
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLELE_SIZE_IDX = 0
REF_SIZE_IDX = 0
TEST_CHROMS = {"chr21", "chr22", "chrX"}
MATCH_TOLERANCE = 15


def get_mode(arr):
    int_arr = np.round(arr).astype(int)
    counts = np.bincount(int_arr - int_arr.min())
    return float(int_arr.min() + np.argmax(counts))


def mode_round_genotype(allele_sizes, ref_size):
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    a1 = get_mode(sizes[:mid])
    a2 = get_mode(sizes[mid:])
    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def adaptive_strict_genotype(allele_sizes, ref_size):
    """Collapse ONLY when modes are exactly equal (threshold=0)."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def adaptive_mode_genotype(allele_sizes, ref_size):
    """Original adaptive: collapse when |m1-m2| <= 1."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if abs(m1 - m2) <= 1:
        m_all = get_mode(sizes)
        d = round(ref_size - m_all)
        return d, d
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def proportion_adaptive_genotype(allele_sizes, ref_size):
    """Collapse when modes within 1bp AND second mode has <15% of reads.

    This avoids collapsing close hets where both alleles have significant read support.
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        # Check if it's a true het or noise-induced split
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < 0.15:
            # Likely noise: collapse to mode of all reads
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
        else:
            # Significant support for both: keep as het
            d1 = round(ref_size - m1)
            d2 = round(ref_size - m2)
            return sorted([d1, d2])
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def proportion_v2_genotype(allele_sizes, ref_size):
    """Same as proportion_adaptive but with 20% threshold."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < 0.20:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
        else:
            d1 = round(ref_size - m1)
            d2 = round(ref_size - m2)
            return sorted([d1, d2])
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def proportion_v3_genotype(allele_sizes, ref_size):
    """Proportion-based with 25% threshold."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < 0.25:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
        else:
            d1 = round(ref_size - m1)
            d2 = round(ref_size - m2)
            return sorted([d1, d2])
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def weighted_mode_genotype(allele_sizes, ref_size):
    """Like mode_round but weight reads towards centers when computing modes.

    For each half, instead of raw mode, use the mode of the inner 80% of reads
    (trimming extremes that may be noise).
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    mid = n // 2
    half1 = sizes[:mid]
    half2 = sizes[mid:]

    def trimmed_mode(arr, trim_frac=0.1):
        if len(arr) <= 3:
            return get_mode(arr)
        n_trim = max(1, int(len(arr) * trim_frac))
        trimmed = arr[n_trim:-n_trim] if n_trim < len(arr) // 2 else arr
        return get_mode(trimmed)

    m1 = trimmed_mode(half1)
    m2 = trimmed_mode(half2)

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < 0.20:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
        else:
            d1 = round(ref_size - m1)
            d2 = round(ref_size - m2)
            return sorted([d1, d2])
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def peak_finder_genotype(allele_sizes, ref_size, motif_len):
    """Direct peak finding in allele size histogram.

    1. Compute bincount of rounded allele sizes
    2. Find all peaks (local maxima with >= 3 reads support)
    3. If 1 peak: hom
    4. If 2+ peaks: het (top 2 peaks)
    """
    sizes = allele_sizes
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    max_val = int_sizes.max()

    if min_val == max_val:
        d = round(ref_size - float(min_val))
        return d, d

    counts = np.bincount(int_sizes - min_val)

    # Find peaks: bins with counts > both neighbors
    peaks = []
    for i in range(len(counts)):
        if counts[i] < max(2, n * 0.05):
            continue
        left = counts[i-1] if i > 0 else 0
        right = counts[i+1] if i < len(counts)-1 else 0
        if counts[i] >= left and counts[i] >= right:
            peaks.append((counts[i], min_val + i))

    if len(peaks) == 0:
        # Fallback to mode
        mode_val = min_val + np.argmax(counts)
        d = round(ref_size - float(mode_val))
        return d, d
    elif len(peaks) == 1:
        d = round(ref_size - float(peaks[0][1]))
        return d, d
    else:
        # Sort by count descending, take top 2
        peaks.sort(reverse=True)
        a1 = float(peaks[0][1])
        a2 = float(peaks[1][1])
        # Check if peaks are adjacent (likely same allele + noise)
        if abs(a1 - a2) <= 1:
            # Merge: use the one with higher count
            d = round(ref_size - a1)
            return d, d
        else:
            d1 = round(ref_size - a1)
            d2 = round(ref_size - a2)
            return sorted([d1, d2])


def parse_longtr_vcf(vcf_path, chroms):
    result = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            if chrom not in chroms:
                continue
            pos = int(parts[1]) - 1
            ref = parts[3]
            alts = parts[4].split(",")
            gt_field = parts[9].split(":")
            gt = gt_field[0].replace("|", "/").split("/")
            alleles_seq = [ref] + alts
            ref_size = len(ref)
            diffs = []
            for g in gt:
                if g == ".":
                    diffs.append(0)
                else:
                    a = alleles_seq[int(g)]
                    diffs.append(ref_size - len(a))
            if len(diffs) == 1:
                diffs = diffs * 2
            result[(chrom, pos)] = (sorted(diffs)[0], sorted(diffs)[1], ref_size)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--longtr-vcf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
    n_test = len(test_indices)
    logger.info(f"Test set: {n_test:,} loci")

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
        true_diffs.append(sorted([h1, h2]))
        motif_lens.append(ml)
        ref_sizes.append(all_locus_features[idx, REF_SIZE_IDX])
        is_tp.append("TP" in tp_statuses[idx])

    true_diffs = np.array(true_diffs)
    motif_lens = np.array(motif_lens)
    is_tp = np.array(is_tp)
    ref_sizes = np.array(ref_sizes)

    # Compute all methods
    methods_simple = [
        ("mode_round", mode_round_genotype),
        ("adaptive_strict", adaptive_strict_genotype),
        ("adaptive_1bp", adaptive_mode_genotype),
        ("prop_15pct", proportion_adaptive_genotype),
        ("prop_20pct", proportion_v2_genotype),
        ("prop_25pct", proportion_v3_genotype),
        ("weighted_mode", weighted_mode_genotype),
    ]

    methods_motif = [
        ("peak_finder", peak_finder_genotype),
    ]

    all_diffs = {}
    for name, fn in methods_simple:
        logger.info(f"Computing {name}...")
        diffs = np.zeros((n_test, 2))
        for i, idx in enumerate(test_indices):
            offset = all_read_offsets[idx]
            count = all_read_counts[idx]
            reads = all_read_features[offset:offset + count]
            allele_sizes = reads[:, ALLELE_SIZE_IDX]
            ref_size = all_locus_features[idx, REF_SIZE_IDX]
            d1, d2 = fn(allele_sizes, ref_size)
            diffs[i] = [d1, d2]
        all_diffs[name] = diffs

    for name, fn in methods_motif:
        logger.info(f"Computing {name}...")
        diffs = np.zeros((n_test, 2))
        for i, idx in enumerate(test_indices):
            offset = all_read_offsets[idx]
            count = all_read_counts[idx]
            reads = all_read_features[offset:offset + count]
            allele_sizes = reads[:, ALLELE_SIZE_IDX]
            ref_size = all_locus_features[idx, REF_SIZE_IDX]
            ml = int(motif_lens[i])
            d1, d2 = fn(allele_sizes, ref_size, ml)
            diffs[i] = [d1, d2]
        all_diffs[name] = diffs

    h5.close()

    # Parse LongTR
    logger.info("Parsing LongTR VCF...")
    longtr_dict = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)

    by_chrom = defaultdict(list)
    for (chrom, start), val in longtr_dict.items():
        by_chrom[chrom].append((start, val))
    for chrom in by_chrom:
        by_chrom[chrom].sort()

    longtr_mask = np.zeros(n_test, dtype=bool)
    longtr_diffs = np.zeros((n_test, 2))

    for i, (chrom, start, end) in enumerate(h5_keys):
        candidates = by_chrom.get(chrom, [])
        if not candidates:
            continue
        h5_ref = ref_sizes[i]
        pos = bisect.bisect_left(candidates, (start,))
        best_dist = MATCH_TOLERANCE + 1
        best_val = None
        for j in range(max(0, pos - 2), min(len(candidates), pos + 3)):
            dist = abs(candidates[j][0] - start)
            tool_ref = candidates[j][1][2]
            if h5_ref > 0 and tool_ref > 0:
                ratio = tool_ref / h5_ref
                if ratio < 0.5 or ratio > 2.0:
                    continue
            if dist < best_dist:
                best_dist = dist
                best_val = candidates[j][1]
        if best_dist <= MATCH_TOLERANCE and best_val is not None:
            longtr_mask[i] = True
            longtr_diffs[i] = [best_val[0], best_val[1]]

    # ── Results ──
    lines = []
    def p(s=""):
        lines.append(s)

    THRESH = 0.5
    is_het_truth = np.abs(true_diffs[:, 1] - true_diffs[:, 0]) > 0.5
    tp_ltr = is_tp & longtr_mask
    tn_ltr = longtr_mask & ~is_tp
    n_tp = tp_ltr.sum()

    p("=" * 90)
    p("  TP GAP ANALYSIS v3: Refined Adaptive Strategies")
    p(f"  Test set: chr21, chr22, chrX (n={n_test:,})")
    p("=" * 90)
    p()

    # Main comparison table
    p(f"{'Method':<22} {'Overall':>8} {'TP':>8} {'TN':>8} {'Het-TP':>8} {'Hom-TP':>8}  {'TP≤1bp':>8} {'TP MAE':>8}")
    p("-" * 90)

    het_m = tp_ltr & is_het_truth
    hom_m = tp_ltr & ~is_het_truth

    for name in list(all_diffs.keys()) + ["LongTR"]:
        diffs = all_diffs[name] if name != "LongTR" else longtr_diffs

        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH

        ov = exact[longtr_mask].mean()
        tp = exact[tp_ltr].mean()
        tn = exact[tn_ltr].mean()
        het = exact[het_m].mean() if het_m.sum() > 0 else 0
        hom = exact[hom_m].mean() if hom_m.sum() > 0 else 0

        flat_errs = np.abs(diffs[tp_ltr] - true_diffs[tp_ltr]).flatten()
        w1 = (flat_errs <= 1.0).mean()
        mae = flat_errs.mean()

        p(f"{name:<22} {ov:>7.1%} {tp:>7.1%} {tn:>7.1%} {het:>7.1%} {hom:>7.1%}  {w1:>7.1%} {mae:>7.2f}")

    p()
    p(f"  Het TP: n={het_m.sum():,}, Hom TP: n={hom_m.sum():,}")
    p()

    # Full test set
    p("=" * 90)
    p("  FULL TEST SET")
    p("=" * 90)
    p()
    p(f"{'Method':<22} {'Overall':>8} {'TP':>8} {'TN':>8}")
    p("-" * 50)

    for name, diffs in all_diffs.items():
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH
        p(f"{name:<22} {exact.mean():>7.1%} {exact[is_tp].mean():>7.1%} {exact[~is_tp].mean():>7.1%}")
    p()

    # Motif breakdown
    p("=" * 90)
    p("  MOTIF BREAKDOWN (TP, LongTR set)")
    p("=" * 90)
    p()

    for motif_name, motif_fn in [
        ("Dinuc (2bp)", lambda: motif_lens == 2),
        ("STR (3-6bp)", lambda: (motif_lens >= 3) & (motif_lens <= 6)),
        ("VNTR (7+bp)", lambda: motif_lens >= 7),
    ]:
        mm = tp_ltr & motif_fn()
        n_m = mm.sum()
        if n_m == 0:
            continue
        p(f"  {motif_name} (n={n_m:,}):")
        for name in list(all_diffs.keys()) + ["LongTR"]:
            diffs = all_diffs[name] if name != "LongTR" else longtr_diffs
            errs = np.max(np.abs(diffs[mm] - true_diffs[mm]), axis=1)
            p(f"    {name:<22} {(errs < THRESH).mean():.1%}")
        p()

    # Net gain/loss vs mode_round for best method
    p("=" * 90)
    p("  NET ANALYSIS: Each method vs mode_round (TP)")
    p("=" * 90)
    p()

    mr_ex = np.max(np.abs(all_diffs["mode_round"] - true_diffs), axis=1) < THRESH
    for name in all_diffs:
        if name == "mode_round":
            continue
        n_ex = np.max(np.abs(all_diffs[name] - true_diffs), axis=1) < THRESH
        gains = (tp_ltr & n_ex & ~mr_ex).sum()
        losses = (tp_ltr & mr_ex & ~n_ex).sum()
        p(f"  {name:<22} gains={gains:>3}, losses={losses:>3}, net={gains-losses:>+4}")

    p()

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report: {output_path}")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
