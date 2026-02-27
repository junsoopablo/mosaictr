#!/usr/bin/env python3
"""Analyze TP MAE gap between HPMedian and LongTR.

Focus: Identify the loci driving the MAE difference and test targeted fixes.
"""

import argparse
import bisect
import gzip
import pickle
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ALLELE_SIZE_IDX = 0
REF_SIZE_IDX = 0
MOTIF_LEN_IDX = 1
TEST_CHROMS = {"chr21", "chr22", "chrX"}
MATCH_TOLERANCE = 15

def parse_longtr_vcf(vcf_path, chroms):
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
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
    return results


def match_loci(h5_keys, h5_ref_sizes, tool_dict, tolerance=MATCH_TOLERANCE):
    by_chrom = defaultdict(list)
    for (chrom, start), val in tool_dict.items():
        by_chrom[chrom].append((start, val))
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    matched_idx = []
    matched_vals = []
    for i, (chrom, start, end) in enumerate(h5_keys):
        candidates = by_chrom.get(chrom, [])
        if not candidates:
            continue
        h5_ref = h5_ref_sizes[i]
        pos = bisect.bisect_left(candidates, (start,))
        best_dist = tolerance + 1
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
        if best_dist <= tolerance and best_val is not None:
            matched_idx.append(i)
            matched_vals.append(best_val)
    return matched_idx, matched_vals


# ── Genotyping methods ──

def prop_mode_genotype(allele_sizes, ref_size, collapse_threshold=0.30):
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
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < collapse_threshold:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def mode_round_genotype(allele_sizes, ref_size):
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
    a1 = get_mode(sizes[:mid])
    a2 = get_mode(sizes[mid:])
    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def hp_median_genotype(sizes_hp, ref_size):
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
    if motif_len >= 7:
        return hp_median_genotype(sizes_hp, ref_size)
    else:
        return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)


# ── NEW methods to reduce MAE ──

def hp_trimmed_median_genotype(sizes_hp, ref_size, motif_len, trim_frac=0.1):
    """HP-aware median with IQR-based outlier removal within each HP group."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def trimmed_med(arr):
        if len(arr) < 4:
            return np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) == 0:
            return np.median(arr)
        return np.median(trimmed)

    if len(hp1) == 0 and len(hp2) == 0:
        if motif_len < 7:
            return prop_mode_genotype(all_sizes, ref_size, 0.25)
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - trimmed_med(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - trimmed_med(hp1))
        return d, d

    med1 = trimmed_med(hp1)
    med2 = trimmed_med(hp2)
    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)

    # Apply same conditional collapse for dinuc/STR
    if motif_len < 7 and d1 != d2 and abs(d1 - d2) == 1:
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(int_sizes == int(round(med1)))
        c2 = np.sum(int_sizes == int(round(med2)))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < 0.25:
            d = round(ref_size - trimmed_med(all_sizes))
            return d, d

    return sorted([d1, d2])


def hp_biweight_genotype(sizes_hp, ref_size, motif_len):
    """HP-aware biweight midcorrelation location — robust to outliers."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def biweight_location(arr, c=6.0, max_iter=10):
        """Tukey's biweight location estimator — highly robust to outliers."""
        if len(arr) < 3:
            return np.median(arr)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-10:
            return med
        for _ in range(max_iter):
            u = (arr - med) / (c * mad)
            mask = np.abs(u) < 1
            if mask.sum() == 0:
                return med
            w = (1 - u**2)**2 * mask
            new_med = np.sum(w * arr) / np.sum(w)
            if abs(new_med - med) < 0.01:
                break
            med = new_med
        return med

    if len(hp1) == 0 and len(hp2) == 0:
        if motif_len < 7:
            return prop_mode_genotype(all_sizes, ref_size, 0.25)
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - biweight_location(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - biweight_location(hp1))
        return d, d

    bw1 = biweight_location(hp1)
    bw2 = biweight_location(hp2)
    d1 = round(ref_size - bw1)
    d2 = round(ref_size - bw2)

    if motif_len < 7 and d1 != d2 and abs(d1 - d2) == 1:
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(int_sizes == int(round(bw1)))
        c2 = np.sum(int_sizes == int(round(bw2)))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < 0.25:
            d = round(ref_size - biweight_location(all_sizes))
            return d, d

    return sorted([d1, d2])


def hp_consensus_genotype(sizes_hp, ref_size, motif_len):
    """HP-aware median with cross-group consistency check.

    If one HP group has high variance (IQR > 2*motif_len for STR/dinuc,
    IQR > 0.3*ref_size for VNTR), fall back to the more consistent group.
    """
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        if motif_len < 7:
            return prop_mode_genotype(all_sizes, ref_size, 0.25)
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - np.median(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - np.median(hp1))
        return d, d

    med1 = np.median(hp1)
    med2 = np.median(hp2)

    # Check consistency: use IQR / median as CV-like measure
    def group_quality(arr):
        if len(arr) < 3:
            return 0.5  # neutral
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        return iqr

    iqr1 = group_quality(hp1)
    iqr2 = group_quality(hp2)

    # If one group is much noisier, downweight it
    noise_threshold = max(2.0 * motif_len, 10.0)
    if iqr1 > noise_threshold and iqr2 < noise_threshold and len(hp2) >= 3:
        # HP1 is noisy, trust HP2 for both
        d = round(ref_size - med2)
        return d, d
    if iqr2 > noise_threshold and iqr1 < noise_threshold and len(hp1) >= 3:
        d = round(ref_size - med1)
        return d, d

    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)

    if motif_len < 7 and d1 != d2 and abs(d1 - d2) == 1:
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(int_sizes == int(round(med1)))
        c2 = np.sum(int_sizes == int(round(med2)))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < 0.25:
            d = round(ref_size - np.median(all_sizes))
            return d, d

    return sorted([d1, d2])


def hp_mode_median_hybrid(sizes_hp, ref_size, motif_len):
    """Use mode for allele identification + median for allele sizing.

    Key idea: mode gives better exact match, but when mode fails,
    the error tends to be large. Use mode when reads cluster tightly
    (clear peak), otherwise fallback to median.
    """
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def smart_center(arr):
        """Use mode if clear peak, else median."""
        if len(arr) < 3:
            return np.median(arr)
        int_arr = np.round(arr).astype(int)
        if int_arr.max() == int_arr.min():
            return float(int_arr[0])
        counts = np.bincount(int_arr - int_arr.min())
        mode_val = float(int_arr.min() + np.argmax(counts))
        mode_count = counts.max()
        mode_frac = mode_count / len(arr)
        # If mode covers >40% of reads, it's a clear peak
        if mode_frac >= 0.40:
            return mode_val
        else:
            return np.median(arr)

    if len(hp1) == 0 and len(hp2) == 0:
        if motif_len < 7:
            return prop_mode_genotype(all_sizes, ref_size, 0.25)
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - smart_center(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - smart_center(hp1))
        return d, d

    c1 = smart_center(hp1)
    c2 = smart_center(hp2)
    d1 = round(ref_size - c1)
    d2 = round(ref_size - c2)

    if motif_len < 7 and d1 != d2 and abs(d1 - d2) == 1:
        int_sizes = np.round(all_sizes).astype(int)
        n1 = np.sum(int_sizes == int(round(c1)))
        n2 = np.sum(int_sizes == int(round(c2)))
        minor_frac = min(n1, n2) / len(all_sizes)
        if minor_frac < 0.25:
            d = round(ref_size - smart_center(all_sizes))
            return d, d

    return sorted([d1, d2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--longtr-vcf", required=True)
    parser.add_argument("--hp-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load H5
    print("Loading HDF5...")
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
    print(f"Test set: {len(test_indices):,}")

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

    # Load LongTR
    print("Loading LongTR VCF...")
    longtr = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)
    matched_idx, matched_vals = match_loci(h5_keys, ref_sizes, longtr)
    longtr_mask = np.zeros(len(h5_keys), dtype=bool)
    longtr_diffs = np.zeros((len(h5_keys), 2))
    for j, val in zip(matched_idx, matched_vals):
        longtr_mask[j] = True
        longtr_diffs[j] = [val[0], val[1]]

    # Load HP cache
    print("Loading HP cache...")
    with open(args.hp_cache, "rb") as f:
        all_reads_hp = pickle.load(f)["all_reads_hp"]

    # PropMode (for fallback)
    print("Computing PropMode baseline...")
    propmode_diffs = np.zeros_like(true_diffs)
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref = all_locus_features[idx, REF_SIZE_IDX]
        propmode_diffs[i] = prop_mode_genotype(allele_sizes, ref)

    h5.close()

    # Focus on TP loci in LongTR set
    tp_ltr = longtr_mask & is_tp
    tp_indices = np.where(tp_ltr)[0]
    print(f"TP in LongTR set: {len(tp_indices):,}")

    # Compute genotypes for all methods
    methods = {
        "hp_cond": lambda i: hp_cond_genotype(
            all_reads_hp[i], ref_sizes[i], motif_lens[i]),
        "hp_trimmed": lambda i: hp_trimmed_median_genotype(
            all_reads_hp[i], ref_sizes[i], motif_lens[i]),
        "hp_biweight": lambda i: hp_biweight_genotype(
            all_reads_hp[i], ref_sizes[i], motif_lens[i]),
        "hp_consensus": lambda i: hp_consensus_genotype(
            all_reads_hp[i], ref_sizes[i], motif_lens[i]),
        "hp_mode_med": lambda i: hp_mode_median_hybrid(
            all_reads_hp[i], ref_sizes[i], motif_lens[i]),
    }

    results = {}
    for name, fn in methods.items():
        diffs = np.zeros_like(true_diffs)
        for i in range(len(h5_keys)):
            reads_hp = all_reads_hp[i]
            if reads_hp:
                diffs[i] = fn(i)
            else:
                diffs[i] = propmode_diffs[i]
        results[name] = diffs

    # ── Analysis ──

    L = []
    def w(s=""):
        L.append(s)

    w("=" * 80)
    w("  TP MAE GAP ANALYSIS")
    w(f"  TP loci in LongTR set: {len(tp_indices):,}")
    w("=" * 80)

    # 1. Per-method MAE on TP
    w()
    w("  METHOD COMPARISON (TP, LongTR set)")
    w()
    w(f"  {'Method':<25} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'MedAE':>7} {'P90':>7} {'P95':>7} {'Max':>7}")
    w("  " + "-" * 74)

    # LongTR
    ltr_errs = np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr]).flatten()
    ltr_exact = np.all(np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr]) < 0.5, axis=1).mean()
    ltr_w1 = (ltr_errs <= 1.0).mean()
    w(f"  {'LongTR':<25} {ltr_exact:>6.1%} {ltr_w1:>6.1%} {ltr_errs.mean():>6.2f} "
      f"{np.median(ltr_errs):>6.2f} {np.percentile(ltr_errs, 90):>6.1f} "
      f"{np.percentile(ltr_errs, 95):>6.1f} {ltr_errs.max():>6.0f}")

    for name in methods:
        diffs = results[name]
        errs = np.abs(true_diffs[tp_ltr] - diffs[tp_ltr]).flatten()
        exact = np.all(np.abs(true_diffs[tp_ltr] - diffs[tp_ltr]) < 0.5, axis=1).mean()
        w1 = (errs <= 1.0).mean()
        w(f"  {name:<25} {exact:>6.1%} {w1:>6.1%} {errs.mean():>6.2f} "
          f"{np.median(errs):>6.2f} {np.percentile(errs, 90):>6.1f} "
          f"{np.percentile(errs, 95):>6.1f} {errs.max():>6.0f}")

    # 2. Error distribution comparison: HPMedian vs LongTR
    w()
    w("=" * 80)
    w("  ERROR DISTRIBUTION (TP alleles)")
    w("=" * 80)
    w()

    hp_errs_all = np.abs(true_diffs[tp_ltr] - results["hp_cond"][tp_ltr]).flatten()
    ltr_errs_all = ltr_errs

    for threshold in [0, 1, 2, 5, 10, 20, 50, 100, 500]:
        hp_above = (hp_errs_all > threshold).sum()
        ltr_above = (ltr_errs_all > threshold).sum()
        w(f"  Error > {threshold:>3}bp:  HPMedian={hp_above:>5} ({hp_above/len(hp_errs_all):.2%})  "
          f"LongTR={ltr_above:>5} ({ltr_above/len(ltr_errs_all):.2%})  "
          f"Diff={hp_above-ltr_above:>+5}")

    # 3. Top error loci for HPMedian
    w()
    w("=" * 80)
    w("  TOP 30 ERROR LOCI (HPMedian)")
    w("=" * 80)
    w()

    hp_locus_errs = np.abs(true_diffs[tp_ltr] - results["hp_cond"][tp_ltr])
    hp_locus_mae = hp_locus_errs.mean(axis=1)
    ltr_locus_errs = np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr])
    ltr_locus_mae = ltr_locus_errs.mean(axis=1)

    worst_30 = np.argsort(hp_locus_mae)[-30:][::-1]
    w(f"  {'Chrom':<6} {'Start':>10} {'RefSz':>6} {'Motif':>5} "
      f"{'True':>12} {'HPMed':>12} {'LTR':>12} {'HPErr':>6} {'LTRErr':>6} {'NReads':>6}")
    w("  " + "-" * 96)

    for rank_idx in worst_30:
        glob_idx = tp_indices[rank_idx]
        chrom, start, end = h5_keys[glob_idx]
        ref = ref_sizes[glob_idx]
        ml = motif_lens[glob_idx]
        td = true_diffs[glob_idx]
        hp_d = results["hp_cond"][glob_idx]
        ltr_d = longtr_diffs[glob_idx]
        hp_e = hp_locus_mae[rank_idx]
        ltr_e = ltr_locus_mae[rank_idx]
        nr = len(all_reads_hp[glob_idx])
        w(f"  {chrom:<6} {start:>10} {ref:>6.0f} {ml:>5.0f} "
          f"  ({td[0]:>4.0f},{td[1]:>4.0f})"
          f"  ({hp_d[0]:>4.0f},{hp_d[1]:>4.0f})"
          f"  ({ltr_d[0]:>4.0f},{ltr_d[1]:>4.0f})"
          f"  {hp_e:>5.0f} {ltr_e:>5.0f} {nr:>6}")

    # 4. MAE contribution: top N alleles
    w()
    w("=" * 80)
    w("  CUMULATIVE MAE CONTRIBUTION")
    w("=" * 80)
    w()

    total_hp_err = hp_errs_all.sum()
    total_ltr_err = ltr_errs_all.sum()
    sorted_hp_errs = np.sort(hp_errs_all)[::-1]
    sorted_ltr_errs = np.sort(ltr_errs_all)[::-1]

    for n in [10, 20, 50, 100, 200, 500]:
        hp_top = sorted_hp_errs[:n].sum()
        ltr_top = sorted_ltr_errs[:n].sum()
        w(f"  Top {n:>3} alleles:  HPMedian={hp_top:>8.0f}bp ({hp_top/total_hp_err:.1%} of total)  "
          f"LongTR={ltr_top:>8.0f}bp ({ltr_top/total_ltr_err:.1%} of total)")

    # 5. Net gains/losses vs LongTR (TP exact match) for all methods
    w()
    w("=" * 80)
    w("  NET GAINS/LOSSES vs LongTR (TP exact)")
    w("=" * 80)
    w()

    ltr_correct = np.all(np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr]) < 0.5, axis=1)
    for name in methods:
        diffs = results[name]
        hp_correct = np.all(np.abs(true_diffs[tp_ltr] - diffs[tp_ltr]) < 0.5, axis=1)
        gains = (~ltr_correct & hp_correct).sum()
        losses = (ltr_correct & ~hp_correct).sum()
        net = gains - losses
        w(f"  {name:<25} gains={gains:>4} losses={losses:>4} net={net:>+5}  "
          f"Exact={hp_correct.mean():.1%}  MAE={np.abs(true_diffs[tp_ltr]-diffs[tp_ltr]).flatten().mean():.2f}")

    # 6. Motif breakdown: MAE
    w()
    w("=" * 80)
    w("  MAE BY MOTIF TYPE (TP, LongTR set)")
    w("=" * 80)
    w()

    motif_groups = {
        "Dinuc (2bp)": motif_lens == 2,
        "STR (3-6bp)": (motif_lens >= 3) & (motif_lens <= 6),
        "VNTR (7+bp)": motif_lens >= 7,
    }

    w(f"  {'Motif':<15} {'Method':<25} {'N':>5} {'Exact':>7} {'MAE':>7}")
    w("  " + "-" * 64)

    for gn, gm_base in motif_groups.items():
        gm = tp_ltr & gm_base
        if gm.sum() == 0:
            continue
        # LongTR
        errs = np.abs(true_diffs[gm] - longtr_diffs[gm]).flatten()
        exact = np.all(np.abs(true_diffs[gm] - longtr_diffs[gm]) < 0.5, axis=1).mean()
        w(f"  {gn:<15} {'LongTR':<25} {gm.sum():>5} {exact:>6.1%} {errs.mean():>6.2f}")
        for name in methods:
            diffs = results[name]
            errs = np.abs(true_diffs[gm] - diffs[gm]).flatten()
            exact = np.all(np.abs(true_diffs[gm] - diffs[gm]) < 0.5, axis=1).mean()
            w(f"  {'':<15} {name:<25} {gm.sum():>5} {exact:>6.1%} {errs.mean():>6.2f}")
        w()

    # 7. Ref size breakdown: MAE
    w()
    w("=" * 80)
    w("  MAE BY REF SIZE (TP, LongTR set)")
    w("=" * 80)
    w()

    size_groups = {
        "<100bp": ref_sizes < 100,
        "100-500bp": (ref_sizes >= 100) & (ref_sizes < 500),
        "500-1000bp": (ref_sizes >= 500) & (ref_sizes < 1000),
        ">1000bp": ref_sizes >= 1000,
    }

    w(f"  {'Size':<15} {'Method':<25} {'N':>5} {'Exact':>7} {'MAE':>7}")
    w("  " + "-" * 64)

    for gn, gm_base in size_groups.items():
        gm = tp_ltr & gm_base
        if gm.sum() == 0:
            continue
        errs = np.abs(true_diffs[gm] - longtr_diffs[gm]).flatten()
        exact = np.all(np.abs(true_diffs[gm] - longtr_diffs[gm]) < 0.5, axis=1).mean()
        w(f"  {gn:<15} {'LongTR':<25} {gm.sum():>5} {exact:>6.1%} {errs.mean():>6.2f}")
        for name in methods:
            diffs = results[name]
            errs = np.abs(true_diffs[gm] - diffs[gm]).flatten()
            exact = np.all(np.abs(true_diffs[gm] - diffs[gm]) < 0.5, axis=1).mean()
            w(f"  {'':<15} {name:<25} {gm.sum():>5} {exact:>6.1%} {errs.mean():>6.2f}")
        w()

    report = "\n".join(L)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
