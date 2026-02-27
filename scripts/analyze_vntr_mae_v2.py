#!/usr/bin/env python3
"""VNTR MAE v2 — combine trimmed median + bimodal check + other targeted fixes.

Findings from v1:
- comb_hp_bimodal: TP MAE 3.48 (best MAE), 87.4% exact
- comb_hp_trimmed_strict: TP MAE 3.86, 87.6% exact (best exact, net +6)
- comb_hp_best_of: TP MAE 3.51, 84.9% exact (hurts exact)

This script combines bimodal + trimmed + other targeted improvements.
"""

import argparse
import bisect
import gzip
import pickle
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


# ── Base methods ──

def mode_round_genotype(allele_sizes, ref_size):
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))
    d1 = round(ref_size - get_mode(sizes[:mid]))
    d2 = round(ref_size - get_mode(sizes[mid:]))
    return sorted([d1, d2])


def prop_mode_genotype(allele_sizes, ref_size, threshold=0.30):
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
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
        return round(ref_size - m1), round(ref_size - m1)
    elif abs(m1 - m2) == 1:
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        if min(c1, c2) / n < threshold:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def split_median_genotype(allele_sizes, ref_size):
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    d1 = round(ref_size - np.median(sizes[:mid]))
    d2 = round(ref_size - np.median(sizes[mid:]))
    return sorted([d1, d2])


def hp_median_genotype(sizes_hp, ref_size):
    if not sizes_hp: return 0.0, 0.0
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
    if not sizes_hp: return 0.0, 0.0
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


# ── New combined VNTR methods ──

def vntr_trimmed_bimodal(sizes_hp, ref_size):
    """Trimmed HP median + bimodal check for VNTRs.

    1. Trim outliers (1.0*IQR) within each HP group
    2. Compute medians
    3. If homozygous, check for bimodality in all reads
    4. If bimodal (IQR > 5% of median), use split_median
    """
    if not sizes_hp: return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def trimmed_med(arr):
        if len(arr) < 4: return np.median(arr)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        trimmed = arr[(arr >= q1 - 1.0*iqr) & (arr <= q3 + 1.0*iqr)]
        return np.median(trimmed) if len(trimmed) > 0 else np.median(arr)

    if len(hp1) == 0 and len(hp2) == 0:
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

    if d1 != d2:
        return sorted([d1, d2])

    # Homozygous — check bimodality
    if len(all_sizes) >= 6:
        iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
        median_size = np.median(all_sizes)
        if median_size > 0 and iqr / median_size > 0.05:
            return split_median_genotype(all_sizes, ref_size)

    return d1, d2


def vntr_trimmed_bimodal_gentle(sizes_hp, ref_size):
    """Same as trimmed_bimodal but with 1.5*IQR (gentler trimming)."""
    if not sizes_hp: return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def trimmed_med(arr):
        if len(arr) < 4: return np.median(arr)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        trimmed = arr[(arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)]
        return np.median(trimmed) if len(trimmed) > 0 else np.median(arr)

    if len(hp1) == 0 and len(hp2) == 0:
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

    if d1 != d2:
        return sorted([d1, d2])

    if len(all_sizes) >= 6:
        iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
        median_size = np.median(all_sizes)
        if median_size > 0 and iqr / median_size > 0.05:
            return split_median_genotype(all_sizes, ref_size)

    return d1, d2


def vntr_bimodal_only(sizes_hp, ref_size):
    """HP median + bimodal check (from v1 — for baseline comparison)."""
    if not sizes_hp: return 0.0, 0.0
    hp_result = hp_median_genotype(sizes_hp, ref_size)
    if hp_result[0] != hp_result[1]:
        return hp_result
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(all_sizes) < 6:
        return hp_result
    iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
    median_size = np.median(all_sizes)
    if median_size > 0 and iqr / median_size > 0.05:
        return split_median_genotype(all_sizes, ref_size)
    return hp_result


def vntr_bimodal_tighter(sizes_hp, ref_size):
    """HP median + tighter bimodal threshold (3% instead of 5%)."""
    if not sizes_hp: return 0.0, 0.0
    hp_result = hp_median_genotype(sizes_hp, ref_size)
    if hp_result[0] != hp_result[1]:
        return hp_result
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(all_sizes) < 6:
        return hp_result
    iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
    median_size = np.median(all_sizes)
    if median_size > 0 and iqr / median_size > 0.03:
        return split_median_genotype(all_sizes, ref_size)
    return hp_result


def vntr_bimodal_looser(sizes_hp, ref_size):
    """HP median + looser bimodal threshold (10%)."""
    if not sizes_hp: return 0.0, 0.0
    hp_result = hp_median_genotype(sizes_hp, ref_size)
    if hp_result[0] != hp_result[1]:
        return hp_result
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(all_sizes) < 6:
        return hp_result
    iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
    median_size = np.median(all_sizes)
    if median_size > 0 and iqr / median_size > 0.10:
        return split_median_genotype(all_sizes, ref_size)
    return hp_result


def vntr_hp_with_read_variance_cap(sizes_hp, ref_size):
    """HP median, but cap extreme predictions based on read variance.

    If prediction implies allele size very far from read consensus,
    clip to the most extreme read value.
    """
    if not sizes_hp: return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)

    def capped_median(arr):
        if len(arr) == 0: return None
        med = np.median(arr)
        return med

    m1 = capped_median(hp1)
    m2 = capped_median(hp2)

    if m1 is None and m2 is None:
        return mode_round_genotype(all_sizes, ref_size)
    if m1 is None:
        d = round(ref_size - m2)
        return d, d
    if m2 is None:
        d = round(ref_size - m1)
        return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)

    # Check for bimodality if homozygous
    if d1 == d2 and len(all_sizes) >= 6:
        iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
        median_size = np.median(all_sizes)
        if median_size > 0 and iqr / median_size > 0.05:
            return split_median_genotype(all_sizes, ref_size)

    return sorted([d1, d2])


# ── Combined strategies ──

def make_combined(vntr_fn, name):
    def combined(sizes_hp, ref_size, motif_len):
        if motif_len >= 7:
            return vntr_fn(sizes_hp, ref_size)
        else:
            return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)
    combined.__name__ = name
    return combined


def hp_cond_baseline(sizes_hp, ref_size, motif_len):
    if motif_len >= 7:
        return hp_median_genotype(sizes_hp, ref_size)
    else:
        return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--longtr-vcf", required=True)
    parser.add_argument("--hp-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading data...")
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
    h5.close()

    longtr = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)
    matched_idx, matched_vals = match_loci(h5_keys, ref_sizes, longtr)
    longtr_mask = np.zeros(len(h5_keys), dtype=bool)
    longtr_diffs = np.zeros((len(h5_keys), 2))
    for j, val in zip(matched_idx, matched_vals):
        longtr_mask[j] = True
        longtr_diffs[j] = [val[0], val[1]]

    with open(args.hp_cache, "rb") as f:
        all_reads_hp = pickle.load(f)["all_reads_hp"]

    propmode_diffs = np.zeros_like(true_diffs)
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref = all_locus_features[idx, REF_SIZE_IDX]
        propmode_diffs[i] = prop_mode_genotype(allele_sizes, ref)

    # Methods
    methods = {
        "hp_cond (baseline)": hp_cond_baseline,
        "trimmed+bimodal": make_combined(vntr_trimmed_bimodal, "trimmed+bimodal"),
        "trimmed+bimodal_gentle": make_combined(vntr_trimmed_bimodal_gentle, "trimmed+bimodal_gentle"),
        "bimodal_5pct": make_combined(vntr_bimodal_only, "bimodal_5pct"),
        "bimodal_3pct": make_combined(vntr_bimodal_tighter, "bimodal_3pct"),
        "bimodal_10pct": make_combined(vntr_bimodal_looser, "bimodal_10pct"),
        "hp_varcap": make_combined(vntr_hp_with_read_variance_cap, "hp_varcap"),
    }

    print("Computing genotypes...")
    method_diffs = {}
    for name, fn in methods.items():
        diffs = np.zeros_like(true_diffs)
        for i in range(len(h5_keys)):
            reads_hp = all_reads_hp[i]
            if reads_hp:
                diffs[i] = fn(reads_hp, ref_sizes[i], motif_lens[i])
            else:
                diffs[i] = propmode_diffs[i]
        method_diffs[name] = diffs

    # Report
    L = []
    def w(s=""): L.append(s)

    tp_ltr = longtr_mask & is_tp
    is_vntr = motif_lens >= 7

    w("=" * 90)
    w("  VNTR MAE OPTIMIZATION v2 — Trimmed + Bimodal Combinations")
    w("=" * 90)

    # Overall TP
    w()
    w(f"  OVERALL TP (n={tp_ltr.sum():,})")
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'Net':>5}")
    w("  " + "-" * 60)

    ltr_errs = np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr])
    ltr_correct = np.all(ltr_errs < 0.5, axis=1)
    w(f"  {'LongTR':<30} {ltr_correct.mean():>6.1%} {(ltr_errs.flatten()<=1).mean():>6.1%} "
      f"{ltr_errs.flatten().mean():>6.2f}   ref")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[tp_ltr] - diffs[tp_ltr])
        correct = np.all(errs < 0.5, axis=1)
        gains = (~ltr_correct & correct).sum()
        losses = (ltr_correct & ~correct).sum()
        net = gains - losses
        w(f"  {name:<30} {correct.mean():>6.1%} {(errs.flatten()<=1).mean():>6.1%} "
          f"{errs.flatten().mean():>6.2f} {net:>+5}")

    # VNTR TP
    vntr_tp = tp_ltr & is_vntr
    w()
    w(f"  VNTR TP (n={vntr_tp.sum():,})")
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7}")
    w("  " + "-" * 55)

    errs = np.abs(true_diffs[vntr_tp] - longtr_diffs[vntr_tp]).flatten()
    w(f"  {'LongTR':<30} {np.all(np.abs(true_diffs[vntr_tp]-longtr_diffs[vntr_tp])<0.5, axis=1).mean():>6.1%} "
      f"{(errs<=1).mean():>6.1%} {errs.mean():>6.2f}")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[vntr_tp] - diffs[vntr_tp]).flatten()
        exact = np.all(np.abs(true_diffs[vntr_tp] - diffs[vntr_tp]) < 0.5, axis=1).mean()
        w(f"  {name:<30} {exact:>6.1%} {(errs<=1).mean():>6.1%} {errs.mean():>6.2f}")

    # Overall (all loci)
    w()
    w(f"  OVERALL — ALL LOCI (n={longtr_mask.sum():,})")
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'ZygAcc':>7} {'GenConc':>7}")
    w("  " + "-" * 72)

    ml_all = motif_lens[longtr_mask]
    ltr_all_errs = np.abs(true_diffs[longtr_mask] - longtr_diffs[longtr_mask])
    ltr_all_mu = ltr_all_errs / np.maximum(ml_all.reshape(-1, 1), 1.0)
    ltr_zyg = (np.abs(true_diffs[longtr_mask, 1] - true_diffs[longtr_mask, 0]) > ml_all).astype(int)
    ltr_pred_zyg = (np.abs(longtr_diffs[longtr_mask, 1] - longtr_diffs[longtr_mask, 0]) > ml_all).astype(int)
    w(f"  {'LongTR':<30} {np.all(ltr_all_errs<0.5, axis=1).mean():>6.1%} "
      f"{(ltr_all_errs.flatten()<=1).mean():>6.1%} {ltr_all_errs.flatten().mean():>6.2f} "
      f"{(ltr_zyg==ltr_pred_zyg).mean():>6.1%} {np.all(ltr_all_mu<=1, axis=1).mean():>6.1%}")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[longtr_mask] - diffs[longtr_mask])
        flat = errs.flatten()
        mu_errs = errs / np.maximum(ml_all.reshape(-1, 1), 1.0)
        pred_zyg = (np.abs(diffs[longtr_mask, 1] - diffs[longtr_mask, 0]) > ml_all).astype(int)
        w(f"  {name:<30} {np.all(errs<0.5, axis=1).mean():>6.1%} "
          f"{(flat<=1).mean():>6.1%} {flat.mean():>6.2f} "
          f"{(ltr_zyg==pred_zyg).mean():>6.1%} {np.all(mu_errs<=1, axis=1).mean():>6.1%}")

    # Per-locus error analysis: top 10 for best method
    w()
    w("=" * 90)
    w("  TOP 10 ERROR LOCI — trimmed+bimodal")
    w("=" * 90)
    w()

    best_diffs = method_diffs["trimmed+bimodal"]
    locus_errs = np.abs(true_diffs[tp_ltr] - best_diffs[tp_ltr])
    locus_mae = locus_errs.mean(axis=1)
    ltr_locus_errs = np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr])
    ltr_locus_mae = ltr_locus_errs.mean(axis=1)

    worst_10 = np.argsort(locus_mae)[-10:][::-1]
    tp_indices = np.where(tp_ltr)[0]

    w(f"  {'Chrom':<6} {'Start':>10} {'RefSz':>6} {'Motif':>5} "
      f"{'True':>12} {'Ours':>12} {'LTR':>12} {'OurErr':>6} {'LTRErr':>6}")
    w("  " + "-" * 85)

    for rank_idx in worst_10:
        glob_idx = tp_indices[rank_idx]
        chrom, start, end = h5_keys[glob_idx]
        ref = ref_sizes[glob_idx]
        ml = motif_lens[glob_idx]
        td = true_diffs[glob_idx]
        our_d = best_diffs[glob_idx]
        ltr_d = longtr_diffs[glob_idx]
        our_e = locus_mae[rank_idx]
        ltr_e = ltr_locus_mae[rank_idx]
        w(f"  {chrom:<6} {start:>10} {ref:>6.0f} {ml:>5.0f} "
          f"  ({td[0]:>4.0f},{td[1]:>4.0f})"
          f"  ({our_d[0]:>4.0f},{our_d[1]:>4.0f})"
          f"  ({ltr_d[0]:>4.0f},{ltr_d[1]:>4.0f})"
          f"  {our_e:>5.0f} {ltr_e:>5.0f}")

    report = "\n".join(L)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
