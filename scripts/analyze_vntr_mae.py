#!/usr/bin/env python3
"""Focused VNTR MAE analysis — try different strategies for VNTR loci only.

The overall TP MAE gap (3.87 vs LongTR 2.60) is entirely driven by VNTR loci.
Dinuc and STR already beat LongTR on MAE. This script tests VNTR-specific strategies.
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


# ── Base genotyping methods ──

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


def split_median_genotype(allele_sizes, ref_size):
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


# ── HP-aware base methods ──

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


# ── VNTR-specific strategies ──

def vntr_split_median(sizes_hp, ref_size):
    """Ignore HP tags for VNTRs, just use sorted split-median."""
    all_sizes = np.array([s for s, _ in sizes_hp]) if sizes_hp else np.array([])
    if len(all_sizes) == 0:
        return 0.0, 0.0
    return split_median_genotype(all_sizes, ref_size)


def vntr_hp_with_bimodal_check(sizes_hp, ref_size):
    """HP median, but check for bimodality when HP says homozygous.

    If HP-median returns homozygous but reads have high variance,
    fall back to split_median to capture potential het.
    """
    if not sizes_hp:
        return 0.0, 0.0

    # First get HP-median result
    hp_result = hp_median_genotype(sizes_hp, ref_size)

    # If het, trust HP
    if hp_result[0] != hp_result[1]:
        return hp_result

    # Homozygous — check if reads actually look bimodal
    all_sizes = np.array([s for s, _ in sizes_hp])
    if len(all_sizes) < 6:
        return hp_result

    iqr = np.percentile(all_sizes, 75) - np.percentile(all_sizes, 25)
    median_size = np.median(all_sizes)

    # If high relative variance, likely het mis-called as hom
    # Threshold: IQR > 5% of median allele size
    if median_size > 0 and iqr / median_size > 0.05:
        return split_median_genotype(all_sizes, ref_size)

    return hp_result


def vntr_hp_trimmed_strict(sizes_hp, ref_size):
    """HP median with strict outlier removal (1.0*IQR instead of 1.5*IQR)."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def strict_trimmed_med(arr):
        if len(arr) < 4:
            return np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lo = q1 - 1.0 * iqr
        hi = q3 + 1.0 * iqr
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) == 0:
            return np.median(arr)
        return np.median(trimmed)

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - strict_trimmed_med(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - strict_trimmed_med(hp1))
        return d, d
    d1 = round(ref_size - strict_trimmed_med(hp1))
    d2 = round(ref_size - strict_trimmed_med(hp2))
    return sorted([d1, d2])


def vntr_hp_with_split_fallback(sizes_hp, ref_size):
    """HP median, but if both HP groups have same median, check split_median.

    If split_median shows het with reasonable separation, use it.
    Otherwise keep HP homozygous call.
    """
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

    med1 = np.median(hp1)
    med2 = np.median(hp2)
    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)

    # If HP gives het, trust it
    if d1 != d2:
        return sorted([d1, d2])

    # HP says homozygous — check split_median
    sm = split_median_genotype(all_sizes, ref_size)
    if sm[0] != sm[1]:
        # Split-median found het — use it if the separation is meaningful
        sep = abs(sm[0] - sm[1])
        if sep > 1:  # More than 1bp difference
            return sm

    return d1, d2


def vntr_hp_best_of(sizes_hp, ref_size):
    """Try both HP median and split-median, pick lower intra-group variance."""
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    # HP median
    hp_res = hp_median_genotype(sizes_hp, ref_size)

    # Split median (no HP)
    sm_res = split_median_genotype(all_sizes, ref_size)

    # If they agree, return either
    if hp_res == sm_res or (hp_res[0] == sm_res[0] and hp_res[1] == sm_res[1]):
        return hp_res

    # Compute "fit" of each: sum of squared distances from each read to nearest allele
    def fit_score(d1, d2):
        a1 = ref_size - d1
        a2 = ref_size - d2
        dists = np.minimum(np.abs(all_sizes - a1), np.abs(all_sizes - a2))
        return np.sum(dists**2)

    hp_fit = fit_score(*hp_res)
    sm_fit = fit_score(*sm_res)

    return hp_res if hp_fit <= sm_fit else sm_res


def vntr_hp_robust_center(sizes_hp, ref_size):
    """Use the center method that gives lowest within-group variance for each HP group.

    For each HP group, compute: mode, median, trimmed_median.
    Pick the one where residuals are smallest.
    """
    if not sizes_hp:
        return 0.0, 0.0
    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    def best_center(arr):
        if len(arr) < 3:
            return np.median(arr)
        med = np.median(arr)
        int_arr = np.round(arr).astype(int)
        if int_arr.max() == int_arr.min():
            return float(int_arr[0])
        counts = np.bincount(int_arr - int_arr.min())
        mode = float(int_arr.min() + np.argmax(counts))
        # Trimmed median
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        trimmed = arr[(arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)]
        tmed = np.median(trimmed) if len(trimmed) > 0 else med
        # Pick center with lowest sum of abs residuals
        candidates = [med, mode, tmed]
        best = min(candidates, key=lambda c: np.sum(np.abs(arr - c)))
        return best

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)
    if len(hp1) == 0:
        d = round(ref_size - best_center(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - best_center(hp1))
        return d, d
    d1 = round(ref_size - best_center(hp1))
    d2 = round(ref_size - best_center(hp2))
    return sorted([d1, d2])


# ── Combined strategies (use HP for dinuc/STR, different VNTR strategy) ──

def make_combined(vntr_fn, name):
    """Create a combined method: hp_median_prop for dinuc/STR, vntr_fn for VNTR."""
    def combined(sizes_hp, ref_size, motif_len):
        if motif_len >= 7:
            return vntr_fn(sizes_hp, ref_size)
        else:
            return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)
    combined.__name__ = name
    return combined


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

    # LongTR
    longtr = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)
    matched_idx, matched_vals = match_loci(h5_keys, ref_sizes, longtr)
    longtr_mask = np.zeros(len(h5_keys), dtype=bool)
    longtr_diffs = np.zeros((len(h5_keys), 2))
    for j, val in zip(matched_idx, matched_vals):
        longtr_mask[j] = True
        longtr_diffs[j] = [val[0], val[1]]

    # HP cache
    with open(args.hp_cache, "rb") as f:
        all_reads_hp = pickle.load(f)["all_reads_hp"]

    # PropMode fallback
    propmode_diffs = np.zeros_like(true_diffs)
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref = all_locus_features[idx, REF_SIZE_IDX]
        propmode_diffs[i] = prop_mode_genotype(allele_sizes, ref)

    # Define combined strategies
    vntr_strategies = {
        "hp_median": hp_median_genotype,
        "split_median": vntr_split_median,
        "hp_bimodal": vntr_hp_with_bimodal_check,
        "hp_trimmed_strict": vntr_hp_trimmed_strict,
        "hp_split_fallback": vntr_hp_with_split_fallback,
        "hp_best_of": vntr_hp_best_of,
        "hp_robust_center": vntr_hp_robust_center,
    }

    combined_methods = {}
    for vname, vfn in vntr_strategies.items():
        combined_methods[f"comb_{vname}"] = make_combined(vfn, f"comb_{vname}")

    # Also keep hp_cond (baseline) for comparison
    def hp_cond_genotype(sizes_hp, ref_size, motif_len):
        if motif_len >= 7:
            return hp_median_genotype(sizes_hp, ref_size)
        else:
            return hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.25)

    combined_methods["hp_cond (baseline)"] = hp_cond_genotype

    # Compute all methods
    print("Computing genotypes for all methods...")
    method_diffs = {}
    for name, fn in combined_methods.items():
        diffs = np.zeros_like(true_diffs)
        for i in range(len(h5_keys)):
            reads_hp = all_reads_hp[i]
            if reads_hp:
                diffs[i] = fn(reads_hp, ref_sizes[i], motif_lens[i])
            else:
                diffs[i] = propmode_diffs[i]
        method_diffs[name] = diffs

    # ── Report ──
    L = []
    def w(s=""):
        L.append(s)

    tp_ltr = longtr_mask & is_tp
    is_vntr = motif_lens >= 7
    is_str = (motif_lens >= 3) & (motif_lens <= 6)
    is_dinuc = motif_lens == 2

    w("=" * 90)
    w("  VNTR-FOCUSED MAE OPTIMIZATION")
    w(f"  Total TP in LongTR set: {tp_ltr.sum():,}")
    w(f"  VNTR TP: {(tp_ltr & is_vntr).sum():,}")
    w(f"  STR TP: {(tp_ltr & is_str).sum():,}")
    w(f"  Dinuc TP: {(tp_ltr & is_dinuc).sum():,}")
    w("=" * 90)

    # Overall TP metrics
    w()
    w("  OVERALL TP (LongTR set, n={:,})".format(tp_ltr.sum()))
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'MedAE':>7} {'≤1mu':>7} {'ZygAcc':>7}")
    w("  " + "-" * 79)

    ltr_errs = np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr])
    ltr_flat = ltr_errs.flatten()
    ltr_exact = np.all(ltr_errs < 0.5, axis=1).mean()
    ltr_w1 = (ltr_flat <= 1.0).mean()
    ml_tp = motif_lens[tp_ltr]
    ltr_mu_errs = ltr_errs / np.maximum(ml_tp.reshape(-1, 1), 1.0)
    ltr_w1mu = (ltr_mu_errs.flatten() <= 1.0).mean()
    ltr_zyg = (np.abs(true_diffs[tp_ltr, 1] - true_diffs[tp_ltr, 0]) > ml_tp).astype(int)
    ltr_pred_zyg = (np.abs(longtr_diffs[tp_ltr, 1] - longtr_diffs[tp_ltr, 0]) > ml_tp).astype(int)
    ltr_zygacc = (ltr_zyg == ltr_pred_zyg).mean()
    w(f"  {'LongTR':<30} {ltr_exact:>6.1%} {ltr_w1:>6.1%} {ltr_flat.mean():>6.2f} "
      f"{np.median(ltr_flat):>6.2f} {ltr_w1mu:>6.1%} {ltr_zygacc:>6.1%}")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[tp_ltr] - diffs[tp_ltr])
        flat = errs.flatten()
        exact = np.all(errs < 0.5, axis=1).mean()
        w1 = (flat <= 1.0).mean()
        mu_errs = errs / np.maximum(ml_tp.reshape(-1, 1), 1.0)
        w1mu = (mu_errs.flatten() <= 1.0).mean()
        pred_zyg = (np.abs(diffs[tp_ltr, 1] - diffs[tp_ltr, 0]) > ml_tp).astype(int)
        zygacc = (ltr_zyg == pred_zyg).mean()
        w(f"  {name:<30} {exact:>6.1%} {w1:>6.1%} {flat.mean():>6.2f} "
          f"{np.median(flat):>6.2f} {w1mu:>6.1%} {zygacc:>6.1%}")

    # VNTR TP only
    vntr_tp = tp_ltr & is_vntr
    w()
    w(f"  VNTR TP ONLY (n={vntr_tp.sum():,})")
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'MedAE':>7}")
    w("  " + "-" * 55)

    errs = np.abs(true_diffs[vntr_tp] - longtr_diffs[vntr_tp]).flatten()
    exact = np.all(np.abs(true_diffs[vntr_tp] - longtr_diffs[vntr_tp]) < 0.5, axis=1).mean()
    w(f"  {'LongTR':<30} {exact:>6.1%} {(errs<=1).mean():>6.1%} {errs.mean():>6.2f} {np.median(errs):>6.2f}")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[vntr_tp] - diffs[vntr_tp]).flatten()
        exact = np.all(np.abs(true_diffs[vntr_tp] - diffs[vntr_tp]) < 0.5, axis=1).mean()
        w(f"  {name:<30} {exact:>6.1%} {(errs<=1).mean():>6.1%} {errs.mean():>6.2f} {np.median(errs):>6.2f}")

    # Dinuc + STR TP (to make sure we don't regress)
    for gn, gm_base in [("Dinuc TP", is_dinuc), ("STR TP", is_str)]:
        gm = tp_ltr & gm_base
        if gm.sum() == 0:
            continue
        w()
        w(f"  {gn} (n={gm.sum():,})")
        w()
        w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7}")
        w("  " + "-" * 55)

        errs = np.abs(true_diffs[gm] - longtr_diffs[gm]).flatten()
        exact = np.all(np.abs(true_diffs[gm] - longtr_diffs[gm]) < 0.5, axis=1).mean()
        w(f"  {'LongTR':<30} {exact:>6.1%} {(errs<=1).mean():>6.1%} {errs.mean():>6.2f}")

        for name in sorted(method_diffs.keys()):
            diffs = method_diffs[name]
            errs = np.abs(true_diffs[gm] - diffs[gm]).flatten()
            exact = np.all(np.abs(true_diffs[gm] - diffs[gm]) < 0.5, axis=1).mean()
            w(f"  {name:<30} {exact:>6.1%} {(errs<=1).mean():>6.1%} {errs.mean():>6.2f}")

    # Net vs LongTR
    w()
    w("=" * 90)
    w("  NET GAINS/LOSSES vs LongTR (TP exact)")
    w("=" * 90)
    w()

    ltr_correct = np.all(np.abs(true_diffs[tp_ltr] - longtr_diffs[tp_ltr]) < 0.5, axis=1)
    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        hp_correct = np.all(np.abs(true_diffs[tp_ltr] - diffs[tp_ltr]) < 0.5, axis=1)
        gains = (~ltr_correct & hp_correct).sum()
        losses = (ltr_correct & ~hp_correct).sum()
        w(f"  {name:<30} gains={gains:>4} losses={losses:>4} net={gains-losses:>+5}")

    # OVERALL metrics (not just TP)
    w()
    w("=" * 90)
    w(f"  OVERALL (all loci in LongTR set, n={longtr_mask.sum():,})")
    w("=" * 90)
    w()
    w(f"  {'Method':<30} {'Exact':>7} {'≤1bp':>7} {'MAE':>7} {'ZygAcc':>7} {'GenConc':>7}")
    w("  " + "-" * 70)

    ltr_all_errs = np.abs(true_diffs[longtr_mask] - longtr_diffs[longtr_mask])
    ml_all = motif_lens[longtr_mask]
    ltr_all_flat = ltr_all_errs.flatten()
    ltr_all_exact = np.all(ltr_all_errs < 0.5, axis=1).mean()
    ltr_all_mu = ltr_all_errs / np.maximum(ml_all.reshape(-1, 1), 1.0)
    ltr_all_zyg = (np.abs(true_diffs[longtr_mask, 1] - true_diffs[longtr_mask, 0]) > ml_all).astype(int)
    ltr_all_pred_zyg = (np.abs(longtr_diffs[longtr_mask, 1] - longtr_diffs[longtr_mask, 0]) > ml_all).astype(int)
    w(f"  {'LongTR':<30} {ltr_all_exact:>6.1%} {(ltr_all_flat<=1).mean():>6.1%} "
      f"{ltr_all_flat.mean():>6.2f} {(ltr_all_zyg==ltr_all_pred_zyg).mean():>6.1%} "
      f"{np.all(ltr_all_mu<=1, axis=1).mean():>6.1%}")

    for name in sorted(method_diffs.keys()):
        diffs = method_diffs[name]
        errs = np.abs(true_diffs[longtr_mask] - diffs[longtr_mask])
        flat = errs.flatten()
        exact = np.all(errs < 0.5, axis=1).mean()
        mu_errs = errs / np.maximum(ml_all.reshape(-1, 1), 1.0)
        pred_zyg = (np.abs(diffs[longtr_mask, 1] - diffs[longtr_mask, 0]) > ml_all).astype(int)
        zygacc = (ltr_all_zyg == pred_zyg).mean()
        gc = np.all(mu_errs <= 1, axis=1).mean()
        w(f"  {name:<30} {exact:>6.1%} {(flat<=1).mean():>6.1%} "
          f"{flat.mean():>6.2f} {zygacc:>6.1%} {gc:>6.1%}")

    report = "\n".join(L)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
