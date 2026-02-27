#!/usr/bin/env python3
"""TP gap analysis v4: EM-mode + threshold sweep + combined strategies.

From v3: mode_round is best for het (85.3%), adaptive_1bp best for hom (88.8%).
adaptive_strict = mode_round (midpoint always gives different modes).
Need to close the 4.8%p het gap vs LongTR (90.1%).

New strategies:
1. EM-mode: Initial midpoint split + mode, then re-assign reads to nearest mode, recompute
2. Fine threshold sweep (10-40%) for proportion-based collapse
3. Hom-single: use single mode for everything, detect het only when gap is very large
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
    if len(int_arr) == 0:
        return float(np.median(arr)) if len(arr) > 0 else 0.0
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


def em_mode_genotype(allele_sizes, ref_size):
    """EM-mode: Midpoint split → mode → re-assign reads to nearest mode → recompute.

    Solves: skewed coverage cases where midpoint puts reads from dominant allele in both halves.
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)

    # Step 1: Initial estimate via midpoint split
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    # If modes are the same, it's hom
    if m1 == m2:
        d = round(ref_size - m1)
        return d, d

    # Step 2: Re-assign reads to nearest mode
    for _ in range(3):  # Up to 3 EM iterations
        m1_int = round(m1)
        m2_int = round(m2)

        # Assign each read to the nearest center
        dist1 = np.abs(int_sizes - m1_int)
        dist2 = np.abs(int_sizes - m2_int)
        assign1 = dist1 <= dist2  # ties go to cluster 1 (lower)

        # Need at least 1 read per cluster
        if assign1.sum() == 0 or assign1.sum() == n:
            break

        # Step 3: Recompute modes
        new_m1 = get_mode(sizes[assign1])
        new_m2 = get_mode(sizes[~assign1])

        if new_m1 == m1 and new_m2 == m2:
            break  # Converged
        m1, m2 = new_m1, new_m2

    # Final: collapse if modes within 1bp
    if abs(m1 - m2) <= 1:
        m_all = get_mode(sizes)
        d = round(ref_size - m_all)
        return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def em_mode_strict_genotype(allele_sizes, ref_size):
    """EM-mode without collapse — always return both modes.

    This should improve het accuracy by better read assignment
    without the hom collapse.
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)

    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d

    for _ in range(3):
        m1_int = round(m1)
        m2_int = round(m2)
        dist1 = np.abs(int_sizes - m1_int)
        dist2 = np.abs(int_sizes - m2_int)
        assign1 = dist1 <= dist2

        if assign1.sum() == 0 or assign1.sum() == n:
            break

        new_m1 = get_mode(sizes[assign1])
        new_m2 = get_mode(sizes[~assign1])

        if new_m1 == m1 and new_m2 == m2:
            break
        m1, m2 = new_m1, new_m2

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def em_mode_prop_genotype(allele_sizes, ref_size):
    """EM-mode with proportion-based collapse.

    After EM convergence, collapse to hom only if minor cluster < 20% of reads.
    """
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)

    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d

    for _ in range(3):
        m1_int = round(m1)
        m2_int = round(m2)
        dist1 = np.abs(int_sizes - m1_int)
        dist2 = np.abs(int_sizes - m2_int)
        assign1 = dist1 <= dist2

        if assign1.sum() == 0 or assign1.sum() == n:
            break

        new_m1 = get_mode(sizes[assign1])
        new_m2 = get_mode(sizes[~assign1])

        if new_m1 == m1 and new_m2 == m2:
            break
        m1, m2 = new_m1, new_m2

    # Collapse decision based on proportion
    if abs(m1 - m2) <= 1:
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < 0.20:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def prop_threshold_genotype(allele_sizes, ref_size, threshold):
    """Parametric proportion-based collapse."""
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
        if minor_frac < threshold:
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

    h5_keys, true_diffs, motif_lens, is_tp, ref_sizes = [], [], [], [], []
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

    # Compute methods
    def compute_all(genotype_fn, needs_extra=False, extra_arg=None):
        diffs = np.zeros((n_test, 2))
        for i, idx in enumerate(test_indices):
            offset = all_read_offsets[idx]
            count = all_read_counts[idx]
            reads = all_read_features[offset:offset + count]
            allele_sizes = reads[:, ALLELE_SIZE_IDX]
            ref_size = all_locus_features[idx, REF_SIZE_IDX]
            if needs_extra:
                d1, d2 = genotype_fn(allele_sizes, ref_size, extra_arg)
            else:
                d1, d2 = genotype_fn(allele_sizes, ref_size)
            diffs[i] = [d1, d2]
        return diffs

    methods = {}

    logger.info("Computing mode_round...")
    methods["mode_round"] = compute_all(mode_round_genotype)

    logger.info("Computing em_mode...")
    methods["em_mode"] = compute_all(em_mode_genotype)

    logger.info("Computing em_strict...")
    methods["em_strict"] = compute_all(em_mode_strict_genotype)

    logger.info("Computing em_prop...")
    methods["em_prop"] = compute_all(em_mode_prop_genotype)

    # Proportion threshold sweep
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        name = f"prop_{int(t*100)}pct"
        logger.info(f"Computing {name}...")
        methods[name] = compute_all(
            lambda sizes, ref, thresh=t: prop_threshold_genotype(sizes, ref, thresh))

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
        if not candidates: continue
        h5_ref = ref_sizes[i]
        pos = bisect.bisect_left(candidates, (start,))
        best_dist = MATCH_TOLERANCE + 1
        best_val = None
        for j in range(max(0, pos - 2), min(len(candidates), pos + 3)):
            dist = abs(candidates[j][0] - start)
            tool_ref = candidates[j][1][2]
            if h5_ref > 0 and tool_ref > 0:
                ratio = tool_ref / h5_ref
                if ratio < 0.5 or ratio > 2.0: continue
            if dist < best_dist:
                best_dist = dist
                best_val = candidates[j][1]
        if best_dist <= MATCH_TOLERANCE and best_val is not None:
            longtr_mask[i] = True
            longtr_diffs[i] = [best_val[0], best_val[1]]

    # Results
    lines = []
    def p(s=""): lines.append(s)

    THRESH = 0.5
    is_het_truth = np.abs(true_diffs[:, 1] - true_diffs[:, 0]) > 0.5
    tp_ltr = is_tp & longtr_mask
    tn_ltr = longtr_mask & ~is_tp
    het_m = tp_ltr & is_het_truth
    hom_m = tp_ltr & ~is_het_truth
    n_tp = tp_ltr.sum()

    p("=" * 100)
    p("  TP GAP ANALYSIS v4: EM-mode + Threshold Sweep")
    p(f"  TP n={n_tp:,} (Het={het_m.sum():,}, Hom={hom_m.sum():,}), LongTR matched")
    p("=" * 100)
    p()
    p(f"{'Method':<18} {'Overall':>8} {'TP':>8} {'TN':>8} {'Het-TP':>8} {'Hom-TP':>8}  {'TPw1bp':>7} {'TPMAE':>7}")
    p("-" * 95)

    for name in list(methods.keys()) + ["LongTR"]:
        diffs = methods[name] if name != "LongTR" else longtr_diffs
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH
        ov = exact[longtr_mask].mean()
        tp = exact[tp_ltr].mean()
        tn = exact[tn_ltr].mean()
        het = exact[het_m].mean()
        hom = exact[hom_m].mean()
        flat = np.abs(diffs[tp_ltr] - true_diffs[tp_ltr]).flatten()
        w1 = (flat <= 1.0).mean()
        mae = flat.mean()
        p(f"{name:<18} {ov:>7.1%} {tp:>7.1%} {tn:>7.1%} {het:>7.1%} {hom:>7.1%}  {w1:>6.1%} {mae:>6.2f}")

    p()

    # Full test set
    p("=" * 100)
    p("  FULL TEST SET")
    p("=" * 100)
    p()
    p(f"{'Method':<18} {'Overall':>8} {'TP':>8} {'TN':>8}")
    p("-" * 42)
    for name, diffs in methods.items():
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH
        p(f"{name:<18} {exact.mean():>7.1%} {exact[is_tp].mean():>7.1%} {exact[~is_tp].mean():>7.1%}")
    p()

    # Net gains vs mode_round
    p("=" * 100)
    p("  NET vs mode_round (TP, LongTR set)")
    p("=" * 100)
    p()
    mr_exact = np.max(np.abs(methods["mode_round"] - true_diffs), axis=1) < THRESH
    for name in methods:
        if name == "mode_round": continue
        n_exact = np.max(np.abs(methods[name] - true_diffs), axis=1) < THRESH
        gains = (tp_ltr & n_exact & ~mr_exact).sum()
        losses = (tp_ltr & mr_exact & ~n_exact).sum()
        p(f"  {name:<18} gains={gains:>3} losses={losses:>3} net={gains-losses:>+4}")

    p()

    # Oracle analysis: what if we knew het/hom?
    p("=" * 100)
    p("  ORACLE ANALYSIS: Best per-zygosity method")
    p("=" * 100)
    p()

    # For each method, compute het and hom TP exact
    best_het_name = max(methods.keys(), key=lambda n: (np.max(np.abs(methods[n] - true_diffs), axis=1) < THRESH)[het_m].mean())
    best_hom_name = max(methods.keys(), key=lambda n: (np.max(np.abs(methods[n] - true_diffs), axis=1) < THRESH)[hom_m].mean())

    best_het_exact = (np.max(np.abs(methods[best_het_name] - true_diffs), axis=1) < THRESH)
    best_hom_exact = (np.max(np.abs(methods[best_hom_name] - true_diffs), axis=1) < THRESH)

    het_acc = best_het_exact[het_m].mean()
    hom_acc = best_hom_exact[hom_m].mean()
    oracle_tp = (best_het_exact[het_m].sum() + best_hom_exact[hom_m].sum()) / n_tp

    p(f"  Best Het method: {best_het_name} ({het_acc:.1%})")
    p(f"  Best Hom method: {best_hom_name} ({hom_acc:.1%})")
    p(f"  Oracle TP (het+hom combined): {oracle_tp:.1%}")
    p(f"  LongTR TP: {(np.max(np.abs(longtr_diffs - true_diffs), axis=1) < THRESH)[tp_ltr].mean():.1%}")
    p()

    # Per-locus oracle: best across ALL methods
    p("  Per-locus oracle (best method per locus):")
    any_correct = np.zeros(n_test, dtype=bool)
    for name, diffs in methods.items():
        exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
        any_correct |= exact
    p(f"    Any method correct (TP): {any_correct[tp_ltr].mean():.1%} ({any_correct[tp_ltr].sum()}/{n_tp})")
    p(f"    LongTR correct (TP): {(np.max(np.abs(longtr_diffs - true_diffs), axis=1) < THRESH)[tp_ltr].mean():.1%}")
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
