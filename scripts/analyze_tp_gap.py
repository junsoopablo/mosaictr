#!/usr/bin/env python3
"""Analyze TP exact match gap between mode_round and LongTR.

Computes win/loss categories and stratifies by locus properties
to identify exactly where mode_round fails and LongTR succeeds.
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
MOTIF_LEN_IDX = 1
TEST_CHROMS = {"chr21", "chr22", "chrX"}
MATCH_TOLERANCE = 15


def mode_round_genotype(allele_sizes, ref_size):
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


def gap_mode_genotype(allele_sizes, ref_size, motif_len):
    """Gap-based split + mode: detect het by largest gap, else use all reads."""
    sizes = np.sort(allele_sizes)
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        if len(int_arr) == 0:
            return float(np.median(arr))
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    # Find largest gap
    gaps = np.diff(sizes)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]

    # Het detection: gap must be >= motif_len and each side >= 2 reads
    min_gap = max(1.0, motif_len * 0.75)
    left_n = max_gap_idx + 1
    right_n = n - left_n

    if max_gap >= min_gap and left_n >= 2 and right_n >= 2:
        # Het: split at gap
        a1 = get_mode(sizes[:left_n])
        a2 = get_mode(sizes[left_n:])
    else:
        # Hom: use all reads, single mode
        a_all = get_mode(sizes)
        a1 = a2 = a_all

    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def quality_mode_genotype(allele_sizes, mapqs, flank_lefts, flank_rights, ref_size, motif_len):
    """Mode with quality filtering: exclude poor reads before mode computation."""
    # Filter: mapq >= 20, both flanks >= 0.7
    mask = (mapqs >= 20) & (flank_lefts >= 0.7) & (flank_rights >= 0.7)
    if mask.sum() < 2:
        mask = mapqs >= 5  # fallback to permissive
    if mask.sum() == 0:
        return 0.0, 0.0

    filtered = np.sort(allele_sizes[mask])
    return _mode_split(filtered, ref_size, motif_len)


def _mode_split(sizes, ref_size, motif_len):
    """Common mode-split logic (gap-based)."""
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    gaps = np.diff(sizes)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]
    min_gap = max(1.0, motif_len * 0.75)
    left_n = max_gap_idx + 1
    right_n = n - left_n

    if max_gap >= min_gap and left_n >= 2 and right_n >= 2:
        a1 = get_mode(sizes[:left_n])
        a2 = get_mode(sizes[left_n:])
    else:
        a_all = get_mode(sizes)
        a1 = a2 = a_all

    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return sorted([d1, d2])


def combined_genotype(allele_sizes, mapqs, flank_lefts, flank_rights, ref_size, motif_len):
    """Quality filtering + gap-based split + mode."""
    mask = (mapqs >= 20) & (flank_lefts >= 0.7) & (flank_rights >= 0.7)
    if mask.sum() < 3:
        mask = mapqs >= 5
    if mask.sum() == 0:
        return 0.0, 0.0
    filtered = np.sort(allele_sizes[mask])
    return _mode_split(filtered, ref_size, motif_len)


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
    n_test = len(test_indices)

    # Compute per-locus features
    logger.info("Computing per-locus read stats...")
    allele_stds = np.zeros(n_test)
    read_counts = np.zeros(n_test, dtype=int)
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        read_counts[i] = count
        if count > 1:
            sizes = all_read_features[offset:offset+count, ALLELE_SIZE_IDX]
            allele_stds[i] = np.std(sizes)

    # Compute mode_round diffs
    logger.info("Computing mode_round diffs...")
    mr_diffs = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset+count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        d1, d2 = mode_round_genotype(allele_sizes, ref_size)
        mr_diffs[i] = [d1, d2]

    # Compute gap_mode diffs
    logger.info("Computing gap_mode diffs...")
    gm_diffs = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset+count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        ml = int(motif_lens[i])
        d1, d2 = gap_mode_genotype(allele_sizes, ref_size, ml)
        gm_diffs[i] = [d1, d2]

    # Compute quality_mode diffs
    logger.info("Computing quality_mode diffs...")
    qm_diffs = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset+count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        mapqs = reads[:, 7]
        flank_l = reads[:, 10]
        flank_r = reads[:, 11]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        ml = int(motif_lens[i])
        d1, d2 = quality_mode_genotype(allele_sizes, mapqs, flank_l, flank_r, ref_size, ml)
        qm_diffs[i] = [d1, d2]

    # Compute combined diffs (quality + gap + mode)
    logger.info("Computing combined diffs...")
    cb_diffs = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset+count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        mapqs = reads[:, 7]
        flank_l = reads[:, 10]
        flank_r = reads[:, 11]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]
        ml = int(motif_lens[i])
        d1, d2 = combined_genotype(allele_sizes, mapqs, flank_l, flank_r, ref_size, ml)
        cb_diffs[i] = [d1, d2]

    h5.close()

    # Parse LongTR
    logger.info("Parsing LongTR VCF...")
    longtr_dict = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)

    # Match LongTR
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

    # ── Analysis on LongTR-matched TP loci ──
    tp_ltr = is_tp & longtr_mask
    n_tp_ltr = tp_ltr.sum()
    logger.info(f"TP loci matched by LongTR: {n_tp_ltr:,}")

    # Errors
    THRESH = 0.5
    mr_errs = np.max(np.abs(mr_diffs - true_diffs), axis=1)
    lr_errs = np.max(np.abs(longtr_diffs - true_diffs), axis=1)
    gm_errs = np.max(np.abs(gm_diffs - true_diffs), axis=1)
    qm_errs = np.max(np.abs(qm_diffs - true_diffs), axis=1)
    cb_errs = np.max(np.abs(cb_diffs - true_diffs), axis=1)

    mr_exact = mr_errs < THRESH
    lr_exact = lr_errs < THRESH
    gm_exact = gm_errs < THRESH
    qm_exact = qm_errs < THRESH
    cb_exact = cb_errs < THRESH

    # Win/loss categories
    lr_wins = tp_ltr & lr_exact & ~mr_exact  # LongTR right, mode_round wrong
    mr_wins = tp_ltr & mr_exact & ~lr_exact  # mode_round right, LongTR wrong
    both_right = tp_ltr & lr_exact & mr_exact
    both_wrong = tp_ltr & ~lr_exact & ~mr_exact

    # Het/hom detection from truth
    truth_diff_range = np.abs(true_diffs[:, 1] - true_diffs[:, 0])
    is_het_truth = truth_diff_range > 0.5  # any allele difference

    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 70)
    p("  TP GAP ANALYSIS: mode_round vs LongTR")
    p(f"  Test set: chr21, chr22, chrX — TP loci matched by LongTR (n={n_tp_ltr:,})")
    p("=" * 70)
    p()

    p(f"  mode_round TP exact: {mr_exact[tp_ltr].mean():.1%} ({mr_exact[tp_ltr].sum():,}/{n_tp_ltr:,})")
    p(f"  LongTR    TP exact: {lr_exact[tp_ltr].mean():.1%} ({lr_exact[tp_ltr].sum():,}/{n_tp_ltr:,})")
    p(f"  gap_mode  TP exact: {gm_exact[tp_ltr].mean():.1%} ({gm_exact[tp_ltr].sum():,}/{n_tp_ltr:,})")
    p(f"  qual_mode TP exact: {qm_exact[tp_ltr].mean():.1%} ({qm_exact[tp_ltr].sum():,}/{n_tp_ltr:,})")
    p(f"  combined  TP exact: {cb_exact[tp_ltr].mean():.1%} ({cb_exact[tp_ltr].sum():,}/{n_tp_ltr:,})")
    p()
    p(f"  LongTR wins (LR right, MR wrong): {lr_wins.sum():,} ({lr_wins.sum()/n_tp_ltr:.1%})")
    p(f"  mode_round wins (MR right, LR wrong): {mr_wins.sum():,} ({mr_wins.sum()/n_tp_ltr:.1%})")
    p(f"  Both right: {both_right.sum():,} ({both_right.sum()/n_tp_ltr:.1%})")
    p(f"  Both wrong: {both_wrong.sum():,} ({both_wrong.sum()/n_tp_ltr:.1%})")
    p()

    # Can new methods recover LongTR wins?
    gm_recovers = lr_wins & gm_exact  # gap_mode fixes cases where mode_round failed
    qm_recovers = lr_wins & qm_exact
    cb_recovers = lr_wins & cb_exact
    p(f"  gap_mode recovers from LR_wins: {gm_recovers.sum():,}/{lr_wins.sum():,} ({gm_recovers.sum()/max(1,lr_wins.sum()):.1%})")
    p(f"  qual_mode recovers from LR_wins: {qm_recovers.sum():,}/{lr_wins.sum():,} ({qm_recovers.sum()/max(1,lr_wins.sum()):.1%})")
    p(f"  combined recovers from LR_wins: {cb_recovers.sum():,}/{lr_wins.sum():,} ({cb_recovers.sum()/max(1,lr_wins.sum()):.1%})")
    p()

    # Does gap_mode/combined break things mode_round got right?
    gm_breaks = mr_wins & ~gm_exact  # cases mode_round won but gap_mode loses
    cb_breaks = mr_wins & ~cb_exact
    p(f"  gap_mode breaks MR_wins: {(tp_ltr & mr_exact & ~gm_exact).sum():,}/{mr_exact[tp_ltr].sum():,}")
    p(f"  combined breaks MR_wins: {(tp_ltr & mr_exact & ~cb_exact).sum():,}/{mr_exact[tp_ltr].sum():,}")
    p()

    # ── Stratify LongTR wins ──
    def stratify(mask, name):
        p(f"--- {name} ---")
        n = mask.sum()
        if n == 0:
            p("  (none)")
            return

        # Motif breakdown
        ml = motif_lens[mask]
        dinuc = (ml == 2).sum()
        str_m = ((ml >= 3) & (ml <= 6)).sum()
        vntr = (ml >= 7).sum()
        p(f"  Motif: dinuc={dinuc} ({dinuc/n:.0%}), STR={str_m} ({str_m/n:.0%}), VNTR={vntr} ({vntr/n:.0%})")

        # Ref size
        rs = ref_sizes[mask]
        lt100 = (rs < 100).sum()
        r100_500 = ((rs >= 100) & (rs < 500)).sum()
        r500_1k = ((rs >= 500) & (rs < 1000)).sum()
        gt1k = (rs >= 1000).sum()
        p(f"  RefSize: <100={lt100} ({lt100/n:.0%}), 100-500={r100_500} ({r100_500/n:.0%}), "
          f"500-1k={r500_1k} ({r500_1k/n:.0%}), >1k={gt1k} ({gt1k/n:.0%})")

        # Het/hom
        het = is_het_truth[mask]
        n_het = het.sum()
        n_hom = n - n_het
        p(f"  Zygosity: het={n_het} ({n_het/n:.0%}), hom={n_hom} ({n_hom/n:.0%})")

        # Read count
        rc = read_counts[mask]
        p(f"  ReadCount: mean={rc.mean():.1f}, median={np.median(rc):.0f}, <10={( rc<10).sum()}, <20={(rc<20).sum()}")

        # Allele STD
        astd = allele_stds[mask]
        p(f"  AlleleSTD: mean={astd.mean():.2f}, median={np.median(astd):.2f}")

        # Error magnitude
        mr_e = mr_errs[mask]
        lr_e = lr_errs[mask]
        p(f"  MR error: mean={mr_e.mean():.2f}, median={np.median(mr_e):.2f}")
        p(f"  LR error: mean={lr_e.mean():.2f}, median={np.median(lr_e):.2f}")

        # Truth diff magnitude (variant size)
        td = np.max(np.abs(true_diffs[mask]), axis=1)
        p(f"  TruthDiff(max): mean={td.mean():.1f}, median={np.median(td):.1f}, <5bp={(td<5).sum()}")
        p()

    p("=" * 70)
    p("  STRATIFICATION BY WIN/LOSS CATEGORY")
    p("=" * 70)
    p()

    stratify(lr_wins, "LongTR wins (n=" + str(lr_wins.sum()) + ")")
    stratify(mr_wins, "mode_round wins (n=" + str(mr_wins.sum()) + ")")
    stratify(both_wrong, "Both wrong (n=" + str(both_wrong.sum()) + ")")
    stratify(both_right, "Both right (n=" + str(both_right.sum()) + ")")

    # ── Overall comparison across all methods ──
    p("=" * 70)
    p("  METHOD COMPARISON ON TP (LongTR set)")
    p("=" * 70)
    p()
    p(f"{'Method':<25} {'TP Exact':>10} {'TP ≤1bp':>10} {'TP MAE':>10} {'Overall Exact':>15} {'TN Exact':>10}")
    p("-" * 80)

    for name, diffs in [("mode_round", mr_diffs), ("gap_mode", gm_diffs),
                         ("quality_mode", qm_diffs), ("combined", cb_diffs)]:
        errs_all = np.max(np.abs(diffs - true_diffs), axis=1)
        flat_errs = np.abs(diffs - true_diffs).flatten()
        tp_m = tp_ltr
        tn_m = longtr_mask & ~is_tp

        tp_ex = (errs_all[tp_m] < 0.5).mean()
        tp_w1 = (np.abs(diffs[tp_m] - true_diffs[tp_m]).flatten() <= 1.0).mean()
        tp_mae = np.abs(diffs[tp_m] - true_diffs[tp_m]).flatten().mean()
        ov_ex = (errs_all[longtr_mask] < 0.5).mean()
        tn_ex = (errs_all[tn_m] < 0.5).mean()

        p(f"{name:<25} {tp_ex:>9.1%} {tp_w1:>9.1%} {tp_mae:>9.2f} {ov_ex:>14.1%} {tn_ex:>9.1%}")

    # LongTR row
    errs_all = np.max(np.abs(longtr_diffs - true_diffs), axis=1)
    tp_m = tp_ltr
    tn_m = longtr_mask & ~is_tp
    tp_ex = (errs_all[tp_m] < 0.5).mean()
    tp_w1 = (np.abs(longtr_diffs[tp_m] - true_diffs[tp_m]).flatten() <= 1.0).mean()
    tp_mae = np.abs(longtr_diffs[tp_m] - true_diffs[tp_m]).flatten().mean()
    ov_ex = (errs_all[longtr_mask] < 0.5).mean()
    tn_ex = (errs_all[tn_m] < 0.5).mean()
    p(f"{'LongTR':<25} {tp_ex:>9.1%} {tp_w1:>9.1%} {tp_mae:>9.2f} {ov_ex:>14.1%} {tn_ex:>9.1%}")
    p()

    # ── Het vs Hom TP comparison ──
    p("=" * 70)
    p("  HET vs HOM TP BREAKDOWN")
    p("=" * 70)
    p()

    for zyg_name, zyg_mask_val in [("Het", True), ("Hom", False)]:
        zyg_m = tp_ltr & (is_het_truth == zyg_mask_val)
        n_z = zyg_m.sum()
        if n_z == 0:
            continue
        p(f"  {zyg_name} TP (n={n_z:,}):")
        for name, diffs in [("mode_round", mr_diffs), ("gap_mode", gm_diffs),
                             ("combined", cb_diffs), ("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[zyg_m] - true_diffs[zyg_m]), axis=1)
            ex = (errs < 0.5).mean()
            w1 = (np.abs(diffs[zyg_m] - true_diffs[zyg_m]).flatten() <= 1.0).mean()
            mae = np.abs(diffs[zyg_m] - true_diffs[zyg_m]).flatten().mean()
            p(f"    {name:<20} exact={ex:.1%}  ≤1bp={w1:.1%}  MAE={mae:.2f}")
        p()

    # ── Motif type TP breakdown ──
    p("=" * 70)
    p("  MOTIF TYPE TP BREAKDOWN")
    p("=" * 70)
    p()

    for motif_name, motif_mask_fn in [
        ("Dinuc (2bp)", lambda: motif_lens == 2),
        ("STR (3-6bp)", lambda: (motif_lens >= 3) & (motif_lens <= 6)),
        ("VNTR (7+bp)", lambda: motif_lens >= 7),
    ]:
        mm = tp_ltr & motif_mask_fn()
        n_m = mm.sum()
        if n_m == 0:
            continue
        p(f"  {motif_name} TP (n={n_m:,}):")
        for name, diffs in [("mode_round", mr_diffs), ("gap_mode", gm_diffs),
                             ("combined", cb_diffs), ("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[mm] - true_diffs[mm]), axis=1)
            ex = (errs < 0.5).mean()
            p(f"    {name:<20} exact={ex:.1%}")
        p()

    # ── Ref size TP breakdown ──
    p("=" * 70)
    p("  REF SIZE TP BREAKDOWN")
    p("=" * 70)
    p()

    for sz_name, sz_fn in [
        ("<100bp", lambda: ref_sizes < 100),
        ("100-500bp", lambda: (ref_sizes >= 100) & (ref_sizes < 500)),
        ("500-1000bp", lambda: (ref_sizes >= 500) & (ref_sizes < 1000)),
        (">1000bp", lambda: ref_sizes >= 1000),
    ]:
        sm = tp_ltr & sz_fn()
        n_s = sm.sum()
        if n_s == 0:
            continue
        p(f"  {sz_name} TP (n={n_s:,}):")
        for name, diffs in [("mode_round", mr_diffs), ("gap_mode", gm_diffs),
                             ("combined", cb_diffs), ("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[sm] - true_diffs[sm]), axis=1)
            ex = (errs < 0.5).mean()
            p(f"    {name:<20} exact={ex:.1%}")
        p()

    # ── Coverage TP breakdown ──
    p("=" * 70)
    p("  COVERAGE TP BREAKDOWN")
    p("=" * 70)
    p()

    for cov_name, cov_fn in [
        ("<15x", lambda: read_counts < 15),
        ("15-30x", lambda: (read_counts >= 15) & (read_counts < 30)),
        ("30-50x", lambda: (read_counts >= 30) & (read_counts < 50)),
        (">50x", lambda: read_counts >= 50),
    ]:
        cm = tp_ltr & cov_fn()
        n_c = cm.sum()
        if n_c == 0:
            continue
        p(f"  {cov_name} TP (n={n_c:,}):")
        for name, diffs in [("mode_round", mr_diffs), ("gap_mode", gm_diffs),
                             ("combined", cb_diffs), ("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[cm] - true_diffs[cm]), axis=1)
            ex = (errs < 0.5).mean()
            p(f"    {name:<20} exact={ex:.1%}")
        p()

    # ── Sample some LongTR wins for inspection ──
    p("=" * 70)
    p("  SAMPLE: LongTR wins (first 20)")
    p("=" * 70)
    p()

    lr_win_idx = np.where(lr_wins)[0]
    for j, i in enumerate(lr_win_idx[:20]):
        idx = test_indices[i]
        chrom, start, end = h5_keys[i]
        p(f"  [{j+1}] {chrom}:{start}-{end}")
        p(f"      Truth:   {true_diffs[i]}")
        p(f"      MR pred: {mr_diffs[i]}  (err={mr_errs[i]:.2f})")
        p(f"      LR pred: {longtr_diffs[i]}  (err={lr_errs[i]:.2f})")
        p(f"      GM pred: {gm_diffs[i]}  (err={gm_errs[i]:.2f})")
        p(f"      CB pred: {cb_diffs[i]}  (err={cb_errs[i]:.2f})")
        p(f"      motif={int(motif_lens[i])}, ref={ref_sizes[i]:.0f}, "
          f"reads={read_counts[i]}, std={allele_stds[i]:.2f}, "
          f"het={'Y' if is_het_truth[i] else 'N'}")
        p()

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")

    # Print key results
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
