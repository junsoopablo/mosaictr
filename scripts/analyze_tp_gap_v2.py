#!/usr/bin/env python3
"""TP gap analysis v2: Test new genotyping strategies designed to beat LongTR.

Key insight from v1 analysis:
- mode_round: Hom 77.2%, Het 85.3%
- gap_mode: Hom 88.8% (great!), Het 53.5% (terrible)
- LongTR: Hom 79.0%, Het 90.1%

Strategy: combine gap_mode's hom strength with mode_round's het handling.
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


# ── Existing baselines ────────────────────────────────────────────

def mode_round_genotype(allele_sizes, ref_size):
    """Current best: midpoint split + mode."""
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


# ── New strategies ────────────────────────────────────────────────

def adaptive_mode_genotype(allele_sizes, ref_size):
    """Midpoint split + mode, but collapse to single mode when halves agree within 1bp.

    Fixes: hom loci where forced split gives ±1bp noise between halves.
    """
    sizes = np.sort(allele_sizes)
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

    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])

    if abs(m1 - m2) <= 1:
        # Likely hom: use single mode of ALL reads
        m_all = get_mode(sizes)
        d = round(ref_size - m_all)
        return d, d
    else:
        d1 = round(ref_size - m1)
        d2 = round(ref_size - m2)
        return sorted([d1, d2])


def bimodal_mode_genotype(allele_sizes, ref_size, motif_len):
    """Find top-2 peaks in allele size distribution directly.

    Instead of splitting reads, finds the two most common integer allele sizes.
    This handles close hets better than midpoint split.
    """
    sizes = allele_sizes
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    counts = np.bincount(int_sizes - min_val)

    # Find primary peak
    peak1_idx = np.argmax(counts)
    peak1_val = min_val + peak1_idx
    peak1_count = counts[peak1_idx]

    # Mask out ±1 around peak1 and find secondary peak
    masked_counts = counts.copy()
    for offset in range(-1, 2):
        idx = peak1_idx + offset
        if 0 <= idx < len(masked_counts):
            masked_counts[idx] = 0

    if masked_counts.sum() == 0:
        # No second peak: homozygous
        d = round(ref_size - float(peak1_val))
        return d, d

    peak2_idx = np.argmax(masked_counts)
    peak2_val = min_val + peak2_idx
    peak2_count = masked_counts[peak2_idx]

    # Second peak must have reasonable support
    min_support = max(2, n * 0.1)  # at least 10% of reads or 2
    if peak2_count >= min_support:
        # Het: two peaks
        d1 = round(ref_size - float(peak1_val))
        d2 = round(ref_size - float(peak2_val))
        return sorted([d1, d2])
    else:
        # Hom: only one significant peak
        d = round(ref_size - float(peak1_val))
        return d, d


def bimodal_mode_v2_genotype(allele_sizes, ref_size, motif_len):
    """Bimodal v2: wider exclusion zone (±2bp) for finding second peak.

    Handles cases where read noise spans 3 adjacent integers.
    """
    sizes = allele_sizes
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    counts = np.bincount(int_sizes - min_val)

    peak1_idx = np.argmax(counts)
    peak1_val = min_val + peak1_idx

    # Mask out ±2 around peak1
    masked_counts = counts.copy()
    for offset in range(-2, 3):
        idx = peak1_idx + offset
        if 0 <= idx < len(masked_counts):
            masked_counts[idx] = 0

    if masked_counts.sum() == 0:
        d = round(ref_size - float(peak1_val))
        return d, d

    peak2_idx = np.argmax(masked_counts)
    peak2_val = min_val + peak2_idx
    peak2_count = masked_counts[peak2_idx]

    min_support = max(2, n * 0.1)
    if peak2_count >= min_support:
        d1 = round(ref_size - float(peak1_val))
        d2 = round(ref_size - float(peak2_val))
        return sorted([d1, d2])
    else:
        d = round(ref_size - float(peak1_val))
        return d, d


def cluster_mode_genotype(allele_sizes, ref_size, motif_len):
    """Cluster-based: assign reads to nearest peak, then refine modes.

    1. Find primary peak
    2. Assign reads within ±1bp to cluster 1
    3. From remaining reads, find secondary peak
    4. If secondary peak has enough support, report both
    """
    sizes = allele_sizes
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    counts = np.bincount(int_sizes - min_val)

    # Primary peak
    peak1_idx = np.argmax(counts)
    peak1_val = min_val + peak1_idx

    # Assign reads to cluster 1 (within ±1bp of peak)
    cluster1_mask = np.abs(int_sizes - peak1_val) <= 1
    remaining = int_sizes[~cluster1_mask]

    if len(remaining) < max(2, n * 0.1):
        # Not enough remaining reads for second allele
        d = round(ref_size - float(peak1_val))
        return d, d

    # Find secondary peak from remaining
    rem_counts = np.bincount(remaining - remaining.min())
    peak2_idx = np.argmax(rem_counts)
    peak2_val = remaining.min() + peak2_idx

    d1 = round(ref_size - float(peak1_val))
    d2 = round(ref_size - float(peak2_val))
    return sorted([d1, d2])


def adaptive_bimodal_genotype(allele_sizes, ref_size, motif_len):
    """Best of both: bimodal for het detection, all-read mode for hom.

    1. Compute bincount of integer allele sizes
    2. Find top peak (mode)
    3. Check if distribution is bimodal (second peak ≥ 15% of reads, ≥ 3bp from first)
    4. If bimodal → midpoint split + mode (mode_round)  [het]
    5. If unimodal → single mode of all reads [hom]
    """
    sizes = np.sort(allele_sizes)
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

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    counts = np.bincount(int_sizes - min_val)

    peak1_idx = np.argmax(counts)

    # Mask out ±2 around peak1
    masked = counts.copy()
    for offset in range(-2, 3):
        idx = peak1_idx + offset
        if 0 <= idx < len(masked):
            masked[idx] = 0

    has_second_peak = masked.max() >= max(2, n * 0.12)

    if has_second_peak:
        # Het: use midpoint split + mode (as mode_round)
        mid = n // 2
        a1 = get_mode(sizes[:mid])
        a2 = get_mode(sizes[mid:])
        d1 = round(ref_size - a1)
        d2 = round(ref_size - a2)
        return sorted([d1, d2])
    else:
        # Hom: single mode of all reads
        a = get_mode(sizes)
        d = round(ref_size - a)
        return d, d


def best_of_both_genotype(allele_sizes, ref_size, motif_len):
    """Adaptive threshold: bimodal detection → mode_round for het, all-mode for hom.

    Uses a more sophisticated bimodality test: the sum of reads in the
    secondary peak region must be ≥ 20% of total.
    """
    sizes = np.sort(allele_sizes)
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

    int_sizes = np.round(sizes).astype(int)
    min_val = int_sizes.min()
    counts = np.bincount(int_sizes - min_val)

    peak1_idx = np.argmax(counts)

    # Sum reads in ±1 neighborhood of peak1
    peak1_total = 0
    for offset in range(-1, 2):
        idx = peak1_idx + offset
        if 0 <= idx < len(counts):
            peak1_total += counts[idx]

    # Reads NOT near peak1
    remaining_total = n - peak1_total

    if remaining_total >= max(3, n * 0.2):
        # Likely het: use midpoint split + mode
        mid = n // 2
        a1 = get_mode(sizes[:mid])
        a2 = get_mode(sizes[mid:])
        d1 = round(ref_size - a1)
        d2 = round(ref_size - a2)
        return sorted([d1, d2])
    else:
        # Likely hom: single mode
        a = get_mode(sizes)
        d = round(ref_size - a)
        return d, d


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


def compute_diffs(test_indices, all_read_features, all_read_offsets, all_read_counts,
                  all_locus_features, motif_lens, genotype_fn, needs_motif=False, needs_reads=False):
    """Compute diffs for all test loci using given genotype function."""
    n_test = len(test_indices)
    diffs = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = all_locus_features[idx, REF_SIZE_IDX]

        if needs_motif:
            d1, d2 = genotype_fn(allele_sizes, ref_size, int(motif_lens[i]))
        else:
            d1, d2 = genotype_fn(allele_sizes, ref_size)
        diffs[i] = [d1, d2]
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--longtr-vcf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load H5
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
    read_counts = np.zeros(n_test, dtype=int)

    for j, idx in enumerate(test_indices):
        chrom = all_chroms[idx]
        start = int(all_starts[idx])
        end = int(all_ends[idx])
        h5_keys.append((chrom, start, end))
        h1, h2, ml = all_labels[idx]
        true_diffs.append(sorted([h1, h2]))
        motif_lens.append(ml)
        ref_sizes.append(all_locus_features[idx, REF_SIZE_IDX])
        is_tp.append("TP" in tp_statuses[idx])
        read_counts[j] = all_read_counts[idx]

    true_diffs = np.array(true_diffs)
    motif_lens = np.array(motif_lens)
    is_tp = np.array(is_tp)
    ref_sizes = np.array(ref_sizes)

    # Compute all methods
    methods = {
        "mode_round": (mode_round_genotype, False),
        "adaptive_mode": (adaptive_mode_genotype, False),
        "bimodal_mode": (bimodal_mode_genotype, True),
        "bimodal_v2": (bimodal_mode_v2_genotype, True),
        "cluster_mode": (cluster_mode_genotype, True),
        "adaptive_bimodal": (adaptive_bimodal_genotype, True),
        "best_of_both": (best_of_both_genotype, True),
    }

    all_diffs = {}
    for name, (fn, needs_motif) in methods.items():
        logger.info(f"Computing {name}...")
        all_diffs[name] = compute_diffs(
            test_indices, all_read_features, all_read_offsets, all_read_counts,
            all_locus_features, motif_lens, fn, needs_motif=needs_motif)

    h5.close()

    # Parse and match LongTR
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

    # Compute errors
    lr_errs = np.max(np.abs(longtr_diffs - true_diffs), axis=1)
    lr_exact = lr_errs < THRESH
    method_exact = {}
    method_errs = {}
    for name, diffs in all_diffs.items():
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        method_errs[name] = errs
        method_exact[name] = errs < THRESH

    p("=" * 80)
    p("  TP GAP ANALYSIS v2: New Genotyping Strategies")
    p(f"  Test set: chr21, chr22, chrX (n={n_test:,})")
    p("=" * 80)
    p()

    # ── Overall comparison (LongTR set) ──
    tp_ltr = is_tp & longtr_mask
    tn_ltr = longtr_mask & ~is_tp
    n_tp = tp_ltr.sum()
    n_tn = tn_ltr.sum()

    p("=" * 80)
    p(f"  OVERALL — LongTR set (n={longtr_mask.sum():,})")
    p("=" * 80)
    p()
    p(f"{'Method':<22} {'Overall':>8} {'TP':>8} {'TN':>8} {'Het-TP':>8} {'Hom-TP':>8}  {'TP w1bp':>8}")
    p("-" * 80)

    for name, diffs in list(all_diffs.items()) + [("LongTR", longtr_diffs)]:
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH

        ov = exact[longtr_mask].mean()
        tp = exact[tp_ltr].mean()
        tn = exact[tn_ltr].mean()

        het_tp_m = tp_ltr & is_het_truth
        hom_tp_m = tp_ltr & ~is_het_truth
        het = exact[het_tp_m].mean() if het_tp_m.sum() > 0 else 0
        hom = exact[hom_tp_m].mean() if hom_tp_m.sum() > 0 else 0

        flat_errs = np.abs(diffs[tp_ltr] - true_diffs[tp_ltr]).flatten()
        w1 = (flat_errs <= 1.0).mean()

        marker = " *" if name != "mode_round" and name != "LongTR" and tp > method_exact["mode_round"][tp_ltr].mean() else ""
        p(f"{name:<22} {ov:>7.1%} {tp:>7.1%} {tn:>7.1%} {het:>7.1%} {hom:>7.1%}  {w1:>7.1%}{marker}")

    p()
    p(f"  (* = TP improved over mode_round)")
    p(f"  Het TP: n={( tp_ltr & is_het_truth).sum():,}, Hom TP: n={(tp_ltr & ~is_het_truth).sum():,}")
    p()

    # ── Full test set comparison ──
    p("=" * 80)
    p(f"  FULL TEST SET (n={n_test:,})")
    p("=" * 80)
    p()
    p(f"{'Method':<22} {'Overall':>8} {'TP':>8} {'TN':>8}")
    p("-" * 50)

    for name, diffs in all_diffs.items():
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH
        ov = exact.mean()
        tp = exact[is_tp].mean()
        tn = exact[~is_tp].mean()
        p(f"{name:<22} {ov:>7.1%} {tp:>7.1%} {tn:>7.1%}")
    p()

    # ── Win/loss vs LongTR for best new method ──
    # Find best new method on TP
    best_name = max(
        [n for n in all_diffs if n != "mode_round"],
        key=lambda n: method_exact[n][tp_ltr].mean()
    )
    best_exact = method_exact[best_name]

    p("=" * 80)
    p(f"  WIN/LOSS: {best_name} vs LongTR (TP, n={n_tp:,})")
    p("=" * 80)
    p()

    new_wins = tp_ltr & best_exact & ~lr_exact
    lr_wins = tp_ltr & lr_exact & ~best_exact
    both_r = tp_ltr & best_exact & lr_exact
    both_w = tp_ltr & ~best_exact & ~lr_exact

    p(f"  {best_name} wins: {new_wins.sum():,} ({new_wins.sum()/n_tp:.1%})")
    p(f"  LongTR wins: {lr_wins.sum():,} ({lr_wins.sum()/n_tp:.1%})")
    p(f"  Both right: {both_r.sum():,} ({both_r.sum()/n_tp:.1%})")
    p(f"  Both wrong: {both_w.sum():,} ({both_w.sum()/n_tp:.1%})")
    p()

    # Also compare best_name vs mode_round
    mr_ex = method_exact["mode_round"]
    mr_to_best_gains = tp_ltr & best_exact & ~mr_ex  # cases best fixes that MR missed
    mr_to_best_losses = tp_ltr & mr_ex & ~best_exact  # cases best breaks that MR had

    p(f"  vs mode_round:")
    p(f"    Gains (new fixes what MR missed): {mr_to_best_gains.sum():,}")
    p(f"    Losses (new breaks what MR had):  {mr_to_best_losses.sum():,}")
    p(f"    Net: +{mr_to_best_gains.sum() - mr_to_best_losses.sum()}")
    p()

    # ── Stratified by motif ──
    p("=" * 80)
    p("  MOTIF BREAKDOWN (TP)")
    p("=" * 80)
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
        for name, diffs in list(all_diffs.items()) + [("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[mm] - true_diffs[mm]), axis=1)
            ex = (errs < THRESH).mean()
            p(f"    {name:<22} {ex:.1%}")
        p()

    # ── Stratified by ref size ──
    p("=" * 80)
    p("  REF SIZE BREAKDOWN (TP)")
    p("=" * 80)
    p()

    for sz_name, sz_fn in [
        ("<100bp", lambda: ref_sizes < 100),
        ("100-500bp", lambda: (ref_sizes >= 100) & (ref_sizes < 500)),
        ("500-1000bp", lambda: (ref_sizes >= 500) & (ref_sizes < 1000)),
    ]:
        sm = tp_ltr & sz_fn()
        n_s = sm.sum()
        if n_s == 0:
            continue
        p(f"  {sz_name} (n={n_s:,}):")
        for name, diffs in list(all_diffs.items()) + [("LongTR", longtr_diffs)]:
            errs = np.max(np.abs(diffs[sm] - true_diffs[sm]), axis=1)
            ex = (errs < THRESH).mean()
            p(f"    {name:<22} {ex:.1%}")
        p()

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
