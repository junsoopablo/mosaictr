"""Diagnose TP variant performance in DeepTR.

Answers the key question: Is TP failure a model/training problem or a feature
quality (CIGAR) limit?

Analyses:
1. Oracle ceiling — read-median accuracy for TP loci (feature quality upper bound)
2. TP error by expansion size bucket
3. TP error by motif type
4. Het vs Hom variant analysis
5. Feature importance — per-feature correlation with truth allele_diff

Convention note:
  - Read feature `allele_size_bp` = absolute repeat allele size in bp
  - Labels `hap_diff` = ref_size - allele_size (positive = contraction)
  - Oracle diff = ref_size - median(allele_size_bp)
"""

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
H5_PATH = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/train_v1/deeptr_features.h5"
OUT_PATH = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/tp_diagnosis_report.txt"

# Read feature indices
ALLELE_SIZE_IDX = 0
N_INS_IDX = 1
N_DEL_IDX = 2
TOTAL_INS_IDX = 3
TOTAL_DEL_IDX = 4
MAX_INS_IDX = 5
MAX_DEL_IDX = 6
MAPQ_IDX = 7
SOFTCLIP_L_IDX = 8
SOFTCLIP_R_IDX = 9
FLANK_L_IDX = 10
FLANK_R_IDX = 11
READ_LEN_IDX = 12
STRAND_IDX = 13
IS_SUPP_IDX = 14

READ_FEATURE_NAMES = [
    "allele_size_bp", "n_insertions", "n_deletions", "total_ins_bp",
    "total_del_bp", "max_single_ins", "max_single_del", "mapq",
    "softclip_left", "softclip_right", "left_flank_match", "right_flank_match",
    "read_length", "strand", "is_supplementary",
]

# Locus feature indices
REF_SIZE_IDX = 0
MOTIF_LEN_IDX = 1


def load_data():
    """Load HDF5 data into memory."""
    logger.info("Loading HDF5: %s", H5_PATH)
    with h5py.File(H5_PATH, "r") as h5:
        labels = h5["labels"][:]
        read_features = h5["read_features"][:]
        read_counts = h5["read_counts"][:]
        read_offsets = h5["read_offsets"][:]
        locus_features = h5["locus_features"][:]
        chroms = np.array([c.decode() if isinstance(c, bytes) else c for c in h5["chroms"][:]])
        tp_statuses = np.array([t.decode() if isinstance(t, bytes) else t for t in h5["tp_statuses"][:]])
    logger.info("Loaded %d loci, %d reads", len(labels), len(read_features))
    return labels, read_features, read_counts, read_offsets, locus_features, chroms, tp_statuses


def get_reads(read_features, read_offsets, read_counts, idx):
    """Get per-read features for locus idx."""
    offset = read_offsets[idx]
    count = read_counts[idx]
    return read_features[offset:offset + count]


def _oracle_two_allele(allele_sizes, ref_size):
    """Split-median oracle: convert absolute allele sizes to diffs.

    Returns (oracle_diff_1, oracle_diff_2) where diff = ref_size - allele_size.
    Sorted so diff_1 <= diff_2.
    """
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n >= 2:
        mid = n // 2
        a1_abs = np.median(alleles[:mid])
        a2_abs = np.median(alleles[mid:])
    elif n == 1:
        a1_abs = a2_abs = alleles[0]
    else:
        a1_abs = a2_abs = ref_size  # no reads → predict ref (diff=0)
    # Convert absolute allele sizes to diffs
    d1 = ref_size - a1_abs
    d2 = ref_size - a2_abs
    return min(d1, d2), max(d1, d2)


def oracle_analysis(labels, read_features, read_offsets, read_counts,
                    locus_features, tp_mask):
    """Compute oracle ceiling: best achievable from read medians alone."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  1. ORACLE CEILING — Read-median accuracy for TP loci")
    lines.append("=" * 70)

    tp_indices = np.where(tp_mask)[0]
    n_tp = len(tp_indices)

    # Arrays for results
    oracle_diffs = np.zeros((n_tp, 2), dtype=np.float32)
    truth_diffs = np.zeros((n_tp, 2), dtype=np.float32)
    read_stds = np.zeros(n_tp, dtype=np.float32)
    read_counts_tp = np.zeros(n_tp, dtype=np.int32)
    # Also: single-value oracle (overall median → diff)
    oracle_single_diff = np.zeros(n_tp, dtype=np.float32)

    for i, idx in enumerate(tp_indices):
        reads = get_reads(read_features, read_offsets, read_counts, idx)
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = locus_features[idx, REF_SIZE_IDX]

        # Two-allele oracle
        d1, d2 = _oracle_two_allele(allele_sizes, ref_size)
        oracle_diffs[i] = [d1, d2]

        # Single-value oracle
        oracle_single_diff[i] = ref_size - np.median(allele_sizes)

        # Read noise (in allele-size space)
        read_stds[i] = np.std(allele_sizes) if len(allele_sizes) > 1 else 0
        read_counts_tp[i] = len(allele_sizes)

        # Truth (sorted)
        h1, h2 = labels[idx, 0], labels[idx, 1]
        truth_diffs[i] = sorted([h1, h2])

    # Single-value oracle: compare to closest haplotype
    oracle_err_h1 = np.abs(oracle_single_diff - truth_diffs[:, 0])
    oracle_err_h2 = np.abs(oracle_single_diff - truth_diffs[:, 1])
    oracle_err_best = np.minimum(oracle_err_h1, oracle_err_h2)

    oracle_exact_single = (oracle_err_best < 0.5).mean()
    oracle_w1bp_single = (oracle_err_best <= 1.0).mean()
    oracle_w5bp_single = (oracle_err_best <= 5.0).mean()

    # Two-allele oracle
    oracle_2err = np.abs(oracle_diffs - truth_diffs)
    oracle_exact_both = np.all(oracle_2err < 0.5, axis=1).mean()
    oracle_w1bp_both = (oracle_2err.flatten() <= 1.0).mean()
    oracle_w5bp_both = (oracle_2err.flatten() <= 5.0).mean()
    oracle_mae_both = oracle_2err.flatten().mean()

    lines.append(f"\n  TP loci analyzed: {n_tp:,}")
    lines.append(f"  Mean reads per TP locus: {read_counts_tp.mean():.1f}")
    lines.append(f"  Mean read allele_size std: {read_stds.mean():.2f} bp")
    lines.append(f"  Median read allele_size std: {np.median(read_stds):.2f} bp")

    lines.append(f"\n  --- Single-value oracle (read median diff → closest haplotype) ---")
    lines.append(f"  Exact match (<0.5bp):  {oracle_exact_single:.4f}  ({oracle_exact_single*100:.1f}%)")
    lines.append(f"  Within 1bp:            {oracle_w1bp_single:.4f}  ({oracle_w1bp_single*100:.1f}%)")
    lines.append(f"  Within 5bp:            {oracle_w5bp_single:.4f}  ({oracle_w5bp_single*100:.1f}%)")

    lines.append(f"\n  --- Two-allele oracle (split-median clustering → both alleles) ---")
    lines.append(f"  Exact match (both <0.5bp):  {oracle_exact_both:.4f}  ({oracle_exact_both*100:.1f}%)")
    lines.append(f"  Within 1bp (per-allele):    {oracle_w1bp_both:.4f}  ({oracle_w1bp_both*100:.1f}%)")
    lines.append(f"  Within 5bp (per-allele):    {oracle_w5bp_both:.4f}  ({oracle_w5bp_both*100:.1f}%)")
    lines.append(f"  MAE (per-allele):           {oracle_mae_both:.2f} bp")

    lines.append(f"\n  → Oracle ceiling = {oracle_exact_both*100:.1f}% exact match")
    if oracle_exact_both > 0.30:
        verdict = "MODEL problem — reads have signal, model fails to use it"
    elif oracle_exact_both < 0.25:
        verdict = "FEATURE problem — CIGAR features lack sufficient signal"
    else:
        verdict = "MIXED — both feature quality and model contribute"
    lines.append(f"    (vs model 20.2% → {verdict})")

    return "\n".join(lines), oracle_diffs, truth_diffs, read_stds


def _compute_oracle_for_indices(indices, labels, read_features, read_offsets,
                                read_counts, locus_features):
    """Compute two-allele oracle diffs and truth for a set of locus indices."""
    n = len(indices)
    oracle = np.zeros((n, 2), dtype=np.float32)
    truth = np.zeros((n, 2), dtype=np.float32)
    rstds = np.zeros(n, dtype=np.float32)

    for i, idx in enumerate(indices):
        reads = get_reads(read_features, read_offsets, read_counts, idx)
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = locus_features[idx, REF_SIZE_IDX]
        d1, d2 = _oracle_two_allele(allele_sizes, ref_size)
        oracle[i] = [d1, d2]
        truth[i] = sorted([labels[idx, 0], labels[idx, 1]])
        rstds[i] = np.std(allele_sizes) if len(allele_sizes) > 1 else 0

    return oracle, truth, rstds


def expansion_bucket_analysis(labels, read_features, read_offsets, read_counts,
                              locus_features, tp_mask):
    """TP error breakdown by expansion size."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  2. TP ERROR BY EXPANSION SIZE BUCKET")
    lines.append("=" * 70)

    tp_indices = np.where(tp_mask)[0]
    buckets = [
        ("<5bp", 0, 5),
        ("5-20bp", 5, 20),
        ("20-100bp", 20, 100),
        ("100-500bp", 100, 500),
        (">500bp", 500, 1e9),
    ]

    lines.append(f"\n  {'Bucket':<12} {'N':>7} {'%TP':>6} {'Oracle_exact':>13} {'Oracle_w1bp':>12} {'Oracle_MAE':>11} {'Read_std':>10}")
    lines.append(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*13} {'-'*12} {'-'*11} {'-'*10}")

    for name, lo, hi in buckets:
        h1 = np.abs(labels[tp_indices, 0])
        h2 = np.abs(labels[tp_indices, 1])
        max_diff = np.maximum(h1, h2)
        mask = (max_diff >= lo) & (max_diff < hi)
        n = mask.sum()
        if n == 0:
            lines.append(f"  {name:<12} {0:>7} {0:>5.1f}%")
            continue

        bucket_indices = tp_indices[mask]
        oracle, truth, rstds = _compute_oracle_for_indices(
            bucket_indices, labels, read_features, read_offsets,
            read_counts, locus_features,
        )
        errs = np.abs(oracle - truth)
        exact = np.all(errs < 0.5, axis=1).mean()
        w1bp = (errs.flatten() <= 1.0).mean()
        mae = errs.flatten().mean()
        pct = n / len(tp_indices) * 100

        lines.append(f"  {name:<12} {n:>7,} {pct:>5.1f}% {exact:>12.1%} {w1bp:>11.1%} {mae:>10.2f} {rstds.mean():>9.2f}")

    return "\n".join(lines)


def motif_type_analysis(labels, read_features, read_offsets, read_counts,
                        locus_features, tp_mask):
    """TP error breakdown by motif type."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  3. TP ERROR BY MOTIF TYPE")
    lines.append("=" * 70)

    tp_indices = np.where(tp_mask)[0]
    motif_lens = labels[tp_indices, 2]

    categories = [
        ("Homopolymer (1bp)", motif_lens == 1),
        ("Dinucleotide (2bp)", motif_lens == 2),
        ("Tri-hex (3-6bp)", (motif_lens >= 3) & (motif_lens <= 6)),
        ("VNTR (>6bp)", motif_lens > 6),
    ]

    lines.append(f"\n  {'Motif type':<22} {'N':>7} {'%TP':>6} {'Oracle_exact':>13} {'Oracle_w1bp':>12} {'Oracle_MAE':>11} {'Read_std':>10}")
    lines.append(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*13} {'-'*12} {'-'*11} {'-'*10}")

    for name, cat_mask in categories:
        n = cat_mask.sum()
        if n == 0:
            lines.append(f"  {name:<22} {0:>7}")
            continue

        bucket_indices = tp_indices[cat_mask]
        oracle, truth, rstds = _compute_oracle_for_indices(
            bucket_indices, labels, read_features, read_offsets,
            read_counts, locus_features,
        )
        errs = np.abs(oracle - truth)
        exact = np.all(errs < 0.5, axis=1).mean()
        w1bp = (errs.flatten() <= 1.0).mean()
        mae = errs.flatten().mean()
        pct = n / len(tp_indices) * 100

        lines.append(f"  {name:<22} {n:>7,} {pct:>5.1f}% {exact:>12.1%} {w1bp:>11.1%} {mae:>10.2f} {rstds.mean():>9.2f}")

    return "\n".join(lines)


def het_hom_analysis(labels, read_features, read_offsets, read_counts,
                     locus_features, tp_mask):
    """Het vs Hom variant analysis — bimodality detection."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  4. HET vs HOM VARIANT ANALYSIS")
    lines.append("=" * 70)

    tp_indices = np.where(tp_mask)[0]
    h1 = labels[tp_indices, 0]
    h2 = labels[tp_indices, 1]
    motif_len = labels[tp_indices, 2]

    is_het = np.abs(h1 - h2) > motif_len
    is_hom = ~is_het

    for name, sub_mask in [("Hom variant", is_hom), ("Het variant", is_het)]:
        n = sub_mask.sum()
        if n == 0:
            continue
        sub_indices = tp_indices[sub_mask]

        oracle, truth, rstds = _compute_oracle_for_indices(
            sub_indices, labels, read_features, read_offsets,
            read_counts, locus_features,
        )
        errs = np.abs(oracle - truth)
        exact = np.all(errs < 0.5, axis=1).mean()
        w1bp = (errs.flatten() <= 1.0).mean()
        w5bp = (errs.flatten() <= 5.0).mean()
        mae = errs.flatten().mean()

        lines.append(f"\n  --- {name} (n={n:,}) ---")
        lines.append(f"  Oracle exact match:   {exact:.4f}  ({exact*100:.1f}%)")
        lines.append(f"  Oracle within 1bp:    {w1bp:.4f}  ({w1bp*100:.1f}%)")
        lines.append(f"  Oracle within 5bp:    {w5bp:.4f}  ({w5bp*100:.1f}%)")
        lines.append(f"  Oracle MAE:           {mae:.2f} bp")
        lines.append(f"  Mean read std:        {rstds.mean():.2f} bp")

        if name == "Het variant":
            # Bimodality: check if reads separate into two clusters
            bimodal_count = 0
            for i, idx in enumerate(sub_indices):
                reads = get_reads(read_features, read_offsets, read_counts, idx)
                alleles = np.sort(reads[:, ALLELE_SIZE_IDX])
                mid = len(alleles) // 2
                if mid > 0 and mid < len(alleles):
                    gap = np.median(alleles[mid:]) - np.median(alleles[:mid])
                    if gap > labels[idx, 2]:
                        bimodal_count += 1
            bimodal_pct = bimodal_count / n * 100
            lines.append(f"  Bimodal reads (gap > motif_len): {bimodal_count:,} / {n:,} ({bimodal_pct:.1f}%)")
            lines.append(f"  → {'Read distribution reflects het alleles' if bimodal_pct > 50 else 'Read distribution does NOT clearly separate het alleles'}")

    return "\n".join(lines)


def feature_importance_analysis(labels, read_features, read_offsets, read_counts,
                                locus_features, tp_mask):
    """Feature correlation with truth allele_diff for TP loci."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  5. FEATURE IMPORTANCE — Correlation with truth allele_diff")
    lines.append("=" * 70)

    tp_indices = np.where(tp_mask)[0]
    n_tp = len(tp_indices)

    # Mean truth diff (in same convention as labels: ref - allele)
    truth_mean_diff = (labels[tp_indices, 0] + labels[tp_indices, 1]) / 2.0

    n_features = 15
    feature_medians = np.zeros((n_tp, n_features), dtype=np.float32)
    feature_stds = np.zeros((n_tp, n_features), dtype=np.float32)

    for i, idx in enumerate(tp_indices):
        reads = get_reads(read_features, read_offsets, read_counts, idx)
        feature_medians[i] = np.median(reads, axis=0)
        feature_stds[i] = np.std(reads, axis=0) if reads.shape[0] > 1 else 0

    # Convert allele_size median to diff for correlation
    # diff = ref_size - allele_size
    ref_sizes = locus_features[tp_indices, REF_SIZE_IDX]
    feature_medians_as_diff = feature_medians.copy()
    feature_medians_as_diff[:, ALLELE_SIZE_IDX] = ref_sizes - feature_medians[:, ALLELE_SIZE_IDX]

    lines.append(f"\n  Correlation of per-locus read feature MEDIAN with truth mean_allele_diff:")
    lines.append(f"  (allele_size_bp converted to diff = ref_size - allele_size)")
    lines.append(f"  {'Feature':<22} {'Pearson r':>10} {'p-value':>12} {'Rank':>6}")
    lines.append(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*6}")

    correlations = []
    for j in range(n_features):
        data = feature_medians_as_diff[:, j] if j == ALLELE_SIZE_IDX else feature_medians[:, j]
        if np.std(data) < 1e-8:
            continue
        r, p = scipy_stats.pearsonr(data, truth_mean_diff)
        correlations.append((abs(r), r, p, READ_FEATURE_NAMES[j]))

    correlations.sort(key=lambda x: -x[0])
    for rank, (abs_r, r, p, fname) in enumerate(correlations, 1):
        lines.append(f"  {fname:<22} {r:>10.4f} {p:>12.2e} {rank:>6}")

    # Correlation of feature STD with oracle error
    lines.append(f"\n  Correlation of per-locus read STD with |oracle_error|:")
    lines.append(f"  (higher = more noise → worse oracle)")
    lines.append(f"  {'Feature':<22} {'Pearson r':>10} {'p-value':>12}")
    lines.append(f"  {'-'*22} {'-'*10} {'-'*12}")

    oracle_errors = np.zeros(n_tp, dtype=np.float32)
    for i, idx in enumerate(tp_indices):
        reads = get_reads(read_features, read_offsets, read_counts, idx)
        allele_sizes = reads[:, ALLELE_SIZE_IDX]
        ref_size = locus_features[idx, REF_SIZE_IDX]
        d1, d2 = _oracle_two_allele(allele_sizes, ref_size)
        t1, t2 = sorted([labels[idx, 0], labels[idx, 1]])
        oracle_errors[i] = (abs(d1 - t1) + abs(d2 - t2)) / 2.0

    std_corrs = []
    for j in range(n_features):
        if np.std(feature_stds[:, j]) < 1e-8:
            continue
        r, p = scipy_stats.pearsonr(feature_stds[:, j], oracle_errors)
        std_corrs.append((abs(r), r, p, READ_FEATURE_NAMES[j]))

    std_corrs.sort(key=lambda x: -x[0])
    for abs_r, r, p, fname in std_corrs[:10]:
        lines.append(f"  {fname:<22} {r:>10.4f} {p:>12.2e}")

    return "\n".join(lines)


def tn_baseline(labels, read_features, read_offsets, read_counts,
                locus_features, tn_mask):
    """Quick TN oracle for comparison."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  BASELINE — TN oracle accuracy (for comparison)")
    lines.append("=" * 70)

    tn_indices = np.where(tn_mask)[0]
    rng = np.random.RandomState(42)
    if len(tn_indices) > 50000:
        sample = rng.choice(tn_indices, 50000, replace=False)
    else:
        sample = tn_indices

    oracle, truth, _ = _compute_oracle_for_indices(
        sample, labels, read_features, read_offsets, read_counts, locus_features,
    )
    errs = np.abs(oracle - truth)
    exact = np.all(errs < 0.5, axis=1).mean()
    w1bp = (errs.flatten() <= 1.0).mean()
    mae = errs.flatten().mean()

    lines.append(f"\n  TN sampled: {len(sample):,}")
    lines.append(f"  Oracle exact match:   {exact:.4f}  ({exact*100:.1f}%)")
    lines.append(f"  Oracle within 1bp:    {w1bp:.4f}  ({w1bp*100:.1f}%)")
    lines.append(f"  Oracle MAE:           {mae:.2f} bp")

    return "\n".join(lines)


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    labels, read_features, read_counts, read_offsets, locus_features, chroms, tp_statuses = load_data()

    tp_mask = np.array(["TP" in t for t in tp_statuses])
    tn_mask = np.array(["TN" in t for t in tp_statuses]) & ~tp_mask

    n_total = len(labels)
    n_tp = tp_mask.sum()
    n_tn = tn_mask.sum()

    report = []
    report.append("=" * 70)
    report.append("  DeepTR — TP Variant Performance Diagnosis")
    report.append("=" * 70)
    report.append(f"\n  Total loci: {n_total:,}")
    report.append(f"  TP (variant): {n_tp:,} ({n_tp/n_total*100:.1f}%)")
    report.append(f"  TN (reference): {n_tn:,} ({n_tn/n_total*100:.1f}%)")
    report.append(f"  TN:TP ratio: {n_tn/max(n_tp,1):.1f}:1")

    tp_h1 = labels[tp_mask, 0]
    tp_h2 = labels[tp_mask, 1]
    tp_mean = (np.abs(tp_h1) + np.abs(tp_h2)) / 2.0
    report.append(f"\n  TP mean |allele_diff|: {tp_mean.mean():.2f} bp (median: {np.median(tp_mean):.2f})")
    report.append(f"  TP allele_diff range: [{labels[tp_mask, :2].min():.0f}, {labels[tp_mask, :2].max():.0f}] bp")

    logger.info("Running oracle analysis...")
    oracle_text, _, _, _ = oracle_analysis(
        labels, read_features, read_offsets, read_counts, locus_features, tp_mask,
    )
    report.append(oracle_text)

    logger.info("Running expansion bucket analysis...")
    report.append(expansion_bucket_analysis(
        labels, read_features, read_offsets, read_counts, locus_features, tp_mask,
    ))

    logger.info("Running motif type analysis...")
    report.append(motif_type_analysis(
        labels, read_features, read_offsets, read_counts, locus_features, tp_mask,
    ))

    logger.info("Running het/hom analysis...")
    report.append(het_hom_analysis(
        labels, read_features, read_offsets, read_counts, locus_features, tp_mask,
    ))

    logger.info("Running feature importance analysis...")
    report.append(feature_importance_analysis(
        labels, read_features, read_offsets, read_counts, locus_features, tp_mask,
    ))

    logger.info("Running TN baseline...")
    report.append(tn_baseline(
        labels, read_features, read_offsets, read_counts, locus_features, tn_mask,
    ))

    # Summary
    report.append("\n" + "=" * 70)
    report.append("  SUMMARY & NEXT STEPS")
    report.append("=" * 70)
    report.append("""
  Interpretation guide:
  - If oracle ceiling >> 20.2%: MODEL/TRAINING problem → improve architecture,
    loss function, class weighting, or training strategy
  - If oracle ceiling ≈ 20.2%: FEATURE QUALITY problem → CIGAR-based allele
    sizing is fundamentally limited, need better feature extraction
  - If oracle varies by bucket: targeted improvements possible for specific
    expansion sizes or motif types
""")

    full_report = "\n".join(report)
    print(full_report)

    with open(OUT_PATH, "w") as f:
        f.write(full_report)
    logger.info("Report saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
