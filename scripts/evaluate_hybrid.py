"""Hybrid evaluation: DL classifier (v8C) + classical allele size estimation.

Pipeline:
1. v8C classifier determines variant/reference for each locus
2. For predicted variants (TP): GMM clustering on per-read allele_size_bp → allele diffs
3. For predicted references (TN): predict allele_diff = 0

Also evaluates:
- Oracle baselines (split-median, GMM) without classifier gating
- Classifier-only performance (perfect allele sizing for detected variants)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/qbio/junsoopablo/02_Projects/10_internship/deeptr")
sys.path.insert(0, "/qbio/junsoopablo/02_Projects/10_internship/ensembletr-lr")

from deeptr.model import DeepTR, DeepTRv2
from deeptr.train import DeepTRDataset, collate_fn
from deeptr.utils import FeatureNormalizer, chrom_split
from faststraglr.cluster import cluster_alleles

# Feature indices (raw HDF5, pre-normalization)
ALLELE_SIZE_IDX = 0
REF_SIZE_IDX = 0      # locus_features index
MOTIF_LEN_IDX = 1     # locus_features index


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid evaluation: DL classifier + classical genotyper")
    parser.add_argument("--classifier-model", default=None,
                        help="Classifier model checkpoint (e.g., v8C stage1). If omitted, use ground truth TP/TN.")
    parser.add_argument("--classifier-normalizer", default=None,
                        help="Normalizer for classifier model")
    parser.add_argument("--h5", required=True, help="Features HDF5 path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--variant-threshold", type=float, default=0.5,
                        help="Classifier threshold (default: 0.5)")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    parser.add_argument("--method", default="gmm",
                        choices=["gmm", "split_median", "split_median_round",
                                 "smart", "motif_round", "mode_round",
                                 "motif_mode", "all_new", "all_tp_improve", "both"],
                        help="Allele estimation method (default: gmm)")
    parser.add_argument("--no-gate", action="store_true",
                        help="Disable TP/TN gating — run allele estimation on ALL loci")
    return parser.parse_args()


def load_classifier(model_path, device):
    """Load classifier model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_version = config.get("model_version", "v1")
    ModelClass = DeepTRv2 if model_version == "v2" else DeepTR
    model = ModelClass(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_reads=config["max_reads"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def split_median_genotype(allele_sizes, ref_size):
    """Oracle split-median: sort reads, split at midpoint, median of each half."""
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n >= 2:
        mid = n // 2
        a1 = float(np.median(alleles[:mid]))
        a2 = float(np.median(alleles[mid:]))
    elif n == 1:
        a1 = a2 = float(alleles[0])
    else:
        return 0.0, 0.0  # no reads

    d1 = ref_size - a1
    d2 = ref_size - a2
    return min(d1, d2), max(d1, d2)


def split_median_round_genotype(allele_sizes, ref_size):
    """Split-median with integer rounding (CIGAR gives integer allele sizes)."""
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n >= 2:
        mid = n // 2
        a1 = float(np.median(alleles[:mid]))
        a2 = float(np.median(alleles[mid:]))
    elif n == 1:
        a1 = a2 = float(alleles[0])
    else:
        return 0.0, 0.0

    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return min(d1, d2), max(d1, d2)


def smart_genotype(allele_sizes, ref_size, motif_len):
    """Gap-based splitting + integer rounding.

    - Find largest gap in sorted allele sizes
    - If gap >= threshold: split there (heterozygous)
    - Otherwise: single cluster (homozygous), median of all reads
    - Round diffs to nearest integer
    """
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - float(alleles[0]))
        return d, d

    # Find largest gap
    gaps = np.diff(alleles)
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])

    # Threshold: at least 1bp gap, scaled by motif
    threshold = max(1.0, motif_len * 0.5)

    # Need at least 2 reads in each group
    split_idx = max_gap_idx + 1
    if max_gap >= threshold and split_idx >= 2 and (n - split_idx) >= 2:
        # Heterozygous: split at gap
        a1 = float(np.median(alleles[:split_idx]))
        a2 = float(np.median(alleles[split_idx:]))
    else:
        # Homozygous: single cluster
        med = float(np.median(alleles))
        a1 = a2 = med

    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return min(d1, d2), max(d1, d2)


def motif_round_genotype(allele_sizes, ref_size, motif_len):
    """Split-median with motif-aware rounding.

    Round diffs to nearest motif_len multiple for STRs (motif ≤ 6bp).
    For VNTRs (motif > 6bp), use integer rounding (motif units too large).
    """
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        raw_d = ref_size - float(alleles[0])
        if motif_len <= 6:
            d = round(raw_d / motif_len) * motif_len
        else:
            d = round(raw_d)
        return d, d

    mid = n // 2
    a1 = float(np.median(alleles[:mid]))
    a2 = float(np.median(alleles[mid:]))

    raw_d1 = ref_size - a1
    raw_d2 = ref_size - a2

    if motif_len <= 6:
        d1 = round(raw_d1 / motif_len) * motif_len
        d2 = round(raw_d2 / motif_len) * motif_len
    else:
        d1 = round(raw_d1)
        d2 = round(raw_d2)

    return min(d1, d2), max(d1, d2)


def mode_round_genotype(allele_sizes, ref_size, motif_len):
    """Split-median using MODE (most frequent value) instead of median.

    For integer CIGAR allele sizes, mode is more precise than median
    when the cluster is unimodal and symmetric.
    """
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        d = round(ref_size - float(alleles[0]))
        return d, d

    mid = n // 2
    half1 = alleles[:mid]
    half2 = alleles[mid:]

    def get_mode(arr):
        """Get mode of integer array, fallback to median."""
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    a1 = get_mode(half1)
    a2 = get_mode(half2)

    d1 = round(ref_size - a1)
    d2 = round(ref_size - a2)
    return min(d1, d2), max(d1, d2)


def motif_mode_genotype(allele_sizes, ref_size, motif_len):
    """MODE-based estimation + motif-aware rounding for STRs."""
    alleles = np.sort(allele_sizes)
    n = len(alleles)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        raw_d = ref_size - float(alleles[0])
        if motif_len <= 6:
            d = round(raw_d / motif_len) * motif_len
        else:
            d = round(raw_d)
        return d, d

    mid = n // 2
    half1 = alleles[:mid]
    half2 = alleles[mid:]

    def get_mode(arr):
        int_arr = np.round(arr).astype(int)
        counts = np.bincount(int_arr - int_arr.min())
        return float(int_arr.min() + np.argmax(counts))

    a1 = get_mode(half1)
    a2 = get_mode(half2)

    raw_d1 = ref_size - a1
    raw_d2 = ref_size - a2

    if motif_len <= 6:
        d1 = round(raw_d1 / motif_len) * motif_len
        d2 = round(raw_d2 / motif_len) * motif_len
    else:
        d1 = round(raw_d1)
        d2 = round(raw_d2)

    return min(d1, d2), max(d1, d2)


def gmm_genotype(allele_sizes, ref_size, motif_len):
    """GMM-based genotyping using FastStraglr's cluster_alleles."""
    if len(allele_sizes) < 2:
        if len(allele_sizes) == 1:
            d = ref_size - allele_sizes[0]
            return d, d
        return 0.0, 0.0

    call = cluster_alleles(
        sizes=allele_sizes.tolist(),
        motif_len=max(1, int(motif_len)),
        min_support=2,
        max_clusters=2,
        min_separation_bp=5,
    )

    if call is None:
        d = ref_size - float(np.median(allele_sizes))
        return d, d

    d1 = ref_size - call.allele1_size
    if call.is_homozygous or call.allele2_size is None:
        return d1, d1
    else:
        d2 = ref_size - call.allele2_size
        return min(d1, d2), max(d1, d2)


def compute_metrics(errs, flat_errs, motif_errs, pz, tz, td, pd, label=""):
    """Compute and format evaluation metrics."""
    n = errs.shape[0]
    if n == 0:
        return f"\n{'='*60}\n  {label}  (n=0)\n{'='*60}\n  No data"

    exact = np.all(errs < 0.5, axis=1).mean()
    w1bp = (flat_errs <= 1.0).mean()
    w1mu = (motif_errs.flatten() <= 1.0).mean()
    w5bp = (flat_errs <= 5.0).mean()
    mae = flat_errs.mean()
    median_ae = np.median(flat_errs)

    td_flat = td.flatten()
    pd_flat = pd.flatten()
    ss_res = np.sum((td_flat - pd_flat) ** 2)
    ss_tot = np.sum((td_flat - td_flat.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    zyg_acc = (pz == tz).mean() if len(pz) > 0 else 0.0
    geno_conc = np.all(motif_errs <= 1.0, axis=1).mean()

    lines = [f"\n{'='*60}", f"  {label}  (n={n:,})", f"{'='*60}"]
    lines.append(f"  Exact match (both <0.5bp):  {exact:.4f}  ({exact*100:.1f}%)")
    lines.append(f"  Within 1bp (per-allele):    {w1bp:.4f}  ({w1bp*100:.1f}%)")
    lines.append(f"  Within 1 motif unit:        {w1mu:.4f}  ({w1mu*100:.1f}%)")
    lines.append(f"  Within 5bp (per-allele):    {w5bp:.4f}  ({w5bp*100:.1f}%)")
    lines.append(f"  MAE (per-allele):           {mae:.2f} bp")
    lines.append(f"  Median AE:                  {median_ae:.2f} bp")
    lines.append(f"  R\u00b2:                         {r2:.4f}")
    lines.append(f"  Zygosity accuracy:          {zyg_acc:.4f}  ({zyg_acc*100:.1f}%)")
    lines.append(f"  Genotype concordance:       {geno_conc:.4f}  ({geno_conc*100:.1f}%)")
    return "\n".join(lines)


def run_evaluation(pred_diffs, true_diffs, motif_lens, true_zyg, ref_sizes,
                   n_reads_arr, tp_list, chrom_list, label_prefix=""):
    """Run full stratified evaluation and return report lines."""
    errors = np.abs(pred_diffs - true_diffs)
    flat_errors = errors.flatten()
    motif_expanded = np.stack([motif_lens, motif_lens], axis=1)
    motif_errors = errors / np.maximum(motif_expanded, 1)

    # Predicted zygosity: het if alleles differ by > motif_len
    pred_motif_expanded = np.stack([motif_lens, motif_lens], axis=1)
    pred_diff_range = np.abs(pred_diffs[:, 1] - pred_diffs[:, 0])
    pred_zyg_binary = (pred_diff_range > motif_lens).astype(int)

    report = []

    report.append(compute_metrics(
        errors, flat_errors, motif_errors,
        pred_zyg_binary, true_zyg, true_diffs, pred_diffs,
        f"{label_prefix}OVERALL",
    ))

    # By chromosome
    for chrom in sorted(set(chrom_list)):
        mask = np.array([c == chrom for c in chrom_list])
        if mask.sum() == 0:
            continue
        report.append(compute_metrics(
            errors[mask], errors[mask].flatten(), motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"{label_prefix}Chromosome: {chrom}",
        ))

    # By motif period
    def motif_bin(ml):
        if ml == 1: return "1_homopolymer"
        elif ml == 2: return "2_dinucleotide"
        elif ml <= 6: return "3-6_STR"
        else: return "7+_VNTR"

    bins = np.array([motif_bin(m) for m in motif_lens])
    for b in sorted(set(bins)):
        mask = bins == b
        report.append(compute_metrics(
            errors[mask], errors[mask].flatten(), motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"{label_prefix}Motif period: {b}",
        ))

    # By repeat length
    def rlen_bin(rs):
        if rs < 100: return "<100bp"
        elif rs < 500: return "100-500bp"
        elif rs < 1000: return "500-1000bp"
        else: return ">1000bp"

    rbins = np.array([rlen_bin(r) for r in ref_sizes])
    for b in ["<100bp", "100-500bp", "500-1000bp", ">1000bp"]:
        mask = rbins == b
        if mask.sum() == 0:
            continue
        report.append(compute_metrics(
            errors[mask], errors[mask].flatten(), motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"{label_prefix}Repeat length: {b}",
        ))

    # By variant type (TP/TN)
    for vtype, vlabel in [("TP", "Variant (TP)"), ("TN", "Reference (TN)")]:
        mask = np.array([vtype in t for t in tp_list])
        if mask.sum() == 0:
            continue
        report.append(compute_metrics(
            errors[mask], errors[mask].flatten(), motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"{label_prefix}{vlabel}",
        ))

    # By coverage
    def cov_bin(n):
        if n < 15: return "<15x"
        elif n <= 30: return "15-30x"
        else: return ">30x"

    cbins = np.array([cov_bin(n) for n in n_reads_arr])
    for b in ["<15x", "15-30x", ">30x"]:
        mask = cbins == b
        if mask.sum() == 0:
            continue
        report.append(compute_metrics(
            errors[mask], errors[mask].flatten(), motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"{label_prefix}Coverage: {b}",
        ))

    return report


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load raw HDF5 data ──
    logger.info("Loading HDF5: %s", args.h5)
    with h5py.File(args.h5, "r") as h5:
        all_read_features = h5["read_features"][:]
        all_read_counts = h5["read_counts"][:]
        all_read_offsets = h5["read_offsets"][:]
        all_locus_features = h5["locus_features"][:]
        all_labels = h5["labels"][:]
        all_chroms = [c.decode() if isinstance(c, bytes) else c for c in h5["chroms"][:]]
        all_starts = h5["starts"][:]
        all_ends = h5["ends"][:]
        all_motifs = [m.decode() if isinstance(m, bytes) else m for m in h5["motifs"][:]]
        all_tp = [t.decode() if isinstance(t, bytes) else t for t in h5["tp_statuses"][:]]

    # Filter to test set
    test_indices = [i for i, c in enumerate(all_chroms) if chrom_split(c) == "test"]
    n_test = len(test_indices)
    logger.info("Test set: %d loci", n_test)

    # ── Run classifier (if provided) ──
    if args.classifier_model:
        logger.info("Loading classifier: %s", args.classifier_model)
        classifier, cls_ckpt = load_classifier(args.classifier_model, device)
        cls_normalizer = FeatureNormalizer.load(args.classifier_normalizer)
        logger.info("Classifier: epoch %d, %d params",
                    cls_ckpt["epoch"], classifier.count_parameters())

        cls_test_ds = DeepTRDataset(args.h5, split="test", normalizer=cls_normalizer, augment=False)
        from torch.utils.data import DataLoader
        cls_loader = DataLoader(cls_test_ds, batch_size=512, shuffle=False,
                                collate_fn=collate_fn, num_workers=4, pin_memory=True)

        all_variant_probs = []
        logger.info("Running classifier inference...")
        with torch.no_grad():
            for batch in cls_loader:
                read_feats = batch["read_features"].to(device)
                locus_feats = batch["locus_features"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                _, _, variant_pred = classifier(read_feats, locus_feats, padding_mask)
                all_variant_probs.append(variant_pred.cpu().numpy())

        variant_probs = np.concatenate(all_variant_probs, axis=0)[:, 0]
        is_variant_pred = variant_probs > args.variant_threshold
        logger.info("Classifier: %d/%d predicted as variant (threshold=%.2f)",
                    is_variant_pred.sum(), n_test, args.variant_threshold)
    else:
        # Use ground truth labels for oracle evaluation
        is_variant_pred = np.array(["TP" in all_tp[i] for i in test_indices])
        logger.info("Using ground truth: %d TP, %d TN", is_variant_pred.sum(),
                    (~is_variant_pred).sum())

    # ── Classical allele estimation ──
    methods_to_run = []
    if args.method == "both":
        methods_to_run = ["split_median", "gmm"]
    elif args.method == "all_new":
        methods_to_run = ["split_median", "split_median_round", "smart"]
    elif args.method == "all_tp_improve":
        methods_to_run = ["split_median_round", "motif_round", "mode_round", "motif_mode"]
    else:
        methods_to_run = [args.method]

    # Gather truth
    true_diffs = []
    motif_lens = []
    true_zyg = []
    ref_sizes = []
    n_reads_list = []
    tp_list = []
    chrom_list = []

    for global_i in test_indices:
        h1 = all_labels[global_i, 0]
        h2 = all_labels[global_i, 1]
        ml = all_labels[global_i, 2]
        d1, d2 = sorted([h1, h2])
        true_diffs.append([d1, d2])
        motif_lens.append(ml)
        true_zyg.append(1 if abs(h1 - h2) > ml else 0)
        ref_sizes.append(all_ends[global_i] - all_starts[global_i])
        n_reads_list.append(all_read_counts[global_i])
        tp_list.append(all_tp[global_i])
        chrom_list.append(all_chroms[global_i])

    true_diffs = np.array(true_diffs, dtype=np.float32)
    motif_lens = np.array(motif_lens, dtype=np.float32)
    true_zyg = np.array(true_zyg)
    ref_sizes = np.array(ref_sizes, dtype=np.float32)
    n_reads_arr = np.array(n_reads_list)

    for method in methods_to_run:
        logger.info("Computing allele predictions with method=%s ...", method)
        t0 = time.time()

        pred_diffs = np.zeros((n_test, 2), dtype=np.float32)

        for local_idx, global_i in enumerate(test_indices):
            if not args.no_gate and not is_variant_pred[local_idx]:
                # TN prediction: diff = 0 (gated mode)
                pred_diffs[local_idx] = [0.0, 0.0]
                continue

            # Get per-read allele sizes (raw, un-normalized)
            offset = all_read_offsets[global_i]
            count = all_read_counts[global_i]
            reads = all_read_features[offset:offset + count]
            allele_sizes = reads[:, ALLELE_SIZE_IDX]  # absolute allele sizes (bp)
            ref_size = all_locus_features[global_i, REF_SIZE_IDX]
            motif_len = all_locus_features[global_i, MOTIF_LEN_IDX]

            if method == "split_median":
                d1, d2 = split_median_genotype(allele_sizes, ref_size)
            elif method == "split_median_round":
                d1, d2 = split_median_round_genotype(allele_sizes, ref_size)
            elif method == "smart":
                d1, d2 = smart_genotype(allele_sizes, ref_size, motif_len)
            elif method == "motif_round":
                d1, d2 = motif_round_genotype(allele_sizes, ref_size, motif_len)
            elif method == "mode_round":
                d1, d2 = mode_round_genotype(allele_sizes, ref_size, motif_len)
            elif method == "motif_mode":
                d1, d2 = motif_mode_genotype(allele_sizes, ref_size, motif_len)
            else:  # gmm
                d1, d2 = gmm_genotype(allele_sizes, ref_size, motif_len)

            pred_diffs[local_idx] = [d1, d2]

        elapsed = time.time() - t0
        logger.info("Method %s: %d loci in %.1fs (%.0f loci/s)",
                    method, n_test, elapsed, n_test / elapsed)

        # ── Generate report ──
        gate_label = "no-gate" if args.no_gate else ("DL-classifier" if args.classifier_model else "oracle-gating")
        header = [
            "=" * 60,
            f"  Hybrid Evaluation: {gate_label} + {method}",
            f"  Test set: chr21, chr22, chrX  (n={n_test:,})",
        ]
        if args.classifier_model:
            header.append(f"  Classifier: {args.classifier_model}")
            header.append(f"  Variant threshold: {args.variant_threshold}")
            header.append(f"  Predicted variants: {is_variant_pred.sum()}/{n_test}")
        else:
            header.append(f"  Gating: ground truth TP/TN labels")
        header.append("=" * 60)

        report_lines = header + run_evaluation(
            pred_diffs, true_diffs, motif_lens, true_zyg, ref_sizes,
            n_reads_arr, tp_list, chrom_list,
        )

        report = "\n".join(report_lines)
        print(report)

        # Save
        suffix = f"nogate_{method}" if args.no_gate else method
        report_path = out_dir / f"hybrid_{suffix}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
