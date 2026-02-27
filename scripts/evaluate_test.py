"""Evaluate DeepTR on test set (chr21-22, chrX) using pre-computed features.

Supports:
- DeepTR v1 and v2 model loading (auto-detected from checkpoint config)
- Soft-gated inference: variant_prob < threshold → allele prediction = 0
- Configurable paths via command-line arguments
"""

import argparse
import logging
import sys
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

# Default paths (backward compatible)
DEFAULT_MODEL = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/train_v1/deeptr_best.pt"
DEFAULT_NORM = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/train_v1/normalizer.npz"
DEFAULT_H5 = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/train_v1/deeptr_features.h5"
DEFAULT_OUT = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/eval_test"

sys.path.insert(0, "/qbio/junsoopablo/02_Projects/10_internship/deeptr")
from deeptr.model import DeepTR, DeepTRv2
from deeptr.train import DeepTRDataset, collate_fn
from deeptr.utils import FeatureNormalizer, chrom_split, log_inverse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepTR on test set")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--normalizer", default=DEFAULT_NORM, help="Normalizer path")
    parser.add_argument("--h5", default=DEFAULT_H5, help="Features HDF5 path")
    parser.add_argument("--output-dir", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--variant-threshold", type=float, default=0.0,
                        help="Soft-gating threshold: if variant_prob < this, predict 0. "
                             "0 = disabled (default). Step B recommends 0.3.")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model (auto-detect v1 vs v2)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
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
    logger.info("Loaded %s model (epoch %d, val_loss=%.4f, %d params)",
                ModelClass.__name__, checkpoint["epoch"], checkpoint["val_loss"],
                model.count_parameters())

    # Load normalizer
    normalizer = FeatureNormalizer.load(args.normalizer)

    # Load test dataset
    test_ds = DeepTRDataset(args.h5, split="test", normalizer=normalizer, augment=False)
    logger.info("Test set: %d loci", len(test_ds))

    # Load metadata for stratification
    with h5py.File(args.h5, "r") as h5:
        all_chroms = [c.decode() if isinstance(c, bytes) else c for c in h5["chroms"][:]]
        all_starts = h5["starts"][:]
        all_ends = h5["ends"][:]
        all_motifs = [m.decode() if isinstance(m, bytes) else m for m in h5["motifs"][:]]
        all_labels = h5["labels"][:]
        all_tp = [t.decode() if isinstance(t, bytes) else t for t in h5["tp_statuses"][:]]
        all_read_counts = h5["read_counts"][:]

    # Get test indices
    test_indices = [i for i, c in enumerate(all_chroms) if chrom_split(c) == "test"]

    # Run inference in batches
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)

    all_pred_diffs = []
    all_pred_zyg = []
    all_pred_variant = []

    logger.info("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            read_feats = batch["read_features"].to(device)
            locus_feats = batch["locus_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            allele_pred, zyg_pred, variant_pred = model(read_feats, locus_feats, padding_mask)

            # Soft-gated inference: zero out allele predictions for low variant_prob
            if args.variant_threshold > 0:
                gate = (variant_pred > args.variant_threshold).float()  # (B, 1)
                allele_pred = allele_pred * gate  # zero out non-variant predictions

            all_pred_diffs.append(allele_pred.cpu().numpy())
            all_pred_zyg.append(zyg_pred.cpu().numpy())
            all_pred_variant.append(variant_pred.cpu().numpy())

    pred_diffs_log = np.concatenate(all_pred_diffs, axis=0)  # (N, 2) log-scale
    pred_zyg = np.concatenate(all_pred_zyg, axis=0)          # (N, 1)
    pred_variant = np.concatenate(all_pred_variant, axis=0)   # (N, 1)

    # Convert log-scale predictions to bp
    pred_diffs = np.sign(pred_diffs_log) * np.expm1(np.abs(pred_diffs_log))

    # Sort predictions so allele1 <= allele2
    pred_diffs = np.sort(pred_diffs, axis=1)

    logger.info("Inference complete: %d loci (variant_threshold=%.2f)",
                pred_diffs.shape[0], args.variant_threshold)

    # Gather truth labels
    true_diffs = []
    motif_lens = []
    true_zyg = []
    ref_sizes = []
    n_reads_list = []
    tp_list = []
    chrom_list = []

    for idx, global_i in enumerate(test_indices):
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
    ref_sizes = np.array(ref_sizes)
    n_reads_arr = np.array(n_reads_list)
    pred_zyg_binary = (pred_zyg[:, 0] > 0.5).astype(int)

    # ── Overall metrics ──
    errors = np.abs(pred_diffs - true_diffs)
    flat_errors = errors.flatten()
    motif_expanded = np.stack([motif_lens, motif_lens], axis=1)
    motif_errors = errors / np.maximum(motif_expanded, 1)

    def compute_metrics(errs, flat_errs, motif_errs, pz, tz, td, pd, label=""):
        n = errs.shape[0]
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

        zyg_acc = (pz == tz).mean()
        geno_conc = np.all(motif_errs <= 1.0, axis=1).mean()

        lines = [f"\n{'='*60}", f"  {label}  (n={n:,})", f"{'='*60}"]
        lines.append(f"  Exact match (both <0.5bp):  {exact:.4f}  ({exact*100:.1f}%)")
        lines.append(f"  Within 1bp (per-allele):    {w1bp:.4f}  ({w1bp*100:.1f}%)")
        lines.append(f"  Within 1 motif unit:        {w1mu:.4f}  ({w1mu*100:.1f}%)")
        lines.append(f"  Within 5bp (per-allele):    {w5bp:.4f}  ({w5bp*100:.1f}%)")
        lines.append(f"  MAE (per-allele):           {mae:.2f} bp")
        lines.append(f"  Median AE:                  {median_ae:.2f} bp")
        lines.append(f"  R²:                         {r2:.4f}")
        lines.append(f"  Zygosity accuracy:          {zyg_acc:.4f}  ({zyg_acc*100:.1f}%)")
        lines.append(f"  Genotype concordance:       {geno_conc:.4f}  ({geno_conc*100:.1f}%)")
        return "\n".join(lines)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"  DeepTR Test Set Evaluation (chr21, chr22, chrX)")
    report_lines.append(f"  Model: {args.model}")
    if args.variant_threshold > 0:
        report_lines.append(f"  Variant threshold: {args.variant_threshold}")
    report_lines.append("=" * 60)

    report_lines.append(compute_metrics(
        errors, flat_errors, motif_errors,
        pred_zyg_binary, true_zyg, true_diffs, pred_diffs,
        "OVERALL",
    ))

    # ── By chromosome ──
    for chrom in sorted(set(chrom_list)):
        mask = np.array([c == chrom for c in chrom_list])
        if mask.sum() == 0:
            continue
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Chromosome: {chrom}",
        ))

    # ── By motif period ──
    def motif_bin(ml):
        if ml == 1: return "1_homopolymer"
        elif ml == 2: return "2_dinucleotide"
        elif ml <= 6: return "3-6_STR"
        else: return "7+_VNTR"

    bins = np.array([motif_bin(m) for m in motif_lens])
    for b in sorted(set(bins)):
        mask = bins == b
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Motif period: {b}",
        ))

    # ── By repeat length ──
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
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Repeat length: {b}",
        ))

    # ── By variant type ──
    for vtype, label in [("TP", "Variant (TP)"), ("TN", "Reference (TN)")]:
        mask = np.array([vtype in t for t in tp_list])
        if mask.sum() == 0:
            continue
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            label,
        ))

    # ── By coverage ──
    def cov_bin(n):
        if n < 15: return "<15x"
        elif n <= 30: return "15-30x"
        else: return ">30x"

    cbins = np.array([cov_bin(n) for n in n_reads_arr])
    for b in ["<15x", "15-30x", ">30x"]:
        mask = cbins == b
        if mask.sum() == 0:
            continue
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Coverage: {b}",
        ))

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = str(out_dir / "test_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
