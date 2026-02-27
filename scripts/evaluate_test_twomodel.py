"""Two-model evaluation: classifier (v8C) + regressor (v8D/8E).

For TP loci detected by the classifier (variant_prob > threshold),
use the regressor model for allele prediction.
For TN loci, predict allele_diff = 0.

This enables using a TP-only trained regressor alongside a separate classifier.
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

sys.path.insert(0, "/qbio/junsoopablo/02_Projects/10_internship/deeptr")
from deeptr.model import DeepTR, DeepTRv2
from deeptr.train import DeepTRDataset, collate_fn
from deeptr.utils import FeatureNormalizer, chrom_split, log_inverse


def parse_args():
    parser = argparse.ArgumentParser(description="Two-model evaluation: classifier + regressor")
    parser.add_argument("--classifier-model", required=True,
                        help="Classifier model checkpoint (e.g., v8C stage1)")
    parser.add_argument("--regressor-model", required=True,
                        help="Regressor model checkpoint (e.g., v8D or v8E)")
    parser.add_argument("--classifier-normalizer", required=True,
                        help="Normalizer for classifier model")
    parser.add_argument("--regressor-normalizer", required=True,
                        help="Normalizer for regressor model")
    parser.add_argument("--h5", required=True, help="Features HDF5 path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--variant-threshold", type=float, default=0.5,
                        help="Classifier threshold for variant detection (default: 0.5)")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint, auto-detect v1/v2."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load classifier model
    classifier, cls_ckpt = load_model(args.classifier_model, device)
    logger.info("Classifier: epoch %d, val_loss=%.4f, %d params",
                cls_ckpt["epoch"], cls_ckpt["val_loss"], classifier.count_parameters())

    # Load regressor model
    regressor, reg_ckpt = load_model(args.regressor_model, device)
    logger.info("Regressor: epoch %d, val_loss=%.4f, %d params",
                reg_ckpt["epoch"], reg_ckpt["val_loss"], regressor.count_parameters())

    # Load normalizers
    cls_normalizer = FeatureNormalizer.load(args.classifier_normalizer)
    reg_normalizer = FeatureNormalizer.load(args.regressor_normalizer)

    # Load test datasets with respective normalizers
    cls_test_ds = DeepTRDataset(args.h5, split="test", normalizer=cls_normalizer, augment=False)
    reg_test_ds = DeepTRDataset(args.h5, split="test", normalizer=reg_normalizer, augment=False)
    logger.info("Test set: %d loci", len(cls_test_ds))

    # Load metadata for stratification
    with h5py.File(args.h5, "r") as h5:
        all_chroms = [c.decode() if isinstance(c, bytes) else c for c in h5["chroms"][:]]
        all_starts = h5["starts"][:]
        all_ends = h5["ends"][:]
        all_motifs = [m.decode() if isinstance(m, bytes) else m for m in h5["motifs"][:]]
        all_labels = h5["labels"][:]
        all_tp = [t.decode() if isinstance(t, bytes) else t for t in h5["tp_statuses"][:]]
        all_read_counts = h5["read_counts"][:]

    test_indices = [i for i, c in enumerate(all_chroms) if chrom_split(c) == "test"]

    # Run inference: classifier first, then regressor for predicted variants
    from torch.utils.data import DataLoader

    cls_loader = DataLoader(cls_test_ds, batch_size=512, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    reg_loader = DataLoader(reg_test_ds, batch_size=512, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Step 1: Get classifier predictions
    all_variant_probs = []
    logger.info("Running classifier inference...")
    with torch.no_grad():
        for batch in cls_loader:
            read_feats = batch["read_features"].to(device)
            locus_feats = batch["locus_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            _, _, variant_pred = classifier(read_feats, locus_feats, padding_mask)
            all_variant_probs.append(variant_pred.cpu().numpy())

    variant_probs = np.concatenate(all_variant_probs, axis=0)  # (N, 1)
    is_variant_pred = variant_probs[:, 0] > args.variant_threshold
    n_pred_variant = is_variant_pred.sum()
    logger.info("Classifier: %d/%d predicted as variant (threshold=%.2f)",
                n_pred_variant, len(is_variant_pred), args.variant_threshold)

    # Step 2: Get regressor predictions
    all_pred_diffs = []
    all_pred_zyg = []
    logger.info("Running regressor inference...")
    with torch.no_grad():
        for batch in reg_loader:
            read_feats = batch["read_features"].to(device)
            locus_feats = batch["locus_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            allele_pred, zyg_pred, _ = regressor(read_feats, locus_feats, padding_mask)
            all_pred_diffs.append(allele_pred.cpu().numpy())
            all_pred_zyg.append(zyg_pred.cpu().numpy())

    pred_diffs_log = np.concatenate(all_pred_diffs, axis=0)  # (N, 2) log-scale
    pred_zyg = np.concatenate(all_pred_zyg, axis=0)

    # Apply gating: zero out allele predictions for non-variant loci
    gate = is_variant_pred[:, np.newaxis].astype(np.float32)  # (N, 1)
    pred_diffs_log_gated = pred_diffs_log * np.broadcast_to(gate, pred_diffs_log.shape)

    # Convert log-scale predictions to bp
    pred_diffs = np.sign(pred_diffs_log_gated) * np.expm1(np.abs(pred_diffs_log_gated))
    pred_diffs = np.sort(pred_diffs, axis=1)

    logger.info("Inference complete: %d loci", pred_diffs.shape[0])

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
    # For non-variant loci, zygosity = homozygous (0)
    pred_zyg_binary[~is_variant_pred] = 0

    # Metrics
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
        lines.append(f"  R\u00b2:                         {r2:.4f}")
        lines.append(f"  Zygosity accuracy:          {zyg_acc:.4f}  ({zyg_acc*100:.1f}%)")
        lines.append(f"  Genotype concordance:       {geno_conc:.4f}  ({geno_conc*100:.1f}%)")
        return "\n".join(lines)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"  Two-Model Evaluation (chr21, chr22, chrX)")
    report_lines.append(f"  Classifier: {args.classifier_model}")
    report_lines.append(f"  Regressor:  {args.regressor_model}")
    report_lines.append(f"  Variant threshold: {args.variant_threshold}")
    report_lines.append(f"  Predicted variants: {n_pred_variant}/{len(is_variant_pred)}")
    report_lines.append("=" * 60)

    report_lines.append(compute_metrics(
        errors, flat_errors, motif_errors,
        pred_zyg_binary, true_zyg, true_diffs, pred_diffs,
        "OVERALL",
    ))

    # By chromosome
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

    # By motif period
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
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Repeat length: {b}",
        ))

    # By variant type
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
        report_lines.append(compute_metrics(
            errors[mask], errors[mask].flatten(),
            motif_errors[mask],
            pred_zyg_binary[mask], true_zyg[mask],
            true_diffs[mask], pred_diffs[mask],
            f"Coverage: {b}",
        ))

    report = "\n".join(report_lines)
    print(report)

    report_path = str(out_dir / "test_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
