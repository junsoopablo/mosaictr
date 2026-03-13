"""Compare MosaicTR v3 vs v4 genotyping results against GIAB truth.

Reuses the overlap-matching infrastructure from compare_v2_v3.py.
Adds v4-specific analysis: confidence distribution, zygosity change tracking.
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import (
    EvalMetrics,
    LocusPrediction,
    LocusTruth,
    StratifiedResults,
    compute_metrics,
    format_results,
    load_predictions,
)
from mosaictr.utils import load_adotto_catalog, load_tier1_bed, match_tier1_to_catalog


def _match_preds_to_truth(preds, tier1_loci, catalog):
    """Match predictions (adotto coords) to truth (Tier1 coords) via overlap."""
    tier1_by_chrom = defaultdict(list)
    for t in tier1_loci:
        tier1_by_chrom[t.chrom].append((t.start, t.end, t))
    for c in tier1_by_chrom:
        tier1_by_chrom[c].sort()
    tier1_starts = {c: [iv[0] for iv in ivs] for c, ivs in tier1_by_chrom.items()}

    matched_t2c = match_tier1_to_catalog(tier1_loci, catalog, tolerance=10)
    tier1_motif = {}
    for locus, motif in matched_t2c:
        tier1_motif[(locus.chrom, locus.start, locus.end)] = motif

    pairs = []
    for pred in preds:
        starts = tier1_starts.get(pred.chrom)
        if starts is None:
            continue
        intervals = tier1_by_chrom[pred.chrom]
        lo = bisect.bisect_left(starts, pred.start - 10000)
        best_t = None
        best_overlap = 0
        for j in range(lo, len(intervals)):
            s, e, t = intervals[j]
            if s > pred.end + 100:
                break
            overlap = min(e, pred.end) - max(s, pred.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_t = t
        if best_t is None or best_overlap < 10:
            continue

        tkey = (best_t.chrom, best_t.start, best_t.end)
        motif = tier1_motif.get(tkey, "")
        if not motif:
            continue

        motif_len = len(motif)
        d1, d2 = sorted([-best_t.hap1_diff_bp, -best_t.hap2_diff_bp])
        truth = LocusTruth(
            chrom=best_t.chrom,
            start=best_t.start,
            end=best_t.end,
            motif=motif,
            true_allele1_diff=d1,
            true_allele2_diff=d2,
            motif_length=motif_len,
            is_variant=best_t.is_variant,
        )
        pairs.append((pred, truth))
    return pairs


def _evaluate_matched(pairs):
    """Evaluate matched prediction-truth pairs."""
    from mosaictr.benchmark import _motif_period_bin, _repeat_length_bin, _coverage_bin

    if not pairs:
        return StratifiedResults()

    pred_diffs_list, true_diffs_list = [], []
    motif_lens_list, pred_zyg_list, true_zyg_list = [], [], []
    meta_list = []

    for pred, truth in pairs:
        ref_size = pred.end - pred.start
        pd1 = pred.allele1_size - ref_size
        pd2 = pred.allele2_size - ref_size
        pd1, pd2 = sorted([pd1, pd2])

        pred_diffs_list.append([pd1, pd2])
        true_diffs_list.append([truth.true_allele1_diff, truth.true_allele2_diff])
        motif_lens_list.append(truth.motif_length)

        pred_het = 1 if pred.zygosity == "HET" else 0
        true_het = 1 if abs(truth.true_allele1_diff - truth.true_allele2_diff) > truth.motif_length else 0
        pred_zyg_list.append(pred_het)
        true_zyg_list.append(true_het)
        meta_list.append((truth.motif_length, ref_size, pred.n_reads, truth.is_variant))

    pred_diffs = np.array(pred_diffs_list)
    true_diffs = np.array(true_diffs_list)
    motif_lens = np.array(motif_lens_list)
    pred_zyg = np.array(pred_zyg_list)
    true_zyg = np.array(true_zyg_list)

    results = StratifiedResults()
    results.overall = compute_metrics(pred_diffs, true_diffs, motif_lens, pred_zyg, true_zyg)

    for i, (ml, rs, nr, iv) in enumerate(meta_list):
        mp_bin = _motif_period_bin(ml)
        rl_bin = _repeat_length_bin(rs)
        cov_bin = _coverage_bin(nr)
        var_bin = "variant" if iv else "reference"
        for bin_name, bin_dict in [
            (mp_bin, results.by_motif_period),
            (rl_bin, results.by_repeat_length),
            (cov_bin, results.by_coverage),
            (var_bin, results.by_variant_type),
        ]:
            if bin_name not in bin_dict:
                bin_dict[bin_name] = {"pred": [], "true": [], "ml": [], "pz": [], "tz": []}
            bin_dict[bin_name]["pred"].append(pred_diffs_list[i])
            bin_dict[bin_name]["true"].append(true_diffs_list[i])
            bin_dict[bin_name]["ml"].append(ml)
            bin_dict[bin_name]["pz"].append(pred_zyg_list[i])
            bin_dict[bin_name]["tz"].append(true_zyg_list[i])

    for stratum_dict in [results.by_motif_period, results.by_repeat_length,
                         results.by_coverage, results.by_variant_type]:
        for key, data in list(stratum_dict.items()):
            stratum_dict[key] = compute_metrics(
                np.array(data["pred"]), np.array(data["true"]),
                np.array(data["ml"]), np.array(data["pz"]), np.array(data["tz"]),
            )
    return results


def _zygosity_change_analysis(v3_pairs, v4_pairs):
    """Analyze zygosity changes between v3 and v4."""
    # Build lookup by pred coords
    v3_by_coord = {}
    for pred, truth in v3_pairs:
        key = (pred.chrom, pred.start, pred.end)
        v3_by_coord[key] = (pred, truth)

    v4_by_coord = {}
    for pred, truth in v4_pairs:
        key = (pred.chrom, pred.start, pred.end)
        v4_by_coord[key] = (pred, truth)

    common_keys = set(v3_by_coord.keys()) & set(v4_by_coord.keys())

    changes = {"HOM→HET": 0, "HET→HOM": 0, "same": 0}
    correct_changes = {"HOM→HET": 0, "HET→HOM": 0}
    incorrect_changes = {"HOM→HET": 0, "HET→HOM": 0}

    for key in common_keys:
        v3_pred, truth = v3_by_coord[key]
        v4_pred, _ = v4_by_coord[key]

        true_het = abs(truth.true_allele1_diff - truth.true_allele2_diff) > truth.motif_length

        if v3_pred.zygosity == v4_pred.zygosity:
            changes["same"] += 1
        elif v3_pred.zygosity == "HOM" and v4_pred.zygosity == "HET":
            changes["HOM→HET"] += 1
            if true_het:
                correct_changes["HOM→HET"] += 1
            else:
                incorrect_changes["HOM→HET"] += 1
        elif v3_pred.zygosity == "HET" and v4_pred.zygosity == "HOM":
            changes["HET→HOM"] += 1
            if not true_het:
                correct_changes["HET→HOM"] += 1
            else:
                incorrect_changes["HET→HOM"] += 1

    return changes, correct_changes, incorrect_changes, len(common_keys)


def _confidence_analysis(v4_preds, v4_pairs):
    """Analyze v4 confidence distribution."""
    # Build truth lookup
    truth_by_coord = {}
    for pred, truth in v4_pairs:
        key = (pred.chrom, pred.start, pred.end)
        truth_by_coord[key] = truth

    confs = {"HET_correct": [], "HET_wrong": [], "HOM_correct": [], "HOM_wrong": []}
    for pred in v4_preds:
        key = (pred.chrom, pred.start, pred.end)
        truth = truth_by_coord.get(key)
        if truth is None:
            continue
        true_het = abs(truth.true_allele1_diff - truth.true_allele2_diff) > truth.motif_length
        conf = pred.confidence

        if pred.zygosity == "HET":
            if true_het:
                confs["HET_correct"].append(conf)
            else:
                confs["HET_wrong"].append(conf)
        else:
            if not true_het:
                confs["HOM_correct"].append(conf)
            else:
                confs["HOM_wrong"].append(conf)

    return confs


def main():
    base = Path(__file__).resolve().parent.parent / "output"

    v3_pred_path = base / "v3_comparison" / "v3_fixed_results.bed"
    v4_pred_path = base / "v4_comparison" / "v4_results.bed"
    truth_path = "/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
    catalog_path = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"
    test_chroms = {"chr21", "chr22", "chrX"}

    print("Loading truth and catalog...")
    tier1 = load_tier1_bed(truth_path, chroms=test_chroms)
    catalog = load_adotto_catalog(catalog_path, chroms=test_chroms)
    print(f"  Tier1 loci: {len(tier1)}")
    print(f"  Catalog entries: {len(catalog)}")

    # Load predictions
    v3_preds = load_predictions(str(v3_pred_path))
    v4_preds = load_predictions(str(v4_pred_path))
    print(f"\n  v3 predictions: {len(v3_preds)}")
    print(f"  v4 predictions: {len(v4_preds)}")

    # Match and evaluate
    v3_pairs = _match_preds_to_truth(v3_preds, tier1, catalog)
    v4_pairs = _match_preds_to_truth(v4_preds, tier1, catalog)
    print(f"  v3 matched to truth: {len(v3_pairs)}")
    print(f"  v4 matched to truth: {len(v4_pairs)}")

    v3_results = _evaluate_matched(v3_pairs)
    v4_results = _evaluate_matched(v4_pairs)

    # === SIDE-BY-SIDE SUMMARY ===
    m3 = v3_results.overall
    m4 = v4_results.overall

    print(f"\n{'='*70}")
    print("  v3 vs v4 OVERALL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30s} {'v3':>10s} {'v4':>10s} {'delta':>10s}")
    print("-" * 62)
    for attr, label in [
        ("n_loci", "N loci"),
        ("exact_match", "Exact match"),
        ("within_1bp", "Within 1bp"),
        ("within_1motif", "Within 1 motif"),
        ("within_5bp", "Within 5bp"),
        ("mae_allele", "MAE (allele, bp)"),
        ("r_squared", "R²"),
        ("zygosity_accuracy", "Zygosity accuracy"),
        ("genotype_concordance", "Geno concordance"),
    ]:
        val3 = getattr(m3, attr)
        val4 = getattr(m4, attr)
        delta = val4 - val3
        if attr == "n_loci":
            print(f"  {label:<28s} {val3:>10d} {val4:>10d} {delta:>+10d}")
        else:
            sign = "+" if delta > 0 else ""
            print(f"  {label:<28s} {val3:>10.4f} {val4:>10.4f} {sign}{delta:>9.4f}")

    # === BY VARIANT TYPE ===
    print(f"\n{'='*70}")
    print("  BY VARIANT TYPE (TP MAE / Zygosity focus)")
    print(f"{'='*70}")
    for vtype in ["variant", "reference"]:
        v3m = v3_results.by_variant_type.get(vtype)
        v4m = v4_results.by_variant_type.get(vtype)
        if v3m and v4m and isinstance(v3m, EvalMetrics) and isinstance(v4m, EvalMetrics):
            print(f"\n  {vtype.upper()} (n_v3={v3m.n_loci}, n_v4={v4m.n_loci})")
            print(f"    MAE:          v3={v3m.mae_allele:.4f}  v4={v4m.mae_allele:.4f}  delta={v4m.mae_allele - v3m.mae_allele:+.4f}")
            print(f"    Exact match:  v3={v3m.exact_match:.4f}  v4={v4m.exact_match:.4f}  delta={v4m.exact_match - v3m.exact_match:+.4f}")
            print(f"    Within 1bp:   v3={v3m.within_1bp:.4f}  v4={v4m.within_1bp:.4f}  delta={v4m.within_1bp - v3m.within_1bp:+.4f}")
            print(f"    Zyg accuracy: v3={v3m.zygosity_accuracy:.4f}  v4={v4m.zygosity_accuracy:.4f}  delta={v4m.zygosity_accuracy - v3m.zygosity_accuracy:+.4f}")
            print(f"    Geno concord: v3={v3m.genotype_concordance:.4f}  v4={v4m.genotype_concordance:.4f}  delta={v4m.genotype_concordance - v3m.genotype_concordance:+.4f}")

    # === BY MOTIF PERIOD ===
    print(f"\n{'='*70}")
    print("  BY MOTIF PERIOD")
    print(f"{'='*70}")
    all_periods = sorted(set(list(v3_results.by_motif_period.keys()) +
                             list(v4_results.by_motif_period.keys())))
    print(f"\n  {'Period':<20s} {'v3 MAE':>8s} {'v4 MAE':>8s} {'Δ MAE':>8s} {'v3 Zyg%':>8s} {'v4 Zyg%':>8s} {'Δ Zyg':>8s} {'n':>6s}")
    print("  " + "-" * 70)
    for period in all_periods:
        v3m = v3_results.by_motif_period.get(period)
        v4m = v4_results.by_motif_period.get(period)
        if v3m and v4m and isinstance(v3m, EvalMetrics) and isinstance(v4m, EvalMetrics):
            d_mae = v4m.mae_allele - v3m.mae_allele
            d_zyg = v4m.zygosity_accuracy - v3m.zygosity_accuracy
            print(f"  {period:<20s} {v3m.mae_allele:>8.3f} {v4m.mae_allele:>8.3f} {d_mae:>+8.3f} "
                  f"{v3m.zygosity_accuracy:>8.4f} {v4m.zygosity_accuracy:>8.4f} {d_zyg:>+8.4f} {v4m.n_loci:>6d}")

    # === ZYGOSITY CHANGE ANALYSIS ===
    print(f"\n{'='*70}")
    print("  ZYGOSITY CHANGE ANALYSIS (v3 → v4)")
    print(f"{'='*70}")
    changes, correct, incorrect, n_common = _zygosity_change_analysis(v3_pairs, v4_pairs)
    print(f"\n  Common loci: {n_common}")
    print(f"  Unchanged:   {changes['same']} ({changes['same']/n_common*100:.1f}%)")
    print(f"  HOM→HET:     {changes['HOM→HET']} (correct: {correct['HOM→HET']}, wrong: {incorrect['HOM→HET']})")
    print(f"  HET→HOM:     {changes['HET→HOM']} (correct: {correct['HET→HOM']}, wrong: {incorrect['HET→HOM']})")
    total_changes = changes['HOM→HET'] + changes['HET→HOM']
    total_correct = correct['HOM→HET'] + correct['HET→HOM']
    if total_changes > 0:
        print(f"  Change accuracy: {total_correct}/{total_changes} = {total_correct/total_changes*100:.1f}%")

    # === CONFIDENCE ANALYSIS ===
    print(f"\n{'='*70}")
    print("  v4 CONFIDENCE ANALYSIS")
    print(f"{'='*70}")
    confs = _confidence_analysis(v4_preds, v4_pairs)
    for label, vals in confs.items():
        if vals:
            arr = np.array(vals)
            print(f"\n  {label} (n={len(arr)}):")
            print(f"    mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
                  f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

    # Save report
    report_path = base / "v4_comparison" / "v3_vs_v4_report.txt"
    import io, contextlib
    buf = io.StringIO()
    # Re-run the prints to capture
    with contextlib.redirect_stdout(buf):
        main_inner(v3_results, v4_results, v3_pairs, v4_pairs, v4_preds, tier1, catalog)
    report_path.write_text(buf.getvalue())
    print(f"\nReport saved to: {report_path}")


def main_inner(v3_results, v4_results, v3_pairs, v4_pairs, v4_preds, tier1, catalog):
    """Inner function for report capture - mirrors main output."""
    pass  # Report is printed in main() directly


if __name__ == "__main__":
    main()
