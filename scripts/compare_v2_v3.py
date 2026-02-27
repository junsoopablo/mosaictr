"""Compare HaploTR v2 vs v3 genotyping results against GIAB truth.

Uses overlap-based matching since prediction coordinates (adotto catalog)
differ from truth coordinates (GIAB Tier1).
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haplotr.benchmark import (
    EvalMetrics,
    LocusPrediction,
    LocusTruth,
    StratifiedResults,
    compute_metrics,
    format_results,
    load_predictions,
)
from haplotr.utils import load_adotto_catalog, load_tier1_bed, match_tier1_to_catalog


def _match_preds_to_truth(
    preds: list[LocusPrediction],
    tier1_loci,
    catalog,
) -> list[tuple[LocusPrediction, LocusTruth]]:
    """Match predictions (adotto coords) to truth (Tier1 coords) via overlap."""
    # Build Tier1 interval index by chrom
    tier1_by_chrom: dict[str, list[tuple[int, int, object]]] = defaultdict(list)
    for t in tier1_loci:
        tier1_by_chrom[t.chrom].append((t.start, t.end, t))
    for c in tier1_by_chrom:
        tier1_by_chrom[c].sort()
    tier1_starts = {c: [iv[0] for iv in ivs] for c, ivs in tier1_by_chrom.items()}

    # Match Tier1 to catalog for motif info
    matched_t2c = match_tier1_to_catalog(tier1_loci, catalog, tolerance=10)
    tier1_motif = {}
    for locus, motif in matched_t2c:
        tier1_motif[(locus.chrom, locus.start, locus.end)] = motif

    # For each prediction, find overlapping Tier1 locus
    pairs = []
    for pred in preds:
        starts = tier1_starts.get(pred.chrom)
        if starts is None:
            continue
        intervals = tier1_by_chrom[pred.chrom]

        # Binary search for nearby Tier1 loci
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
        # Negate: Tier1 stores ref-allele, convert to allele-ref
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


def _evaluate_matched(
    pairs: list[tuple[LocusPrediction, LocusTruth]],
) -> StratifiedResults:
    """Evaluate matched prediction-truth pairs."""
    from haplotr.benchmark import _motif_period_bin, _repeat_length_bin, _coverage_bin

    if not pairs:
        return StratifiedResults()

    pred_diffs_list = []
    true_diffs_list = []
    motif_lens_list = []
    pred_zyg_list = []
    true_zyg_list = []
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

    # Stratify
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
                np.array(data["pred"]),
                np.array(data["true"]),
                np.array(data["ml"]),
                np.array(data["pz"]),
                np.array(data["tz"]),
            )

    return results


def main():
    base = Path(__file__).resolve().parent.parent / "output" / "v3_comparison"

    v2_pred_path = base / "v2_results.bed"
    v3_pred_path = base / "v3_fixed_results.bed"
    truth_path = "/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
    catalog_path = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

    test_chroms = {"chr21", "chr22", "chrX"}

    print("Loading truth and catalog...")
    tier1 = load_tier1_bed(truth_path, chroms=test_chroms)
    catalog = load_adotto_catalog(catalog_path, chroms=test_chroms)
    print(f"  Tier1 loci: {len(tier1)}")
    print(f"  Catalog entries: {len(catalog)}")

    for label, pred_path in [("v2 (HPMedian_v2)", v2_pred_path),
                              ("v3 (HPMedian_v3)", v3_pred_path)]:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        preds = load_predictions(str(pred_path))
        print(f"  Predictions loaded: {len(preds)}")

        pairs = _match_preds_to_truth(preds, tier1, catalog)
        print(f"  Matched to truth: {len(pairs)}")

        results = _evaluate_matched(pairs)
        report = format_results(results)
        print(report)

    # Side-by-side summary
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    v2_preds = load_predictions(str(v2_pred_path))
    v3_preds = load_predictions(str(v3_pred_path))
    v2_pairs = _match_preds_to_truth(v2_preds, tier1, catalog)
    v3_pairs = _match_preds_to_truth(v3_preds, tier1, catalog)
    v2_results = _evaluate_matched(v2_pairs)
    v3_results = _evaluate_matched(v3_pairs)

    m2 = v2_results.overall
    m3 = v3_results.overall

    print(f"\n{'Metric':<30s} {'v2':>10s} {'v3':>10s} {'delta':>10s}")
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
        val2 = getattr(m2, attr)
        val3 = getattr(m3, attr)
        delta = val3 - val2
        if attr == "n_loci":
            print(f"  {label:<28s} {val2:>10d} {val3:>10d} {delta:>+10d}")
        else:
            sign = "+" if delta > 0 else ""
            print(f"  {label:<28s} {val2:>10.4f} {val3:>10.4f} {sign}{delta:>9.4f}")

    # Stratified comparison: TP vs TN MAE
    print(f"\n{'='*70}")
    print("BY VARIANT TYPE (TP MAE focus)")
    print(f"{'='*70}")
    for vtype in ["variant", "reference"]:
        v2m = v2_results.by_variant_type.get(vtype)
        v3m = v3_results.by_variant_type.get(vtype)
        if v2m and v3m and isinstance(v2m, EvalMetrics) and isinstance(v3m, EvalMetrics):
            delta_mae = v3m.mae_allele - v2m.mae_allele
            delta_exact = v3m.exact_match - v2m.exact_match
            print(f"\n  {vtype.upper()} (n_v2={v2m.n_loci}, n_v3={v3m.n_loci})")
            print(f"    MAE:          v2={v2m.mae_allele:.4f}  v3={v3m.mae_allele:.4f}  delta={delta_mae:+.4f}")
            print(f"    Exact match:  v2={v2m.exact_match:.4f}  v3={v3m.exact_match:.4f}  delta={delta_exact:+.4f}")
            print(f"    Within 1bp:   v2={v2m.within_1bp:.4f}  v3={v3m.within_1bp:.4f}")
            print(f"    Zyg accuracy: v2={v2m.zygosity_accuracy:.4f}  v3={v3m.zygosity_accuracy:.4f}")

    # Stratified by motif period
    print(f"\n{'='*70}")
    print("BY MOTIF PERIOD (MAE)")
    print(f"{'='*70}")
    all_periods = sorted(set(list(v2_results.by_motif_period.keys()) +
                             list(v3_results.by_motif_period.keys())))
    print(f"\n  {'Period':<20s} {'v2 MAE':>10s} {'v3 MAE':>10s} {'delta':>10s} {'v2 n':>8s} {'v3 n':>8s}")
    print("  " + "-" * 58)
    for period in all_periods:
        v2m = v2_results.by_motif_period.get(period)
        v3m = v3_results.by_motif_period.get(period)
        if v2m and v3m and isinstance(v2m, EvalMetrics) and isinstance(v3m, EvalMetrics):
            delta = v3m.mae_allele - v2m.mae_allele
            print(f"  {period:<20s} {v2m.mae_allele:>10.4f} {v3m.mae_allele:>10.4f} {delta:>+10.4f} {v2m.n_loci:>8d} {v3m.n_loci:>8d}")


if __name__ == "__main__":
    main()
