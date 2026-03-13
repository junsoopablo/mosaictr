"""Diagnose TP loci with high MAE in v2/v3 results."""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import LocusPrediction, LocusTruth, load_predictions
from mosaictr.utils import load_adotto_catalog, load_tier1_bed, match_tier1_to_catalog


def _match_preds_to_truth(preds, tier1_loci, catalog):
    """Match predictions to truth via overlap (same as compare_v2_v3.py)."""
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
        # Negate: Tier1 stores ref-allele, convert to allele-ref
        d1, d2 = sorted([-best_t.hap1_diff_bp, -best_t.hap2_diff_bp])
        truth = LocusTruth(
            chrom=best_t.chrom, start=best_t.start, end=best_t.end,
            motif=motif, true_allele1_diff=d1, true_allele2_diff=d2,
            motif_length=motif_len, is_variant=best_t.is_variant,
        )
        pairs.append((pred, truth))
    return pairs


def main():
    base = Path(__file__).resolve().parent.parent / "output" / "v3_comparison"
    truth_path = "/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
    catalog_path = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"
    test_chroms = {"chr21", "chr22", "chrX"}

    print("Loading data...")
    tier1 = load_tier1_bed(truth_path, chroms=test_chroms)
    catalog = load_adotto_catalog(catalog_path, chroms=test_chroms)

    v3_preds = load_predictions(str(base / "v3_results.bed"))
    pairs = _match_preds_to_truth(v3_preds, tier1, catalog)

    # Filter to TP (variant) only
    tp_pairs = [(p, t) for p, t in pairs if t.is_variant]
    print(f"TP loci: {len(tp_pairs)}")

    # Compute per-locus errors
    errors = []
    for pred, truth in tp_pairs:
        ref_size = pred.end - pred.start
        pd1 = pred.allele1_size - ref_size
        pd2 = pred.allele2_size - ref_size
        pd1, pd2 = sorted([pd1, pd2])
        e1 = abs(pd1 - truth.true_allele1_diff)
        e2 = abs(pd2 - truth.true_allele2_diff)
        mae = (e1 + e2) / 2
        errors.append({
            "pred": pred, "truth": truth,
            "pred_d1": pd1, "pred_d2": pd2,
            "true_d1": truth.true_allele1_diff, "true_d2": truth.true_allele2_diff,
            "e1": e1, "e2": e2, "mae": mae,
            "ref_size": ref_size, "motif_len": truth.motif_length,
        })

    errors.sort(key=lambda x: x["mae"], reverse=True)

    # Error distribution
    maes = np.array([e["mae"] for e in errors])
    print(f"\n{'='*70}")
    print("TP MAE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  Mean: {maes.mean():.2f}")
    print(f"  Median: {np.median(maes):.2f}")
    print(f"  P90: {np.percentile(maes, 90):.2f}")
    print(f"  P95: {np.percentile(maes, 95):.2f}")
    print(f"  P99: {np.percentile(maes, 99):.2f}")
    print(f"  Max: {maes.max():.2f}")

    # How many loci dominate the MAE?
    for thresh in [0, 1, 5, 10, 50, 100, 500]:
        n = np.sum(maes > thresh)
        pct = 100 * n / len(maes)
        contrib = maes[maes > thresh].sum() / maes.sum() * 100
        print(f"  MAE > {thresh:>4d}bp: {n:>5d} loci ({pct:>5.1f}%), contributing {contrib:>5.1f}% of total error")

    # Coordinate mismatch analysis
    print(f"\n{'='*70}")
    print("COORDINATE MISMATCH ANALYSIS")
    print(f"{'='*70}")
    coord_diffs = []
    for e in errors:
        p, t = e["pred"], e["truth"]
        start_diff = abs(p.start - t.start)
        end_diff = abs(p.end - t.end)
        coord_diffs.append((start_diff, end_diff, start_diff + end_diff))
    cd = np.array(coord_diffs)
    print(f"  Start diff: mean={cd[:,0].mean():.1f}, median={np.median(cd[:,0]):.0f}, max={cd[:,0].max():.0f}")
    print(f"  End diff:   mean={cd[:,1].mean():.1f}, median={np.median(cd[:,1]):.0f}, max={cd[:,1].max():.0f}")
    print(f"  Total diff: mean={cd[:,2].mean():.1f}, median={np.median(cd[:,2]):.0f}, max={cd[:,2].max():.0f}")

    # Correlation between coordinate diff and error
    total_coord_diff = cd[:, 2]
    corr = np.corrcoef(total_coord_diff, maes)[0, 1]
    print(f"\n  Correlation(coord_diff, MAE): {corr:.4f}")

    # Group by coordinate match quality
    exact_mask = total_coord_diff == 0
    close_mask = (total_coord_diff > 0) & (total_coord_diff <= 20)
    far_mask = total_coord_diff > 20
    print(f"\n  Exact coords (diff=0):  n={exact_mask.sum():>5d}, MAE={maes[exact_mask].mean():.2f}" if exact_mask.any() else "  Exact coords: 0")
    print(f"  Close coords (diff<=20): n={close_mask.sum():>5d}, MAE={maes[close_mask].mean():.2f}" if close_mask.any() else "  Close coords: 0")
    print(f"  Far coords (diff>20):   n={far_mask.sum():>5d}, MAE={maes[far_mask].mean():.2f}" if far_mask.any() else "  Far coords: 0")

    # Top 30 worst loci
    print(f"\n{'='*70}")
    print("TOP 30 WORST TP LOCI")
    print(f"{'='*70}")
    print(f"  {'Chrom':<6s} {'PredStart':>10s} {'PredEnd':>10s} {'RefSize':>7s} {'Motif':>5s} "
          f"{'PredD1':>8s} {'PredD2':>8s} {'TrueD1':>8s} {'TrueD2':>8s} {'MAE':>8s} "
          f"{'Nreads':>6s} {'CoordDiff':>9s}")
    print("  " + "-" * 110)
    for e in errors[:30]:
        p, t = e["pred"], e["truth"]
        coord_d = abs(p.start - t.start) + abs(p.end - t.end)
        print(f"  {p.chrom:<6s} {p.start:>10d} {p.end:>10d} {e['ref_size']:>7d} {e['motif_len']:>5d} "
              f"{e['pred_d1']:>8.1f} {e['pred_d2']:>8.1f} {e['true_d1']:>8.1f} {e['true_d2']:>8.1f} "
              f"{e['mae']:>8.1f} {p.n_reads:>6d} {coord_d:>9d}")

    # Stratify by error source
    print(f"\n{'='*70}")
    print("ERROR SOURCE ANALYSIS")
    print(f"{'='*70}")

    # Category 1: Coordinate mismatch dominates
    coord_dom = [e for e in errors if cd[errors.index(e)][2] > 20 and e["mae"] > 10]
    # Category 2: Prediction is 0,0 (missed variant)
    missed = [e for e in errors if e["pred_d1"] == 0 and e["pred_d2"] == 0 and e["mae"] > 1]
    # Category 3: Wrong sign / wrong allele assignment
    wrong_sign = [e for e in errors if
                  (e["pred_d1"] * e["true_d1"] < 0 or e["pred_d2"] * e["true_d2"] < 0)
                  and e["mae"] > 1]
    # Category 4: Large ref_size (>500bp) loci
    large_ref = [e for e in errors if e["ref_size"] > 500 and e["mae"] > 10]

    print(f"\n  Coordinate mismatch (diff>20, MAE>10): {len(coord_dom)} loci")
    if coord_dom:
        cm_maes = np.array([e["mae"] for e in coord_dom])
        print(f"    Mean MAE: {cm_maes.mean():.2f}, contribution: {cm_maes.sum()/maes.sum()*100:.1f}%")

    print(f"\n  Missed variants (pred=0,0, MAE>1): {len(missed)} loci")
    if missed:
        mm_maes = np.array([e["mae"] for e in missed])
        print(f"    Mean MAE: {mm_maes.mean():.2f}, contribution: {mm_maes.sum()/maes.sum()*100:.1f}%")

    print(f"\n  Wrong sign (MAE>1): {len(wrong_sign)} loci")
    if wrong_sign:
        ws_maes = np.array([e["mae"] for e in wrong_sign])
        print(f"    Mean MAE: {ws_maes.mean():.2f}, contribution: {ws_maes.sum()/maes.sum()*100:.1f}%")

    print(f"\n  Large ref (>500bp, MAE>10): {len(large_ref)} loci")
    if large_ref:
        lr_maes = np.array([e["mae"] for e in large_ref])
        print(f"    Mean MAE: {lr_maes.mean():.2f}, contribution: {lr_maes.sum()/maes.sum()*100:.1f}%")

    # Motif period breakdown for TP
    print(f"\n{'='*70}")
    print("TP MAE BY MOTIF PERIOD")
    print(f"{'='*70}")
    by_period = defaultdict(list)
    for e in errors:
        ml = e["motif_len"]
        if ml == 1:
            period = "homopolymer"
        elif ml == 2:
            period = "dinucleotide"
        elif ml <= 6:
            period = f"STR_{ml}bp"
        else:
            period = "VNTR_7+"
        by_period[period].append(e["mae"])

    print(f"\n  {'Period':<20s} {'N':>6s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'Max':>8s}")
    print("  " + "-" * 52)
    for period in sorted(by_period.keys()):
        m = np.array(by_period[period])
        print(f"  {period:<20s} {len(m):>6d} {m.mean():>8.2f} {np.median(m):>8.2f} "
              f"{np.percentile(m, 90):>8.2f} {m.max():>8.2f}")

    # Ref size breakdown for TP
    print(f"\n{'='*70}")
    print("TP MAE BY REF SIZE")
    print(f"{'='*70}")
    by_refsize = defaultdict(list)
    for e in errors:
        rs = e["ref_size"]
        if rs < 50:
            b = "<50bp"
        elif rs < 100:
            b = "50-100bp"
        elif rs < 200:
            b = "100-200bp"
        elif rs < 500:
            b = "200-500bp"
        elif rs < 1000:
            b = "500-1000bp"
        else:
            b = ">1000bp"
        by_refsize[b].append(e["mae"])

    print(f"\n  {'RefSize':<20s} {'N':>6s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'Max':>8s}")
    print("  " + "-" * 52)
    for b in ["<50bp", "50-100bp", "100-200bp", "200-500bp", "500-1000bp", ">1000bp"]:
        if b in by_refsize:
            m = np.array(by_refsize[b])
            print(f"  {b:<20s} {len(m):>6d} {m.mean():>8.2f} {np.median(m):>8.2f} "
                  f"{np.percentile(m, 90):>8.2f} {m.max():>8.2f}")


if __name__ == "__main__":
    main()
