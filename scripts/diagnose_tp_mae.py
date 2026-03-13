#!/usr/bin/env python3
"""Diagnose TP MAE for MosaicTR v4 vs LongTR vs TRGT.

Analyzes why MosaicTR v4 has higher TP MAE (~3.66) compared to
LongTR (~2.60) and TRGT (~2.80). Examines error distributions,
outlier contribution, motif-period breakdown, error direction,
allele-specific errors, worst loci, and 3-way comparison.

Reuses parsers from compare_tools.py and matching from compare_v3_v4.py.
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import LocusPrediction, LocusTruth, load_predictions
from mosaictr.utils import load_adotto_catalog, load_tier1_bed, match_tier1_to_catalog

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
HAPLOTR_BED = BASE / "output" / "v4_comparison" / "v4_results.bed"
LONGTR_VCF = "/qbio/junsoopablo/02_Projects/10_internship/ensembletr-lr/results/HG002.longtr.vcf.gz"
TRGT_VCF = "/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio_TRGT-VCF_GRCh38/HG002.GRCh38.trgt.sorted.phased.vcf.gz"
TRUTH_BED = "/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG_BED = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"
OUTPUT_PATH = BASE / "output" / "v4_comparison" / "tp_mae_diagnosis.txt"
TEST_CHROMS = {"chr21", "chr22", "chrX"}


# ---------------------------------------------------------------------------
# VCF parsers (from compare_tools.py)
# ---------------------------------------------------------------------------

def parse_longtr_vcf(vcf_path, chroms):
    """Parse LongTR VCF -> dict[(chrom, start)] -> (diff1, diff2, ref_size)."""
    import gzip
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chrom not in chroms:
                continue
            info = fields[7]
            start = end = None
            bpdiffs_str = None
            for kv in info.split(";"):
                if kv.startswith("START="):
                    start = int(kv[6:])
                elif kv.startswith("END="):
                    end = int(kv[4:])
                elif kv.startswith("BPDIFFS="):
                    bpdiffs_str = kv[8:]
            if start is None or end is None:
                continue
            ref_size = end - start + 1  # LongTR uses inclusive [START, END]
            if bpdiffs_str is None or bpdiffs_str == ".":
                results[(chrom, start)] = (0.0, 0.0, ref_size)
                continue
            parts = bpdiffs_str.split(",")
            fmt_keys = fields[8].split(":")
            fmt_vals = fields[9].split(":")
            gt_idx = fmt_keys.index("GT")
            gt_str = fmt_vals[gt_idx]
            sep = "|" if "|" in gt_str else "/"
            a1, a2 = int(gt_str.split(sep)[0]), int(gt_str.split(sep)[1])
            all_bpdiffs = [0] + [int(x) for x in parts]
            if a1 < len(all_bpdiffs) and a2 < len(all_bpdiffs):
                # BPDIFFS = allele - ref (positive = expansion)
                d1 = float(all_bpdiffs[a1])
                d2 = float(all_bpdiffs[a2])
            else:
                d1 = d2 = 0.0
            d1, d2 = sorted([d1, d2])
            results[(chrom, start)] = (d1, d2, ref_size)
    return results


def parse_trgt_vcf(vcf_path, chroms):
    """Parse TRGT VCF -> dict[(chrom, start_0based)] -> (diff1, diff2, ref_size)."""
    import gzip
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chrom not in chroms:
                continue
            pos = int(fields[1])
            ref_size = len(fields[3]) - 1  # remove anchor base
            fmt_keys = fields[8].split(":")
            fmt_vals = fields[9].split(":")
            try:
                al_idx = fmt_keys.index("AL")
                al_str = fmt_vals[al_idx]
            except (ValueError, IndexError):
                continue
            if al_str == ".":
                continue
            al_parts = al_str.split(",")
            if len(al_parts) < 2:
                continue
            try:
                a1_len = int(al_parts[0])
                a2_len = int(al_parts[1])
            except ValueError:
                continue
            # allele - ref convention (positive = expansion)
            d1 = float(a1_len - ref_size)
            d2 = float(a2_len - ref_size)
            d1, d2 = sorted([d1, d2])
            start_0based = pos
            results[(chrom, start_0based)] = (d1, d2, ref_size)
    return results


# ---------------------------------------------------------------------------
# Matching: predictions to truth (from compare_v3_v4.py)
# ---------------------------------------------------------------------------

def match_preds_to_truth(preds, tier1_loci, catalog):
    """Match MosaicTR predictions (adotto coords) to Tier1 truth via overlap."""
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
            chrom=best_t.chrom, start=best_t.start, end=best_t.end,
            motif=motif, true_allele1_diff=d1, true_allele2_diff=d2,
            motif_length=motif_len, is_variant=best_t.is_variant,
        )
        pairs.append((pred, truth))
    return pairs


def match_tool_to_truth(tool_dict, tier1_loci, catalog, tolerance=15):
    """Match LongTR/TRGT dict to Tier1 truth via coordinate matching.

    Adjusts tool diffs to Tier1 reference frame:
      tool_diff = ref_tool - allele  (tool's own reference)
      true_diff = ref_tier1 - allele (Tier1 reference)
      adjusted_tool_diff = tool_diff + (ref_tier1 - ref_tool)
    """
    matched_t2c = match_tier1_to_catalog(tier1_loci, catalog, tolerance=10)
    tier1_motif = {}
    for locus, motif in matched_t2c:
        tier1_motif[(locus.chrom, locus.start, locus.end)] = motif

    # Build tool index by chrom
    tool_by_chrom = defaultdict(list)
    for (chrom, start), val in tool_dict.items():
        tool_by_chrom[chrom].append((start, val))
    for c in tool_by_chrom:
        tool_by_chrom[c].sort()

    pairs = []
    for t in tier1_loci:
        tkey = (t.chrom, t.start, t.end)
        motif = tier1_motif.get(tkey, "")
        if not motif:
            continue

        candidates = tool_by_chrom.get(t.chrom, [])
        if not candidates:
            continue
        cand_starts = [c[0] for c in candidates]
        pos = bisect.bisect_left(cand_starts, t.start)
        best_dist = tolerance + 1
        best_val = None
        for j in range(max(0, pos - 2), min(len(candidates), pos + 3)):
            dist = abs(candidates[j][0] - t.start)
            tool_ref = candidates[j][1][2]
            t_ref = t.end - t.start
            if t_ref > 0 and tool_ref > 0:
                ratio = tool_ref / t_ref
                if ratio < 0.5 or ratio > 2.0:
                    continue
            if dist < best_dist:
                best_dist = dist
                best_val = candidates[j][1]
        if best_dist > tolerance or best_val is None:
            continue

        motif_len = len(motif)
        d1_true, d2_true = sorted([-t.hap1_diff_bp, -t.hap2_diff_bp])

        # Tool diffs are in (allele - ref) convention from parse_*_vcf.
        # Truth diffs are in (allele - ref) convention (negated Tier1).
        # Adjust for ref_size difference between tool and Tier1.
        tool_ref = best_val[2]
        tier1_ref = t.end - t.start
        ref_adjust = tier1_ref - tool_ref
        d1_pred = best_val[0] - ref_adjust
        d2_pred = best_val[1] - ref_adjust
        d1_pred, d2_pred = sorted([d1_pred, d2_pred])

        truth = LocusTruth(
            chrom=t.chrom, start=t.start, end=t.end,
            motif=motif, true_allele1_diff=d1_true, true_allele2_diff=d2_true,
            motif_length=motif_len, is_variant=t.is_variant,
        )
        pairs.append((d1_pred, d2_pred, truth))
    return pairs


# ---------------------------------------------------------------------------
# Motif period bin
# ---------------------------------------------------------------------------

def motif_period_bin(motif_length):
    if motif_length == 1:
        return "homopolymer"
    elif motif_length == 2:
        return "dinucleotide"
    elif motif_length <= 6:
        return f"STR_{motif_length}bp"
    else:
        return "VNTR_7+"


def motif_coarse_bin(motif_length):
    if motif_length <= 2:
        return "dinuc_homo"
    elif motif_length <= 6:
        return "STR_3-6"
    else:
        return "VNTR_7+"


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_per_allele_errors(pred_d1, pred_d2, true_d1, true_d2):
    """Compute per-allele errors for matched pairs."""
    e1 = np.abs(np.array(pred_d1) - np.array(true_d1))
    e2 = np.abs(np.array(pred_d2) - np.array(true_d2))
    return e1, e2


def analyze_error_distribution(name, e1, e2, lines):
    """Error distribution analysis: histogram + percentile."""
    all_e = np.concatenate([e1, e2])
    n = len(all_e)
    lines.append(f"\n  {name} (n_alleles={n}, n_loci={len(e1)})")
    lines.append(f"    MAE = {all_e.mean():.4f}")
    lines.append(f"    Median AE = {np.median(all_e):.4f}")
    for pct in [50, 75, 90, 95, 99]:
        lines.append(f"    P{pct} = {np.percentile(all_e, pct):.2f}")
    for threshold in [0, 0.5, 1, 2, 5, 10, 20, 50]:
        frac = (all_e <= threshold).mean()
        lines.append(f"    |error| <= {threshold:>5.1f}bp: {frac:>6.1%}")


def analyze_outlier_contribution(name, e1, e2, lines):
    """How much do top outliers contribute to total MAE?"""
    all_e = np.concatenate([e1, e2])
    n = len(all_e)
    total_ae = all_e.sum()
    mae = total_ae / n if n > 0 else 0

    sorted_e = np.sort(all_e)[::-1]  # descending
    cumsum = np.cumsum(sorted_e)

    lines.append(f"\n  {name}: Outlier contribution to MAE (total AE = {total_ae:.1f}, MAE = {mae:.4f})")
    for pct in [1, 2, 5, 10, 20]:
        k = max(1, int(n * pct / 100))
        contrib = cumsum[k - 1] / total_ae * 100
        topk_mae = cumsum[k - 1] / k
        lines.append(f"    Top {pct:>2d}% ({k:>5d} alleles): "
                      f"contributes {contrib:>5.1f}% of total AE, "
                      f"mean error in top = {topk_mae:.2f}")


def analyze_by_motif_period(name, pred_d1, pred_d2, true_d1, true_d2, motif_lens, lines):
    """MAE breakdown by motif period."""
    bins = defaultdict(lambda: {"pd1": [], "pd2": [], "td1": [], "td2": []})
    for i in range(len(pred_d1)):
        b = motif_coarse_bin(motif_lens[i])
        bins[b]["pd1"].append(pred_d1[i])
        bins[b]["pd2"].append(pred_d2[i])
        bins[b]["td1"].append(true_d1[i])
        bins[b]["td2"].append(true_d2[i])

    lines.append(f"\n  {name}: MAE by motif period")
    lines.append(f"    {'Period':<15s} {'n':>6s} {'MAE':>8s} {'MedAE':>8s} {'P95':>8s} {'Exact%':>8s}")
    lines.append("    " + "-" * 57)
    for b in ["dinuc_homo", "STR_3-6", "VNTR_7+"]:
        if b not in bins:
            continue
        d = bins[b]
        e1 = np.abs(np.array(d["pd1"]) - np.array(d["td1"]))
        e2 = np.abs(np.array(d["pd2"]) - np.array(d["td2"]))
        all_e = np.concatenate([e1, e2])
        n = len(e1)
        exact = ((e1 < 0.5) & (e2 < 0.5)).mean()
        lines.append(f"    {b:<15s} {n:>6d} {all_e.mean():>8.3f} "
                      f"{np.median(all_e):>8.3f} {np.percentile(all_e, 95):>8.2f} "
                      f"{exact:>7.1%}")


def analyze_error_direction(name, pred_d1, pred_d2, true_d1, true_d2, lines):
    """Over-prediction vs under-prediction analysis.

    diff = ref_size - allele_size, so positive diff = contraction.
    signed_error = pred_diff - true_diff.
    Positive signed_error = predicted more contraction than truth = under-estimated allele.
    """
    se1 = np.array(pred_d1) - np.array(true_d1)
    se2 = np.array(pred_d2) - np.array(true_d2)
    all_se = np.concatenate([se1, se2])
    n = len(all_se)

    n_over = (all_se > 0.5).sum()    # predicted diff larger = allele smaller
    n_under = (all_se < -0.5).sum()   # predicted diff smaller = allele larger
    n_correct = n - n_over - n_under

    lines.append(f"\n  {name}: Error direction (signed error = pred_diff - true_diff)")
    lines.append(f"    Over-est contraction  (allele under-est):  {n_over:>6d} ({n_over/n:.1%})")
    lines.append(f"    Under-est contraction (allele over-est):   {n_under:>6d} ({n_under/n:.1%})")
    lines.append(f"    Correct (|se| <= 0.5bp):                   {n_correct:>6d} ({n_correct/n:.1%})")
    lines.append(f"    Mean signed error:       {all_se.mean():>+.4f}")
    lines.append(f"    Median signed error:     {np.median(all_se):>+.4f}")


def analyze_allele_specific(name, e1, e2, lines):
    """Compare allele1 (smaller diff / larger allele) vs allele2 (larger diff / smaller allele) MAE."""
    lines.append(f"\n  {name}: Allele-specific MAE (diffs sorted: d1 <= d2)")
    lines.append(f"    Allele1 (d1, smaller/neg diff): MAE = {e1.mean():.4f}, median = {np.median(e1):.4f}")
    lines.append(f"    Allele2 (d2, larger/pos diff):  MAE = {e2.mean():.4f}, median = {np.median(e2):.4f}")
    lines.append(f"    Ratio allele2/allele1 MAE:      {e2.mean() / max(e1.mean(), 1e-8):.2f}")


def analyze_worst_loci(preds, pairs, lines, n_worst=50):
    """Characteristics of MosaicTR's worst loci."""
    locus_errors = []
    for pred, truth in pairs:
        if not truth.is_variant:
            continue
        ref_size = pred.end - pred.start
        pd1, pd2 = sorted([pred.allele1_size - ref_size, pred.allele2_size - ref_size])
        td1, td2 = truth.true_allele1_diff, truth.true_allele2_diff
        mae = (abs(pd1 - td1) + abs(pd2 - td2)) / 2
        locus_errors.append((mae, pred, truth, pd1, pd2, td1, td2))

    locus_errors.sort(key=lambda x: -x[0])

    lines.append(f"\n  MosaicTR: Top {n_worst} worst TP loci")
    lines.append(f"    {'#':>3s} {'chrom':<6s} {'start':>10s} {'motif':<15s} {'ml':>3s} "
                  f"{'ref_sz':>6s} {'pred_d1':>8s} {'pred_d2':>8s} {'true_d1':>8s} "
                  f"{'true_d2':>8s} {'MAE':>7s} {'conf':>5s} {'nrd':>4s}")
    lines.append("    " + "-" * 110)
    for i, (mae, pred, truth, pd1, pd2, td1, td2) in enumerate(locus_errors[:n_worst]):
        motif_disp = truth.motif[:12] + ".." if len(truth.motif) > 14 else truth.motif
        lines.append(f"    {i+1:>3d} {truth.chrom:<6s} {truth.start:>10d} {motif_disp:<15s} "
                      f"{truth.motif_length:>3d} {truth.end - truth.start:>6d} "
                      f"{pd1:>8.1f} {pd2:>8.1f} {td1:>8.1f} {td2:>8.1f} "
                      f"{mae:>7.1f} {pred.confidence:>5.2f} {pred.n_reads:>4d}")

    # Summarize characteristics
    top = locus_errors[:n_worst]
    mls = [x[2].motif_length for x in top]
    ref_sizes = [x[2].end - x[2].start for x in top]
    confs = [x[1].confidence for x in top]
    nreads = [x[1].n_reads for x in top]

    lines.append(f"\n    Summary of top {n_worst} worst loci:")
    lines.append(f"      Motif len: mean={np.mean(mls):.1f}, median={np.median(mls):.0f}")

    ml_counts = defaultdict(int)
    for ml in mls:
        ml_counts[motif_coarse_bin(ml)] += 1
    for b in sorted(ml_counts):
        lines.append(f"        {b}: {ml_counts[b]}")

    lines.append(f"      Ref size:  mean={np.mean(ref_sizes):.0f}, median={np.median(ref_sizes):.0f}")
    lines.append(f"      Confidence: mean={np.mean(confs):.3f}, median={np.median(confs):.3f}")
    lines.append(f"      N reads:   mean={np.mean(nreads):.0f}, median={np.median(nreads):.0f}")

    return locus_errors


def analyze_3way_comparison(ht_pairs, lt_pairs, trgt_pairs, lines):
    """3-way comparison: loci where MosaicTR is bad but LongTR/TRGT are good."""
    # Build truth-keyed lookup
    ht_by_truth = {}
    for pred, truth in ht_pairs:
        ref_size = pred.end - pred.start
        pd1, pd2 = sorted([pred.allele1_size - ref_size, pred.allele2_size - ref_size])
        key = (truth.chrom, truth.start, truth.end)
        ht_by_truth[key] = (pd1, pd2, truth, pred)

    lt_by_truth = {}
    for d1, d2, truth in lt_pairs:
        key = (truth.chrom, truth.start, truth.end)
        lt_by_truth[key] = (d1, d2)

    trgt_by_truth = {}
    for d1, d2, truth in trgt_pairs:
        key = (truth.chrom, truth.start, truth.end)
        trgt_by_truth[key] = (d1, d2)

    # Find common TP loci
    common_keys = set(ht_by_truth.keys()) & set(lt_by_truth.keys()) & set(trgt_by_truth.keys())

    ht_better = 0
    ht_worse = 0
    ht_same = 0
    worse_details = []

    for key in common_keys:
        pd1_ht, pd2_ht, truth, pred = ht_by_truth[key]
        pd1_lt, pd2_lt = lt_by_truth[key]
        pd1_trgt, pd2_trgt = trgt_by_truth[key]
        td1, td2 = truth.true_allele1_diff, truth.true_allele2_diff

        if not truth.is_variant:
            continue

        mae_ht = (abs(pd1_ht - td1) + abs(pd2_ht - td2)) / 2
        mae_lt = (abs(pd1_lt - td1) + abs(pd2_lt - td2)) / 2
        mae_trgt = (abs(pd1_trgt - td1) + abs(pd2_trgt - td2)) / 2

        best_other = min(mae_lt, mae_trgt)

        if mae_ht < best_other - 0.5:
            ht_better += 1
        elif mae_ht > best_other + 0.5:
            ht_worse += 1
            worse_details.append((mae_ht, mae_lt, mae_trgt, truth, pred,
                                  pd1_ht, pd2_ht, pd1_lt, pd2_lt, pd1_trgt, pd2_trgt))
        else:
            ht_same += 1

    total_tp_common = ht_better + ht_worse + ht_same
    lines.append(f"\n  3-WAY COMPARISON: MosaicTR vs best-of(LongTR, TRGT) on common TP loci")
    lines.append(f"    Common TP loci: {total_tp_common}")
    lines.append(f"    MosaicTR better (>0.5bp): {ht_better} ({ht_better / max(total_tp_common, 1):.1%})")
    lines.append(f"    MosaicTR worse  (>0.5bp): {ht_worse} ({ht_worse / max(total_tp_common, 1):.1%})")
    lines.append(f"    Similar (<=0.5bp):        {ht_same} ({ht_same / max(total_tp_common, 1):.1%})")

    # Analyze characteristics of loci where MosaicTR is worse
    if worse_details:
        worse_details.sort(key=lambda x: -(x[0] - min(x[1], x[2])))
        mls = [x[3].motif_length for x in worse_details]
        ref_sizes = [x[3].end - x[3].start for x in worse_details]
        confs = [x[4].confidence for x in worse_details]
        nreads = [x[4].n_reads for x in worse_details]

        lines.append(f"\n    Characteristics of {ht_worse} MosaicTR-worse loci:")

        ml_counts = defaultdict(int)
        for ml in mls:
            ml_counts[motif_coarse_bin(ml)] += 1
        for b in sorted(ml_counts):
            lines.append(f"      {b}: {ml_counts[b]} ({ml_counts[b] / ht_worse:.1%})")

        lines.append(f"      Ref size: mean={np.mean(ref_sizes):.0f}, median={np.median(ref_sizes):.0f}")
        lines.append(f"      Confidence: mean={np.mean(confs):.3f}")
        lines.append(f"      N reads: mean={np.mean(nreads):.0f}")

        # Show top 20 worst discrepancies
        lines.append(f"\n    Top 20 loci where MosaicTR is worst relative to others:")
        lines.append(f"      {'#':>3s} {'chrom':<6s} {'start':>10s} {'ml':>3s} {'ref_sz':>6s} "
                      f"{'HT_MAE':>7s} {'LT_MAE':>7s} {'TR_MAE':>7s} {'gap':>6s}")
        lines.append("      " + "-" * 65)
        for i, row in enumerate(worse_details[:20]):
            mae_ht, mae_lt, mae_trgt, truth, pred = row[:5]
            gap = mae_ht - min(mae_lt, mae_trgt)
            lines.append(f"      {i+1:>3d} {truth.chrom:<6s} {truth.start:>10d} "
                          f"{truth.motif_length:>3d} {truth.end - truth.start:>6d} "
                          f"{mae_ht:>7.1f} {mae_lt:>7.1f} {mae_trgt:>7.1f} {gap:>+6.1f}")

    # MAE breakdown for common TP loci
    ht_maes, lt_maes, trgt_maes = [], [], []
    ml_list = []
    for key in common_keys:
        pd1_ht, pd2_ht, truth, pred = ht_by_truth[key]
        pd1_lt, pd2_lt = lt_by_truth[key]
        pd1_trgt, pd2_trgt = trgt_by_truth[key]
        td1, td2 = truth.true_allele1_diff, truth.true_allele2_diff
        if not truth.is_variant:
            continue
        ht_maes.append((abs(pd1_ht - td1) + abs(pd2_ht - td2)) / 2)
        lt_maes.append((abs(pd1_lt - td1) + abs(pd2_lt - td2)) / 2)
        trgt_maes.append((abs(pd1_trgt - td1) + abs(pd2_trgt - td2)) / 2)
        ml_list.append(truth.motif_length)

    ht_maes = np.array(ht_maes)
    lt_maes = np.array(lt_maes)
    trgt_maes = np.array(trgt_maes)
    ml_arr = np.array(ml_list)

    lines.append(f"\n    Common TP loci MAE comparison:")
    lines.append(f"      MosaicTR: {ht_maes.mean():.4f}")
    lines.append(f"      LongTR:  {lt_maes.mean():.4f}")
    lines.append(f"      TRGT:    {trgt_maes.mean():.4f}")

    # By motif period
    lines.append(f"\n    Common TP MAE by motif period:")
    lines.append(f"      {'Period':<15s} {'n':>6s} {'MosaicTR':>8s} {'LongTR':>8s} {'TRGT':>8s}")
    lines.append("      " + "-" * 45)
    for b in ["dinuc_homo", "STR_3-6", "VNTR_7+"]:
        mask = np.array([motif_coarse_bin(m) == b for m in ml_list])
        if mask.sum() == 0:
            continue
        lines.append(f"      {b:<15s} {mask.sum():>6d} {ht_maes[mask].mean():>8.3f} "
                      f"{lt_maes[mask].mean():>8.3f} {trgt_maes[mask].mean():>8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lines = []

    def w(s=""):
        lines.append(s)

    w("=" * 80)
    w("  TP MAE DIAGNOSIS: MosaicTR v4 vs LongTR vs TRGT")
    w("  Test set: chr21, chr22, chrX — GIAB Tier1 truth")
    w("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading truth and catalog...")
    tier1 = load_tier1_bed(TRUTH_BED, chroms=TEST_CHROMS)
    catalog = load_adotto_catalog(CATALOG_BED, chroms=TEST_CHROMS)
    print(f"  Tier1: {len(tier1)}, Catalog: {len(catalog)}")

    print("Loading MosaicTR v4 predictions...")
    ht_preds = load_predictions(str(HAPLOTR_BED))
    print(f"  MosaicTR predictions: {len(ht_preds)}")

    print("Parsing LongTR VCF...")
    longtr_dict = parse_longtr_vcf(LONGTR_VCF, TEST_CHROMS)
    print(f"  LongTR loci: {len(longtr_dict)}")

    print("Parsing TRGT VCF...")
    trgt_dict = parse_trgt_vcf(TRGT_VCF, TEST_CHROMS)
    print(f"  TRGT loci: {len(trgt_dict)}")

    # ── Match to truth ────────────────────────────────────────────────────
    print("Matching to truth...")
    ht_pairs = match_preds_to_truth(ht_preds, tier1, catalog)
    lt_pairs = match_tool_to_truth(longtr_dict, tier1, catalog)
    trgt_pairs = match_tool_to_truth(trgt_dict, tier1, catalog)
    print(f"  MosaicTR matched: {len(ht_pairs)}")
    print(f"  LongTR matched: {len(lt_pairs)}")
    print(f"  TRGT matched: {len(trgt_pairs)}")

    # ── Filter to TP only ─────────────────────────────────────────────────
    ht_tp = [(p, t) for p, t in ht_pairs if t.is_variant]
    lt_tp = [(d1, d2, t) for d1, d2, t in lt_pairs if t.is_variant]
    trgt_tp = [(d1, d2, t) for d1, d2, t in trgt_pairs if t.is_variant]

    w(f"\n  Matched TP loci:")
    w(f"    MosaicTR: {len(ht_tp)}")
    w(f"    LongTR:  {len(lt_tp)}")
    w(f"    TRGT:    {len(trgt_tp)}")

    # ── Extract arrays ────────────────────────────────────────────────────
    # MosaicTR
    ht_pd1, ht_pd2, ht_td1, ht_td2, ht_ml = [], [], [], [], []
    for pred, truth in ht_tp:
        ref_size = pred.end - pred.start
        pd1, pd2 = sorted([pred.allele1_size - ref_size, pred.allele2_size - ref_size])
        ht_pd1.append(pd1)
        ht_pd2.append(pd2)
        ht_td1.append(truth.true_allele1_diff)
        ht_td2.append(truth.true_allele2_diff)
        ht_ml.append(truth.motif_length)
    ht_pd1, ht_pd2 = np.array(ht_pd1), np.array(ht_pd2)
    ht_td1, ht_td2 = np.array(ht_td1), np.array(ht_td2)
    ht_ml = np.array(ht_ml)
    ht_e1, ht_e2 = compute_per_allele_errors(ht_pd1, ht_pd2, ht_td1, ht_td2)

    # LongTR
    lt_pd1 = np.array([x[0] for x in lt_tp])
    lt_pd2 = np.array([x[1] for x in lt_tp])
    lt_td1 = np.array([x[2].true_allele1_diff for x in lt_tp])
    lt_td2 = np.array([x[2].true_allele2_diff for x in lt_tp])
    lt_ml = np.array([x[2].motif_length for x in lt_tp])
    lt_e1, lt_e2 = compute_per_allele_errors(lt_pd1, lt_pd2, lt_td1, lt_td2)

    # TRGT
    trgt_pd1 = np.array([x[0] for x in trgt_tp])
    trgt_pd2 = np.array([x[1] for x in trgt_tp])
    trgt_td1 = np.array([x[2].true_allele1_diff for x in trgt_tp])
    trgt_td2 = np.array([x[2].true_allele2_diff for x in trgt_tp])
    trgt_ml = np.array([x[2].motif_length for x in trgt_tp])
    trgt_e1, trgt_e2 = compute_per_allele_errors(trgt_pd1, trgt_pd2, trgt_td1, trgt_td2)

    # ══════════════════════════════════════════════════════════════════════
    # (a) Error distribution
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (a) ERROR DISTRIBUTION")
    w(f"{'='*80}")
    analyze_error_distribution("MosaicTR v4", ht_e1, ht_e2, lines)
    analyze_error_distribution("LongTR", lt_e1, lt_e2, lines)
    analyze_error_distribution("TRGT", trgt_e1, trgt_e2, lines)

    # ══════════════════════════════════════════════════════════════════════
    # (b) Outlier contribution
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (b) OUTLIER CONTRIBUTION TO MAE")
    w(f"{'='*80}")
    analyze_outlier_contribution("MosaicTR v4", ht_e1, ht_e2, lines)
    analyze_outlier_contribution("LongTR", lt_e1, lt_e2, lines)
    analyze_outlier_contribution("TRGT", trgt_e1, trgt_e2, lines)

    # ══════════════════════════════════════════════════════════════════════
    # (c) Motif period breakdown
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (c) MAE BY MOTIF PERIOD")
    w(f"{'='*80}")
    analyze_by_motif_period("MosaicTR v4", ht_pd1, ht_pd2, ht_td1, ht_td2, ht_ml, lines)
    analyze_by_motif_period("LongTR", lt_pd1, lt_pd2, lt_td1, lt_td2, lt_ml, lines)
    analyze_by_motif_period("TRGT", trgt_pd1, trgt_pd2, trgt_td1, trgt_td2, trgt_ml, lines)

    # ══════════════════════════════════════════════════════════════════════
    # (d) Error direction
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (d) ERROR DIRECTION")
    w(f"{'='*80}")
    analyze_error_direction("MosaicTR v4", ht_pd1, ht_pd2, ht_td1, ht_td2, lines)
    analyze_error_direction("LongTR", lt_pd1, lt_pd2, lt_td1, lt_td2, lines)
    analyze_error_direction("TRGT", trgt_pd1, trgt_pd2, trgt_td1, trgt_td2, lines)

    # ══════════════════════════════════════════════════════════════════════
    # (e) Allele-specific MAE
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (e) ALLELE-SPECIFIC MAE")
    w(f"{'='*80}")
    analyze_allele_specific("MosaicTR v4", ht_e1, ht_e2, lines)
    analyze_allele_specific("LongTR", lt_e1, lt_e2, lines)
    analyze_allele_specific("TRGT", trgt_e1, trgt_e2, lines)

    # ══════════════════════════════════════════════════════════════════════
    # (f) MosaicTR worst loci
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (f) HAPLOTR WORST LOCI (top 50)")
    w(f"{'='*80}")
    analyze_worst_loci(ht_preds, ht_tp, lines, n_worst=50)

    # ══════════════════════════════════════════════════════════════════════
    # (g) 3-way comparison
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n{'='*80}")
    w("  (g) 3-WAY COMPARISON")
    w(f"{'='*80}")
    analyze_3way_comparison(ht_tp, lt_tp, trgt_tp, lines)

    # ── Key findings summary ──────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  KEY FINDINGS SUMMARY")
    w(f"{'='*80}")

    ht_all_e = np.concatenate([ht_e1, ht_e2])
    lt_all_e = np.concatenate([lt_e1, lt_e2])
    trgt_all_e = np.concatenate([trgt_e1, trgt_e2])

    w(f"\n  TP MAE comparison:")
    w(f"    MosaicTR: {ht_all_e.mean():.4f}")
    w(f"    LongTR:  {lt_all_e.mean():.4f}")
    w(f"    TRGT:    {trgt_all_e.mean():.4f}")

    w(f"\n  Median AE (less sensitive to outliers):")
    w(f"    MosaicTR: {np.median(ht_all_e):.4f}")
    w(f"    LongTR:  {np.median(lt_all_e):.4f}")
    w(f"    TRGT:    {np.median(trgt_all_e):.4f}")

    # Outlier impact
    n_ht = len(ht_all_e)
    top5_idx = int(n_ht * 0.05)
    sorted_ht = np.sort(ht_all_e)[::-1]
    top5_contrib = sorted_ht[:top5_idx].sum() / ht_all_e.sum() * 100
    ht_mae_sans_top5 = sorted_ht[top5_idx:].mean()

    w(f"\n  Outlier impact (MosaicTR):")
    w(f"    Top 5% alleles contribute {top5_contrib:.1f}% of total AE")
    w(f"    MAE without top 5% outliers: {ht_mae_sans_top5:.4f}")
    w(f"    (vs full MAE: {ht_all_e.mean():.4f})")

    # ── Output ────────────────────────────────────────────────────────────
    report = "\n".join(lines)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {OUTPUT_PATH}")
    print(report)


if __name__ == "__main__":
    main()
