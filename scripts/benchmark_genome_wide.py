#!/usr/bin/env python3
"""Genome-wide benchmark: MosaicTR v4 vs LongTR vs TRGT against GIAB truth.

Produces publication-quality tables comparing genotyping accuracy across
all GIAB Tier1 loci. Stratified by variant type, motif period, repeat
length, and coverage. Also reports on STRchive disease loci.

Usage:
  python scripts/benchmark_genome_wide.py \
    --mosaictr output/genome_wide/mosaictr_v4_genome_wide.bed \
    --longtr-vcf /path/to/HG002.longtr.vcf.gz \
    --trgt-vcf /path/to/HG002.trgt.sorted.phased.vcf.gz \
    --output output/genome_wide/benchmark_report.txt
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import (
    EvalMetrics,
    LocusPrediction,
    LocusTruth,
    load_predictions,
    _motif_period_bin,
    _repeat_length_bin,
    _coverage_bin,
)
from mosaictr.utils import (
    load_adotto_catalog,
    load_tier1_bed,
    match_tier1_to_catalog,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

TRUTH_BED = "/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG_BED = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"
LONGTR_VCF = "/qbio/junsoopablo/02_Projects/10_internship/ensembletr-lr/results/HG002.longtr.vcf.gz"
TRGT_VCF = "/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio_TRGT-VCF_GRCh38/HG002.GRCh38.trgt.sorted.phased.vcf.gz"

# STRchive disease-associated loci (GRCh38 coordinates)
# Source: STRchive v1, https://strchive.org
# Format: (chrom, start, end, gene, motif, disease)
STRCHIVE_LOCI = [
    ("chr4", 3074876, 3074933, "HTT", "CAG", "Huntington disease"),
    ("chrX", 147912050, 147912110, "FMR1", "CGG", "Fragile X syndrome"),
    ("chr19", 45770204, 45770264, "DMPK", "CTG", "Myotonic dystrophy 1"),
    ("chr9", 69037286, 69037304, "FXN", "GAA", "Friedreich ataxia"),
    ("chr4", 39348424, 39348479, "RFC1", "AAGGG", "CANVAS"),
    ("chr9", 27573528, 27573546, "C9ORF72", "GGGGCC", "ALS/FTD"),
    ("chr6", 16327633, 16327723, "ATXN1", "CAG", "SCA1"),
    ("chr12", 111598950, 111599019, "ATXN2", "CAG", "SCA2"),
    ("chr14", 92071009, 92071040, "ATXN3", "CAG", "SCA3/MJD"),
    ("chr6", 170870995, 170871114, "TBP", "CAG", "SCA17"),
    ("chr13", 70139383, 70139429, "ATXN8OS", "CTG", "SCA8"),
    ("chr22", 46191235, 46191304, "ATXN10", "ATTCT", "SCA10"),
    ("chr5", 146878727, 146878757, "PPP2R2B", "CAG", "SCA12"),
    ("chr19", 13318670, 13318711, "CACNA1A", "CAG", "SCA6"),
    ("chr16", 87637893, 87637935, "JPH3", "CTG", "HDL2"),
    ("chr12", 6936716, 6936773, "ATN1", "CAG", "DRPLA"),
    ("chrX", 67545316, 67545385, "AR", "CAG", "SBMA"),
    ("chr5", 10356346, 10356412, "CSTB", "CCCCGCCCCGCG", "EPM1"),
    ("chr22", 19766762, 19766817, "TBCE", "GCN", "HMN"),
    ("chr3", 63912684, 63912714, "CNBP", "CCTG", "Myotonic dystrophy 2"),
    ("chr20", 2652733, 2652757, "NOP56", "GGCCTG", "SCA36"),
    ("chr1", 57367043, 57367100, "DAB1", "ATTTC", "SCA37"),
    ("chr12", 50505001, 50505022, "DIP2B", "GGC", "FRA12A MR"),
    ("chr2", 175923218, 175923261, "HOXD13", "GCG", "Synpolydactyly"),
    ("chr4", 41745972, 41746032, "PHOX2B", "GCN", "CCHS"),
    ("chr14", 23321472, 23321490, "PABPN1", "GCG", "OPMD"),
    ("chr7", 27199878, 27199927, "HOXA13", "GCN", "HFGS"),
    ("chr2", 176093058, 176093103, "HOXD13", "GCG", "Brachydactyly"),
    ("chr7", 55248931, 55249016, "EGFR", "CA", "EGFR regulation"),
    ("chr11", 119206289, 119206322, "CBL2", "CGG", "Jacobsen syndrome"),
    ("chr18", 55586155, 55586227, "TCF4", "CAG", "FECD3"),
    ("chr7", 158628630, 158628672, "VIPR2", "GCC", "Schizophrenia risk"),
]


# ---------------------------------------------------------------------------
# VCF parsers
# ---------------------------------------------------------------------------

def parse_longtr_vcf(vcf_path, chroms=None):
    """Parse LongTR VCF -> dict[(chrom, start)] -> (d1, d2, ref_size).

    diff convention: allele - ref (positive = expansion).
    """
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    t0 = time.time()
    n = 0
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chroms is not None and chrom not in chroms:
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
                n += 1
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
                # BPDIFFS = expansion, use directly as (allele - ref)
                d1 = float(all_bpdiffs[a1])
                d2 = float(all_bpdiffs[a2])
            else:
                d1 = d2 = 0.0
            d1, d2 = sorted([d1, d2])
            results[(chrom, start)] = (d1, d2, ref_size)
            n += 1
    logger.info("LongTR: parsed %d loci in %.1fs", n, time.time() - t0)
    return results


def parse_trgt_vcf(vcf_path, chroms=None):
    """Parse TRGT VCF -> dict[(chrom, start, end)] -> (d1, d2, ref_size).

    Uses TRID or END field for the end coordinate. The key includes
    (chrom, start, end) for interval-based matching since TRGT uses
    its own catalog coordinates that may differ from Adotto/Tier1.

    diff convention: allele - ref (positive = expansion).
    """
    results = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    t0 = time.time()
    n = 0
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            if chroms is not None and chrom not in chroms:
                continue
            pos = int(fields[1])
            ref_size = len(fields[3]) - 1  # remove anchor base
            # Parse END from INFO
            info = fields[7]
            end = pos + ref_size  # fallback
            for kv in info.split(";"):
                if kv.startswith("END="):
                    end = int(kv[4:])
                    break
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
            # allele - ref convention
            d1 = float(a1_len - ref_size)
            d2 = float(a2_len - ref_size)
            d1, d2 = sorted([d1, d2])
            results[(chrom, pos, end)] = (d1, d2, ref_size)
            n += 1
    logger.info("TRGT: parsed %d loci in %.1fs", n, time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_preds_to_truth(preds, tier1_loci, catalog):
    """Match MosaicTR predictions (adotto coords) to truth via overlap."""
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
        # Truth: negate Tier1 hap_diff_bp to get (allele - ref) convention
        d1, d2 = sorted([-best_t.hap1_diff_bp, -best_t.hap2_diff_bp])
        truth = LocusTruth(
            chrom=best_t.chrom, start=best_t.start, end=best_t.end,
            motif=motif, true_allele1_diff=d1, true_allele2_diff=d2,
            motif_length=motif_len, is_variant=best_t.is_variant,
        )
        pairs.append((pred, truth))
    return pairs


def match_tool_to_truth(tool_dict, tier1_loci, catalog, tolerance=15):
    """Match LongTR/TRGT to truth, adjusting for ref frame differences.

    Supports two key formats:
    - (chrom, start) -> (d1, d2, ref_size)  [LongTR]
    - (chrom, start, end) -> (d1, d2, ref_size)  [TRGT with interval]

    For interval keys, uses overlap-based matching (robust to coordinate shifts).
    For start-only keys, uses start-position proximity matching.

    Tool diffs are in (allele - ref_tool) convention.
    Truth diffs are in (allele - ref_tier1) convention.
    Adjustment: adjusted = tool_diff - (tier1_ref - tool_ref)
    """
    matched_t2c = match_tier1_to_catalog(tier1_loci, catalog, tolerance=10)
    tier1_motif = {}
    for locus, motif in matched_t2c:
        tier1_motif[(locus.chrom, locus.start, locus.end)] = motif

    # Detect key format
    sample_key = next(iter(tool_dict), None)
    has_end = sample_key is not None and len(sample_key) == 3

    if has_end:
        # Interval-based matching (TRGT)
        tool_by_chrom = defaultdict(list)
        for (chrom, start, end), val in tool_dict.items():
            tool_by_chrom[chrom].append((start, end, val))
        for c in tool_by_chrom:
            tool_by_chrom[c].sort()
        tool_starts = {c: [iv[0] for iv in ivs] for c, ivs in tool_by_chrom.items()}
    else:
        # Start-position matching (LongTR)
        tool_by_chrom = defaultdict(list)
        for (chrom, start), val in tool_dict.items():
            tool_by_chrom[chrom].append((start, val))
        for c in tool_by_chrom:
            tool_by_chrom[c].sort()
        tool_starts = {c: [iv[0] for iv in ivs] for c, ivs in tool_by_chrom.items()}

    pairs = []
    for t in tier1_loci:
        tkey = (t.chrom, t.start, t.end)
        motif = tier1_motif.get(tkey, "")
        if not motif:
            continue
        candidates = tool_by_chrom.get(t.chrom, [])
        if not candidates:
            continue

        best_val = None

        if has_end:
            # Overlap-based matching for TRGT
            starts = tool_starts.get(t.chrom, [])
            lo = bisect.bisect_left(starts, t.start - 500)
            best_overlap = 0
            for j in range(lo, len(candidates)):
                cs, ce, cv = candidates[j]
                if cs > t.end + 500:
                    break
                overlap = min(ce, t.end) - max(cs, t.start)
                if overlap <= 0:
                    continue
                # Size ratio filter
                tool_ref = cv[2]
                t_ref = t.end - t.start
                if t_ref > 0 and tool_ref > 0:
                    ratio = tool_ref / t_ref
                    if ratio < 0.3 or ratio > 3.0:
                        continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_val = cv
            if best_val is None or best_overlap < max(5, (t.end - t.start) * 0.1):
                continue
        else:
            # Start-position matching for LongTR
            cand_starts = tool_starts.get(t.chrom, [])
            pos = bisect.bisect_left(cand_starts, t.start)
            best_dist = tolerance + 1
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
        # Truth: (allele - ref_tier1)
        d1_true, d2_true = sorted([-t.hap1_diff_bp, -t.hap2_diff_bp])
        # Tool: (allele - ref_tool), adjust to tier1 frame
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
# Metrics computation
# ---------------------------------------------------------------------------

def compute_tool_metrics(pairs, motif_len_source="truth"):
    """Compute metrics from list of (pred_d1, pred_d2, truth) tuples."""
    if not pairs:
        return None
    pred_diffs = np.array([[p[0], p[1]] for p in pairs])
    true_diffs = np.array([[p[2].true_allele1_diff, p[2].true_allele2_diff] for p in pairs])
    motif_lens = np.array([p[2].motif_length for p in pairs])
    pred_zyg = np.array([(1 if abs(p[0] - p[1]) > p[2].motif_length else 0) for p in pairs])
    true_zyg = np.array([(1 if abs(p[2].true_allele1_diff - p[2].true_allele2_diff) > p[2].motif_length else 0) for p in pairs])

    return _compute_metrics_arrays(pred_diffs, true_diffs, motif_lens, pred_zyg, true_zyg)


def compute_mosaictr_metrics(pairs):
    """Compute metrics from list of (LocusPrediction, LocusTruth) tuples."""
    if not pairs:
        return None
    pred_diffs_list = []
    true_diffs_list = []
    motif_lens_list = []
    pred_zyg_list = []
    true_zyg_list = []

    for pred, truth in pairs:
        ref_size = pred.end - pred.start
        pd1, pd2 = sorted([pred.allele1_size - ref_size, pred.allele2_size - ref_size])
        pred_diffs_list.append([pd1, pd2])
        true_diffs_list.append([truth.true_allele1_diff, truth.true_allele2_diff])
        motif_lens_list.append(truth.motif_length)
        pred_zyg_list.append(1 if pred.zygosity == "HET" else 0)
        true_zyg_list.append(1 if abs(truth.true_allele1_diff - truth.true_allele2_diff) > truth.motif_length else 0)

    return _compute_metrics_arrays(
        np.array(pred_diffs_list), np.array(true_diffs_list),
        np.array(motif_lens_list), np.array(pred_zyg_list), np.array(true_zyg_list),
    )


def _compute_metrics_arrays(pred_diffs, true_diffs, motif_lens, pred_zyg, true_zyg):
    """Compute metrics from numpy arrays."""
    n = pred_diffs.shape[0]
    if n == 0:
        return None
    errors = np.abs(pred_diffs - true_diffs)
    flat_errors = errors.flatten()
    motif_expanded = np.stack([motif_lens, motif_lens], axis=1)
    motif_errors = errors / np.maximum(motif_expanded, 1)

    true_flat = true_diffs.flatten()
    pred_flat = pred_diffs.flatten()
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    return {
        "n": n,
        "exact": float(np.all(errors < 0.5, axis=1).mean()),
        "w1bp": float((flat_errors <= 1.0).mean()),
        "w1motif": float((motif_errors.flatten() <= 1.0).mean()),
        "w5bp": float((flat_errors <= 5.0).mean()),
        "mae": float(flat_errors.mean()),
        "median_ae": float(np.median(flat_errors)),
        "r2": float(r2),
        "zyg_acc": float((pred_zyg == true_zyg).mean()),
        "geno_conc": float(np.all(motif_errors <= 1.0, axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

def stratify_pairs(pairs, key_fn):
    """Split pairs into bins using key_fn(pair) -> bin_name."""
    bins = defaultdict(list)
    for p in pairs:
        bins[key_fn(p)].append(p)
    return dict(bins)


def mosaictr_motif_bin(pair):
    _, truth = pair
    return _motif_period_bin(truth.motif_length)


def mosaictr_length_bin(pair):
    pred, truth = pair
    return _repeat_length_bin(truth.end - truth.start)


def mosaictr_coverage_bin(pair):
    pred, truth = pair
    return _coverage_bin(pred.n_reads)


def mosaictr_variant_bin(pair):
    _, truth = pair
    return "TP" if truth.is_variant else "TN"


def tool_motif_bin(pair):
    return _motif_period_bin(pair[2].motif_length)


def tool_length_bin(pair):
    return _repeat_length_bin(pair[2].end - pair[2].start)


def tool_variant_bin(pair):
    return "TP" if pair[2].is_variant else "TN"


# ---------------------------------------------------------------------------
# Disease loci analysis
# ---------------------------------------------------------------------------

def check_disease_loci(ht_pairs, lt_pairs, trgt_pairs):
    """Check how tools perform on STRchive disease-associated loci."""
    # Build truth lookups
    ht_by_truth = {}
    for pred, truth in ht_pairs:
        ref_size = pred.end - pred.start
        pd1, pd2 = sorted([pred.allele1_size - ref_size, pred.allele2_size - ref_size])
        ht_by_truth[(truth.chrom, truth.start, truth.end)] = (pd1, pd2, truth, pred)

    lt_by_truth = {}
    for d1, d2, truth in lt_pairs:
        lt_by_truth[(truth.chrom, truth.start, truth.end)] = (d1, d2, truth)

    trgt_by_truth = {}
    for d1, d2, truth in trgt_pairs:
        trgt_by_truth[(truth.chrom, truth.start, truth.end)] = (d1, d2, truth)

    results = []
    for chrom, start, end, gene, motif, disease in STRCHIVE_LOCI:
        # Find matching truth locus (within 100bp)
        ht_match = lt_match = trgt_match = None
        for key in ht_by_truth:
            if key[0] == chrom and abs(key[1] - start) < 100:
                ht_match = ht_by_truth[key]
                break
        for key in lt_by_truth:
            if key[0] == chrom and abs(key[1] - start) < 100:
                lt_match = lt_by_truth[key]
                break
        for key in trgt_by_truth:
            if key[0] == chrom and abs(key[1] - start) < 100:
                trgt_match = trgt_by_truth[key]
                break

        entry = {
            "gene": gene, "disease": disease, "motif": motif,
            "chrom": chrom, "start": start, "end": end,
        }
        if ht_match:
            pd1, pd2, truth, pred = ht_match
            entry["ht_d1"] = pd1
            entry["ht_d2"] = pd2
            entry["true_d1"] = truth.true_allele1_diff
            entry["true_d2"] = truth.true_allele2_diff
            entry["ht_mae"] = (abs(pd1 - truth.true_allele1_diff) +
                               abs(pd2 - truth.true_allele2_diff)) / 2
            entry["ht_conf"] = pred.confidence
            entry["ht_nreads"] = pred.n_reads
        if lt_match:
            d1, d2, truth = lt_match
            entry["lt_d1"] = d1
            entry["lt_d2"] = d2
            entry["lt_mae"] = (abs(d1 - truth.true_allele1_diff) +
                               abs(d2 - truth.true_allele2_diff)) / 2
        if trgt_match:
            d1, d2, truth = trgt_match
            entry["trgt_d1"] = d1
            entry["trgt_d2"] = d2
            entry["trgt_mae"] = (abs(d1 - truth.true_allele1_diff) +
                                 abs(d2 - truth.true_allele2_diff)) / 2
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

TABLE_HEADER = "{:<25s} {:>6s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
    "Tool", "n", "Exact", "<=1bp", "<=1mu", "MAE", "MedAE", "ZygAcc", "GenConc")
TABLE_SEP = "-" * len(TABLE_HEADER)


def fmt_metric_row(name, m):
    if m is None:
        return f"{name:<25s}   (no data)"
    return "{:<25s} {:>6d} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.2f} {:>6.2f} {:>6.1%} {:>6.1%}".format(
        name, m['n'], m['exact'], m['w1bp'], m['w1motif'],
        m['mae'], m['median_ae'], m['zyg_acc'], m['geno_conc'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genome-wide TR genotyping benchmark")
    parser.add_argument("--mosaictr", required=True, help="MosaicTR v4 BED output")
    parser.add_argument("--mosaictr-v3", default=None, help="MosaicTR v3 BED output (optional)")
    parser.add_argument("--longtr-vcf", default=LONGTR_VCF, help="LongTR VCF")
    parser.add_argument("--trgt-vcf", default=TRGT_VCF, help="TRGT VCF")
    parser.add_argument("--truth", default=TRUTH_BED, help="GIAB Tier1 BED")
    parser.add_argument("--catalog", default=CATALOG_BED, help="Adotto catalog BED")
    parser.add_argument("--output", required=True, help="Output report file")
    parser.add_argument("--chroms", default=None,
                        help="Comma-separated chromosomes (default: all)")
    args = parser.parse_args()

    chrom_set = set(args.chroms.split(",")) if args.chroms else None

    lines = []
    def w(s=""):
        lines.append(s)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading truth and catalog...")
    tier1 = load_tier1_bed(args.truth, chroms=chrom_set)
    catalog = load_adotto_catalog(args.catalog, chroms=chrom_set)
    logger.info("Tier1: %d, Catalog: %d", len(tier1), len(catalog))

    logger.info("Loading MosaicTR v4...")
    ht_preds = load_predictions(args.mosaictr)
    logger.info("MosaicTR v4: %d predictions", len(ht_preds))

    ht_v3_preds = None
    if args.mosaictr_v3:
        logger.info("Loading MosaicTR v3...")
        ht_v3_preds = load_predictions(args.mosaictr_v3)
        logger.info("MosaicTR v3: %d predictions", len(ht_v3_preds))

    logger.info("Parsing LongTR VCF...")
    longtr_dict = parse_longtr_vcf(args.longtr_vcf, chroms=chrom_set)

    logger.info("Parsing TRGT VCF...")
    trgt_dict = parse_trgt_vcf(args.trgt_vcf, chroms=chrom_set)

    # ── Match to truth ────────────────────────────────────────────────────
    logger.info("Matching to truth...")
    ht_pairs = match_preds_to_truth(ht_preds, tier1, catalog)
    lt_pairs = match_tool_to_truth(longtr_dict, tier1, catalog)
    trgt_pairs = match_tool_to_truth(trgt_dict, tier1, catalog)
    logger.info("MosaicTR matched: %d", len(ht_pairs))
    logger.info("LongTR matched: %d", len(lt_pairs))
    logger.info("TRGT matched: %d", len(trgt_pairs))

    ht_v3_pairs = None
    if ht_v3_preds:
        ht_v3_pairs = match_preds_to_truth(ht_v3_preds, tier1, catalog)
        logger.info("MosaicTR v3 matched: %d", len(ht_v3_pairs))

    # ── Report header ─────────────────────────────────────────────────────
    chrom_label = args.chroms if args.chroms else "all"
    w("=" * 80)
    w("  GENOME-WIDE TR GENOTYPING BENCHMARK")
    w(f"  Chromosomes: {chrom_label}")
    w(f"  Truth: GIAB Tier1 ({len(tier1)} loci)")
    w("=" * 80)

    n_tp = sum(1 for t in tier1 if t.is_variant)
    n_tn = len(tier1) - n_tp
    w(f"\n  Tier1: {len(tier1)} total ({n_tp} TP, {n_tn} TN)")
    w(f"  MosaicTR v4 matched: {len(ht_pairs)}")
    if ht_v3_pairs:
        w(f"  MosaicTR v3 matched: {len(ht_v3_pairs)}")
    w(f"  LongTR matched:    {len(lt_pairs)}")
    w(f"  TRGT matched:      {len(trgt_pairs)}")

    # ── Compute metrics ───────────────────────────────────────────────────
    ht_metrics = compute_mosaictr_metrics(ht_pairs)
    lt_metrics = compute_tool_metrics(lt_pairs)
    trgt_metrics = compute_tool_metrics(trgt_pairs)
    ht_v3_metrics = compute_mosaictr_metrics(ht_v3_pairs) if ht_v3_pairs else None

    # ── Overall table ─────────────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  OVERALL (all matched loci per tool)")
    w(f"{'='*80}\n")
    w(TABLE_HEADER)
    w(TABLE_SEP)
    w(fmt_metric_row("MosaicTR v4", ht_metrics))
    if ht_v3_metrics:
        w(fmt_metric_row("MosaicTR v3", ht_v3_metrics))
    w(fmt_metric_row("LongTR", lt_metrics))
    w(fmt_metric_row("TRGT", trgt_metrics))

    # ── By variant type ───────────────────────────────────────────────────
    for vtype in ["TP", "TN"]:
        ht_sub = [(p, t) for p, t in ht_pairs if mosaictr_variant_bin((p, t)) == vtype]
        lt_sub = [p for p in lt_pairs if tool_variant_bin(p) == vtype]
        trgt_sub = [p for p in trgt_pairs if tool_variant_bin(p) == vtype]
        ht_v3_sub = [(p, t) for p, t in (ht_v3_pairs or []) if mosaictr_variant_bin((p, t)) == vtype]

        w(f"\n{'='*80}")
        w(f"  {vtype} (variant)" if vtype == "TP" else f"  {vtype} (reference)")
        w(f"{'='*80}\n")
        w(TABLE_HEADER)
        w(TABLE_SEP)
        w(fmt_metric_row("MosaicTR v4", compute_mosaictr_metrics(ht_sub)))
        if ht_v3_sub:
            w(fmt_metric_row("MosaicTR v3", compute_mosaictr_metrics(ht_v3_sub)))
        w(fmt_metric_row("LongTR", compute_tool_metrics(lt_sub)))
        w(fmt_metric_row("TRGT", compute_tool_metrics(trgt_sub)))

    # ── By motif period ───────────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  BY MOTIF PERIOD")
    w(f"{'='*80}")
    motif_bins = ["homopolymer", "dinucleotide", "STR_3bp", "STR_4bp",
                  "STR_5bp", "STR_6bp", "VNTR_7+"]
    ht_by_motif = stratify_pairs(ht_pairs, mosaictr_motif_bin)
    lt_by_motif = stratify_pairs(lt_pairs, tool_motif_bin)
    trgt_by_motif = stratify_pairs(trgt_pairs, tool_motif_bin)

    for mb in motif_bins:
        ht_sub = ht_by_motif.get(mb, [])
        lt_sub = lt_by_motif.get(mb, [])
        trgt_sub = trgt_by_motif.get(mb, [])
        if not ht_sub and not lt_sub and not trgt_sub:
            continue
        w(f"\n  --- {mb} ---")
        w(f"  {TABLE_HEADER}")
        w(f"  {TABLE_SEP}")
        w(f"  {fmt_metric_row('MosaicTR v4', compute_mosaictr_metrics(ht_sub))}")
        w(f"  {fmt_metric_row('LongTR', compute_tool_metrics(lt_sub))}")
        w(f"  {fmt_metric_row('TRGT', compute_tool_metrics(trgt_sub))}")

    # ── By repeat length ──────────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  BY REPEAT LENGTH")
    w(f"{'='*80}")
    len_bins = ["<100bp", "100-500bp", "500-1000bp", ">1000bp"]
    ht_by_len = stratify_pairs(ht_pairs, mosaictr_length_bin)
    lt_by_len = stratify_pairs(lt_pairs, tool_length_bin)
    trgt_by_len = stratify_pairs(trgt_pairs, tool_length_bin)

    for lb in len_bins:
        ht_sub = ht_by_len.get(lb, [])
        lt_sub = lt_by_len.get(lb, [])
        trgt_sub = trgt_by_len.get(lb, [])
        if not ht_sub and not lt_sub and not trgt_sub:
            continue
        w(f"\n  --- {lb} ---")
        w(f"  {TABLE_HEADER}")
        w(f"  {TABLE_SEP}")
        w(f"  {fmt_metric_row('MosaicTR v4', compute_mosaictr_metrics(ht_sub))}")
        w(f"  {fmt_metric_row('LongTR', compute_tool_metrics(lt_sub))}")
        w(f"  {fmt_metric_row('TRGT', compute_tool_metrics(trgt_sub))}")

    # ── By coverage ───────────────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  BY COVERAGE (MosaicTR only)")
    w(f"{'='*80}")
    cov_bins = ["<15x", "15-30x", ">30x"]
    ht_by_cov = stratify_pairs(ht_pairs, mosaictr_coverage_bin)
    for cb in cov_bins:
        sub = ht_by_cov.get(cb, [])
        if not sub:
            continue
        m = compute_mosaictr_metrics(sub)
        w(f"\n  {cb}: n={m['n']}, MAE={m['mae']:.3f}, Exact={m['exact']:.1%}, "
          f"ZygAcc={m['zyg_acc']:.1%}, GenConc={m['geno_conc']:.1%}")

    # ── Disease loci ──────────────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  DISEASE-ASSOCIATED LOCI (STRchive)")
    w(f"{'='*80}")

    disease_results = check_disease_loci(ht_pairs, lt_pairs, trgt_pairs)
    w(f"\n  {'Gene':<10s} {'Disease':<25s} {'Motif':<8s} "
      f"{'True d1':>8s} {'True d2':>8s} {'HT MAE':>7s} {'LT MAE':>7s} {'TR MAE':>7s} "
      f"{'HT conf':>7s} {'HT nrd':>6s}")
    w("  " + "-" * 100)

    for entry in disease_results:
        td1 = entry.get("true_d1", ".")
        td2 = entry.get("true_d2", ".")
        ht_mae = entry.get("ht_mae", ".")
        lt_mae = entry.get("lt_mae", ".")
        trgt_mae = entry.get("trgt_mae", ".")
        ht_conf = entry.get("ht_conf", ".")
        ht_nrd = entry.get("ht_nreads", ".")

        td1_s = f"{td1:>8.1f}" if isinstance(td1, (int, float)) else f"{td1:>8s}"
        td2_s = f"{td2:>8.1f}" if isinstance(td2, (int, float)) else f"{td2:>8s}"
        ht_s = f"{ht_mae:>7.1f}" if isinstance(ht_mae, (int, float)) else f"{ht_mae:>7s}"
        lt_s = f"{lt_mae:>7.1f}" if isinstance(lt_mae, (int, float)) else f"{lt_mae:>7s}"
        tr_s = f"{trgt_mae:>7.1f}" if isinstance(trgt_mae, (int, float)) else f"{trgt_mae:>7s}"
        cf_s = f"{ht_conf:>7.2f}" if isinstance(ht_conf, (int, float)) else f"{ht_conf:>7s}"
        nr_s = f"{ht_nrd:>6d}" if isinstance(ht_nrd, int) else f"{ht_nrd:>6s}"

        w(f"  {entry['gene']:<10s} {entry['disease']:<25s} {entry['motif']:<8s} "
          f"{td1_s} {td2_s} {ht_s} {lt_s} {tr_s} {cf_s} {nr_s}")

    n_found = sum(1 for e in disease_results if "ht_mae" in e)
    w(f"\n  Disease loci in Tier1: {n_found}/{len(STRCHIVE_LOCI)}")

    # ── Compact paper table ───────────────────────────────────────────────
    w(f"\n{'='*80}")
    w("  PAPER TABLE (compact)")
    w(f"{'='*80}")
    w(f"\n  {'Tool':<15s} {'n':>6s} {'Exact%':>7s} {'<=1bp':>7s} {'MAE':>7s} "
      f"{'Zyg%':>7s} {'GenConc':>7s}")
    w("  " + "-" * 55)
    for name, m in [("MosaicTR v4", ht_metrics), ("LongTR", lt_metrics), ("TRGT", trgt_metrics)]:
        if m:
            w(f"  {name:<15s} {m['n']:>6d} {m['exact']:>6.1%} {m['w1bp']:>6.1%} "
              f"{m['mae']:>6.2f} {m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")

    # TP only
    w(f"\n  TP (variant) loci:")
    w(f"  {'Tool':<15s} {'n':>6s} {'Exact%':>7s} {'<=1bp':>7s} {'MAE':>7s} "
      f"{'Zyg%':>7s} {'GenConc':>7s}")
    w("  " + "-" * 55)
    for name, pairs_list, is_ht in [
        ("MosaicTR v4", ht_pairs, True),
        ("LongTR", lt_pairs, False),
        ("TRGT", trgt_pairs, False),
    ]:
        if is_ht:
            sub = [(p, t) for p, t in pairs_list if t.is_variant]
            m = compute_mosaictr_metrics(sub)
        else:
            sub = [p for p in pairs_list if p[2].is_variant]
            m = compute_tool_metrics(sub)
        if m:
            w(f"  {name:<15s} {m['n']:>6d} {m['exact']:>6.1%} {m['w1bp']:>6.1%} "
              f"{m['mae']:>6.2f} {m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")

    # ── Write output ──────────────────────────────────────────────────────
    report = "\n".join(lines)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", output_path)
    print(report)


if __name__ == "__main__":
    main()
