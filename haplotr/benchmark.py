"""Benchmarking and evaluation for HaploTR.

Compares HaploTR predictions against GIAB truth and other tools.
Provides stratified analysis by motif period, repeat length, coverage, etc.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import Tier1Locus, load_tier1_bed

logger = logging.getLogger(__name__)


@dataclass
class LocusPrediction:
    """Prediction for a single locus."""
    chrom: str
    start: int
    end: int
    motif: str
    allele1_size: float
    allele2_size: float
    zygosity: str
    confidence: float
    n_reads: int


@dataclass
class LocusTruth:
    """Truth for a single locus."""
    chrom: str
    start: int
    end: int
    motif: str
    true_allele1_diff: float  # bp diff from ref
    true_allele2_diff: float
    motif_length: int
    is_variant: bool


@dataclass
class EvalMetrics:
    """Evaluation metrics for a set of loci."""
    n_loci: int = 0
    exact_match: float = 0.0        # both alleles exactly correct
    within_1bp: float = 0.0          # per-allele within 1bp
    within_1motif: float = 0.0       # per-allele within 1 motif unit
    within_5bp: float = 0.0          # per-allele within 5bp
    mae_allele: float = 0.0          # mean absolute error per allele
    r_squared: float = 0.0           # R² for allele sizes
    zygosity_accuracy: float = 0.0   # fraction correctly called hom/het
    genotype_concordance: float = 0.0  # both alleles within 1 motif unit


@dataclass
class StratifiedResults:
    """Results stratified by various categories."""
    overall: EvalMetrics = field(default_factory=EvalMetrics)
    by_motif_period: dict[str, EvalMetrics] = field(default_factory=dict)
    by_repeat_length: dict[str, EvalMetrics] = field(default_factory=dict)
    by_coverage: dict[str, EvalMetrics] = field(default_factory=dict)
    by_variant_type: dict[str, EvalMetrics] = field(default_factory=dict)


def load_predictions(pred_path: str) -> list[LocusPrediction]:
    """Load HaploTR predictions from BED output.

    Supports both 8-column (HaploTR) and 9-column (legacy DeepTR) formats.
    """
    preds = []
    with open(pred_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 8:
                continue
            try:
                if cols[4] == ".":
                    continue
                confidence = float(cols[7]) if len(cols) >= 9 else 0.0
                n_reads = int(cols[8]) if len(cols) >= 9 else int(cols[7])
                preds.append(LocusPrediction(
                    chrom=cols[0],
                    start=int(cols[1]),
                    end=int(cols[2]),
                    motif=cols[3],
                    allele1_size=float(cols[4]),
                    allele2_size=float(cols[5]),
                    zygosity=cols[6],
                    confidence=confidence,
                    n_reads=n_reads,
                ))
            except (ValueError, IndexError):
                continue
    return preds


def prepare_truth(
    tier1_loci: list[Tier1Locus],
    motif_lookup: dict[tuple[str, int, int], str],
) -> list[LocusTruth]:
    """Convert Tier1 loci to LocusTruth with motif information."""
    truths = []
    for locus in tier1_loci:
        key = (locus.chrom, locus.start, locus.end)
        motif = motif_lookup.get(key, "")
        if not motif:
            continue
        motif_len = len(motif)

        # Negate and sort: Tier1 stores ref-allele (positive=deletion),
        # convert to allele-ref (positive=expansion) to match predictions.
        d1, d2 = sorted([-locus.hap1_diff_bp, -locus.hap2_diff_bp])

        truths.append(LocusTruth(
            chrom=locus.chrom,
            start=locus.start,
            end=locus.end,
            motif=motif,
            true_allele1_diff=d1,
            true_allele2_diff=d2,
            motif_length=motif_len,
            is_variant=locus.is_variant,
        ))
    return truths


def _motif_period_bin(motif_length: int) -> str:
    if motif_length == 1:
        return "homopolymer"
    elif motif_length == 2:
        return "dinucleotide"
    elif motif_length <= 6:
        return f"STR_{motif_length}bp"
    else:
        return "VNTR_7+"


def _repeat_length_bin(ref_size: int) -> str:
    if ref_size < 100:
        return "<100bp"
    elif ref_size < 500:
        return "100-500bp"
    elif ref_size < 1000:
        return "500-1000bp"
    else:
        return ">1000bp"


def _coverage_bin(n_reads: int) -> str:
    if n_reads < 15:
        return "<15x"
    elif n_reads <= 30:
        return "15-30x"
    else:
        return ">30x"


def compute_metrics(
    pred_diffs: np.ndarray,
    true_diffs: np.ndarray,
    motif_lens: np.ndarray,
    pred_zyg: np.ndarray,
    true_zyg: np.ndarray,
) -> EvalMetrics:
    """Compute evaluation metrics for a set of loci.

    Args:
        pred_diffs: (N, 2) predicted allele diffs from reference.
        true_diffs: (N, 2) true allele diffs from reference.
        motif_lens: (N,) motif lengths.
        pred_zyg: (N,) predicted zygosity (0=HOM, 1=HET).
        true_zyg: (N,) true zygosity (0=HOM, 1=HET).
    """
    n = pred_diffs.shape[0]
    if n == 0:
        return EvalMetrics()

    # Per-allele errors
    errors = np.abs(pred_diffs - true_diffs)  # (N, 2)
    flat_errors = errors.flatten()

    # Motif unit errors
    motif_expanded = np.stack([motif_lens, motif_lens], axis=1)
    motif_errors = errors / np.maximum(motif_expanded, 1)

    # Exact match: both alleles within 0.5bp
    exact = np.all(errors < 0.5, axis=1).mean()

    # Per-allele within thresholds
    within_1bp = (flat_errors <= 1.0).mean()
    within_1motif = (motif_errors.flatten() <= 1.0).mean()
    within_5bp = (flat_errors <= 5.0).mean()

    # MAE
    mae = flat_errors.mean()

    # R²
    true_flat = true_diffs.flatten()
    pred_flat = pred_diffs.flatten()
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    # Zygosity accuracy
    zyg_acc = (pred_zyg == true_zyg).mean()

    # Genotype concordance: both alleles within 1 motif unit
    geno_conc = np.all(motif_errors <= 1.0, axis=1).mean()

    return EvalMetrics(
        n_loci=n,
        exact_match=float(exact),
        within_1bp=float(within_1bp),
        within_1motif=float(within_1motif),
        within_5bp=float(within_5bp),
        mae_allele=float(mae),
        r_squared=float(r2),
        zygosity_accuracy=float(zyg_acc),
        genotype_concordance=float(geno_conc),
    )


def evaluate(
    predictions: list[LocusPrediction],
    truths: list[LocusTruth],
) -> StratifiedResults:
    """Evaluate predictions against truth with stratification.

    Matches predictions to truths by coordinate and computes metrics
    overall and stratified by motif period, repeat length, coverage.
    """
    # Build truth lookup
    truth_map = {}
    for t in truths:
        truth_map[(t.chrom, t.start, t.end)] = t

    # Match predictions to truth
    matched_pred_diffs = []
    matched_true_diffs = []
    matched_motif_lens = []
    matched_pred_zyg = []
    matched_true_zyg = []
    matched_meta = []  # (motif_length, ref_size, n_reads, is_variant)

    for pred in predictions:
        key = (pred.chrom, pred.start, pred.end)
        truth = truth_map.get(key)
        if truth is None:
            continue

        ref_size = pred.end - pred.start
        pred_d1 = pred.allele1_size - ref_size
        pred_d2 = pred.allele2_size - ref_size
        pred_d1, pred_d2 = sorted([pred_d1, pred_d2])

        matched_pred_diffs.append([pred_d1, pred_d2])
        matched_true_diffs.append([truth.true_allele1_diff, truth.true_allele2_diff])
        matched_motif_lens.append(truth.motif_length)

        pred_is_het = 1 if pred.zygosity == "HET" else 0
        true_is_het = 1 if abs(truth.true_allele1_diff - truth.true_allele2_diff) > truth.motif_length else 0
        matched_pred_zyg.append(pred_is_het)
        matched_true_zyg.append(true_is_het)

        matched_meta.append((truth.motif_length, ref_size, pred.n_reads, truth.is_variant))

    if not matched_pred_diffs:
        logger.warning("No matched loci found")
        return StratifiedResults()

    pred_diffs = np.array(matched_pred_diffs)
    true_diffs = np.array(matched_true_diffs)
    motif_lens = np.array(matched_motif_lens)
    pred_zyg = np.array(matched_pred_zyg)
    true_zyg = np.array(matched_true_zyg)

    logger.info("Matched %d loci for evaluation", len(pred_diffs))

    # Overall metrics
    results = StratifiedResults()
    results.overall = compute_metrics(pred_diffs, true_diffs, motif_lens, pred_zyg, true_zyg)

    # Stratify
    for i, (ml, rs, nr, iv) in enumerate(matched_meta):
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
            bin_dict[bin_name]["pred"].append(matched_pred_diffs[i])
            bin_dict[bin_name]["true"].append(matched_true_diffs[i])
            bin_dict[bin_name]["ml"].append(ml)
            bin_dict[bin_name]["pz"].append(matched_pred_zyg[i])
            bin_dict[bin_name]["tz"].append(matched_true_zyg[i])

    # Compute per-stratum metrics
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


def format_results(results: StratifiedResults) -> str:
    """Format evaluation results as a readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("HaploTR Evaluation Results")
    lines.append("=" * 70)

    def _fmt_metrics(m: EvalMetrics, label: str = "") -> list[str]:
        out = []
        if label:
            out.append(f"\n--- {label} (n={m.n_loci}) ---")
        out.append(f"  Exact match:           {m.exact_match:.4f}")
        out.append(f"  Within 1bp (allele):   {m.within_1bp:.4f}")
        out.append(f"  Within 1 motif unit:   {m.within_1motif:.4f}")
        out.append(f"  Within 5bp (allele):   {m.within_5bp:.4f}")
        out.append(f"  MAE (allele, bp):      {m.mae_allele:.2f}")
        out.append(f"  R²:                    {m.r_squared:.4f}")
        out.append(f"  Zygosity accuracy:     {m.zygosity_accuracy:.4f}")
        out.append(f"  Genotype concordance:  {m.genotype_concordance:.4f}")
        return out

    lines.extend(_fmt_metrics(results.overall, "OVERALL"))

    for title, stratum_dict in [
        ("By Motif Period", results.by_motif_period),
        ("By Repeat Length", results.by_repeat_length),
        ("By Coverage", results.by_coverage),
        ("By Variant Type", results.by_variant_type),
    ]:
        lines.append(f"\n{'=' * 70}")
        lines.append(title)
        lines.append("=" * 70)
        for key in sorted(stratum_dict.keys()):
            if isinstance(stratum_dict[key], EvalMetrics):
                lines.extend(_fmt_metrics(stratum_dict[key], key))

    return "\n".join(lines)
