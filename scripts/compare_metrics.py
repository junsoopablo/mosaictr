#!/usr/bin/env python3
"""Compare instability metrics: MAD/motif (HII), CV, IQR/motif, SD/motif.

Simulates expansion and contraction scenarios at various instability levels,
then evaluates each metric on:
  1. Dose-response linearity (R²)
  2. ROC AUC for stable vs unstable classification
  3. Robustness to outliers
  4. Sensitivity to contraction vs expansion asymmetry

Outputs:
  - metric_comparison_report.txt
  - fig_metric_comparison.png (4-panel figure)

Usage:
  python scripts/compare_metrics.py [--output-dir output/instability/metric_comparison]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric definitions (all operate on raw allele sizes for one haplotype)
# ---------------------------------------------------------------------------

def metric_hii(sizes: np.ndarray, motif_len: int) -> float:
    """HII = MAD / motif_len (MosaicTR's metric)."""
    med = np.median(sizes)
    mad = np.median(np.abs(sizes - med))
    return mad / motif_len


def metric_cv(sizes: np.ndarray, motif_len: int) -> float:
    """CV = std / mean (Owl's metric)."""
    m = np.mean(sizes)
    if m == 0:
        return 0.0
    return np.std(sizes) / abs(m)


def metric_iqr(sizes: np.ndarray, motif_len: int) -> float:
    """IQR / motif_len — interquartile range normalized by motif."""
    q75, q25 = np.percentile(sizes, [75, 25])
    return (q75 - q25) / motif_len


def metric_sd(sizes: np.ndarray, motif_len: int) -> float:
    """SD / motif_len — standard deviation normalized by motif."""
    return np.std(sizes) / motif_len


def metric_entropy(sizes: np.ndarray, motif_len: int) -> float:
    """Shannon entropy of allele size distribution (binned by motif unit)."""
    # Bin sizes into motif-unit bins
    binned = np.round((sizes - np.median(sizes)) / motif_len).astype(int)
    unique, counts = np.unique(binned, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy


METRICS = {
    "HII (MAD/motif)": metric_hii,
    "CV (SD/mean)": metric_cv,
    "IQR/motif": metric_iqr,
    "SD/motif": metric_sd,
    "Entropy": metric_entropy,
}


# ---------------------------------------------------------------------------
# Read generators (same model as simulate_instability.py)
# ---------------------------------------------------------------------------

_ARCSINH_HALF = float(np.arcsinh(0.5))  # ≈ 0.4812


def generate_expansion_reads(
    median: float, n: int, motif_len: int, target_hii: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Expansion-biased exponential model (somatic expansion dominant)."""
    target_mad = target_hii * motif_len
    if target_mad < 0.01:
        return rng.normal(median, 0.5, n)
    scale = target_mad / _ARCSINH_HALF
    expansions = rng.exponential(scale, n)
    noise = rng.normal(0, 0.5, n)
    return median + expansions + noise


def generate_contraction_reads(
    median: float, n: int, motif_len: int, target_hii: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Contraction-biased model (deletions dominant, e.g., MMR-deficient tumors)."""
    target_mad = target_hii * motif_len
    if target_mad < 0.01:
        return rng.normal(median, 0.5, n)
    scale = target_mad / _ARCSINH_HALF
    contractions = rng.exponential(scale, n)
    noise = rng.normal(0, 0.5, n)
    return median - contractions + noise


def generate_bidirectional_reads(
    median: float, n: int, motif_len: int, target_hii: float,
    rng: np.random.Generator, expansion_frac: float = 0.7,
) -> np.ndarray:
    """Mixed expansion + contraction (realistic general instability)."""
    target_mad = target_hii * motif_len
    if target_mad < 0.01:
        return rng.normal(median, 0.5, n)
    scale = target_mad / _ARCSINH_HALF
    deviations = rng.exponential(scale, n)
    # Assign direction: expansion_frac go positive, rest negative
    directions = np.where(rng.random(n) < expansion_frac, 1.0, -1.0)
    noise = rng.normal(0, 0.5, n)
    return median + directions * deviations + noise


def generate_with_outliers(
    median: float, n: int, motif_len: int, target_hii: float,
    rng: np.random.Generator, outlier_frac: float = 0.1,
) -> np.ndarray:
    """Expansion reads with a fraction of extreme outliers (misaligned reads)."""
    sizes = generate_expansion_reads(median, n, motif_len, target_hii, rng)
    n_outliers = max(1, int(n * outlier_frac))
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    # Outliers: random large deviations (10-50 motif units)
    sizes[outlier_idx] = median + rng.uniform(10, 50, n_outliers) * motif_len
    return sizes


# ---------------------------------------------------------------------------
# Test 1: Dose-response linearity
# ---------------------------------------------------------------------------

def run_dose_response(n_reps: int = 30) -> dict:
    """Compare metric dose-response across expansion levels."""
    logger.info("Running dose-response comparison...")

    target_hiis = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    motif_len = 3
    n_reads = 30
    median_size = 120.0

    results = {name: [] for name in METRICS}

    for target_hii in target_hiis:
        metric_values = {name: [] for name in METRICS}
        for rep in range(n_reps):
            rng = np.random.default_rng(1000 + rep)
            sizes = generate_expansion_reads(
                median_size, n_reads, motif_len, target_hii, rng
            )
            for name, func in METRICS.items():
                metric_values[name].append(func(sizes, motif_len))

        for name in METRICS:
            vals = metric_values[name]
            results[name].append({
                "target_hii": target_hii,
                "mean": np.mean(vals),
                "std": np.std(vals),
            })

    # Compute R² for each metric
    r_squared = {}
    for name in METRICS:
        targets = np.array([r["target_hii"] for r in results[name]])
        measured = np.array([r["mean"] for r in results[name]])
        if len(targets) > 2 and np.std(targets) > 0 and np.std(measured) > 0:
            corr = np.corrcoef(targets, measured)[0, 1]
            r_squared[name] = corr ** 2
        else:
            r_squared[name] = 0.0

    return {"results": results, "r_squared": r_squared}


# ---------------------------------------------------------------------------
# Test 2: ROC AUC comparison
# ---------------------------------------------------------------------------

def run_roc_comparison(n_stable: int = 300, n_unstable: int = 300) -> dict:
    """Compare ROC AUC across metrics."""
    logger.info("Running ROC comparison...")

    motif_len = 3
    n_reads = 25
    median_size = 120.0

    all_values = {name: [] for name in METRICS}
    labels = []

    # Stable loci
    for i in range(n_stable):
        rng = np.random.default_rng(2000 + i)
        sizes = rng.normal(median_size, 0.5, n_reads)
        for name, func in METRICS.items():
            all_values[name].append(func(sizes, motif_len))
        labels.append(0)

    # Unstable loci (varying HII)
    for i in range(n_unstable):
        rng = np.random.default_rng(3000 + i)
        target_hii = rng.uniform(1.5, 10.0)
        sizes = generate_expansion_reads(
            median_size, n_reads, motif_len, target_hii, rng
        )
        for name, func in METRICS.items():
            all_values[name].append(func(sizes, motif_len))
        labels.append(1)

    labels = np.array(labels)

    # Compute AUC for each metric
    auc_results = {}
    for name in METRICS:
        values = np.array(all_values[name])
        thresholds = np.linspace(values.min(), values.max(), 500)
        fprs, tprs = [], []
        for thr in thresholds:
            predicted = (values > thr).astype(int)
            tp = np.sum((predicted == 1) & (labels == 1))
            fp = np.sum((predicted == 1) & (labels == 0))
            tn = np.sum((predicted == 0) & (labels == 0))
            fn = np.sum((predicted == 0) & (labels == 1))
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

        fprs = np.array(fprs)
        tprs = np.array(tprs)
        sort_idx = np.argsort(fprs)
        auc = float(np.trapezoid(tprs[sort_idx], fprs[sort_idx]))
        auc_results[name] = {
            "auc": auc,
            "roc_fprs": fprs[sort_idx].tolist(),
            "roc_tprs": tprs[sort_idx].tolist(),
        }

    return auc_results


# ---------------------------------------------------------------------------
# Test 3: Outlier robustness
# ---------------------------------------------------------------------------

def run_outlier_robustness(n_reps: int = 30) -> dict:
    """Compare metric stability when outliers are present."""
    logger.info("Running outlier robustness test...")

    motif_len = 3
    n_reads = 30
    median_size = 120.0
    target_hii = 3.0
    outlier_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    results = {name: [] for name in METRICS}

    for frac in outlier_fracs:
        metric_values = {name: [] for name in METRICS}
        for rep in range(n_reps):
            rng = np.random.default_rng(4000 + rep)
            if frac == 0.0:
                sizes = generate_expansion_reads(
                    median_size, n_reads, motif_len, target_hii, rng
                )
            else:
                sizes = generate_with_outliers(
                    median_size, n_reads, motif_len, target_hii, rng,
                    outlier_frac=frac,
                )
            for name, func in METRICS.items():
                metric_values[name].append(func(sizes, motif_len))

        for name in METRICS:
            vals = metric_values[name]
            results[name].append({
                "outlier_frac": frac,
                "mean": np.mean(vals),
                "std": np.std(vals),
            })

    # Compute relative change from 0% outliers
    relative_change = {}
    for name in METRICS:
        baseline = results[name][0]["mean"]
        if baseline > 0:
            changes = [
                abs(r["mean"] - baseline) / baseline * 100
                for r in results[name]
            ]
        else:
            changes = [0.0] * len(outlier_fracs)
        relative_change[name] = changes

    return {"results": results, "relative_change": relative_change,
            "outlier_fracs": outlier_fracs}


# ---------------------------------------------------------------------------
# Test 4: Expansion vs contraction sensitivity
# ---------------------------------------------------------------------------

def run_directionality_test(n_reps: int = 30) -> dict:
    """Test whether metrics distinguish expansion-dominant vs contraction-dominant."""
    logger.info("Running directionality test...")

    motif_len = 3
    n_reads = 30
    median_size = 120.0
    target_hii = 3.0

    scenarios = {
        "Expansion only": lambda m, n, ml, hii, rng: generate_expansion_reads(m, n, ml, hii, rng),
        "Contraction only": lambda m, n, ml, hii, rng: generate_contraction_reads(m, n, ml, hii, rng),
        "Bidirectional (70/30)": lambda m, n, ml, hii, rng: generate_bidirectional_reads(m, n, ml, hii, rng, 0.7),
        "Symmetric (50/50)": lambda m, n, ml, hii, rng: generate_bidirectional_reads(m, n, ml, hii, rng, 0.5),
    }

    results = {}
    for scenario_name, gen_func in scenarios.items():
        metric_values = {name: [] for name in METRICS}
        for rep in range(n_reps):
            rng = np.random.default_rng(5000 + rep)
            sizes = gen_func(median_size, n_reads, motif_len, target_hii, rng)
            for name, func in METRICS.items():
                metric_values[name].append(func(sizes, motif_len))

        results[scenario_name] = {
            name: {"mean": np.mean(vals), "std": np.std(vals)}
            for name, vals in metric_values.items()
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(dose_data, roc_data, outlier_data, direction_data, output_path):
    """Generate 2x2 comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "HII (MAD/motif)": "#4CAF50",
        "CV (SD/mean)": "#E53935",
        "IQR/motif": "#1E88E5",
        "SD/motif": "#FF9800",
        "Entropy": "#9C27B0",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Panel A: Dose-response linearity ---
    ax = axes[0, 0]
    for name in METRICS:
        data = dose_data["results"][name]
        targets = [r["target_hii"] for r in data]
        means = [r["mean"] for r in data]
        # Normalize to [0, 1] range for comparison
        max_val = max(means) if max(means) > 0 else 1
        normed = [m / max_val for m in means]
        r2 = dose_data["r_squared"][name]
        ax.plot(targets, normed, "o-", color=colors[name], linewidth=1.5,
                markersize=6, label=f"{name} (R²={r2:.3f})")
    ax.plot([0, 20], [0, 1], "k--", alpha=0.3, label="Perfect linear")
    ax.set_xlabel("Target HII (instability level)", fontsize=11)
    ax.set_ylabel("Normalized metric value", fontsize=11)
    ax.set_title("A. Dose-Response Linearity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Panel B: ROC curves ---
    ax = axes[0, 1]
    for name in METRICS:
        auc = roc_data[name]["auc"]
        fprs = roc_data[name]["roc_fprs"]
        tprs = roc_data[name]["roc_tprs"]
        ax.plot(fprs, tprs, color=colors[name], linewidth=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("B. ROC Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- Panel C: Outlier robustness ---
    ax = axes[1, 0]
    fracs = outlier_data["outlier_fracs"]
    for name in METRICS:
        changes = outlier_data["relative_change"][name]
        ax.plot([f * 100 for f in fracs], changes, "o-", color=colors[name],
                linewidth=1.5, markersize=6, label=name)
    ax.set_xlabel("Outlier fraction (%)", fontsize=11)
    ax.set_ylabel("Relative change from baseline (%)", fontsize=11)
    ax.set_title("C. Outlier Robustness (HII=3.0)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel D: Directionality sensitivity ---
    ax = axes[1, 1]
    scenario_names = list(direction_data.keys())
    x = np.arange(len(scenario_names))
    width = 0.15
    for i, name in enumerate(METRICS):
        means = [direction_data[s][name]["mean"] for s in scenario_names]
        stds = [direction_data[s][name]["std"] for s in scenario_names]
        # Normalize by expansion-only value
        exp_val = direction_data["Expansion only"][name]["mean"]
        if exp_val > 0:
            normed_means = [m / exp_val for m in means]
            normed_stds = [s / exp_val for s in stds]
        else:
            normed_means = means
            normed_stds = stds
        ax.bar(x + i * width, normed_means, width, yerr=normed_stds,
               capsize=2, color=colors[name], alpha=0.85, label=name)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([s.replace(" only", "").replace(" (", "\n(")
                        for s in scenario_names], fontsize=8)
    ax.set_ylabel("Metric value (normalized to expansion)", fontsize=11)
    ax.set_title("D. Expansion vs Contraction Sensitivity", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(dose_data, roc_data, outlier_data, direction_data, output_path):
    """Write comparison report."""
    lines = []
    w = lines.append

    w("=" * 90)
    w("  INSTABILITY METRIC COMPARISON REPORT")
    w("=" * 90)
    w(f"\n  Metrics compared: {', '.join(METRICS.keys())}")

    # 1. Dose-response
    w("\n" + "-" * 90)
    w("  1. DOSE-RESPONSE LINEARITY (R²)")
    w("-" * 90)
    for name in METRICS:
        w(f"    {name:20s}  R² = {dose_data['r_squared'][name]:.4f}")
    best = max(dose_data["r_squared"], key=dose_data["r_squared"].get)
    w(f"\n  Best linearity: {best}")

    # 2. ROC AUC
    w("\n" + "-" * 90)
    w("  2. ROC AUC (stable vs unstable classification)")
    w("-" * 90)
    for name in METRICS:
        w(f"    {name:20s}  AUC = {roc_data[name]['auc']:.4f}")
    best_auc = max(roc_data, key=lambda n: roc_data[n]["auc"])
    w(f"\n  Best AUC: {best_auc}")

    # 3. Outlier robustness
    w("\n" + "-" * 90)
    w("  3. OUTLIER ROBUSTNESS (% change at 20% outliers)")
    w("-" * 90)
    for name in METRICS:
        # Index 4 = 20% outlier fraction
        change_20 = outlier_data["relative_change"][name][4]
        w(f"    {name:20s}  {change_20:6.1f}% change")
    best_robust = min(
        METRICS.keys(),
        key=lambda n: outlier_data["relative_change"][n][4]
    )
    w(f"\n  Most robust: {best_robust}")

    # 4. Directionality
    w("\n" + "-" * 90)
    w("  4. DIRECTIONALITY SENSITIVITY")
    w("-" * 90)
    w(f"\n  {'Metric':20s}  {'Expansion':>12s}  {'Contraction':>12s}  "
      f"{'Bidir 70/30':>12s}  {'Symmetric':>12s}")
    w("  " + "-" * 72)
    for name in METRICS:
        vals = [
            direction_data[s][name]["mean"]
            for s in ["Expansion only", "Contraction only",
                       "Bidirectional (70/30)", "Symmetric (50/50)"]
        ]
        w(f"  {name:20s}  {vals[0]:12.3f}  {vals[1]:12.3f}  "
          f"{vals[2]:12.3f}  {vals[3]:12.3f}")

    w("\n  Note: A good instability metric should give similar values for")
    w("  expansion and contraction of equal magnitude (direction-agnostic).")

    # Overall assessment
    w("\n" + "=" * 90)
    w("  OVERALL ASSESSMENT")
    w("=" * 90)
    w("")
    for name in METRICS:
        r2 = dose_data["r_squared"][name]
        auc = roc_data[name]["auc"]
        change_20 = outlier_data["relative_change"][name][4]
        w(f"  {name:20s}  R²={r2:.3f}  AUC={auc:.3f}  "
          f"Outlier sensitivity={change_20:.1f}%")
    w("")
    w("=" * 90)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report saved: %s", output_path)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare instability metrics via simulation",
    )
    parser.add_argument(
        "--output-dir", default="output/instability/metric_comparison",
    )
    parser.add_argument("--n-reps", default=30, type=int)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dose_data = run_dose_response(n_reps=args.n_reps)
    roc_data = run_roc_comparison()
    outlier_data = run_outlier_robustness(n_reps=args.n_reps)
    direction_data = run_directionality_test(n_reps=args.n_reps)

    try:
        plot_comparison(
            dose_data, roc_data, outlier_data, direction_data,
            str(outdir / "fig_metric_comparison.png"),
        )
    except ImportError as e:
        logger.warning("matplotlib not available: %s", e)

    report = write_report(
        dose_data, roc_data, outlier_data, direction_data,
        str(outdir / "metric_comparison_report.txt"),
    )
    print(report)


if __name__ == "__main__":
    main()
