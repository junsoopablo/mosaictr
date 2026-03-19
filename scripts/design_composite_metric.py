#!/usr/bin/env python3
"""Design and evaluate composite instability metrics.

Takes the strengths of individual metrics:
  - IQR: best classification (AUC)
  - CV: best outlier robustness
  - MAD: best linearity + balance
  - SD: good linearity

Candidates:
  1. Trimmed-IQR: MAD-based outlier trim → IQR/motif
     (IQR classification + MAD robustness)
  2. Trimmed-CV: MAD-based outlier trim → CV
     (CV robustness + MAD trim stabilization)
  3. Winsorized-SD: Winsorize at 5th/95th → SD/motif
     (SD linearity + bounded tail sensitivity)
  4. Robust Dispersion Index (RDI):
     RDI = (IQR * (1 - 2*outlier_fraction)) / motif
     (IQR weighted by data quality)
  5. Biweight midvariance / motif
     (robust scale estimator from astronomy, 95% efficiency)

Disease models from published literature (Handsaker 2025, Higham 2013,
Morales 2012, Dischler 2025, Hause 2016).

Usage:
  python scripts/design_composite_metric.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ARCSINH_HALF = float(np.arcsinh(0.5))


# ============================================================================
# Disease models (from compare_metrics_realistic.py)
# ============================================================================

class DiseaseModel:
    def __init__(self, name, motif_len, normal_bp, expanded_bp,
                 expansion_bias, tail_weight):
        self.name = name
        self.motif_len = motif_len
        self.normal_bp = normal_bp
        self.expanded_bp = expanded_bp
        self.expansion_bias = expansion_bias
        self.tail_weight = tail_weight

MODELS = {
    "HD_blood": DiseaseModel("HD (blood)", 3, 60, 132, 0.82, 0.05),
    "HD_brain": DiseaseModel("HD (brain)", 3, 60, 132, 0.82, 0.15),
    "DM1":     DiseaseModel("DM1 (blood)", 3, 45, 600, 0.60, 0.05),
    "FXS":     DiseaseModel("FXS (blood)", 3, 90, 900, 0.55, 0.10),
    "MSI":     DiseaseModel("MSI (cancer)", 1, 20, 20, 0.50, 0.00),
}


def _somatic_deviations(n, motif_len, level, expansion_bias, tail_weight, rng):
    if level < 0.01:
        return np.zeros(n)
    target_mad_bp = level * motif_len
    body_scale = target_mad_bp / _ARCSINH_HALF
    n_tail = int(n * tail_weight)
    n_body = n - n_tail
    body = rng.exponential(body_scale, n_body)
    tail = rng.exponential(body_scale * 5, n_tail) if n_tail > 0 else np.array([])
    devs = np.concatenate([body, tail])
    rng.shuffle(devs)
    dirs = np.where(rng.random(n) < expansion_bias, 1.0, -1.0)
    return dirs * devs


def gen_disease(model, n, level, rng, noise=0.5):
    devs = _somatic_deviations(
        n, model.motif_len, level, model.expansion_bias, model.tail_weight, rng)
    return model.expanded_bp + devs + rng.normal(0, noise, n)


def gen_stable(size, n, rng, noise=0.5):
    return rng.normal(size, noise, n)


def add_outliers(sizes, motif_len, rng, frac=0.1):
    result = sizes.copy()
    n_out = max(1, int(len(result) * frac))
    idx = rng.choice(len(result), n_out, replace=False)
    result[idx] = np.median(result) + rng.uniform(10, 50, n_out) * motif_len
    return result


# ============================================================================
# Base metrics
# ============================================================================

def m_mad(s, ml): return np.median(np.abs(s - np.median(s))) / ml
def m_cv(s, ml):
    mu = np.mean(s)
    return np.std(s) / abs(mu) if mu != 0 else 0.0
def m_iqr(s, ml): return (np.percentile(s, 75) - np.percentile(s, 25)) / ml
def m_sd(s, ml): return np.std(s) / ml


# ============================================================================
# Composite metrics
# ============================================================================

def m_trimmed_iqr(sizes, motif_len):
    """MAD-based outlier trim → IQR/motif.
    Combines: MAD robustness (trim) + IQR classification power."""
    med = np.median(sizes)
    mad = np.median(np.abs(sizes - med))
    if mad == 0:
        mad = 0.5  # minimum for trimming threshold
    threshold = med + 5 * mad
    trimmed = sizes[(sizes >= med - 5 * mad) & (sizes <= threshold)]
    if len(trimmed) < 3:
        trimmed = sizes
    q75, q25 = np.percentile(trimmed, [75, 25])
    return (q75 - q25) / motif_len


def m_trimmed_sd(sizes, motif_len):
    """MAD-based outlier trim → SD/motif.
    Combines: MAD robustness + SD sensitivity."""
    med = np.median(sizes)
    mad = np.median(np.abs(sizes - med))
    if mad == 0:
        mad = 0.5
    trimmed = sizes[np.abs(sizes - med) <= 5 * mad]
    if len(trimmed) < 3:
        trimmed = sizes
    return np.std(trimmed) / motif_len


def m_winsorized_sd(sizes, motif_len):
    """Winsorize at 5th/95th percentile → SD/motif.
    Bounds tail influence while preserving sensitivity."""
    p5, p95 = np.percentile(sizes, [5, 95])
    clipped = np.clip(sizes, p5, p95)
    return np.std(clipped) / motif_len


def m_biweight(sizes, motif_len):
    """Biweight midvariance / motif.
    Robust scale estimator, 95% asymptotic efficiency at Gaussian,
    highly resistant to outliers (breakdown point 50%)."""
    med = np.median(sizes)
    mad = np.median(np.abs(sizes - med))
    if mad < 0.01:
        return 0.0
    u = (sizes - med) / (9 * mad)
    mask = np.abs(u) < 1
    if np.sum(mask) < 3:
        return np.std(sizes) / motif_len
    n = len(sizes)
    d = sizes[mask] - med
    u_m = u[mask]
    numer = n * np.sum(d**2 * (1 - u_m**2)**4)
    denom = np.sum((1 - u_m**2) * (1 - 5*u_m**2))
    denom = abs(denom)
    if denom < 1e-12:
        return np.std(sizes) / motif_len
    bwmv = np.sqrt(numer / denom)
    return bwmv / motif_len


def m_gini_mad(sizes, motif_len):
    """Gini's mean difference / motif.
    Average absolute difference between all pairs of observations.
    More efficient than MAD (asymptotic efficiency 98% vs 37% for MAD at Gaussian)
    while still robust to outliers."""
    n = len(sizes)
    if n < 2:
        return 0.0
    # Efficient computation via sorted order statistics
    sorted_s = np.sort(sizes)
    # Gini = (2 * sum(i * x_i) / (n*(n-1))) - (n+1)/(n-1) * mean
    # Simplified: mean of all |x_i - x_j|
    # Use vectorized approach for moderate n
    if n <= 200:
        diffs = np.abs(sizes[:, None] - sizes[None, :])
        gini = np.mean(diffs) / 2
    else:
        # Efficient O(n log n) via sorted array
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * sorted_s) / n - (n + 1) * np.mean(sorted_s)) / n
    return gini / motif_len


def m_qn(sizes, motif_len):
    """Qn estimator / motif (Rousseeuw & Croux 1993).
    First quartile of |x_i - x_j| for all pairs.
    Breakdown point 50%, Gaussian efficiency 82%."""
    n = len(sizes)
    if n < 4:
        return np.std(sizes) / motif_len
    if n <= 200:
        diffs = np.abs(sizes[:, None] - sizes[None, :])
        # Extract upper triangle
        upper = diffs[np.triu_indices(n, k=1)]
        # Qn = 2.2219 * first quartile of pairwise distances
        qn = 2.2219 * np.percentile(upper, 25)
    else:
        # Approximate with subsample
        idx = np.random.default_rng(42).choice(n, min(n, 200), replace=False)
        sub = sizes[idx]
        diffs = np.abs(sub[:, None] - sub[None, :])
        upper = diffs[np.triu_indices(len(sub), k=1)]
        qn = 2.2219 * np.percentile(upper, 25)
    return qn / motif_len


ALL_METRICS = {
    # Base
    "MAD/motif (HII)":      m_mad,
    "CV (SD/mean)":          m_cv,
    "IQR/motif":             m_iqr,
    "SD/motif":              m_sd,
    # Composite
    "Trimmed-IQR":           m_trimmed_iqr,
    "Trimmed-SD":            m_trimmed_sd,
    "Winsorized-SD":         m_winsorized_sd,
    "Biweight/motif":        m_biweight,
    "Gini-MAD/motif":        m_gini_mad,
    "Qn/motif":              m_qn,
}


# ============================================================================
# Evaluation
# ============================================================================

def eval_dose_response(n_reps=50):
    logger.info("Evaluating dose-response linearity...")
    levels = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n = 30
    r2_all = {met: {} for met in ALL_METRICS}

    for mname, model in MODELS.items():
        for met_name, met_func in ALL_METRICS.items():
            means = []
            for level in levels:
                vals = []
                for rep in range(n_reps):
                    rng = np.random.default_rng(1000 + hash(mname) % 9999 + rep)
                    sizes = gen_disease(model, n, level, rng)
                    vals.append(met_func(sizes, model.motif_len))
                means.append(np.mean(vals))
            targets = np.array(levels)
            measured = np.array(means)
            if np.std(measured) > 0:
                r2 = float(np.corrcoef(targets, measured)[0, 1] ** 2)
            else:
                r2 = 0.0
            r2_all[met_name][mname] = r2

    return r2_all


def eval_roc(n_stable=300, n_unstable=300):
    logger.info("Evaluating ROC AUC...")
    n = 25
    auc_all = {met: {} for met in ALL_METRICS}

    for mname, model in MODELS.items():
        values = {met: [] for met in ALL_METRICS}
        labels = []
        for i in range(n_stable):
            rng = np.random.default_rng(2000 + i)
            sizes = gen_stable(model.normal_bp, n, rng)
            for met_name, met_func in ALL_METRICS.items():
                values[met_name].append(met_func(sizes, model.motif_len))
            labels.append(0)
        for i in range(n_unstable):
            rng = np.random.default_rng(3000 + i)
            level = rng.uniform(0.5, 10.0)
            sizes = gen_disease(model, n, level, rng)
            for met_name, met_func in ALL_METRICS.items():
                values[met_name].append(met_func(sizes, model.motif_len))
            labels.append(1)
        labels = np.array(labels)
        for met_name in ALL_METRICS:
            v = np.array(values[met_name])
            thr = np.linspace(v.min() - 0.01, v.max() + 0.01, 500)
            fprs, tprs = [], []
            for t in thr:
                p = (v > t).astype(int)
                tp = np.sum((p == 1) & (labels == 1))
                fp = np.sum((p == 1) & (labels == 0))
                tn = np.sum((p == 0) & (labels == 0))
                fn = np.sum((p == 0) & (labels == 1))
                tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            fprs, tprs = np.array(fprs), np.array(tprs)
            idx = np.argsort(fprs)
            auc_all[met_name][mname] = float(np.trapezoid(tprs[idx], fprs[idx]))

    return auc_all


def eval_outlier_robustness(n_reps=50):
    logger.info("Evaluating outlier robustness...")
    n = 30
    level = 3.0
    fracs = [0.0, 0.10, 0.20]
    change_all = {}

    for met_name, met_func in ALL_METRICS.items():
        changes = []
        for mname, model in MODELS.items():
            baseline_vals = []
            outlier_vals = []
            for rep in range(n_reps):
                rng = np.random.default_rng(4000 + rep)
                sizes = gen_disease(model, n, level, rng)
                baseline_vals.append(met_func(sizes, model.motif_len))

                rng2 = np.random.default_rng(4000 + rep)
                sizes2 = gen_disease(model, n, level, rng2)
                sizes2 = add_outliers(sizes2, model.motif_len, rng2, frac=0.20)
                outlier_vals.append(met_func(sizes2, model.motif_len))

            bl = np.mean(baseline_vals)
            ol = np.mean(outlier_vals)
            if bl > 0:
                changes.append(abs(ol - bl) / bl * 100)
        change_all[met_name] = np.mean(changes)

    return change_all


def eval_subtle_detection(n_reps=100):
    """Test detection at subtle instability levels near threshold (HII 0.25-1.0).
    This is the clinically relevant range for blood-based instability."""
    logger.info("Evaluating subtle instability detection...")
    n = 30
    threshold_percentile = 95  # use 95th percentile of stable as threshold

    results = {}
    for met_name, met_func in ALL_METRICS.items():
        detection_rates = {}
        for mname, model in MODELS.items():
            # Establish threshold from stable distribution
            stable_vals = []
            for i in range(500):
                rng = np.random.default_rng(5000 + i)
                sizes = gen_stable(model.expanded_bp, n, rng)
                stable_vals.append(met_func(sizes, model.motif_len))
            thr = np.percentile(stable_vals, threshold_percentile)

            # Test subtle levels
            for level in [0.25, 0.5, 0.75, 1.0]:
                detected = 0
                for rep in range(n_reps):
                    rng = np.random.default_rng(6000 + rep)
                    sizes = gen_disease(model, n, level, rng)
                    if met_func(sizes, model.motif_len) > thr:
                        detected += 1
                key = f"{mname}_level{level}"
                detection_rates[key] = detected / n_reps

        results[met_name] = detection_rates

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/instability/composite_metric")
    parser.add_argument("--n-reps", default=50, type=int)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    r2_data = eval_dose_response(n_reps=args.n_reps)
    auc_data = eval_roc()
    outlier_data = eval_outlier_robustness(n_reps=args.n_reps)
    subtle_data = eval_subtle_detection(n_reps=100)

    # --- Report ---
    lines = []
    w = lines.append
    w("=" * 100)
    w("  COMPOSITE INSTABILITY METRIC EVALUATION")
    w("=" * 100)
    w(f"\n  {'Metric':22s}  {'Avg R²':>8s}  {'Avg AUC':>8s}  {'Outlier%':>10s}  "
      f"{'Subtle Det':>10s}  {'Score':>8s}")
    w("  " + "-" * 76)

    scores = {}
    for met_name in ALL_METRICS:
        avg_r2 = np.mean(list(r2_data[met_name].values()))
        avg_auc = np.mean(list(auc_data[met_name].values()))
        outlier_pct = outlier_data[met_name]

        # Average subtle detection across all models and levels
        subtle_vals = list(subtle_data[met_name].values())
        avg_subtle = np.mean(subtle_vals)

        # Composite score: R² * 0.2 + AUC * 0.3 + (1-outlier/max) * 0.2 + subtle * 0.3
        max_out = max(outlier_data.values())
        outlier_score = 1 - outlier_pct / max_out if max_out > 0 else 1
        score = avg_r2 * 0.2 + avg_auc * 0.3 + outlier_score * 0.2 + avg_subtle * 0.3
        scores[met_name] = score

        w(f"  {met_name:22s}  {avg_r2:>8.4f}  {avg_auc:>8.4f}  {outlier_pct:>10.1f}"
          f"  {avg_subtle:>10.1%}  {score:>8.4f}")

    ranked = sorted(scores, key=scores.get, reverse=True)
    w(f"\n  RANKING:")
    for i, met in enumerate(ranked, 1):
        is_composite = met not in ["MAD/motif (HII)", "CV (SD/mean)", "IQR/motif", "SD/motif"]
        tag = " [COMPOSITE]" if is_composite else " [BASE]"
        w(f"    #{i}: {met}{tag}  (score={scores[met]:.4f})")

    # Detailed subtle detection
    w("\n" + "-" * 100)
    w("  SUBTLE INSTABILITY DETECTION RATES (at 95th percentile threshold)")
    w("-" * 100)
    w(f"\n  Top 5 metrics at level=0.5 (critical threshold region):")
    level_05_rates = {}
    for met_name in ALL_METRICS:
        rates = [v for k, v in subtle_data[met_name].items() if "level0.5" in k]
        level_05_rates[met_name] = np.mean(rates)
    for i, (met, rate) in enumerate(sorted(level_05_rates.items(), key=lambda x: -x[1])[:5], 1):
        w(f"    #{i}: {met:22s}  {rate:.1%}")

    w(f"\n  Top 5 metrics at level=0.25 (very subtle, blood-level instability):")
    level_025_rates = {}
    for met_name in ALL_METRICS:
        rates = [v for k, v in subtle_data[met_name].items() if "level0.25" in k]
        level_025_rates[met_name] = np.mean(rates)
    for i, (met, rate) in enumerate(sorted(level_025_rates.items(), key=lambda x: -x[1])[:5], 1):
        w(f"    #{i}: {met:22s}  {rate:.1%}")

    w("\n" + "=" * 100)
    w(f"  RECOMMENDED METRIC: {ranked[0]}")
    w("=" * 100)

    report = "\n".join(lines)
    report_path = str(outdir / "composite_metric_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    logger.info("Report: %s", report_path)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics_sorted = ranked[:10]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # A: Score ranking
        ax = axes[0, 0]
        score_vals = [scores[m] for m in metrics_sorted]
        base_mask = [m in ["MAD/motif (HII)", "CV (SD/mean)", "IQR/motif", "SD/motif"]
                     for m in metrics_sorted]
        bar_colors = ["#90CAF9" if b else "#4CAF50" for b in base_mask]
        ax.barh(range(len(metrics_sorted)), score_vals, color=bar_colors, edgecolor="black")
        ax.set_yticks(range(len(metrics_sorted)))
        ax.set_yticklabels(metrics_sorted, fontsize=8)
        ax.set_xlabel("Composite Score", fontsize=10)
        ax.set_title("A. Overall Ranking (blue=base, green=composite)", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        # B: R² vs AUC scatter
        ax = axes[0, 1]
        for i, met in enumerate(metrics_sorted):
            r2 = np.mean(list(r2_data[met].values()))
            auc = np.mean(list(auc_data[met].values()))
            ax.scatter(r2, auc, c=[colors[i]], s=100, zorder=5)
            ax.annotate(met.split("/")[0][:10], (r2, auc), fontsize=7,
                       textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Average R² (linearity)", fontsize=10)
        ax.set_ylabel("Average AUC (classification)", fontsize=10)
        ax.set_title("B. Linearity vs Classification", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # C: Outlier robustness
        ax = axes[1, 0]
        out_vals = [outlier_data[m] for m in metrics_sorted]
        ax.barh(range(len(metrics_sorted)), out_vals, color=bar_colors, edgecolor="black")
        ax.set_yticks(range(len(metrics_sorted)))
        ax.set_yticklabels(metrics_sorted, fontsize=8)
        ax.set_xlabel("Relative change at 20% outliers (%)", fontsize=10)
        ax.set_title("C. Outlier Sensitivity (lower = better)", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        # D: Subtle detection
        ax = axes[1, 1]
        subtle_05 = [level_05_rates[m] * 100 for m in metrics_sorted]
        subtle_025 = [level_025_rates[m] * 100 for m in metrics_sorted]
        x = np.arange(len(metrics_sorted))
        ax.barh(x - 0.2, subtle_05, 0.35, color="#1E88E5", label="Level=0.5")
        ax.barh(x + 0.2, subtle_025, 0.35, color="#FF9800", label="Level=0.25")
        ax.set_yticks(x)
        ax.set_yticklabels(metrics_sorted, fontsize=8)
        ax.set_xlabel("Detection rate (%)", fontsize=10)
        ax.set_title("D. Subtle Instability Detection", fontweight="bold")
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        fig.savefig(str(outdir / "fig_composite_metric.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
