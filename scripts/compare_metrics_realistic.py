#!/usr/bin/env python3
"""Biologically realistic simulation of TR somatic instability.

Models somatic expansion/contraction distributions based on published data:
  - HD (CAG): 82% expansion bias, two-phase heavy tail (Handsaker 2025, Higham 2013)
  - DM1 (CTG): 60% expansion bias, right-skewed unimodal (Morales 2012, Higham 2013)
  - FXS (CGG): bidirectional broad mosaicism (Dischler 2025)
  - MSI (cancer): symmetric small indels at mononucleotide repeats (Hause 2016)

Evaluates:
  1. Metric comparison (HII, CV, IQR/motif, SD/motif) across disease models
  2. ROC for stable vs unstable classification per disease type
  3. Long-read vs short-read detection performance
  4. Dose-response linearity per disease model
  5. Outlier robustness

Usage:
  python scripts/compare_metrics_realistic.py [--output-dir output/instability/realistic_simulation]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mosaictr.genotype import ReadInfo
from mosaictr.instability import compute_instability


# ============================================================================
# Metric definitions
# ============================================================================

def metric_hii(sizes: np.ndarray, motif_len: int) -> float:
    """MAD / motif_len (MosaicTR)."""
    return np.median(np.abs(sizes - np.median(sizes))) / motif_len

def metric_cv(sizes: np.ndarray, motif_len: int) -> float:
    """SD / mean (Owl-style)."""
    m = np.mean(sizes)
    return np.std(sizes) / abs(m) if m != 0 else 0.0

def metric_iqr(sizes: np.ndarray, motif_len: int) -> float:
    """IQR / motif_len."""
    return (np.percentile(sizes, 75) - np.percentile(sizes, 25)) / motif_len

def metric_sd(sizes: np.ndarray, motif_len: int) -> float:
    """SD / motif_len."""
    return np.std(sizes) / motif_len

METRICS = {
    "HII (MAD/motif)": metric_hii,
    "CV (SD/mean)": metric_cv,
    "IQR/motif": metric_iqr,
    "SD/motif": metric_sd,
}
METRIC_COLORS = {
    "HII (MAD/motif)": "#4CAF50",
    "CV (SD/mean)": "#E53935",
    "IQR/motif": "#1E88E5",
    "SD/motif": "#FF9800",
}


# ============================================================================
# Disease-specific somatic expansion models
# ============================================================================

def _somatic_deviations(
    n: int,
    motif_len: int,
    instability_level: float,
    expansion_bias: float,
    tail_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate somatic repeat length deviations from inherited size.

    Uses a mixture of exponential (body) and Pareto (heavy tail) to model
    the right-skewed, heavy-tailed distributions observed in repeat expansion
    disorders (Handsaker 2025, Higham & Monckton 2013).

    Args:
        n: Number of reads.
        motif_len: Repeat unit length in bp.
        instability_level: Controls overall spread (analogous to target HII).
            In motif units: level=1 means ~1 motif unit MAD.
        expansion_bias: Fraction of changes that are expansions [0, 1].
            HD brain: 0.82, DM1 blood: 0.60, MSI: 0.50.
        tail_weight: Fraction of reads from heavy tail [0, 1].
            HD brain: 0.15 (two-phase), DM1: 0.05, MSI: 0.0.
        rng: Random number generator.

    Returns:
        Array of deviations in bp from inherited allele size.
    """
    if instability_level < 0.01:
        return np.zeros(n)

    target_mad_bp = instability_level * motif_len
    arcsinh_half = float(np.arcsinh(0.5))

    # Body: exponential distribution (moderate somatic changes)
    body_scale = target_mad_bp / arcsinh_half
    n_tail = int(n * tail_weight)
    n_body = n - n_tail

    # Body deviations (exponential)
    body_devs = rng.exponential(body_scale, n_body)

    # Heavy tail deviations (Pareto-like: much larger expansions)
    # Models the second phase of HD expansion (>80 CAGs, Handsaker 2025)
    if n_tail > 0:
        tail_scale = body_scale * 5  # 5x larger than body
        tail_devs = rng.exponential(tail_scale, n_tail)
    else:
        tail_devs = np.array([])

    all_devs = np.concatenate([body_devs, tail_devs])
    rng.shuffle(all_devs)

    # Assign direction: expansion (+) or contraction (-)
    directions = np.where(rng.random(n) < expansion_bias, 1.0, -1.0)
    return directions * all_devs


class DiseaseModel:
    """Parameters for a disease-specific somatic instability model."""
    def __init__(self, name: str, motif: str, motif_len: int,
                 normal_size_bp: float, expanded_size_bp: float,
                 expansion_bias: float, tail_weight: float,
                 description: str):
        self.name = name
        self.motif = motif
        self.motif_len = motif_len
        self.normal_size_bp = normal_size_bp
        self.expanded_size_bp = expanded_size_bp
        self.expansion_bias = expansion_bias
        self.tail_weight = tail_weight
        self.description = description


# Published parameters from literature
DISEASE_MODELS = {
    "HD_blood": DiseaseModel(
        "HD (blood)", "CAG", 3,
        normal_size_bp=60.0,       # ~20 CAG (normal)
        expanded_size_bp=132.0,    # ~44 CAG (pathogenic)
        expansion_bias=0.82,       # Higham & Monckton 2013
        tail_weight=0.05,          # modest tail in blood
        description="Huntington disease blood: 82% expansion bias, modest tail",
    ),
    "HD_brain": DiseaseModel(
        "HD (brain)", "CAG", 3,
        normal_size_bp=60.0,
        expanded_size_bp=132.0,
        expansion_bias=0.82,
        tail_weight=0.15,          # prominent heavy tail (Handsaker 2025)
        description="Huntington disease striatum: two-phase expansion with heavy tail",
    ),
    "DM1": DiseaseModel(
        "DM1 (blood)", "CTG", 3,
        normal_size_bp=45.0,       # ~15 CTG (normal)
        expanded_size_bp=600.0,    # ~200 CTG (mild DM1)
        expansion_bias=0.60,       # Higham & Monckton 2013
        tail_weight=0.05,
        description="Myotonic dystrophy 1 blood: 60% expansion bias, right-skewed",
    ),
    "FXS": DiseaseModel(
        "FXS (blood)", "CGG", 3,
        normal_size_bp=90.0,       # ~30 CGG (normal)
        expanded_size_bp=900.0,    # ~300 CGG (full mutation)
        expansion_bias=0.55,       # bidirectional (Dischler 2025)
        tail_weight=0.10,
        description="Fragile X full mutation: bidirectional broad mosaicism",
    ),
    "MSI": DiseaseModel(
        "MSI (cancer)", "A", 1,
        normal_size_bp=20.0,       # mononucleotide repeat
        expanded_size_bp=20.0,     # same (somatic indels)
        expansion_bias=0.50,       # symmetric (Hause 2016)
        tail_weight=0.0,
        description="Microsatellite instability: symmetric small indels",
    ),
}


def generate_disease_reads(
    model: DiseaseModel,
    n_reads: int,
    instability_level: float,
    rng: np.random.Generator,
    seq_noise_std: float = 0.5,
) -> np.ndarray:
    """Generate reads for the expanded haplotype of a disease carrier.

    Args:
        model: Disease-specific parameters.
        n_reads: Number of reads.
        instability_level: Instability in motif units (target HII).
        rng: RNG.
        seq_noise_std: Sequencing noise in bp (HiFi ~0.5, ONT ~3.0).

    Returns:
        Array of allele sizes in bp.
    """
    base = model.expanded_size_bp
    deviations = _somatic_deviations(
        n_reads, model.motif_len, instability_level,
        model.expansion_bias, model.tail_weight, rng,
    )
    seq_noise = rng.normal(0, seq_noise_std, n_reads)
    return base + deviations + seq_noise


def generate_stable_reads(
    size_bp: float, n_reads: int, rng: np.random.Generator,
    seq_noise_std: float = 0.5,
) -> np.ndarray:
    """Generate reads for a stable haplotype."""
    return rng.normal(size_bp, seq_noise_std, n_reads)


def add_short_read_artifacts(
    sizes: np.ndarray, motif_len: int, rng: np.random.Generator,
    stutter_rate: float = 0.35,
) -> np.ndarray:
    """Add PCR stutter artifacts typical of short-read sequencing.

    Published cumulative stutter rate: 30-40% for trinucleotides after
    ~25 PCR cycles (Shinde 2003, Lai & Sun 2003).
    """
    result = sizes.copy()
    stutter_mask = rng.random(len(result)) < stutter_rate
    stutter_dir = rng.choice([-1, 1], size=len(result))
    result[stutter_mask] += stutter_dir[stutter_mask] * motif_len
    return result


# ============================================================================
# Test 1: Dose-response per disease model
# ============================================================================

def run_dose_response(n_reps: int = 30) -> dict:
    """Dose-response linearity across disease models."""
    logger.info("Test 1: Dose-response per disease model...")

    instability_levels = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_reads = 30

    results = {}
    for model_name, model in DISEASE_MODELS.items():
        metric_results = {m: [] for m in METRICS}
        for level in instability_levels:
            metric_vals = {m: [] for m in METRICS}
            for rep in range(n_reps):
                rng = np.random.default_rng(1000 + hash(model_name) % 10000 + rep)
                sizes = generate_disease_reads(model, n_reads, level, rng)
                for mname, mfunc in METRICS.items():
                    metric_vals[mname].append(mfunc(sizes, model.motif_len))
            for mname in METRICS:
                metric_results[mname].append({
                    "level": level,
                    "mean": float(np.mean(metric_vals[mname])),
                    "std": float(np.std(metric_vals[mname])),
                })

        # R² per metric
        r2 = {}
        for mname in METRICS:
            targets = np.array([r["level"] for r in metric_results[mname]])
            measured = np.array([r["mean"] for r in metric_results[mname]])
            if np.std(targets) > 0 and np.std(measured) > 0:
                r2[mname] = float(np.corrcoef(targets, measured)[0, 1] ** 2)
            else:
                r2[mname] = 0.0

        results[model_name] = {
            "model": model.name,
            "description": model.description,
            "metric_results": metric_results,
            "r_squared": r2,
        }

    return results


# ============================================================================
# Test 2: ROC per disease model
# ============================================================================

def run_roc(n_stable: int = 300, n_unstable: int = 300) -> dict:
    """ROC comparison across metrics and disease models."""
    logger.info("Test 2: ROC per disease model...")

    n_reads = 25
    results = {}

    for model_name, model in DISEASE_MODELS.items():
        all_values = {m: [] for m in METRICS}
        labels = []

        # Stable loci
        for i in range(n_stable):
            rng = np.random.default_rng(2000 + i)
            sizes = generate_stable_reads(model.normal_size_bp, n_reads, rng)
            for mname, mfunc in METRICS.items():
                all_values[mname].append(mfunc(sizes, model.motif_len))
            labels.append(0)

        # Unstable loci (varying instability 0.5 to 10.0)
        for i in range(n_unstable):
            rng = np.random.default_rng(3000 + i)
            level = rng.uniform(0.5, 10.0)
            sizes = generate_disease_reads(model, n_reads, level, rng)
            for mname, mfunc in METRICS.items():
                all_values[mname].append(mfunc(sizes, model.motif_len))
            labels.append(1)

        labels = np.array(labels)
        auc_results = {}
        for mname in METRICS:
            values = np.array(all_values[mname])
            auc = _compute_auc(values, labels)
            auc_results[mname] = auc

        results[model_name] = {"model": model.name, "auc": auc_results}

    return results


def _compute_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC from metric values and binary labels."""
    thresholds = np.linspace(values.min() - 0.01, values.max() + 0.01, 500)
    fprs, tprs = [], []
    for thr in thresholds:
        pred = (values > thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    idx = np.argsort(fprs)
    return float(np.trapezoid(tprs[idx], fprs[idx]))


# ============================================================================
# Test 3: Long-read vs short-read detection
# ============================================================================

def run_lr_vs_sr(n_reps: int = 50) -> dict:
    """Compare long-read vs short-read instability detection.

    Short-read limitations:
    1. PCR stutter artifacts (35% cumulative rate)
    2. Cannot span repeats >100bp
    3. No haplotype resolution (pooled analysis)
    """
    logger.info("Test 3: Long-read vs short-read comparison...")

    n_reads = 30
    results = {}

    for model_name, model in DISEASE_MODELS.items():
        # Test across instability levels including subtle ones
        levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        lr_data = []
        sr_data = []

        for level in levels:
            lr_hiis, sr_hiis = [], []
            lr_detected, sr_detected = 0, 0

            for rep in range(n_reps):
                rng = np.random.default_rng(4000 + rep)

                # --- Long-read (HiFi): haplotype-resolved ---
                hp1_sizes = generate_stable_reads(
                    model.normal_size_bp, n_reads, rng, seq_noise_std=0.5)
                hp2_sizes = generate_disease_reads(
                    model, n_reads, level, rng, seq_noise_std=0.5)
                # Compute HII per haplotype
                hii_hp2 = metric_hii(hp2_sizes, model.motif_len)
                lr_hiis.append(hii_hp2)
                if hii_hp2 > 0.45:
                    lr_detected += 1

                # --- Short-read: pooled (no HP) + PCR stutter ---
                rng2 = np.random.default_rng(5000 + rep)
                sr_hp1 = generate_stable_reads(
                    model.normal_size_bp, n_reads, rng2, seq_noise_std=0.5)
                sr_hp2 = generate_disease_reads(
                    model, n_reads, level, rng2, seq_noise_std=0.5)

                # Can short reads span this repeat?
                repeat_size = model.expanded_size_bp
                if repeat_size > 100:
                    # Short reads can't span — only partial reads, much noisier
                    # Model: reads that partially overlap give noisy size estimates
                    sr_hp2 = rng2.normal(
                        np.median(sr_hp2),
                        max(10.0, model.motif_len * 3),
                        n_reads,
                    )

                # Add PCR stutter
                sr_hp1 = add_short_read_artifacts(sr_hp1, model.motif_len, rng2)
                sr_hp2 = add_short_read_artifacts(sr_hp2, model.motif_len, rng2)

                # Pooled analysis (no haplotype separation)
                pooled = np.concatenate([sr_hp1, sr_hp2])
                hii_pooled = metric_hii(pooled, model.motif_len)
                sr_hiis.append(hii_pooled)
                if hii_pooled > 0.45:
                    sr_detected += 1

            lr_data.append({
                "level": level,
                "mean_hii": float(np.mean(lr_hiis)),
                "std_hii": float(np.std(lr_hiis)),
                "detection_rate": lr_detected / n_reps,
            })
            sr_data.append({
                "level": level,
                "mean_hii": float(np.mean(sr_hiis)),
                "std_hii": float(np.std(sr_hiis)),
                "detection_rate": sr_detected / n_reps,
            })

        results[model_name] = {
            "model": model.name,
            "long_read": lr_data,
            "short_read": sr_data,
        }

    return results


# ============================================================================
# Test 4: Outlier robustness per disease model
# ============================================================================

def run_outlier_robustness(n_reps: int = 30) -> dict:
    """Test metric robustness to misaligned/chimeric read outliers."""
    logger.info("Test 4: Outlier robustness...")

    n_reads = 30
    outlier_fracs = [0.0, 0.05, 0.10, 0.15, 0.20]
    level = 3.0

    results = {}
    for model_name, model in DISEASE_MODELS.items():
        metric_data = {m: [] for m in METRICS}
        for frac in outlier_fracs:
            vals = {m: [] for m in METRICS}
            for rep in range(n_reps):
                rng = np.random.default_rng(6000 + rep)
                sizes = generate_disease_reads(model, n_reads, level, rng)
                # Add outliers
                if frac > 0:
                    n_out = max(1, int(n_reads * frac))
                    idx = rng.choice(n_reads, n_out, replace=False)
                    sizes[idx] = model.expanded_size_bp + rng.uniform(
                        10, 50, n_out) * model.motif_len
                for mname, mfunc in METRICS.items():
                    vals[mname].append(mfunc(sizes, model.motif_len))
            for mname in METRICS:
                metric_data[mname].append({
                    "outlier_frac": frac,
                    "mean": float(np.mean(vals[mname])),
                    "std": float(np.std(vals[mname])),
                })

        # Relative change at 20% outliers
        rel_change = {}
        for mname in METRICS:
            baseline = metric_data[mname][0]["mean"]
            val_20 = metric_data[mname][-1]["mean"]
            rel_change[mname] = abs(val_20 - baseline) / baseline * 100 if baseline > 0 else 0
        results[model_name] = {
            "model": model.name,
            "metric_data": metric_data,
            "rel_change_20pct": rel_change,
        }

    return results


# ============================================================================
# Test 5: MosaicTR pipeline (HP-tagged) vs pooled
# ============================================================================

def run_hp_vs_pooled(n_reps: int = 50) -> dict:
    """Compare HP-tagged (MosaicTR) vs pooled analysis at HET loci."""
    logger.info("Test 5: HP-tagged vs pooled analysis...")

    n_reads = 30
    results = {}

    for model_name, model in DISEASE_MODELS.items():
        levels = [0.0, 0.5, 1.0, 2.0, 5.0]
        hp_fp, pooled_fp = [], []  # false positive rates at stable level
        hp_tp, pooled_tp = [], []  # true positive rates at unstable levels

        for level in levels:
            hp_detected, pooled_detected = 0, 0
            for rep in range(n_reps):
                rng = np.random.default_rng(7000 + rep)
                hp1 = generate_stable_reads(model.normal_size_bp, n_reads, rng)
                hp2 = generate_disease_reads(model, n_reads, level, rng)

                # HP-tagged: per-haplotype HII
                reads = ([ReadInfo(allele_size=float(s), hp=1, mapq=60) for s in hp1] +
                         [ReadInfo(allele_size=float(s), hp=2, mapq=60) for s in hp2])
                result = compute_instability(reads, model.normal_size_bp, model.motif_len)
                if result:
                    hp_max = max(result["hii_h1"], result["hii_h2"])
                    if hp_max > 0.45:
                        hp_detected += 1

                # Pooled: all reads mixed
                pooled_sizes = np.concatenate([hp1, hp2])
                pooled_hii = metric_hii(pooled_sizes, model.motif_len)
                if pooled_hii > 0.45:
                    pooled_detected += 1

            rate_hp = hp_detected / n_reps
            rate_pooled = pooled_detected / n_reps
            if level == 0.0:
                hp_fp.append(rate_hp)
                pooled_fp.append(rate_pooled)
            else:
                hp_tp.append({"level": level, "rate": rate_hp})
                pooled_tp.append({"level": level, "rate": rate_pooled})

        results[model_name] = {
            "model": model.name,
            "hp_fp_rate": hp_fp[0] if hp_fp else 0,
            "pooled_fp_rate": pooled_fp[0] if pooled_fp else 0,
            "hp_tp": hp_tp,
            "pooled_tp": pooled_tp,
        }

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_all(dose_data, roc_data, lr_sr_data, outlier_data, hp_pooled_data, outdir):
    """Generate comprehensive figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Figure 1: Dose-response R² heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(dose_data.keys())
    metrics = list(METRICS.keys())
    r2_matrix = np.array([
        [dose_data[m]["r_squared"][met] for met in metrics]
        for m in models
    ])
    im = ax.imshow(r2_matrix, cmap="YlGn", vmin=0.9, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([dose_data[m]["model"] for m in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{r2_matrix[i,j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="R²")
    ax.set_title("Dose-Response Linearity (R²) by Disease Model", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(outdir / "fig1_dose_response_r2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: ROC AUC heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    auc_matrix = np.array([
        [roc_data[m]["auc"][met] for met in metrics]
        for m in models
    ])
    im = ax.imshow(auc_matrix, cmap="YlOrRd", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([roc_data[m]["model"] for m in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{auc_matrix[i,j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="AUC")
    ax.set_title("ROC AUC by Disease Model and Metric", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(outdir / "fig2_roc_auc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Long-read vs short-read detection ---
    n_models = len(lr_sr_data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (model_name, data) in enumerate(lr_sr_data.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        lr_levels = [d["level"] for d in data["long_read"]]
        lr_rates = [d["detection_rate"] * 100 for d in data["long_read"]]
        sr_rates = [d["detection_rate"] * 100 for d in data["short_read"]]
        ax.plot(lr_levels, lr_rates, "o-", color="#1E88E5", linewidth=2,
                markersize=7, label="Long-read (HP-tagged)")
        ax.plot(lr_levels, sr_rates, "s--", color="#E53935", linewidth=2,
                markersize=7, label="Short-read (pooled+stutter)")
        ax.set_xlabel("Instability level (motif units)", fontsize=10)
        ax.set_ylabel("Detection rate (%)", fontsize=10)
        ax.set_title(data["model"], fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for idx in range(len(lr_sr_data), 6):
        axes[idx].set_visible(False)
    fig.suptitle("Detection Rate: Long-Read vs Short-Read", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(outdir / "fig3_lr_vs_sr.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Outlier robustness ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Average across disease models
    avg_change = {m: 0 for m in METRICS}
    for model_name in outlier_data:
        for mname in METRICS:
            avg_change[mname] += outlier_data[model_name]["rel_change_20pct"][mname]
    for mname in METRICS:
        avg_change[mname] /= len(outlier_data)

    bars = ax.bar(range(len(METRICS)), [avg_change[m] for m in METRICS],
                  color=[METRIC_COLORS[m] for m in METRICS], alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(list(METRICS.keys()), fontsize=10)
    ax.set_ylabel("Mean relative change at 20% outliers (%)", fontsize=11)
    ax.set_title("Outlier Robustness (averaged across disease models)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [avg_change[m] for m in METRICS]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(outdir / "fig4_outlier_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 5: HP-tagged vs pooled ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: False positive rates
    ax = axes[0]
    model_labels = [hp_pooled_data[m]["model"] for m in hp_pooled_data]
    hp_fps = [hp_pooled_data[m]["hp_fp_rate"] * 100 for m in hp_pooled_data]
    pooled_fps = [hp_pooled_data[m]["pooled_fp_rate"] * 100 for m in hp_pooled_data]
    x = np.arange(len(model_labels))
    ax.bar(x - 0.2, hp_fps, 0.35, color="#1E88E5", label="HP-tagged (MosaicTR)")
    ax.bar(x + 0.2, pooled_fps, 0.35, color="#E53935", label="Pooled")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("False positive rate (%)", fontsize=11)
    ax.set_title("A. False Positives at Stable Loci", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: True positive rates (averaged across unstable levels)
    ax = axes[1]
    hp_avg_tp = []
    pooled_avg_tp = []
    for m in hp_pooled_data:
        hp_avg_tp.append(np.mean([d["rate"] for d in hp_pooled_data[m]["hp_tp"]]) * 100)
        pooled_avg_tp.append(np.mean([d["rate"] for d in hp_pooled_data[m]["pooled_tp"]]) * 100)
    ax.bar(x - 0.2, hp_avg_tp, 0.35, color="#1E88E5", label="HP-tagged (MosaicTR)")
    ax.bar(x + 0.2, pooled_avg_tp, 0.35, color="#E53935", label="Pooled")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("True positive rate (%)", fontsize=11)
    ax.set_title("B. True Positives at Unstable Loci", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(outdir / "fig5_hp_vs_pooled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 6: Combined summary (for manuscript) ---
    fig = plt.figure(figsize=(18, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # A: R² heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(r2_matrix, cmap="YlGn", vmin=0.9, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.split("(")[0].strip() for m in metrics], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([dose_data[m]["model"] for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{r2_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("A. Dose-Response R²", fontweight="bold", fontsize=11)

    # B: AUC heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(auc_matrix, cmap="YlOrRd", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.split("(")[0].strip() for m in metrics], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([roc_data[m]["model"] for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{auc_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("B. Classification AUC", fontweight="bold", fontsize=11)

    # C: Outlier robustness
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(range(len(METRICS)), [avg_change[m] for m in METRICS],
                  color=[METRIC_COLORS[m] for m in METRICS], alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m.split("(")[0].strip() for m in METRICS], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative change (%)", fontsize=9)
    ax.set_title("C. Outlier Robustness (20%)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # D: LR vs SR (HD brain)
    ax = fig.add_subplot(gs[1, 0])
    hd_data = lr_sr_data.get("HD_brain", list(lr_sr_data.values())[0])
    lr_levels = [d["level"] for d in hd_data["long_read"]]
    lr_rates = [d["detection_rate"] * 100 for d in hd_data["long_read"]]
    sr_rates = [d["detection_rate"] * 100 for d in hd_data["short_read"]]
    ax.plot(lr_levels, lr_rates, "o-", color="#1E88E5", linewidth=2, markersize=7, label="Long-read")
    ax.plot(lr_levels, sr_rates, "s--", color="#E53935", linewidth=2, markersize=7, label="Short-read")
    ax.set_xlabel("Instability level", fontsize=10)
    ax.set_ylabel("Detection rate (%)", fontsize=10)
    ax.set_title(f"D. LR vs SR: {hd_data['model']}", fontweight="bold", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # E: LR vs SR (MSI)
    ax = fig.add_subplot(gs[1, 1])
    msi_data = lr_sr_data.get("MSI", list(lr_sr_data.values())[-1])
    lr_levels = [d["level"] for d in msi_data["long_read"]]
    lr_rates = [d["detection_rate"] * 100 for d in msi_data["long_read"]]
    sr_rates = [d["detection_rate"] * 100 for d in msi_data["short_read"]]
    ax.plot(lr_levels, lr_rates, "o-", color="#1E88E5", linewidth=2, markersize=7, label="Long-read")
    ax.plot(lr_levels, sr_rates, "s--", color="#E53935", linewidth=2, markersize=7, label="Short-read")
    ax.set_xlabel("Instability level", fontsize=10)
    ax.set_ylabel("Detection rate (%)", fontsize=10)
    ax.set_title(f"E. LR vs SR: {msi_data['model']}", fontweight="bold", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # F: HP vs Pooled FP/TP
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(model_labels))
    ax.bar(x - 0.2, hp_fps, 0.35, color="#1E88E5", alpha=0.7, label="HP-tagged FP")
    ax.bar(x + 0.2, pooled_fps, 0.35, color="#E53935", alpha=0.7, label="Pooled FP")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("False positive rate (%)", fontsize=9)
    ax.set_title("F. HP-Tagged vs Pooled (FP)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(str(outdir / "fig_combined_realistic.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("All figures saved to %s", outdir)


# ============================================================================
# Report
# ============================================================================

def write_report(dose_data, roc_data, lr_sr_data, outlier_data, hp_pooled_data,
                 output_path):
    """Write comprehensive report."""
    lines = []
    w = lines.append

    w("=" * 95)
    w("  BIOLOGICALLY REALISTIC INSTABILITY METRIC COMPARISON")
    w("=" * 95)
    w("\n  Disease models based on published somatic expansion distributions:")
    for name, model in DISEASE_MODELS.items():
        w(f"    {model.name:20s}  motif={model.motif:5s}  exp_bias={model.expansion_bias:.0%}"
          f"  tail={model.tail_weight:.0%}")

    # 1. Dose-response R²
    w("\n" + "-" * 95)
    w("  1. DOSE-RESPONSE LINEARITY (R²)")
    w("-" * 95)
    w(f"\n  {'Model':20s}  " + "  ".join(f"{m:>16s}" for m in METRICS))
    w("  " + "-" * 88)
    for model_name in dose_data:
        r2s = dose_data[model_name]["r_squared"]
        w(f"  {dose_data[model_name]['model']:20s}  " +
          "  ".join(f"{r2s[m]:>16.4f}" for m in METRICS))

    # Average R²
    avg_r2 = {m: np.mean([dose_data[mn]["r_squared"][m] for mn in dose_data]) for m in METRICS}
    w(f"\n  {'AVERAGE':20s}  " + "  ".join(f"{avg_r2[m]:>16.4f}" for m in METRICS))
    best_r2 = max(avg_r2, key=avg_r2.get)
    w(f"\n  Best average R²: {best_r2}")

    # 2. ROC AUC
    w("\n" + "-" * 95)
    w("  2. ROC AUC (stable vs unstable)")
    w("-" * 95)
    w(f"\n  {'Model':20s}  " + "  ".join(f"{m:>16s}" for m in METRICS))
    w("  " + "-" * 88)
    for model_name in roc_data:
        aucs = roc_data[model_name]["auc"]
        w(f"  {roc_data[model_name]['model']:20s}  " +
          "  ".join(f"{aucs[m]:>16.4f}" for m in METRICS))

    avg_auc = {m: np.mean([roc_data[mn]["auc"][m] for mn in roc_data]) for m in METRICS}
    w(f"\n  {'AVERAGE':20s}  " + "  ".join(f"{avg_auc[m]:>16.4f}" for m in METRICS))
    best_auc = max(avg_auc, key=avg_auc.get)
    w(f"\n  Best average AUC: {best_auc}")

    # 3. LR vs SR
    w("\n" + "-" * 95)
    w("  3. LONG-READ vs SHORT-READ DETECTION")
    w("-" * 95)
    for model_name, data in lr_sr_data.items():
        w(f"\n  {data['model']}:")
        w(f"    {'Level':>8s}  {'LR detect%':>12s}  {'SR detect%':>12s}  {'LR advantage':>14s}")
        for lr, sr in zip(data["long_read"], data["short_read"]):
            advantage = lr["detection_rate"] - sr["detection_rate"]
            w(f"    {lr['level']:>8.2f}  {lr['detection_rate']*100:>12.1f}"
              f"  {sr['detection_rate']*100:>12.1f}  {advantage*100:>+14.1f} pp")

    # 4. Outlier robustness
    w("\n" + "-" * 95)
    w("  4. OUTLIER ROBUSTNESS (% change at 20% outliers)")
    w("-" * 95)
    avg_outlier = {}
    for mname in METRICS:
        vals = [outlier_data[mn]["rel_change_20pct"][mname] for mn in outlier_data]
        avg_outlier[mname] = np.mean(vals)
        w(f"    {mname:20s}  {np.mean(vals):6.1f}% (range: {min(vals):.1f}-{max(vals):.1f}%)")
    best_robust = min(avg_outlier, key=avg_outlier.get)
    w(f"\n  Most robust: {best_robust}")

    # 5. HP vs Pooled
    w("\n" + "-" * 95)
    w("  5. HP-TAGGED vs POOLED ANALYSIS")
    w("-" * 95)
    for model_name, data in hp_pooled_data.items():
        fp_reduction = data["pooled_fp_rate"] - data["hp_fp_rate"]
        w(f"  {data['model']:20s}  HP FP={data['hp_fp_rate']*100:.1f}%"
          f"  Pooled FP={data['pooled_fp_rate']*100:.1f}%"
          f"  FP reduction={fp_reduction*100:+.1f} pp")

    # Overall ranking
    w("\n" + "=" * 95)
    w("  OVERALL METRIC RANKING")
    w("=" * 95)
    w(f"\n  {'Metric':20s}  {'Avg R²':>8s}  {'Avg AUC':>8s}  {'Outlier%':>10s}  {'Rank':>6s}")
    w("  " + "-" * 60)

    # Simple ranking: weighted score
    scores = {}
    for mname in METRICS:
        r2_score = avg_r2[mname]
        auc_score = avg_auc[mname]
        # Lower outlier change is better, normalize to [0,1]
        max_out = max(avg_outlier.values())
        outlier_score = 1 - (avg_outlier[mname] / max_out) if max_out > 0 else 1
        scores[mname] = r2_score * 0.3 + auc_score * 0.4 + outlier_score * 0.3

    ranked = sorted(scores, key=scores.get, reverse=True)
    for rank, mname in enumerate(ranked, 1):
        w(f"  {mname:20s}  {avg_r2[mname]:>8.4f}  {avg_auc[mname]:>8.4f}"
          f"  {avg_outlier[mname]:>10.1f}  {'#' + str(rank):>6s}")

    w(f"\n  RECOMMENDED METRIC: {ranked[0]}")
    w("=" * 95)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report: %s", output_path)
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Biologically realistic instability metric comparison",
    )
    parser.add_argument("--output-dir", default="output/instability/realistic_simulation")
    parser.add_argument("--n-reps", default=50, type=int)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dose_data = run_dose_response(n_reps=args.n_reps)
    roc_data = run_roc(n_stable=300, n_unstable=300)
    lr_sr_data = run_lr_vs_sr(n_reps=args.n_reps)
    outlier_data = run_outlier_robustness(n_reps=args.n_reps)
    hp_pooled_data = run_hp_vs_pooled(n_reps=args.n_reps)

    try:
        plot_all(dose_data, roc_data, lr_sr_data, outlier_data, hp_pooled_data, outdir)
    except ImportError as e:
        logger.warning("matplotlib not available: %s", e)

    report = write_report(
        dose_data, roc_data, lr_sr_data, outlier_data, hp_pooled_data,
        str(outdir / "realistic_simulation_report.txt"),
    )
    print(report)


if __name__ == "__main__":
    main()
