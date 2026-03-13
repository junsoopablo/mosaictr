#!/usr/bin/env python3
"""Simulation-based validation of MosaicTR instability detection.

Generates synthetic read distributions with known instability parameters,
runs compute_instability(), and compares input vs measured metrics.

Scenarios:
  1. HII dose-response: HII 0 → 0.5 → 1 → 2 → 5 → 10 → 20
  2. Coverage sweep: HII=1.5 at 5x, 10x, 15x, 20x, 30x, 40x, 60x, 80x
  3. ROC: mixture of stable vs unstable loci, threshold sweep
  4. Long-read advantage: spanning limitation + PCR stutter false-positive HII

Unstable reads use an expansion-biased right-skewed model:
  each read = median + |X| + noise, where X ~ Exponential(1 / (target_HII * motif_len))

Outputs (to output/instability/simulation/):
  - simulation_report.txt           — quantitative results
  - fig_hii_dose_response.png       — HII sensitivity curve + linearity
  - fig_coverage_sweep.png          — detection power vs coverage
  - fig_roc.png                     — ROC curve for stable/unstable classification
  - fig_longread_advantage.png      — spanning + stutter panels
  - fig_simulation_combined.png     — 2x2 combined figure

Usage:
  python scripts/simulate_instability.py [--output-dir output/instability/simulation]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.genotype import ReadInfo
from mosaictr.instability import compute_instability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic read generators
# ---------------------------------------------------------------------------

def _ri(size: float, hp: int, mapq: int = 60) -> ReadInfo:
    """Shorthand ReadInfo constructor."""
    return ReadInfo(allele_size=size, hp=hp, mapq=mapq)


def generate_stable_reads(
    median: float, n: int, hp: int, noise_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[ReadInfo]:
    """Generate reads for a stable haplotype (near-zero instability).

    Small Gaussian noise simulates HiFi stutter (~0.5bp std).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sizes = rng.normal(median, noise_std, n)
    return [_ri(float(s), hp) for s in sizes]


def generate_unstable_reads(
    median: float,
    n: int,
    hp: int,
    motif_len: int,
    target_hii: float,
    rng: np.random.Generator | None = None,
) -> list[ReadInfo]:
    """Generate reads for an unstable haplotype with expansion-biased model.

    Each read = median + |X| + noise, where:
      X ~ Exponential(lambda), lambda = 1 / (target_HII * motif_len)
      noise ~ N(0, 0.5) for measurement noise

    This models somatic expansion: peak at inherited size with an exponential
    tail toward larger expansions. Rarer large expansions, common small ones.
    HII is computed from MAD of the resulting distribution.

    Args:
        median: Center of the distribution (bp), inherited allele size.
        n: Number of reads.
        hp: Haplotype tag (1 or 2).
        motif_len: Motif length in bp.
        target_hii: Target HII value (controls expansion magnitude).
        rng: Random number generator.

    Returns:
        List of ReadInfo with expansion-biased instability.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    target_mad = target_hii * motif_len

    if target_mad < 0.01:
        return generate_stable_reads(median, n, hp, noise_std=0.3, rng=rng)

    # For Exponential(scale), MAD = arcsinh(0.5) * scale ≈ 0.4812 * scale.
    # Calibrate scale so that the theoretical MAD equals target_MAD,
    # leaving only MAD-based outlier trimming as the source of attenuation.
    _ARCSINH_HALF = float(np.arcsinh(0.5))  # ≈ 0.4812
    scale = target_mad / _ARCSINH_HALF  # calibrated exponential scale
    expansions = rng.exponential(scale, n)  # always >= 0
    noise = rng.normal(0, 0.5, n)  # measurement noise

    sizes = median + expansions + noise
    return [_ri(float(s), hp) for s in sizes]


# ---------------------------------------------------------------------------
# Test 1: HII Dose-Response
# ---------------------------------------------------------------------------

def run_hii_dose_response(n_reps: int = 20) -> dict:
    """Test HII recovery across increasing instability levels.

    Checks: monotonicity, linearity (R²), absolute accuracy.
    """
    logger.info("Running HII dose-response test (%d reps per level)...", n_reps)

    target_hiis = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    motif_len = 3
    n_reads_per_hp = 30

    results = []
    for target_hii in target_hiis:
        measured_hiis = []
        for rep in range(n_reps):
            rng = np.random.default_rng(2000 + rep)
            hp1 = generate_stable_reads(60.0, n_reads_per_hp, hp=1, rng=rng)
            hp2 = generate_unstable_reads(
                120.0, n_reads_per_hp, hp=2, motif_len=motif_len,
                target_hii=target_hii, rng=rng,
            )
            result = compute_instability(hp1 + hp2, 60.0, motif_len)
            if result is None:
                continue
            measured_hiis.append(result["hii_h2"])

        if measured_hiis:
            results.append({
                "target_hii": target_hii,
                "mean_hii": np.mean(measured_hiis),
                "std_hii": np.std(measured_hiis),
                "median_hii": np.median(measured_hiis),
                "n_success": len(measured_hiis),
            })

    # Check monotonicity
    means = [r["mean_hii"] for r in results]
    monotonic = all(means[i] <= means[i + 1] + 0.01 for i in range(len(means) - 1))

    # Linearity (R²)
    targets = np.array([r["target_hii"] for r in results])
    measured = np.array([r["mean_hii"] for r in results])
    if len(targets) > 2:
        corr = np.corrcoef(targets, measured)[0, 1]
        r_squared = corr ** 2
    else:
        r_squared = 0.0

    return {
        "dose_results": results,
        "monotonic": monotonic,
        "r_squared": r_squared,
    }


# ---------------------------------------------------------------------------
# Test 2: Coverage Sweep
# ---------------------------------------------------------------------------

def run_coverage_sweep(n_reps: int = 30) -> dict:
    """Test HII detection at fixed instability across coverage levels.

    HII=1.5 (moderate instability, ~3x noise threshold) at different coverages.
    Tests whether MosaicTR can reliably detect genuine instability even at low
    coverage, where stochastic noise in MAD estimation is highest.
    """
    logger.info("Running coverage sweep test (%d reps per coverage)...", n_reps)

    target_hii = 1.5
    motif_len = 3
    coverages = [5, 10, 15, 20, 30, 40, 60, 80]
    noise_threshold = 0.45

    results = []
    for cov in coverages:
        n_reads = cov  # reads per haplotype
        detected = 0
        measured_hiis = []
        for rep in range(n_reps):
            rng = np.random.default_rng(3000 + rep)
            hp1 = generate_stable_reads(60.0, n_reads, hp=1, rng=rng)
            hp2 = generate_unstable_reads(
                120.0, n_reads, hp=2, motif_len=motif_len,
                target_hii=target_hii, rng=rng,
            )
            result = compute_instability(hp1 + hp2, 60.0, motif_len)
            if result is None:
                continue
            max_hii = max(result["hii_h1"], result["hii_h2"])
            measured_hiis.append(max_hii)
            if max_hii > noise_threshold:
                detected += 1

        total_valid = len(measured_hiis)
        results.append({
            "coverage": cov,
            "detection_rate": detected / total_valid if total_valid > 0 else 0.0,
            "mean_hii": np.mean(measured_hiis) if measured_hiis else 0.0,
            "std_hii": np.std(measured_hiis) if measured_hiis else 0.0,
            "n_valid": total_valid,
        })

    return {"coverage_results": results}


# ---------------------------------------------------------------------------
# Test 3: ROC Analysis
# ---------------------------------------------------------------------------

def run_roc_analysis(n_stable: int = 200, n_unstable: int = 200) -> dict:
    """ROC analysis for stable vs unstable locus classification.

    Generate stable (HII~0) and unstable (HII=1.5-10) loci,
    sweep HII threshold to compute sensitivity/specificity.
    """
    logger.info("Running ROC analysis (%d stable + %d unstable loci)...",
                n_stable, n_unstable)

    motif_len = 3
    n_reads_per_hp = 25

    labels = []  # 0=stable, 1=unstable
    max_hiis = []

    # Generate stable loci
    for i in range(n_stable):
        rng = np.random.default_rng(4000 + i)
        hp1 = generate_stable_reads(60.0, n_reads_per_hp, hp=1, rng=rng)
        hp2 = generate_stable_reads(90.0, n_reads_per_hp, hp=2, rng=rng)
        result = compute_instability(hp1 + hp2, 60.0, motif_len)
        if result is None:
            continue
        max_hiis.append(max(result["hii_h1"], result["hii_h2"]))
        labels.append(0)

    # Generate unstable loci (varying HII from 1.5 to 10.0)
    for i in range(n_unstable):
        rng = np.random.default_rng(5000 + i)
        target_hii = rng.uniform(1.5, 10.0)
        hp1 = generate_stable_reads(60.0, n_reads_per_hp, hp=1, rng=rng)
        hp2 = generate_unstable_reads(
            120.0, n_reads_per_hp, hp=2, motif_len=motif_len,
            target_hii=target_hii, rng=rng,
        )
        result = compute_instability(hp1 + hp2, 60.0, motif_len)
        if result is None:
            continue
        max_hiis.append(max(result["hii_h1"], result["hii_h2"]))
        labels.append(1)

    labels = np.array(labels)
    max_hiis = np.array(max_hiis)

    # Sweep thresholds
    thresholds = np.linspace(0, 5.0, 200)
    roc_points = []
    for thr in thresholds:
        predicted = (max_hiis > thr).astype(int)
        tp = np.sum((predicted == 1) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        tn = np.sum((predicted == 0) & (labels == 0))
        fn = np.sum((predicted == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # 1-specificity
        roc_points.append({"threshold": float(thr), "tpr": tpr, "fpr": fpr})

    # AUC (trapezoidal)
    fprs = np.array([p["fpr"] for p in roc_points])
    tprs = np.array([p["tpr"] for p in roc_points])
    sort_idx = np.argsort(fprs)
    fprs_sorted = fprs[sort_idx]
    tprs_sorted = tprs[sort_idx]
    auc = float(np.trapezoid(tprs_sorted, fprs_sorted))

    # Metrics at default threshold (0.45)
    pred_default = (max_hiis > 0.45).astype(int)
    tp_d = int(np.sum((pred_default == 1) & (labels == 1)))
    fp_d = int(np.sum((pred_default == 1) & (labels == 0)))
    tn_d = int(np.sum((pred_default == 0) & (labels == 0)))
    fn_d = int(np.sum((pred_default == 0) & (labels == 1)))
    sens_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
    spec_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0.0

    return {
        "roc_points": roc_points,
        "auc": auc,
        "default_threshold": {
            "threshold": 0.45,
            "sensitivity": sens_d,
            "specificity": spec_d,
            "tp": tp_d, "fp": fp_d, "tn": tn_d, "fn": fn_d,
        },
        "n_stable": int(np.sum(labels == 0)),
        "n_unstable": int(np.sum(labels == 1)),
    }


# ---------------------------------------------------------------------------
# Test 4: Long-read Advantage Simulation
# ---------------------------------------------------------------------------

def run_longread_advantage(n_reps: int = 30) -> dict:
    """Simulate long-read vs short-read advantages for TR instability.

    Panel A: Spanning limitation (analytical).
    Panel B: PCR stutter false-positive HII (simulation).
    """
    logger.info("Running long-read advantage simulation...")

    # --- Panel A: Spanning limitation (analytical) ---
    tr_lengths = np.arange(50, 5001, 10)
    # Short-read (150bp): need flanking sequence (~25bp each side) to anchor alignment
    # Effective insert for spanning = 150 - 2*25 = 100bp usable
    short_read_len = 150
    flank_req = 25  # minimum flanking bp needed on each side
    short_span = np.clip(
        (short_read_len - 2 * flank_req - tr_lengths) / (short_read_len - 2 * flank_req),
        0, 1,
    )
    # Long-read (15kb HiFi): virtually all TRs spanned
    long_read_len = 15000
    long_span = np.clip(
        (long_read_len - tr_lengths) / long_read_len,
        0, 1,
    )

    spanning_data = {
        "tr_lengths": tr_lengths.tolist(),
        "short_span": short_span.tolist(),
        "long_span": long_span.tolist(),
    }

    # --- Panel B: PCR stutter false-positive HII ---
    motif_lengths = [3, 4, 5, 6]
    n_reads_per_hp = 30
    # PCR stutter: cumulative across ~25 PCR cycles. Published rates 5-15% per
    # cycle for trinucleotides (Shinde 2003, Lai & Sun 2003). After amplification,
    # ~30-40% of fragments are stuttered. We use 35% as a realistic cumulative rate.
    stutter_prob = 0.35
    noise_std = 0.5

    stutter_results = []
    for ml in motif_lengths:
        longread_hiis = []
        shortread_hiis = []

        for rep in range(n_reps):
            rng = np.random.default_rng(7000 + ml * 100 + rep)

            # --- Long-read: Gaussian noise only ---
            lr_sizes_h1 = rng.normal(60.0, noise_std, n_reads_per_hp)
            lr_sizes_h2 = rng.normal(90.0, noise_std, n_reads_per_hp)
            lr_reads = (
                [_ri(float(s), 1) for s in lr_sizes_h1]
                + [_ri(float(s), 2) for s in lr_sizes_h2]
            )
            lr_result = compute_instability(lr_reads, 60.0, ml)
            if lr_result is not None:
                longread_hiis.append(max(lr_result["hii_h1"], lr_result["hii_h2"]))

            # --- Short-read: PCR stutter + Gaussian noise ---
            sr_sizes_h1 = rng.normal(60.0, noise_std, n_reads_per_hp)
            sr_sizes_h2 = rng.normal(90.0, noise_std, n_reads_per_hp)
            # Apply stutter: each read has stutter_prob chance of +/-1 motif unit
            for arr in [sr_sizes_h1, sr_sizes_h2]:
                stutter_mask = rng.random(len(arr)) < stutter_prob
                stutter_dir = rng.choice([-1, 1], size=len(arr))
                arr[stutter_mask] += stutter_dir[stutter_mask] * ml
            sr_reads = (
                [_ri(float(s), 1) for s in sr_sizes_h1]
                + [_ri(float(s), 2) for s in sr_sizes_h2]
            )
            sr_result = compute_instability(sr_reads, 60.0, ml)
            if sr_result is not None:
                shortread_hiis.append(max(sr_result["hii_h1"], sr_result["hii_h2"]))

        stutter_results.append({
            "motif_len": ml,
            "longread_mean_hii": float(np.mean(longread_hiis)) if longread_hiis else 0.0,
            "longread_std_hii": float(np.std(longread_hiis)) if longread_hiis else 0.0,
            "shortread_mean_hii": float(np.mean(shortread_hiis)) if shortread_hiis else 0.0,
            "shortread_std_hii": float(np.std(shortread_hiis)) if shortread_hiis else 0.0,
            "n_reps_lr": len(longread_hiis),
            "n_reps_sr": len(shortread_hiis),
        })

    return {
        "spanning_data": spanning_data,
        "stutter_results": stutter_results,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_hii_dose_response(dose_results: list[dict], r_squared: float,
                           output_path: str):
    """HII dose-response curve with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    targets = [r["target_hii"] for r in dose_results]
    means = [r["mean_hii"] for r in dose_results]
    stds = [r["std_hii"] for r in dose_results]

    # Left: dose-response curve
    ax = axes[0]
    ax.errorbar(targets, means, yerr=stds, fmt="o-", capsize=4,
                color="#4CAF50", markersize=8, linewidth=1.5)
    ax.plot([0, 22], [0, 22], "k--", alpha=0.4, label="Perfect recovery")
    ax.set_xlabel("Target HII", fontsize=12)
    ax.set_ylabel("Measured HII (mean +/- SD)", fontsize=12)
    ax.set_title("HII Dose-Response", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.92, f"R2 = {r_squared:.3f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right: residuals
    ax = axes[1]
    rel_errors = [abs(m - t) / max(t, 0.01) * 100 for t, m in zip(targets, means)]
    ax.bar(range(len(targets)), rel_errors, color="#FF9800", alpha=0.8,
           tick_label=[f"{t:.1f}" for t in targets])
    ax.set_xlabel("Target HII", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("HII Recovery Accuracy", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_coverage_sweep(cov_results: list[dict], output_path: str):
    """Coverage vs detection rate plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    coverages = [r["coverage"] for r in cov_results]
    det_rates = [r["detection_rate"] * 100 for r in cov_results]
    mean_hiis = [r["mean_hii"] for r in cov_results]
    std_hiis = [r["std_hii"] for r in cov_results]

    # Left: detection rate
    ax = axes[0]
    ax.plot(coverages, det_rates, "o-", color="#E91E63", markersize=8, linewidth=2)
    ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% detection")
    ax.set_xlabel("Coverage (reads per haplotype)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Detection Power vs Coverage\n(HII=1.5, threshold=0.45)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: HII stability
    ax = axes[1]
    ax.errorbar(coverages, mean_hiis, yerr=std_hiis, fmt="o-", capsize=4,
                color="#9C27B0", markersize=8, linewidth=1.5)
    ax.axhline(1.5, color="gray", linestyle="--", alpha=0.5, label="Target HII=1.5")
    ax.axhline(0.45, color="red", linestyle=":", alpha=0.5, label="Noise threshold=0.45")
    ax.set_xlabel("Coverage (reads per haplotype)", fontsize=12)
    ax.set_ylabel("Measured HII (mean +/- SD)", fontsize=12)
    ax.set_title("HII Estimation Stability vs Coverage",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_roc(roc_data: dict, output_path: str):
    """ROC curve for stable vs unstable classification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    fprs = [p["fpr"] for p in roc_data["roc_points"]]
    tprs = [p["tpr"] for p in roc_data["roc_points"]]

    ax.plot(fprs, tprs, color="#2196F3", linewidth=2.5,
            label=f'HII threshold (AUC = {roc_data["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    # Mark default threshold
    dt = roc_data["default_threshold"]
    ax.plot(1 - dt["specificity"], dt["sensitivity"], "r*", markersize=15,
            label=f'Default t=0.45\n(Sens={dt["sensitivity"]:.1%}, Spec={dt["specificity"]:.1%})')

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(
        f'ROC: Stable vs Unstable Classification\n'
        f'(n={roc_data["n_stable"]} stable + {roc_data["n_unstable"]} unstable)',
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_longread_advantage(longread_data: dict, output_path: str):
    """Long-read advantage figure with spanning limitation and stutter panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: Spanning limitation ---
    ax = axes[0]
    tr_lengths = np.array(longread_data["spanning_data"]["tr_lengths"])
    short_span = np.array(longread_data["spanning_data"]["short_span"])
    long_span = np.array(longread_data["spanning_data"]["long_span"])

    ax.plot(tr_lengths, short_span * 100, color="#E53935", linewidth=2.5,
            label="Short-read (150 bp)")
    ax.plot(tr_lengths, long_span * 100, color="#1E88E5", linewidth=2.5,
            label="Long-read (15 kb HiFi)")

    # Disease expansion ranges as shaded regions
    ax.axvspan(120, 600, alpha=0.12, color="#FF9800", label="HD (120-600 bp)")
    ax.axvspan(150, 5000, alpha=0.08, color="#9C27B0", label="DM1 (150-15,000 bp)")
    ax.axvspan(600, 5000, alpha=0.08, color="#4CAF50", label="FXS (600-6,000 bp)")

    ax.set_xlabel("TR Length (bp)", fontsize=12)
    ax.set_ylabel("Reads Spanning Entire Repeat (%)", fontsize=12)
    ax.set_title("A. Spanning Limitation", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(50, 5000)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    # --- Panel B: PCR stutter false-positive HII ---
    ax = axes[1]
    stutter_results = longread_data["stutter_results"]
    motif_lens = [r["motif_len"] for r in stutter_results]
    lr_means = [r["longread_mean_hii"] for r in stutter_results]
    lr_stds = [r["longread_std_hii"] for r in stutter_results]
    sr_means = [r["shortread_mean_hii"] for r in stutter_results]
    sr_stds = [r["shortread_std_hii"] for r in stutter_results]

    x = np.arange(len(motif_lens))
    width = 0.35

    bars_lr = ax.bar(x - width / 2, lr_means, width, yerr=lr_stds, capsize=4,
                     color="#1E88E5", edgecolor="#0D47A1", alpha=0.85,
                     label="Long-read (no PCR)")
    bars_sr = ax.bar(x + width / 2, sr_means, width, yerr=sr_stds, capsize=4,
                     color="#E53935", edgecolor="#B71C1C", alpha=0.85,
                     label="Short-read (10% stutter)")

    ax.axhline(0.45, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Noise threshold (0.45)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ml} bp" for ml in motif_lens])
    ax.set_xlabel("Motif Length", fontsize=12)
    ax.set_ylabel("Measured HII (stable loci, true HII = 0)", fontsize=12)
    ax.set_title("B. PCR Stutter Inflates HII", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_combined(dose_data: dict, cov_data: dict, roc_data: dict,
                  longread_data: dict, output_path: str):
    """Generate 2x2 combined simulation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # --- Panel A: HII dose-response ---
    ax_a = fig.add_subplot(gs[0, 0])
    dose_results = dose_data["dose_results"]
    targets = [r["target_hii"] for r in dose_results]
    means = [r["mean_hii"] for r in dose_results]
    stds = [r["std_hii"] for r in dose_results]

    ax_a.errorbar(targets, means, yerr=stds, fmt="o-", capsize=4,
                  color="#4CAF50", markersize=8, linewidth=1.5)
    ax_a.plot([0, 22], [0, 22], "k--", alpha=0.4, label="Perfect recovery")
    ax_a.set_xlabel("Target HII", fontsize=11)
    ax_a.set_ylabel("Measured HII (mean +/- SD)", fontsize=11)
    ax_a.set_title("A. HII Dose-Response", fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)
    ax_a.text(0.05, 0.92, f"R2 = {dose_data['r_squared']:.3f}",
              transform=ax_a.transAxes, fontsize=10,
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Panel B: ROC ---
    ax_b = fig.add_subplot(gs[0, 1])
    fprs = [p["fpr"] for p in roc_data["roc_points"]]
    tprs = [p["tpr"] for p in roc_data["roc_points"]]

    ax_b.plot(fprs, tprs, color="#2196F3", linewidth=2.5,
              label=f'AUC = {roc_data["auc"]:.3f}')
    ax_b.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    dt = roc_data["default_threshold"]
    ax_b.plot(1 - dt["specificity"], dt["sensitivity"], "r*", markersize=15,
              label=f't=0.45 (Sens={dt["sensitivity"]:.1%}, Spec={dt["specificity"]:.1%})')

    ax_b.set_xlabel("False Positive Rate", fontsize=11)
    ax_b.set_ylabel("True Positive Rate", fontsize=11)
    ax_b.set_title("B. ROC Curve", fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=9, loc="lower right")
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.02)

    # --- Panel C: Coverage sweep ---
    ax_c = fig.add_subplot(gs[1, 0])
    cov_results = cov_data["coverage_results"]
    coverages = [r["coverage"] for r in cov_results]
    det_rates = [r["detection_rate"] * 100 for r in cov_results]

    ax_c.plot(coverages, det_rates, "o-", color="#E91E63", markersize=8, linewidth=2)
    ax_c.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% detection")
    ax_c.set_xlabel("Coverage (reads per haplotype)", fontsize=11)
    ax_c.set_ylabel("Detection Rate (%)", fontsize=11)
    ax_c.set_title("C. Detection Power vs Coverage (HII=1.5)",
                   fontsize=13, fontweight="bold")
    ax_c.set_ylim(0, 105)
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # --- Panel D: Long-read advantage (2 sub-panels) ---
    gs_d = gs[1, 1].subgridspec(2, 1, hspace=0.45)

    # D-top: Spanning limitation
    ax_d1 = fig.add_subplot(gs_d[0])
    tr_lengths = np.array(longread_data["spanning_data"]["tr_lengths"])
    short_span = np.array(longread_data["spanning_data"]["short_span"])
    long_span = np.array(longread_data["spanning_data"]["long_span"])

    ax_d1.plot(tr_lengths, short_span * 100, color="#E53935", linewidth=2,
               label="Short-read (150 bp)")
    ax_d1.plot(tr_lengths, long_span * 100, color="#1E88E5", linewidth=2,
               label="Long-read (15 kb)")

    ax_d1.axvspan(120, 600, alpha=0.12, color="#FF9800")
    ax_d1.axvspan(600, 5000, alpha=0.08, color="#4CAF50")
    ax_d1.text(250, 50, "HD", fontsize=8, color="#E65100", fontweight="bold")
    ax_d1.text(1500, 50, "FXS", fontsize=8, color="#2E7D32", fontweight="bold")

    ax_d1.set_xlabel("TR Length (bp)", fontsize=10)
    ax_d1.set_ylabel("Spanning (%)", fontsize=10)
    ax_d1.set_title("D. Long-Read Advantage", fontsize=13, fontweight="bold")
    ax_d1.set_xscale("log")
    ax_d1.set_xlim(50, 5000)
    ax_d1.set_ylim(-2, 105)
    ax_d1.legend(fontsize=7, loc="center right")
    ax_d1.grid(True, alpha=0.3)

    # D-bottom: PCR stutter
    ax_d2 = fig.add_subplot(gs_d[1])
    stutter_results = longread_data["stutter_results"]
    motif_lens = [r["motif_len"] for r in stutter_results]
    lr_means = [r["longread_mean_hii"] for r in stutter_results]
    lr_stds = [r["longread_std_hii"] for r in stutter_results]
    sr_means = [r["shortread_mean_hii"] for r in stutter_results]
    sr_stds = [r["shortread_std_hii"] for r in stutter_results]

    x = np.arange(len(motif_lens))
    width = 0.35
    ax_d2.bar(x - width / 2, lr_means, width, yerr=lr_stds, capsize=3,
              color="#1E88E5", edgecolor="#0D47A1", alpha=0.85, label="Long-read")
    ax_d2.bar(x + width / 2, sr_means, width, yerr=sr_stds, capsize=3,
              color="#E53935", edgecolor="#B71C1C", alpha=0.85, label="Short-read")
    ax_d2.axhline(0.45, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                  label="Threshold (0.45)")

    ax_d2.set_xticks(x)
    ax_d2.set_xticklabels([f"{ml} bp" for ml in motif_lens])
    ax_d2.set_xlabel("Motif Length", fontsize=10)
    ax_d2.set_ylabel("HII (stable loci)", fontsize=10)
    ax_d2.legend(fontsize=7, loc="upper left")
    ax_d2.grid(True, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    dose_data: dict,
    cov_data: dict,
    roc_data: dict,
    longread_data: dict,
    output_path: str,
):
    """Write comprehensive text report."""
    lines = []
    w = lines.append

    w("=" * 90)
    w("  MOSAICTR INSTABILITY SIMULATION REPORT")
    w("=" * 90)

    # --- 1. HII Dose-Response ---
    w("\n" + "-" * 90)
    w("  1. HII DOSE-RESPONSE TEST")
    w("-" * 90)
    w(f"\n  {'Target HII':>12s}  {'Measured HII':>14s}  {'Std':>8s}  {'Rel.Err%':>10s}  {'N':>4s}")
    w("  " + "-" * 56)
    for r in dose_data["dose_results"]:
        rel_err = abs(r["mean_hii"] - r["target_hii"]) / max(r["target_hii"], 0.01) * 100
        w(f"  {r['target_hii']:>12.2f}  {r['mean_hii']:>14.4f}  {r['std_hii']:>8.4f}"
          f"  {rel_err:>10.1f}  {r['n_success']:>4d}")

    w(f"\n  Monotonic: {'YES' if dose_data['monotonic'] else 'NO'}")
    w(f"  Linearity R2 = {dose_data['r_squared']:.4f}")
    dose_pass = dose_data["monotonic"] and dose_data["r_squared"] > 0.95
    w(f"  >>> {'PASS' if dose_pass else 'FAIL'} (monotonic + R2 > 0.95 required)")

    # --- 2. Coverage Sweep ---
    w("\n" + "-" * 90)
    w("  2. COVERAGE SWEEP TEST (HII=1.5, threshold=0.45)")
    w("-" * 90)
    w(f"\n  {'Coverage':>10s}  {'Detect%':>10s}  {'Mean HII':>10s}  {'Std HII':>10s}")
    w("  " + "-" * 46)
    for r in cov_data["coverage_results"]:
        w(f"  {r['coverage']:>10d}  {r['detection_rate']*100:>10.1f}"
          f"  {r['mean_hii']:>10.4f}  {r['std_hii']:>10.4f}")

    min_cov_80 = None
    for r in cov_data["coverage_results"]:
        if r["detection_rate"] >= 0.80:
            min_cov_80 = r["coverage"]
            break
    if min_cov_80:
        w(f"\n  Minimum coverage for >=80% detection: {min_cov_80}x per haplotype")
    else:
        w("\n  80% detection not reached at any tested coverage")
    cov_pass = min_cov_80 is not None and min_cov_80 <= 40
    w(f"  >>> {'PASS' if cov_pass else 'FAIL'} (>=80% detection at <=40x required)")

    # --- 3. ROC ---
    w("\n" + "-" * 90)
    w("  3. ROC ANALYSIS (stable vs unstable classification)")
    w("-" * 90)
    dt = roc_data["default_threshold"]
    w(f"\n  Loci: {roc_data['n_stable']} stable + {roc_data['n_unstable']} unstable")
    w(f"  AUC = {roc_data['auc']:.4f}")
    w(f"\n  At default threshold (HII > 0.45):")
    w(f"    Sensitivity: {dt['sensitivity']:.1%} ({dt['tp']}/{dt['tp']+dt['fn']})")
    w(f"    Specificity: {dt['specificity']:.1%} ({dt['tn']}/{dt['tn']+dt['fp']})")
    w(f"    TP={dt['tp']}, FP={dt['fp']}, TN={dt['tn']}, FN={dt['fn']}")
    roc_pass = roc_data["auc"] > 0.95 and dt["sensitivity"] > 0.90 and dt["specificity"] > 0.90
    w(f"  >>> {'PASS' if roc_pass else 'FAIL'} (AUC>0.95, Sens>90%, Spec>90% required)")

    # --- 4. Long-read Advantage ---
    w("\n" + "-" * 90)
    w("  4. LONG-READ ADVANTAGE SIMULATION")
    w("-" * 90)

    w("\n  Panel A: Spanning limitation")
    w("    Short-read (150 bp) cannot span TRs > ~100 bp")
    w("    Long-read (15 kb HiFi) spans virtually all disease-relevant TRs")

    w("\n  Panel B: PCR stutter false-positive HII (stable loci, true HII = 0)")
    w(f"\n  {'Motif':>8s}  {'LR mean HII':>14s}  {'LR std':>10s}  {'SR mean HII':>14s}  {'SR std':>10s}")
    w("  " + "-" * 62)
    any_sr_above = False
    for r in longread_data["stutter_results"]:
        w(f"  {r['motif_len']:>5d} bp  {r['longread_mean_hii']:>14.4f}"
          f"  {r['longread_std_hii']:>10.4f}  {r['shortread_mean_hii']:>14.4f}"
          f"  {r['shortread_std_hii']:>10.4f}")
        if r["shortread_mean_hii"] > 0.45:
            any_sr_above = True

    lr_all_below = all(
        r["longread_mean_hii"] < 0.45 for r in longread_data["stutter_results"]
    )
    longread_pass = lr_all_below
    w(f"\n  Long-read HII all below threshold: {'YES' if lr_all_below else 'NO'}")
    if any_sr_above:
        w("  Short-read stutter pushes HII above noise threshold for some motif lengths")
    w(f"  >>> {'PASS' if longread_pass else 'FAIL'} (all long-read HII < 0.45 required)")

    # --- Overall Summary ---
    all_pass = dose_pass and cov_pass and roc_pass and longread_pass
    n_pass = sum([dose_pass, cov_pass, roc_pass, longread_pass])
    w("\n" + "=" * 90)
    w("  OVERALL SIMULATION SUMMARY")
    w("=" * 90)
    w(f"  1. HII Dose-Response:   {'PASS' if dose_pass else 'FAIL'} "
      f"(R2={dose_data['r_squared']:.3f}, mono={'Y' if dose_data['monotonic'] else 'N'})")
    w(f"  2. Coverage Sweep:      {'PASS' if cov_pass else 'FAIL'} "
      f"(min cov for 80%: {min_cov_80 or '>80'}x)")
    w(f"  3. ROC Analysis:        {'PASS' if roc_pass else 'FAIL'} "
      f"(AUC={roc_data['auc']:.3f})")
    w(f"  4. Long-read Advantage: {'PASS' if longread_pass else 'FAIL'} "
      f"(LR HII below threshold: {'Y' if lr_all_below else 'N'})")
    w(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'} ({n_pass}/4 passed)")
    w("=" * 90)

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", output_path)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulation-based validation of MosaicTR instability detection",
    )
    parser.add_argument(
        "--output-dir", default="output/instability/simulation",
        help="Output directory for figures and report",
    )
    parser.add_argument(
        "--n-reps", default=30, type=int,
        help="Repetitions per condition (default: 30)",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_reps = args.n_reps

    # Run all scenarios
    dose_data = run_hii_dose_response(n_reps=n_reps)
    cov_data = run_coverage_sweep(n_reps=n_reps)
    roc_data = run_roc_analysis(n_stable=300, n_unstable=300)
    longread_data = run_longread_advantage(n_reps=n_reps)

    # Generate individual figures
    try:
        plot_hii_dose_response(
            dose_data["dose_results"], dose_data["r_squared"],
            str(outdir / "fig_hii_dose_response.png"),
        )
        plot_coverage_sweep(
            cov_data["coverage_results"],
            str(outdir / "fig_coverage_sweep.png"),
        )
        plot_roc(roc_data, str(outdir / "fig_roc.png"))
        plot_longread_advantage(longread_data, str(outdir / "fig_longread_advantage.png"))

        # Generate combined 2x2 figure
        plot_combined(dose_data, cov_data, roc_data, longread_data,
                      str(outdir / "fig_simulation_combined.png"))

        logger.info("All figures saved to %s", outdir)
    except ImportError as e:
        logger.warning("matplotlib not available, skipping figures: %s", e)

    # Write report
    report = write_report(
        dose_data, cov_data, roc_data, longread_data,
        str(outdir / "simulation_report.txt"),
    )
    print(report)


if __name__ == "__main__":
    main()
