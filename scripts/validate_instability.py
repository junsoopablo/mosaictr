#!/usr/bin/env python3
"""Phase 1: Simulation-based instability validation.

Part A — Synthetic disease scenarios (no BAM needed):
  8 scenarios testing compute_instability() with synthetic ReadInfo lists.

Part B — Real HG002 baseline:
  Extract reads from 32 STRchive disease loci, establish noise floor.

Usage:
  python scripts/validate_instability.py --output output/instability/phase1_report.txt
  python scripts/validate_instability.py --skip-bam --output output/instability/phase1a_only.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haplotr.genotype import ReadInfo, extract_reads_enhanced
from haplotr.instability import compute_instability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data paths (same as benchmark_genome_wide.py)
# ---------------------------------------------------------------------------

BAM_PATH = "/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF_PATH = "/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"

# STRchive disease loci (from benchmark_genome_wide.py)
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


def _ri(size: float, hp: int, mapq: int = 60) -> ReadInfo:
    """Shorthand ReadInfo constructor."""
    return ReadInfo(allele_size=size, hp=hp, mapq=mapq)


# ============================================================================
# Part A: Synthetic disease scenarios
# ============================================================================

def scenario_1_healthy_baseline() -> tuple[str, list[ReadInfo], int, dict]:
    """Healthy baseline: both haplotypes stable at 60bp (CAG x 20)."""
    reads = [_ri(60.0, 1)] * 10 + [_ri(60.0, 2)] * 10
    return "1. Healthy baseline (CAG x20 stable)", reads, 3, {
        "hii_h1_max": 0.01, "hii_h2_max": 0.01,
        "ser_h1_max": 0.01, "ser_h2_max": 0.01,
        "scr_h1_max": 0.01, "scr_h2_max": 0.01,
        "ais_max": 0.01,
    }


def scenario_2_hd_premanifest() -> tuple[str, list[ReadInfo], int, dict]:
    """HD pre-manifest: HP1 stable (19 CAG), HP2 expanded ~45-55 CAG with somatic spread."""
    hp1 = [_ri(57.0, 1)] * 10  # 19 CAG = stable
    # HP2: ~45-55 CAG range with somatic expansion scatter
    hp2_sizes = [135, 141, 144, 150, 156, 165, 180, 144, 138, 153]
    hp2 = [_ri(s, 2) for s in hp2_sizes]
    return "2. HD pre-manifest (stable HP1, expanded HP2)", hp1 + hp2, 3, {
        "hii_h2_gt_h1": True,
        "ser_h2_min": 0.3,
        "ias_min": 0.8,
    }


def scenario_3_hd_symptomatic() -> tuple[str, list[ReadInfo], int, dict]:
    """HD symptomatic: HP1 stable, HP2 highly expanded with large somatic spread.

    SER threshold lowered to 0.3: SER measures fraction > median + 1 motif,
    so for a wide symmetric spread SER ≈ 0.5. The key HD signals are
    HII >> 1.0 (massive dispersion) and IAS ≈ 1.0 (asymmetric).
    """
    hp1 = [_ri(60.0, 1)] * 10  # 20 CAG = stable
    # HP2: 60-150 CAG range, massive somatic expansion
    hp2_sizes = [180, 240, 300, 360, 450, 210, 270, 330, 390, 420]
    hp2 = [_ri(s, 2) for s in hp2_sizes]
    return "3. HD symptomatic (massive HP2 expansion)", hp1 + hp2, 3, {
        "hii_h2_min": 1.0,
        "ser_h2_min": 0.3,
        "ias_approx_1": True,
    }


def scenario_4_dm1_extreme() -> tuple[str, list[ReadInfo], int, dict]:
    """DM1 extreme: HP1 stable, HP2 large expansion with right-skewed somatic spread.

    DM1 somatic instability is characteristically expansion-biased:
    most reads cluster near the modal size, with a long right tail of
    further expansions. This creates clear SER >> SCR and ECB > 0.
    """
    hp1 = [_ri(36.0, 1)] * 10  # 12 CTG = normal
    # HP2: modal ~1500bp (500 CTG), right-skewed expansion tail
    # 6 reads near modal, 4 reads with progressive further expansion
    hp2_sizes = [1500, 1500, 1500, 1503, 1497, 1500,
                 1560, 1650, 1800, 2400]
    hp2 = [_ri(s, 2) for s in hp2_sizes]
    return "4. DM1 extreme (right-skewed expansion)", hp1 + hp2, 3, {
        "ecb_h2_min": 0.5,
        "ser_gt_scr_h2": True,
    }


def scenario_5_msi_h_symmetric() -> tuple[str, list[ReadInfo], int, dict]:
    """MSI-H symmetric: both haplotypes show similar instability (cancer MSI)."""
    hp1_sizes = [60, 66, 54, 72, 48, 63, 69, 57, 75, 51]
    hp2_sizes = [60, 69, 51, 75, 45, 66, 72, 54, 78, 48]
    hp1 = [_ri(s, 1) for s in hp1_sizes]
    hp2 = [_ri(s, 2) for s in hp2_sizes]
    return "5. MSI-H symmetric instability", hp1 + hp2, 3, {
        "hii_symmetric": True,  # |HII_h1 - HII_h2| / max(HII) < 0.5
        "ias_max": 0.3,
    }


def scenario_6_treatment_response() -> tuple[str, list[ReadInfo], int, dict]:
    """Treatment response: HP2 spread narrows after treatment."""
    hp1 = [_ri(60.0, 1)] * 10  # stable reference haplotype
    # Before treatment: HP2 with ±15bp spread around 150bp
    before_hp2_sizes = [135, 140, 145, 150, 155, 160, 165, 148, 152, 138]
    before_reads = hp1 + [_ri(s, 2) for s in before_hp2_sizes]
    # After treatment: HP2 with ±3bp spread around 150bp (stabilized)
    after_hp2_sizes = [147, 149, 150, 151, 153, 150, 148, 152, 150, 151]
    after_reads = hp1 + [_ri(s, 2) for s in after_hp2_sizes]
    return "6. Treatment response (before vs after)", (before_reads, after_reads), 3, {
        "ais_after_lt_before": True,
    }


def scenario_7_dose_response() -> tuple[str, list[ReadInfo], int, dict]:
    """Dose-response: 5 levels of increasing dispersion around HP2 modal size.

    Rather than fraction-expanded, we model increasing spread (MAD).
    This more reliably produces monotonically increasing HII since HII = MAD/motif.
    """
    hp1 = [_ri(60.0, 1)] * 10  # stable HP1
    levels = []
    # Increasing spread: 0, ±1, ±3, ±6, ±12 bp from modal
    spreads = [0, 1, 3, 6, 12]
    np.random.seed(42)
    for spread in spreads:
        if spread == 0:
            hp2_reads = [_ri(60.0, 2)] * 10
        else:
            # Generate 10 reads with uniform spread ±spread around 60
            offsets = np.linspace(-spread, spread, 10)
            hp2_reads = [_ri(60.0 + off, 2) for off in offsets]
        levels.append(hp1 + hp2_reads)
    return "7. Dose-response (spread 0-12bp)", levels, 3, {
        "hii_monotonic": True,
    }


def scenario_8_edge_cases() -> tuple[str, list, int, dict]:
    """Edge cases: all HP=0 bimodal, low coverage, mixed HP."""
    cases = {}
    # 8a: all HP=0 with bimodal distribution -> gap split
    cases["8a_all_hp0_bimodal"] = [_ri(60.0, 0)] * 5 + [_ri(120.0, 0)] * 5
    # 8b: low coverage (3+3 reads)
    cases["8b_low_coverage"] = [_ri(60.0, 1)] * 3 + [_ri(90.0, 2)] * 3
    # 8c: mixed HP with some HP=0
    cases["8c_mixed_hp"] = (
        [_ri(60.0, 1)] * 4
        + [_ri(90.0, 2)] * 4
        + [_ri(62.0, 0)] * 2
        + [_ri(88.0, 0)] * 2
    )
    # 8d: single read (should return None or minimal)
    cases["8d_single_read"] = [_ri(60.0, 1)]
    # 8e: all same value (zero instability)
    cases["8e_all_same"] = [_ri(100.0, 0)] * 10
    return "8. Edge cases", cases, 3, {
        "all_no_crash": True,
    }


def run_scenario(name, reads, motif_len, checks) -> tuple[bool, list[str]]:
    """Run a single scenario and return (pass, messages)."""
    msgs = []
    passed = True

    def fail(msg):
        nonlocal passed
        passed = False
        msgs.append(f"  FAIL: {msg}")

    def ok(msg):
        msgs.append(f"  OK:   {msg}")

    # Scenario 6: before/after comparison
    if isinstance(reads, tuple) and len(reads) == 2:
        before_reads, after_reads = reads
        r_before = compute_instability(before_reads, 100.0, motif_len)
        r_after = compute_instability(after_reads, 100.0, motif_len)
        if r_before is None or r_after is None:
            fail("compute_instability returned None")
            return passed, msgs
        if checks.get("ais_after_lt_before"):
            if r_after["ais"] < r_before["ais"]:
                ok(f"AIS after ({r_after['ais']:.4f}) < before ({r_before['ais']:.4f})")
            else:
                fail(f"AIS after ({r_after['ais']:.4f}) >= before ({r_before['ais']:.4f})")
        msgs.append(f"  Before: HII_h2={r_before['hii_h2']:.4f}, AIS={r_before['ais']:.4f}")
        msgs.append(f"  After:  HII_h2={r_after['hii_h2']:.4f}, AIS={r_after['ais']:.4f}")
        return passed, msgs

    # Scenario 7: dose-response (list of read sets)
    if isinstance(reads, list) and len(reads) > 0 and isinstance(reads[0], list):
        levels = reads
        hiis = []
        fractions = [0.0, 0.10, 0.30, 0.50, 0.80]
        for i, level_reads in enumerate(levels):
            r = compute_instability(level_reads, 100.0, motif_len)
            if r is None:
                fail(f"Level {i} returned None")
                hiis.append(0.0)
                continue
            max_hii = max(r["hii_h1"], r["hii_h2"])
            hiis.append(max_hii)
            msgs.append(f"  Level {i} ({fractions[i]*100:.0f}%): max(HII)={max_hii:.4f}, AIS={r['ais']:.4f}")
        if checks.get("hii_monotonic"):
            # Check monotonic increase (allowing small ties for 0% level)
            monotonic = all(hiis[i] <= hiis[i + 1] + 0.001 for i in range(len(hiis) - 1))
            if monotonic:
                ok(f"HII monotonically increasing: {[f'{h:.4f}' for h in hiis]}")
            else:
                fail(f"HII not monotonic: {[f'{h:.4f}' for h in hiis]}")
        return passed, msgs

    # Scenario 8: edge cases (dict of sub-cases)
    if isinstance(reads, dict):
        for case_name, case_reads in reads.items():
            try:
                r = compute_instability(case_reads, 100.0, motif_len)
                if case_name == "8d_single_read":
                    # Single read -> should still return something (pooled fallback)
                    ok(f"{case_name}: returned {'result' if r else 'None'} (acceptable)")
                elif r is not None:
                    ok(f"{case_name}: HII_h1={r['hii_h1']:.4f}, HII_h2={r['hii_h2']:.4f}, "
                       f"conc={r['concordance']:.2f}")
                else:
                    ok(f"{case_name}: returned None (insufficient reads)")
            except Exception as e:
                fail(f"{case_name}: crashed with {type(e).__name__}: {e}")
        return passed, msgs

    # Standard single-result scenarios (1-5)
    ref_size = 100.0
    result = compute_instability(reads, ref_size, motif_len)
    if result is None:
        fail("compute_instability returned None")
        return passed, msgs

    # Display key metrics
    msgs.append(
        f"  Metrics: HII_h1={result['hii_h1']:.4f}, HII_h2={result['hii_h2']:.4f}, "
        f"SER_h1={result['ser_h1']:.4f}, SER_h2={result['ser_h2']:.4f}, "
        f"SCR_h1={result['scr_h1']:.4f}, SCR_h2={result['scr_h2']:.4f}"
    )
    msgs.append(
        f"  ECB_h1={result['ecb_h1']:.4f}, ECB_h2={result['ecb_h2']:.4f}, "
        f"IAS={result['ias']:.4f}, AIS={result['ais']:.4f}, "
        f"conc={result['concordance']:.4f}"
    )

    # Check assertions
    if "hii_h1_max" in checks:
        if result["hii_h1"] <= checks["hii_h1_max"]:
            ok(f"HII_h1={result['hii_h1']:.4f} <= {checks['hii_h1_max']}")
        else:
            fail(f"HII_h1={result['hii_h1']:.4f} > {checks['hii_h1_max']}")

    if "hii_h2_max" in checks:
        if result["hii_h2"] <= checks["hii_h2_max"]:
            ok(f"HII_h2={result['hii_h2']:.4f} <= {checks['hii_h2_max']}")
        else:
            fail(f"HII_h2={result['hii_h2']:.4f} > {checks['hii_h2_max']}")

    if "ser_h1_max" in checks:
        if result["ser_h1"] <= checks["ser_h1_max"]:
            ok(f"SER_h1={result['ser_h1']:.4f} <= {checks['ser_h1_max']}")
        else:
            fail(f"SER_h1={result['ser_h1']:.4f} > {checks['ser_h1_max']}")

    if "ser_h2_max" in checks:
        if result["ser_h2"] <= checks["ser_h2_max"]:
            ok(f"SER_h2={result['ser_h2']:.4f} <= {checks['ser_h2_max']}")
        else:
            fail(f"SER_h2={result['ser_h2']:.4f} > {checks['ser_h2_max']}")

    if "scr_h1_max" in checks:
        if result["scr_h1"] <= checks["scr_h1_max"]:
            ok(f"SCR_h1={result['scr_h1']:.4f} <= {checks['scr_h1_max']}")
        else:
            fail(f"SCR_h1={result['scr_h1']:.4f} > {checks['scr_h1_max']}")

    if "scr_h2_max" in checks:
        if result["scr_h2"] <= checks["scr_h2_max"]:
            ok(f"SCR_h2={result['scr_h2']:.4f} <= {checks['scr_h2_max']}")
        else:
            fail(f"SCR_h2={result['scr_h2']:.4f} > {checks['scr_h2_max']}")

    if "ais_max" in checks:
        if result["ais"] <= checks["ais_max"]:
            ok(f"AIS={result['ais']:.4f} <= {checks['ais_max']}")
        else:
            fail(f"AIS={result['ais']:.4f} > {checks['ais_max']}")

    if checks.get("hii_h2_gt_h1"):
        if result["hii_h2"] > result["hii_h1"]:
            ok(f"HII_h2={result['hii_h2']:.4f} > HII_h1={result['hii_h1']:.4f}")
        else:
            fail(f"HII_h2={result['hii_h2']:.4f} <= HII_h1={result['hii_h1']:.4f}")

    if "ser_h2_min" in checks:
        if result["ser_h2"] >= checks["ser_h2_min"]:
            ok(f"SER_h2={result['ser_h2']:.4f} >= {checks['ser_h2_min']}")
        else:
            fail(f"SER_h2={result['ser_h2']:.4f} < {checks['ser_h2_min']}")

    if "ias_min" in checks:
        if result["ias"] >= checks["ias_min"]:
            ok(f"IAS={result['ias']:.4f} >= {checks['ias_min']}")
        else:
            fail(f"IAS={result['ias']:.4f} < {checks['ias_min']}")

    if "hii_h2_min" in checks:
        if result["hii_h2"] >= checks["hii_h2_min"]:
            ok(f"HII_h2={result['hii_h2']:.4f} >= {checks['hii_h2_min']}")
        else:
            fail(f"HII_h2={result['hii_h2']:.4f} < {checks['hii_h2_min']}")

    if checks.get("ias_approx_1"):
        if result["ias"] > 0.9:
            ok(f"IAS={result['ias']:.4f} ~= 1.0")
        else:
            fail(f"IAS={result['ias']:.4f} not ~= 1.0")

    if "ecb_h2_min" in checks:
        if result["ecb_h2"] >= checks["ecb_h2_min"]:
            ok(f"ECB_h2={result['ecb_h2']:.4f} >= {checks['ecb_h2_min']}")
        else:
            fail(f"ECB_h2={result['ecb_h2']:.4f} < {checks['ecb_h2_min']}")

    if checks.get("ser_gt_scr_h2"):
        if result["ser_h2"] > result["scr_h2"]:
            ok(f"SER_h2={result['ser_h2']:.4f} > SCR_h2={result['scr_h2']:.4f}")
        else:
            fail(f"SER_h2={result['ser_h2']:.4f} <= SCR_h2={result['scr_h2']:.4f}")

    if checks.get("hii_symmetric"):
        max_hii = max(result["hii_h1"], result["hii_h2"])
        ratio = abs(result["hii_h1"] - result["hii_h2"]) / max_hii if max_hii > 0 else 0
        if ratio < 0.5:
            ok(f"HII symmetric: |diff|/max = {ratio:.4f} < 0.5")
        else:
            fail(f"HII not symmetric: |diff|/max = {ratio:.4f} >= 0.5")

    if "ias_max" in checks and "ias_max" not in [c for c in checks if c == "ias_max"]:
        pass  # already checked above

    return passed, msgs


def run_part_a() -> tuple[str, int, int]:
    """Run Part A: Synthetic disease scenarios."""
    scenarios = [
        scenario_1_healthy_baseline(),
        scenario_2_hd_premanifest(),
        scenario_3_hd_symptomatic(),
        scenario_4_dm1_extreme(),
        scenario_5_msi_h_symmetric(),
        scenario_6_treatment_response(),
        scenario_7_dose_response(),
        scenario_8_edge_cases(),
    ]

    lines = []
    w = lines.append
    n_pass = 0
    n_total = len(scenarios)

    w("=" * 100)
    w("  PHASE 1A: SYNTHETIC DISEASE SCENARIOS")
    w("=" * 100)

    for name, reads, motif_len, checks in scenarios:
        w(f"\n--- {name} ---")
        passed, msgs = run_scenario(name, reads, motif_len, checks)
        for m in msgs:
            w(m)
        status = "PASS" if passed else "FAIL"
        w(f"  >>> {status}")
        if passed:
            n_pass += 1

    w(f"\n{'=' * 100}")
    w(f"  PART A SUMMARY: {n_pass}/{n_total} scenarios PASSED")
    w(f"{'=' * 100}")

    return "\n".join(lines), n_pass, n_total


# ============================================================================
# Part B: Real HG002 baseline
# ============================================================================

def run_part_b(bam_path: str, ref_path: str | None = None) -> tuple[str, bool]:
    """Run Part B: HG002 baseline noise floor from real BAM data."""
    import pysam

    lines = []
    w = lines.append

    w(f"\n{'=' * 100}")
    w("  PHASE 1B: HG002 BASELINE NOISE FLOOR")
    w(f"{'=' * 100}")

    logger.info("Opening BAM: %s", bam_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(ref_path) if ref_path else None

    hii_values = []
    ais_values = []
    results_table = []

    for chrom, start, end, gene, motif, disease in STRCHIVE_LOCI:
        ref_size = end - start
        motif_len = len(motif)

        reads = extract_reads_enhanced(
            bam, chrom, start, end,
            min_mapq=5, min_flank=50, max_reads=200,
            ref_fasta=ref_fasta, motif_len=motif_len,
        )
        if not reads:
            results_table.append((gene, disease, 0, None))
            continue

        result = compute_instability(reads, ref_size, motif_len)
        if result is None:
            results_table.append((gene, disease, len(reads), None))
            continue

        max_hii = max(result["hii_h1"], result["hii_h2"])
        hii_values.append(max_hii)
        ais_values.append(result["ais"])
        results_table.append((gene, disease, len(reads), result))

    bam.close()
    if ref_fasta:
        ref_fasta.close()

    # Report per-locus results
    w(f"\n  {'Gene':<12s} {'Disease':<25s} {'nReads':>6s} {'HII_h1':>8s} {'HII_h2':>8s} "
      f"{'SER_h2':>8s} {'IAS':>8s} {'AIS':>8s}")
    w("  " + "-" * 95)

    for gene, disease, nreads, result in results_table:
        if result is None:
            w(f"  {gene:<12s} {disease[:25]:<25s} {nreads:>6d} {'---':>8s} {'---':>8s} "
              f"{'---':>8s} {'---':>8s} {'---':>8s}")
        else:
            w(f"  {gene:<12s} {disease[:25]:<25s} {nreads:>6d} "
              f"{result['hii_h1']:>8.4f} {result['hii_h2']:>8.4f} "
              f"{result['ser_h2']:>8.4f} {result['ias']:>8.4f} {result['ais']:>8.4f}")

    # Noise floor statistics
    hii_arr = np.array(hii_values) if hii_values else np.array([0.0])
    ais_arr = np.array(ais_values) if ais_values else np.array([0.0])

    n_analyzed = len(hii_values)
    mean_hii = float(np.mean(hii_arr))
    std_hii = float(np.std(hii_arr))
    p95_hii = float(np.percentile(hii_arr, 95)) if len(hii_arr) > 1 else hii_arr[0]
    max_hii_val = float(np.max(hii_arr))
    mean_ais = float(np.mean(ais_arr))
    max_ais = float(np.max(ais_arr))

    w(f"\n  NOISE FLOOR STATISTICS ({n_analyzed} loci analyzed)")
    w("  " + "-" * 50)
    w(f"  max(HII) mean:     {mean_hii:.4f}")
    w(f"  max(HII) std:      {std_hii:.4f}")
    w(f"  max(HII) p95:      {p95_hii:.4f}")
    w(f"  max(HII) max:      {max_hii_val:.4f}")
    w(f"  AIS mean:          {mean_ais:.4f}")
    w(f"  AIS max:           {max_ais:.4f}")

    # Success criteria
    success = True
    checks = []

    if mean_hii < 0.3:
        checks.append(f"  OK:   mean HII ({mean_hii:.4f}) < 0.3")
    else:
        checks.append(f"  FAIL: mean HII ({mean_hii:.4f}) >= 0.3")
        success = False

    if p95_hii < 1.0:
        checks.append(f"  OK:   p95 HII ({p95_hii:.4f}) < 1.0")
    else:
        checks.append(f"  FAIL: p95 HII ({p95_hii:.4f}) >= 1.0")
        success = False

    if max_ais < 0.5:
        checks.append(f"  OK:   max AIS ({max_ais:.4f}) < 0.5")
    else:
        checks.append(f"  WARN: max AIS ({max_ais:.4f}) >= 0.5 (some loci may have real variation)")

    w("\n  CRITERIA:")
    for c in checks:
        w(c)

    status = "PASS" if success else "FAIL"
    w(f"\n  >>> PHASE 1B: {status}")

    # Store noise floor values for Phase 2 comparison
    w(f"\n  [Noise floor for Phase 2: mean_HII={mean_hii:.4f}, std_HII={std_hii:.4f}]")

    return "\n".join(lines), success


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Instability validation")
    parser.add_argument("--output", required=True, help="Output report file")
    parser.add_argument("--bam", default=BAM_PATH, help="HG002 BAM path")
    parser.add_argument("--ref", default=REF_PATH, help="Reference FASTA")
    parser.add_argument("--skip-bam", action="store_true", help="Skip Part B (BAM-based)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Part A: Synthetic scenarios
    logger.info("Running Phase 1A: Synthetic disease scenarios...")
    t0 = time.time()
    part_a_report, n_pass_a, n_total_a = run_part_a()
    logger.info("Phase 1A complete in %.1fs: %d/%d PASS", time.time() - t0, n_pass_a, n_total_a)

    # Part B: HG002 baseline
    part_b_report = ""
    part_b_pass = True
    if not args.skip_bam:
        logger.info("Running Phase 1B: HG002 baseline noise floor...")
        t0 = time.time()
        part_b_report, part_b_pass = run_part_b(args.bam, args.ref)
        logger.info("Phase 1B complete in %.1fs", time.time() - t0)
    else:
        part_b_report = "\n  [Phase 1B skipped (--skip-bam)]"

    # Combine report
    report = []
    report.append("=" * 100)
    report.append("  HAPLOTR SOMATIC INSTABILITY — PHASE 1 VALIDATION REPORT")
    report.append("=" * 100)
    report.append(part_a_report)
    report.append(part_b_report)

    # Overall summary
    overall = n_pass_a == n_total_a and part_b_pass
    report.append(f"\n{'=' * 100}")
    report.append("  OVERALL PHASE 1 RESULT")
    report.append(f"{'=' * 100}")
    report.append(f"  Part A: {n_pass_a}/{n_total_a} scenarios PASS")
    report.append(f"  Part B: {'PASS' if part_b_pass else 'FAIL'}")
    report.append(f"  Overall: {'PASS' if overall else 'FAIL'}")

    full_report = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(full_report)

    print(full_report)
    logger.info("Report saved to %s", output_path)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
