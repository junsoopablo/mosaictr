#!/usr/bin/env python3
"""Check real carrier read distributions vs exponential model.

Extracts per-read allele sizes from carrier BAMs at the ATXN10 locus,
fits exponential distribution to the expanded haplotype, and compares.
"""

import sys
from pathlib import Path

import numpy as np
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.genotype import extract_reads_enhanced, ReadInfo

# ATXN10 locus (SCA10): chr22:45795354-45795424, motif ATTCT (5bp)
ATXN10 = ("chr22", 45795354, 45795424, "ATTCT")

CARRIERS = {
    "HG01122": "output/instability/1000g/HG01122_ATXN10_region.bam",
    "HG02252": "output/instability/1000g/HG02252_ATXN10_region.bam",
    "HG02345": "output/instability/1000g/HG02345_ATXN10_region.bam",
}


def analyze_carrier(name, bam_path):
    chrom, start, end, motif = ATXN10
    ref_size = end - start
    motif_len = len(motif)

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        reads = extract_reads_enhanced(bam, chrom, start, end, motif_len=motif_len)
    finally:
        bam.close()

    if not reads:
        print(f"  {name}: no reads extracted")
        return None

    # Separate by HP
    hp1 = [r.allele_size for r in reads if r.hp == 1]
    hp2 = [r.allele_size for r in reads if r.hp == 2]
    hp0 = [r.allele_size for r in reads if r.hp == 0]

    print(f"\n{'='*60}")
    print(f"  {name} — ATXN10 (ref={ref_size}bp, motif={motif})")
    print(f"{'='*60}")
    print(f"  Reads: HP1={len(hp1)}, HP2={len(hp2)}, HP0={len(hp0)}")

    from scipy.stats import skew, kurtosis, expon, kstest
    from scipy.stats import gamma as gamma_dist

    for label, sizes in [("HP1", hp1), ("HP2", hp2)]:
        if len(sizes) < 3:
            print(f"  {label}: too few reads ({len(sizes)})")
            continue

        arr = np.array(sizes)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        hii = mad / motif_len
        mean_val = np.mean(arr)
        std_val = np.std(arr)

        print(f"\n  {label}: n={len(arr)}")
        print(f"    Median = {med:.1f} bp ({med/motif_len:.1f} RU)")
        print(f"    Mean   = {mean_val:.1f} bp")
        print(f"    MAD    = {mad:.1f} bp, HII = {hii:.2f}")
        print(f"    SD     = {std_val:.1f} bp")
        print(f"    Range  = [{np.min(arr):.0f}, {np.max(arr):.0f}] bp")

        # Skewness
        sk = skew(arr)
        ku = kurtosis(arr)
        print(f"    Skewness = {sk:.3f} (>0 = right-skewed/expansion-biased)")
        print(f"    Kurtosis = {ku:.3f}")

        # Fraction above vs below median
        above = np.sum(arr > med + motif_len)
        below = np.sum(arr < med - motif_len)
        print(f"    Reads > median+1unit: {above} ({above/len(arr)*100:.1f}%)")
        print(f"    Reads < median-1unit: {below} ({below/len(arr)*100:.1f}%)")

        # Fit distributions to unstable haplotype
        if hii > 0.5:
            deviations = arr - np.min(arr)
            deviations = deviations[deviations > 0]
            if len(deviations) > 5:
                # Exponential fit
                loc_e, scale_e = expon.fit(deviations, floc=0)
                ks_e, p_e = kstest(deviations, 'expon', args=(0, scale_e))
                print(f"\n    Exponential fit (deviations from min):")
                print(f"      Scale (mean) = {scale_e:.1f} bp")
                print(f"      KS stat = {ks_e:.3f}, p = {p_e:.4f}")
                print(f"      {'CONSISTENT' if p_e > 0.05 else 'REJECTED'} with exponential (p>0.05)")

                # Gamma fit
                a_gam, loc_gam, scale_gam = gamma_dist.fit(deviations, floc=0)
                ks_g, p_g = kstest(deviations, 'gamma', args=(a_gam, 0, scale_gam))
                print(f"\n    Gamma fit:")
                print(f"      Shape={a_gam:.2f}, Scale={scale_gam:.1f}")
                print(f"      KS stat = {ks_g:.3f}, p = {p_g:.4f}")
                print(f"      {'CONSISTENT' if p_g > 0.05 else 'REJECTED'} with gamma")

                # Percentiles
                pcts = [5, 25, 50, 75, 95]
                print(f"\n    Percentiles (bp): ", end="")
                for p in pcts:
                    print(f"p{p}={np.percentile(arr, p):.0f}", end="  ")
                print()

    return True


def main():
    print("Analyzing real carrier read distributions at ATXN10 locus")
    print("Checking if expansion-biased exponential model is realistic\n")

    for name, bam in CARRIERS.items():
        if Path(bam).exists():
            try:
                analyze_carrier(name, bam)
            except Exception as e:
                import traceback
                print(f"  {name}: ERROR - {e}")
                traceback.print_exc()
        else:
            print(f"  {name}: BAM not found at {bam}")


if __name__ == "__main__":
    main()
