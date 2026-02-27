#!/usr/bin/env python3
"""TP gap analysis v5: Haplotag-aware genotyping.

Key insight: LongTR uses HP:i: haplotag annotations to deterministically
separate reads into haplotypes. This turns het genotyping from a statistical
problem (midpoint split, EM) into a trivial grouping problem.

The HG002 BAM has 54-74% of reads tagged with HP:i:1 or HP:i:2.

Strategies:
1. hp_mode: separate by HP tag → mode of each group
2. hp_propmode: HP separation + proportion-based hom collapse
3. hp_fallback: HP when available, PropMode otherwise
4. hp_em: HP tags as EM initialization
"""

import argparse
import bisect
import gzip
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pysam

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_CHROMS = {"chr21", "chr22", "chrX"}
MATCH_TOLERANCE = 15
MIN_FLANK = 50
MIN_MAPQ = 5
MAX_READS = 200


# ── CIGAR-based allele size ──────────────────────────────────────────

def compute_allele_size(aln, locus_start, locus_end):
    """Compute allele size (query bases within locus) from CIGAR."""
    if aln.cigartuples is None:
        return None
    ref_pos = aln.reference_start
    qbases = 0
    for op, length in aln.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            ov_start = max(ref_pos, locus_start)
            ov_end = min(ref_pos + length, locus_end)
            if ov_start < ov_end:
                qbases += (ov_end - ov_start)
            ref_pos += length
        elif op == 1:  # I
            if locus_start <= ref_pos <= locus_end:
                qbases += length
        elif op in (2, 3):  # D, N
            ref_pos += length
        # 4=S, 5=H: skip
    return qbases if qbases > 0 else None


# ── Read extraction from BAM ─────────────────────────────────────────

def extract_reads_with_hp(bam, chrom, start, end):
    """Extract allele sizes and HP tags from BAM reads at a locus.

    Returns list of (allele_size_bp, hp_tag) tuples.
    hp_tag: 0=untagged, 1=hap1, 2=hap2
    """
    reads = []
    seen = set()
    try:
        fetched = bam.fetch(chrom, max(0, start - MIN_FLANK), end + MIN_FLANK)
    except ValueError:
        return reads

    for aln in fetched:
        if len(reads) >= MAX_READS:
            break
        if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
            continue
        if aln.mapping_quality < MIN_MAPQ:
            continue
        if aln.query_name in seen:
            continue
        seen.add(aln.query_name)

        # Check span
        if aln.reference_start is None or aln.reference_end is None:
            continue
        if aln.reference_start > start - MIN_FLANK:
            continue
        if aln.reference_end < end + MIN_FLANK:
            continue

        allele_size = compute_allele_size(aln, start, end)
        if allele_size is None:
            continue

        try:
            hp = aln.get_tag('HP')
        except KeyError:
            hp = 0

        reads.append((float(allele_size), int(hp)))

    return reads


# ── Genotyping methods ───────────────────────────────────────────────

def get_mode(arr):
    if len(arr) == 0:
        return 0.0
    int_arr = np.round(arr).astype(int)
    counts = np.bincount(int_arr - int_arr.min())
    return float(int_arr.min() + np.argmax(counts))


def mode_round_genotype(sizes, ref_size):
    """Baseline: midpoint split + mode."""
    sizes = np.sort(sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def prop_mode_genotype(sizes, ref_size, threshold=0.30):
    """PropMode (prop_30pct): best classical method."""
    sizes = np.sort(sizes)
    n = len(sizes)
    if n == 0: return 0.0, 0.0
    if n == 1:
        d = round(ref_size - sizes[0])
        return d, d
    mid = n // 2
    m1 = get_mode(sizes[:mid])
    m2 = get_mode(sizes[mid:])
    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        int_sizes = np.round(sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / n
        if minor_frac < threshold:
            m_all = get_mode(sizes)
            d = round(ref_size - m_all)
            return d, d
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_mode_genotype(sizes_hp, ref_size):
    """Haplotag-aware mode: separate by HP tag, mode of each group.

    sizes_hp: list of (allele_size, hp_tag) tuples
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    untagged = np.array([s for s, h in sizes_hp if h == 0])
    all_sizes = np.array([s for s, _ in sizes_hp])

    # If no tagged reads, fall back to mode_round
    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)

    # If only one haplotype has reads
    if len(hp1) == 0:
        m = get_mode(hp2)
        d = round(ref_size - m)
        return d, d
    if len(hp2) == 0:
        m = get_mode(hp1)
        d = round(ref_size - m)
        return d, d

    # Both haplotypes have reads → mode of each
    m1 = get_mode(hp1)
    m2 = get_mode(hp2)
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_propmode_genotype(sizes_hp, ref_size, threshold=0.30):
    """Haplotag mode + proportion-based collapse.

    When HP tags separate reads, compute modes. If modes differ by only 1bp,
    check proportion to decide het vs hom.
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return prop_mode_genotype(all_sizes, ref_size, threshold)

    if len(hp1) == 0:
        m = get_mode(hp2)
        d = round(ref_size - m)
        return d, d
    if len(hp2) == 0:
        m = get_mode(hp1)
        d = round(ref_size - m)
        return d, d

    m1 = get_mode(hp1)
    m2 = get_mode(hp2)

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) == 1:
        # Use proportion across ALL reads (tagged + untagged)
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(int_sizes == int(m1))
        c2 = np.sum(int_sizes == int(m2))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < threshold:
            m_all = get_mode(all_sizes)
            d = round(ref_size - m_all)
            return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_em_genotype(sizes_hp, ref_size):
    """HP-initialized EM: use HP tags to initialize, then EM with all reads.

    Key improvement over basic EM: initialization uses ground-truth phasing
    instead of midpoint split.
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)

    # Initial centers from HP tags
    if len(hp1) == 0:
        m1 = get_mode(hp2)
        m2 = m1
    elif len(hp2) == 0:
        m1 = get_mode(hp1)
        m2 = m1
    else:
        m1 = get_mode(hp1)
        m2 = get_mode(hp2)

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d

    # EM: re-assign ALL reads (including untagged) to nearest center
    int_sizes = np.round(all_sizes).astype(int)
    for _ in range(5):
        m1_int = round(m1)
        m2_int = round(m2)
        dist1 = np.abs(int_sizes - m1_int)
        dist2 = np.abs(int_sizes - m2_int)
        assign1 = dist1 <= dist2

        n = len(all_sizes)
        if assign1.sum() == 0 or assign1.sum() == n:
            break

        new_m1 = get_mode(all_sizes[assign1])
        new_m2 = get_mode(all_sizes[~assign1])
        if new_m1 == m1 and new_m2 == m2:
            break
        m1, m2 = new_m1, new_m2

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_fallback_genotype(sizes_hp, ref_size, min_tagged_frac=0.3):
    """HP mode when enough reads are tagged, PropMode otherwise."""
    if not sizes_hp:
        return 0.0, 0.0

    all_sizes = np.array([s for s, _ in sizes_hp])
    n_tagged = sum(1 for _, h in sizes_hp if h > 0)
    tagged_frac = n_tagged / len(sizes_hp) if sizes_hp else 0

    if tagged_frac >= min_tagged_frac:
        return hp_mode_genotype(sizes_hp, ref_size)
    else:
        return prop_mode_genotype(all_sizes, ref_size)


def hp_assign_untagged_genotype(sizes_hp, ref_size):
    """HP mode + assign untagged reads to nearest allele.

    1. Get allele centers from HP:1 and HP:2 reads
    2. Assign untagged reads to nearest center
    3. Recompute modes with augmented groups
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = [s for s, h in sizes_hp if h == 1]
    hp2 = [s for s, h in sizes_hp if h == 2]
    untagged = [s for s, h in sizes_hp if h == 0]
    all_sizes = np.array([s for s, _ in sizes_hp])

    if not hp1 and not hp2:
        return prop_mode_genotype(all_sizes, ref_size)

    if not hp1:
        m = get_mode(np.array(hp2))
        d = round(ref_size - m)
        return d, d
    if not hp2:
        m = get_mode(np.array(hp1))
        d = round(ref_size - m)
        return d, d

    # Initial centers
    m1 = get_mode(np.array(hp1))
    m2 = get_mode(np.array(hp2))

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d

    # Assign untagged reads to nearest center
    augmented1 = list(hp1)
    augmented2 = list(hp2)
    for s in untagged:
        if abs(s - m1) <= abs(s - m2):
            augmented1.append(s)
        else:
            augmented2.append(s)

    # Recompute modes
    m1 = get_mode(np.array(augmented1))
    m2 = get_mode(np.array(augmented2))

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def get_motif_mode(arr, ref_size, motif_len):
    """Mode after rounding allele sizes to nearest motif-multiple position.

    For a repeat with motif_len=3, valid allele sizes relative to reference
    are: ref ± 0, ±3, ±6, ±9, ... This filters out ±1-2bp sequencing noise.
    """
    if len(arr) == 0:
        return ref_size
    if motif_len <= 1:
        return get_mode(arr)  # homopolymers: every bp is a motif multiple

    # Express as diffs from reference, round to nearest motif multiple
    diffs = arr - ref_size  # negative = deletion, positive = insertion
    rounded_diffs = np.round(diffs / motif_len) * motif_len
    rounded_sizes = ref_size + rounded_diffs

    # Take mode of rounded sizes
    return get_mode(rounded_sizes)


def hp_motif_mode_genotype(sizes_hp, ref_size, motif_len):
    """HP separation + motif-aware mode within each group.

    Mimics LongTR's stutter model: true alleles differ from reference
    by exact multiples of motif length.
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        m = get_motif_mode(all_sizes, ref_size, motif_len)
        d = round(ref_size - m)
        return d, d

    if len(hp1) == 0:
        m = get_motif_mode(hp2, ref_size, motif_len)
        d = round(ref_size - m)
        return d, d
    if len(hp2) == 0:
        m = get_motif_mode(hp1, ref_size, motif_len)
        d = round(ref_size - m)
        return d, d

    m1 = get_motif_mode(hp1, ref_size, motif_len)
    m2 = get_motif_mode(hp2, ref_size, motif_len)
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_motif_propmode_genotype(sizes_hp, ref_size, motif_len, threshold=0.30):
    """HP + motif-aware mode + proportion-based collapse."""
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return prop_mode_genotype(all_sizes, ref_size, threshold)

    if len(hp1) == 0:
        m = get_motif_mode(hp2, ref_size, motif_len)
        d = round(ref_size - m)
        return d, d
    if len(hp2) == 0:
        m = get_motif_mode(hp1, ref_size, motif_len)
        d = round(ref_size - m)
        return d, d

    m1 = get_motif_mode(hp1, ref_size, motif_len)
    m2 = get_motif_mode(hp2, ref_size, motif_len)

    if m1 == m2:
        d = round(ref_size - m1)
        return d, d
    elif abs(m1 - m2) <= motif_len:
        # Collapse if minor allele has low proportion
        int_sizes = np.round(all_sizes).astype(int)
        c1 = np.sum(np.abs(int_sizes - int(m1)) < motif_len / 2)
        c2 = np.sum(np.abs(int_sizes - int(m2)) < motif_len / 2)
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < threshold:
            m_all = get_motif_mode(all_sizes, ref_size, motif_len)
            d = round(ref_size - m_all)
            return d, d

    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_median_genotype(sizes_hp, ref_size):
    """HP separation + median within each group (more robust to outliers)."""
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return mode_round_genotype(all_sizes, ref_size)

    if len(hp1) == 0:
        d = round(ref_size - np.median(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - np.median(hp1))
        return d, d

    m1 = np.median(hp1)
    m2 = np.median(hp2)
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


def hp_median_prop_genotype(sizes_hp, ref_size, threshold=0.30):
    """HP median + proportion-based hom collapse.

    Best of both worlds: hp_median's het strength + PropMode's hom strength.
    1. Compute median of each HP group
    2. If medians round to same integer: hom
    3. If medians round to values differing by 1: check proportion → collapse
    4. Otherwise: het
    """
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return prop_mode_genotype(all_sizes, ref_size, threshold)

    if len(hp1) == 0:
        d = round(ref_size - np.median(hp2))
        return d, d
    if len(hp2) == 0:
        d = round(ref_size - np.median(hp1))
        return d, d

    med1 = np.median(hp1)
    med2 = np.median(hp2)
    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)

    if d1 == d2:
        return d1, d2
    elif abs(d1 - d2) == 1:
        # Check proportion of reads supporting each allele
        int_sizes = np.round(all_sizes).astype(int)
        s1 = round(med1)
        s2 = round(med2)
        c1 = np.sum(int_sizes == int(s1))
        c2 = np.sum(int_sizes == int(s2))
        minor_frac = min(c1, c2) / len(all_sizes)
        if minor_frac < threshold:
            m_all = np.median(all_sizes)
            d = round(ref_size - m_all)
            return d, d

    return sorted([d1, d2])


def hp_median_assign_genotype(sizes_hp, ref_size):
    """HP median + assign untagged reads to nearest allele + recompute median."""
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = [s for s, h in sizes_hp if h == 1]
    hp2 = [s for s, h in sizes_hp if h == 2]
    untagged = [s for s, h in sizes_hp if h == 0]
    all_sizes = np.array([s for s, _ in sizes_hp])

    if not hp1 and not hp2:
        return prop_mode_genotype(all_sizes, ref_size)

    if not hp1:
        d = round(ref_size - np.median(hp2))
        return d, d
    if not hp2:
        d = round(ref_size - np.median(hp1))
        return d, d

    med1 = np.median(hp1)
    med2 = np.median(hp2)

    # Assign untagged reads to nearest center
    aug1 = list(hp1)
    aug2 = list(hp2)
    for s in untagged:
        if abs(s - med1) <= abs(s - med2):
            aug1.append(s)
        else:
            aug2.append(s)

    med1 = np.median(aug1)
    med2 = np.median(aug2)
    d1 = round(ref_size - med1)
    d2 = round(ref_size - med2)
    return sorted([d1, d2])


def hp_trimmed_mode_genotype(sizes_hp, ref_size, trim_frac=0.1):
    """HP separation + trimmed mode (remove outlier reads before mode)."""
    if not sizes_hp:
        return 0.0, 0.0

    hp1 = np.array([s for s, h in sizes_hp if h == 1])
    hp2 = np.array([s for s, h in sizes_hp if h == 2])
    all_sizes = np.array([s for s, _ in sizes_hp])

    if len(hp1) == 0 and len(hp2) == 0:
        return prop_mode_genotype(all_sizes, ref_size)

    def trimmed_mode(arr):
        if len(arr) <= 3:
            return get_mode(arr)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return get_mode(arr)
        # Remove reads > 3 MAD from median
        mask = np.abs(arr - med) <= 3 * max(mad, 1)
        if mask.sum() == 0:
            return get_mode(arr)
        return get_mode(arr[mask])

    if len(hp1) == 0:
        m = trimmed_mode(hp2)
        d = round(ref_size - m)
        return d, d
    if len(hp2) == 0:
        m = trimmed_mode(hp1)
        d = round(ref_size - m)
        return d, d

    m1 = trimmed_mode(hp1)
    m2 = trimmed_mode(hp2)
    d1 = round(ref_size - m1)
    d2 = round(ref_size - m2)
    return sorted([d1, d2])


# ── LongTR VCF parsing ──────────────────────────────────────────────

def parse_longtr_vcf(vcf_path, chroms):
    result = {}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            if chrom not in chroms:
                continue
            pos = int(parts[1]) - 1
            ref = parts[3]
            alts = parts[4].split(",")
            gt_field = parts[9].split(":")
            gt = gt_field[0].replace("|", "/").split("/")
            alleles_seq = [ref] + alts
            ref_size = len(ref)
            diffs = []
            for g in gt:
                if g == ".":
                    diffs.append(0)
                else:
                    a = alleles_seq[int(g)]
                    diffs.append(ref_size - len(a))
            if len(diffs) == 1:
                diffs = diffs * 2
            result[(chrom, pos)] = (sorted(diffs)[0], sorted(diffs)[1], ref_size)
    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TP gap v5: Haplotag-aware genotyping")
    parser.add_argument("--h5", required=True, help="HDF5 features file")
    parser.add_argument("--bam", required=True, help="BAM file with HP tags")
    parser.add_argument("--longtr-vcf", required=True, help="LongTR VCF")
    parser.add_argument("--output", required=True, help="Output report path")
    parser.add_argument("--cache", default=None, help="Cache file for extracted reads (pickle)")
    args = parser.parse_args()

    # Load HDF5 for locus positions and ground truth
    logger.info(f"Loading HDF5: {args.h5}")
    h5 = h5py.File(args.h5, "r")
    all_chroms = [c.decode() for c in h5["chroms"][:]]
    all_starts = h5["starts"][:]
    all_ends = h5["ends"][:]
    all_labels = h5["labels"][:]
    all_locus_features = h5["locus_features"][:]
    all_read_features = h5["read_features"][:]
    all_read_offsets = h5["read_offsets"][:]
    all_read_counts = h5["read_counts"][:]
    tp_statuses = [s.decode() for s in h5["tp_statuses"][:]]

    test_indices = [i for i, c in enumerate(all_chroms) if c in TEST_CHROMS]
    n_test = len(test_indices)
    logger.info(f"Test set: {n_test:,} loci")

    # Collect locus info
    h5_keys = []
    true_diffs = []
    motif_lens = []
    is_tp = []
    ref_sizes = []
    for idx in test_indices:
        chrom = all_chroms[idx]
        start = int(all_starts[idx])
        end = int(all_ends[idx])
        h5_keys.append((chrom, start, end))
        h1, h2, ml = all_labels[idx]
        true_diffs.append(sorted([h1, h2]))
        motif_lens.append(ml)
        ref_sizes.append(all_locus_features[idx, 0])
        is_tp.append("TP" in tp_statuses[idx])

    true_diffs = np.array(true_diffs)
    motif_lens = np.array(motif_lens)
    is_tp = np.array(is_tp)
    ref_sizes = np.array(ref_sizes)

    # Compute HDF5-based methods (mode_round, prop_30pct) for comparison
    logger.info("Computing HDF5-based baselines (mode_round, prop_30pct)...")
    h5_mode_round = np.zeros((n_test, 2))
    h5_prop30 = np.zeros((n_test, 2))
    for i, idx in enumerate(test_indices):
        offset = all_read_offsets[idx]
        count = all_read_counts[idx]
        reads = all_read_features[offset:offset + count]
        allele_sizes = reads[:, 0]
        ref = all_locus_features[idx, 0]
        h5_mode_round[i] = mode_round_genotype(allele_sizes, ref)
        h5_prop30[i] = prop_mode_genotype(allele_sizes, ref)

    h5.close()

    # Extract reads with HP tags from BAM (with caching)
    cache_path = Path(args.cache) if args.cache else Path(args.output).with_suffix(".cache.pkl")
    if cache_path.exists():
        logger.info(f"Loading cached reads from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        all_reads_hp = cache["all_reads_hp"]
        logger.info(f"Loaded {len(all_reads_hp)} loci from cache")
    else:
        logger.info(f"Opening BAM: {args.bam}")
        bam = pysam.AlignmentFile(args.bam, "rb")
        logger.info("Extracting reads with HP tags from BAM...")
        all_reads_hp = []
        for i, (chrom, start, end) in enumerate(h5_keys):
            if i % 5000 == 0 and i > 0:
                logger.info(f"  Processed {i:,}/{n_test:,} loci...")
            reads_hp = extract_reads_with_hp(bam, chrom, start, end)
            all_reads_hp.append(reads_hp)
        bam.close()

        # Save cache
        logger.info(f"Saving cache to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"all_reads_hp": all_reads_hp}, f)

    # Compute all methods
    logger.info("Computing genotypes for all methods...")
    hp_methods = {
        "hp_mode": np.zeros((n_test, 2)),
        "hp_propmode": np.zeros((n_test, 2)),
        "hp_em": np.zeros((n_test, 2)),
        "hp_fallback30": np.zeros((n_test, 2)),
        "hp_assign": np.zeros((n_test, 2)),
        "hp_motif_mode": np.zeros((n_test, 2)),
        "hp_motif_prop": np.zeros((n_test, 2)),
        "hp_median": np.zeros((n_test, 2)),
        "hp_median_prop": np.zeros((n_test, 2)),
        "hp_median_assign": np.zeros((n_test, 2)),
        "hp_trimmed": np.zeros((n_test, 2)),
        "hp_med_p15": np.zeros((n_test, 2)),
        "hp_med_p20": np.zeros((n_test, 2)),
        "hp_med_p25": np.zeros((n_test, 2)),
        "hp_med_p35": np.zeros((n_test, 2)),
        "hp_med_p40": np.zeros((n_test, 2)),
        "hp_cond_v1": np.zeros((n_test, 2)),
        "hp_cond_v2": np.zeros((n_test, 2)),
    }
    bam_mode_round = np.zeros((n_test, 2))
    bam_prop30 = np.zeros((n_test, 2))

    # Stats
    n_with_hp = 0
    n_hp_both = 0
    total_reads = 0
    tagged_reads = 0
    loci_read_counts = []

    for i in range(n_test):
        reads_hp = all_reads_hp[i]
        ref = ref_sizes[i]
        ml = motif_lens[i]
        loci_read_counts.append(len(reads_hp))

        if not reads_hp:
            bam_mode_round[i] = h5_mode_round[i]
            bam_prop30[i] = h5_prop30[i]
            for name in hp_methods:
                hp_methods[name][i] = h5_prop30[i]
            continue

        all_sizes = np.array([s for s, _ in reads_hp])
        n_tagged = sum(1 for _, h in reads_hp if h > 0)
        has_hp1 = any(h == 1 for _, h in reads_hp)
        has_hp2 = any(h == 2 for _, h in reads_hp)

        total_reads += len(reads_hp)
        tagged_reads += n_tagged
        if n_tagged > 0:
            n_with_hp += 1
        if has_hp1 and has_hp2:
            n_hp_both += 1

        # BAM-based baselines
        bam_mode_round[i] = mode_round_genotype(all_sizes, ref)
        bam_prop30[i] = prop_mode_genotype(all_sizes, ref)

        # HP methods
        hp_methods["hp_mode"][i] = hp_mode_genotype(reads_hp, ref)
        hp_methods["hp_propmode"][i] = hp_propmode_genotype(reads_hp, ref)
        hp_methods["hp_em"][i] = hp_em_genotype(reads_hp, ref)
        hp_methods["hp_fallback30"][i] = hp_fallback_genotype(reads_hp, ref, 0.3)
        hp_methods["hp_assign"][i] = hp_assign_untagged_genotype(reads_hp, ref)
        # New methods
        hp_methods["hp_motif_mode"][i] = hp_motif_mode_genotype(reads_hp, ref, ml)
        hp_methods["hp_motif_prop"][i] = hp_motif_propmode_genotype(reads_hp, ref, ml)
        hp_methods["hp_median"][i] = hp_median_genotype(reads_hp, ref)
        hp_methods["hp_median_prop"][i] = hp_median_prop_genotype(reads_hp, ref)
        hp_methods["hp_median_assign"][i] = hp_median_assign_genotype(reads_hp, ref)
        hp_methods["hp_trimmed"][i] = hp_trimmed_mode_genotype(reads_hp, ref)
        # Threshold sweep for hp_median_prop
        for t in [0.15, 0.20, 0.25, 0.35, 0.40]:
            name = f"hp_med_p{int(t*100)}"
            hp_methods[name][i] = hp_median_prop_genotype(reads_hp, ref, threshold=t)
        # Conditional: hp_median for VNTR (7+), hp_med_p25 for dinuc/STR
        if ml >= 7:
            hp_methods["hp_cond_v1"][i] = hp_median_genotype(reads_hp, ref)
        else:
            hp_methods["hp_cond_v1"][i] = hp_median_prop_genotype(reads_hp, ref, threshold=0.25)
        # Conditional v2: hp_median for VNTR, hp_med_p30 for rest
        if ml >= 7:
            hp_methods["hp_cond_v2"][i] = hp_median_genotype(reads_hp, ref)
        else:
            hp_methods["hp_cond_v2"][i] = hp_median_prop_genotype(reads_hp, ref, threshold=0.30)

    # Parse LongTR
    logger.info("Parsing LongTR VCF...")
    longtr_dict = parse_longtr_vcf(args.longtr_vcf, TEST_CHROMS)
    by_chrom = defaultdict(list)
    for (chrom, start), val in longtr_dict.items():
        by_chrom[chrom].append((start, val))
    for chrom in by_chrom:
        by_chrom[chrom].sort()

    longtr_mask = np.zeros(n_test, dtype=bool)
    longtr_diffs = np.zeros((n_test, 2))
    for i, (chrom, start, end) in enumerate(h5_keys):
        candidates = by_chrom.get(chrom, [])
        if not candidates:
            continue
        h5_ref = ref_sizes[i]
        pos = bisect.bisect_left(candidates, (start,))
        best_dist = MATCH_TOLERANCE + 1
        best_val = None
        for j in range(max(0, pos - 2), min(len(candidates), pos + 3)):
            dist = abs(candidates[j][0] - start)
            tool_ref = candidates[j][1][2]
            if h5_ref > 0 and tool_ref > 0:
                ratio = tool_ref / h5_ref
                if ratio < 0.5 or ratio > 2.0:
                    continue
            if dist < best_dist:
                best_dist = dist
                best_val = candidates[j][1]
        if best_dist <= MATCH_TOLERANCE and best_val is not None:
            longtr_mask[i] = True
            longtr_diffs[i] = [best_val[0], best_val[1]]

    # ── Report ───────────────────────────────────────────────────────
    lines = []
    def p(s=""): lines.append(s)

    THRESH = 0.5
    is_het_truth = np.abs(true_diffs[:, 1] - true_diffs[:, 0]) > 0.5
    tp_ltr = is_tp & longtr_mask
    tn_ltr = longtr_mask & ~is_tp
    het_m = tp_ltr & is_het_truth
    hom_m = tp_ltr & ~is_het_truth
    n_tp = tp_ltr.sum()

    p("=" * 110)
    p("  TP GAP ANALYSIS v5: HAPLOTAG-AWARE GENOTYPING")
    p(f"  Test set: chr21, chr22, chrX (n={n_test:,})")
    p(f"  TP n={n_tp:,} (Het={het_m.sum():,}, Hom={hom_m.sum():,})")
    p("=" * 110)
    p()

    # HP coverage stats
    p("  HP COVERAGE STATS:")
    p(f"    Loci with any HP-tagged reads: {n_with_hp:,}/{n_test:,} ({n_with_hp/n_test:.1%})")
    p(f"    Loci with both HP:1 and HP:2:  {n_hp_both:,}/{n_test:,} ({n_hp_both/n_test:.1%})")
    p(f"    Total reads: {total_reads:,}, Tagged: {tagged_reads:,} ({tagged_reads/total_reads:.1%})")
    rc = np.array(loci_read_counts)
    p(f"    Reads/locus: median={np.median(rc):.0f}, mean={rc.mean():.1f}, p25={np.percentile(rc,25):.0f}, p75={np.percentile(rc,75):.0f}")
    p()

    # Sanity check: BAM vs HDF5 mode_round
    bam_mr_exact = np.max(np.abs(bam_mode_round - true_diffs), axis=1) < THRESH
    h5_mr_exact = np.max(np.abs(h5_mode_round - true_diffs), axis=1) < THRESH
    agree = (bam_mr_exact == h5_mr_exact).mean()
    p(f"  SANITY CHECK: BAM vs HDF5 mode_round agreement = {agree:.1%}")
    p(f"    BAM mode_round overall: {bam_mr_exact.mean():.1%}")
    p(f"    HDF5 mode_round overall: {h5_mr_exact.mean():.1%}")
    p()

    # All methods
    all_methods = {
        "mode_round(H5)": h5_mode_round,
        "prop_30pct(H5)": h5_prop30,
        "mode_round(BAM)": bam_mode_round,
        "prop_30pct(BAM)": bam_prop30,
        **hp_methods,
    }

    # ── LongTR set results ────────────────────────────────────────
    p("=" * 110)
    p(f"  OVERALL — LongTR set (n={longtr_mask.sum():,})")
    p("=" * 110)
    p()
    p(f"{'Method':<22} {'Overall':>8} {'TP':>8} {'TN':>8} {'Het-TP':>8} {'Hom-TP':>8}  {'TPw1bp':>7} {'TPMAE':>7}")
    p("-" * 100)

    for name in list(all_methods.keys()) + ["LongTR"]:
        diffs = all_methods[name] if name != "LongTR" else longtr_diffs
        errs = np.max(np.abs(diffs - true_diffs), axis=1)
        exact = errs < THRESH
        ov = exact[longtr_mask].mean()
        tp = exact[tp_ltr].mean()
        tn = exact[tn_ltr].mean()
        het = exact[het_m].mean() if het_m.sum() > 0 else 0
        hom = exact[hom_m].mean() if hom_m.sum() > 0 else 0
        flat = np.abs(diffs[tp_ltr] - true_diffs[tp_ltr]).flatten()
        w1 = (flat <= 1.0).mean()
        mae = flat.mean()
        p(f"{name:<22} {ov:>7.1%} {tp:>7.1%} {tn:>7.1%} {het:>7.1%} {hom:>7.1%}  {w1:>6.1%} {mae:>6.2f}")

    p()

    # ── NET gains vs prop_30pct ───────────────────────────────────
    p("=" * 110)
    p("  NET vs prop_30pct(H5) (TP, LongTR set)")
    p("=" * 110)
    p()
    ref_exact = np.max(np.abs(h5_prop30 - true_diffs), axis=1) < THRESH
    for name, diffs in hp_methods.items():
        n_exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
        gains = (tp_ltr & n_exact & ~ref_exact).sum()
        losses = (tp_ltr & ref_exact & ~n_exact).sum()
        p(f"  {name:<20} gains={gains:>4} losses={losses:>4} net={gains-losses:>+5}")

    p()

    # ── NET gains vs LongTR ──────────────────────────────────────
    p("=" * 110)
    p("  NET vs LongTR (TP, LongTR set)")
    p("=" * 110)
    p()
    ltr_exact = np.max(np.abs(longtr_diffs - true_diffs), axis=1) < THRESH
    for name, diffs in all_methods.items():
        n_exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
        gains = (tp_ltr & n_exact & ~ltr_exact).sum()
        losses = (tp_ltr & ltr_exact & ~n_exact).sum()
        p(f"  {name:<22} gains={gains:>4} losses={losses:>4} net={gains-losses:>+5}")

    p()

    # ── Per-locus oracle: HP methods + classical ──────────────────
    p("=" * 110)
    p("  ORACLE ANALYSIS")
    p("=" * 110)
    p()

    any_correct = np.zeros(n_test, dtype=bool)
    for name, diffs in all_methods.items():
        exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
        any_correct |= exact
    p(f"  Any method correct (TP): {any_correct[tp_ltr].mean():.1%} ({any_correct[tp_ltr].sum()}/{n_tp})")
    p(f"  LongTR correct (TP): {ltr_exact[tp_ltr].mean():.1%}")
    p()

    # Best method by het/hom
    for tag, mask in [("Het-TP", het_m), ("Hom-TP", hom_m)]:
        if mask.sum() == 0:
            continue
        p(f"  {tag} breakdown:")
        for name in list(all_methods.keys()) + ["LongTR"]:
            diffs = all_methods[name] if name != "LongTR" else longtr_diffs
            exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
            p(f"    {name:<22} {exact[mask].mean():.1%}")
        p()

    # ── Motif breakdown ──────────────────────────────────────────
    p("=" * 110)
    p("  MOTIF BREAKDOWN (TP, LongTR set)")
    p("=" * 110)
    p()

    for motif_name, lo, hi in [("Dinuc (2bp)", 2, 2), ("STR (3-6bp)", 3, 6), ("VNTR (7+bp)", 7, 999)]:
        mask = tp_ltr & (motif_lens >= lo) & (motif_lens <= hi)
        if mask.sum() == 0:
            continue
        p(f"  {motif_name} (n={mask.sum()}):")
        for name in list(all_methods.keys()) + ["LongTR"]:
            diffs = all_methods[name] if name != "LongTR" else longtr_diffs
            exact = np.max(np.abs(diffs - true_diffs), axis=1) < THRESH
            p(f"    {name:<22} {exact[mask].mean():.1%}")
        p()

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report: {output_path}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
