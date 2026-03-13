"""Utilities for MosaicTR: data loading, I/O helpers."""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GIAB Tier1 BED parser
# ---------------------------------------------------------------------------

@dataclass
class Tier1Locus:
    """A single locus from the GIAB Tier1 BED file."""
    chrom: str
    start: int
    end: int
    tier: str
    tp_status: str      # e.g. "TP_TP_TP" or "TN_TN_TN"
    col6: int            # column 6 (motif period or similar)
    col7: float          # column 7 (quality/score)
    hap1_diff_bp: int    # haplotype 1 allele size diff from reference
    hap2_diff_bp: int    # haplotype 2 allele size diff from reference

    @property
    def is_variant(self) -> bool:
        return "TP" in self.tp_status

    @property
    def is_het(self) -> bool:
        return self.hap1_diff_bp != self.hap2_diff_bp


def load_tier1_bed(
    bed_path: str,
    chroms: Optional[set[str]] = None,
) -> list[Tier1Locus]:
    """Load GIAB Tier1 BED file.

    Format: chrom start end Tier1 TP_status col6 col7 hap1_diff hap2_diff

    Args:
        bed_path: Path to Tier1 BED (supports .gz).
        chroms: If provided, only load loci on these chromosomes.

    Returns:
        List of Tier1Locus objects.
    """
    loci = []
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 9:
                continue
            try:
                chrom = cols[0]
                if chroms is not None and chrom not in chroms:
                    continue
                loci.append(Tier1Locus(
                    chrom=chrom,
                    start=int(cols[1]),
                    end=int(cols[2]),
                    tier=cols[3],
                    tp_status=cols[4],
                    col6=int(cols[5]),
                    col7=float(cols[6]),
                    hap1_diff_bp=int(cols[7]),
                    hap2_diff_bp=int(cols[8]),
                ))
            except (ValueError, IndexError):
                continue

    logger.info("Loaded %d Tier1 loci from %s", len(loci), bed_path)
    return loci


# ---------------------------------------------------------------------------
# Adotto catalog loader
# ---------------------------------------------------------------------------

def load_adotto_catalog(
    bed_path: str,
    chroms: Optional[set[str]] = None,
) -> dict[tuple[str, int, int], str]:
    """Load adotto catalog as {(chrom, start, end): motif} lookup.

    Format: chrom start end motif (4-column BED).

    Args:
        bed_path: Path to adotto catalog BED.
        chroms: If provided, only load loci on these chromosomes.

    Returns:
        Dictionary mapping (chrom, start, end) to motif string.
    """
    catalog = {}
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            try:
                chrom = cols[0]
                if chroms is not None and chrom not in chroms:
                    continue
                start = int(cols[1])
                end = int(cols[2])
                motif = cols[3].strip()
                if motif:
                    catalog[(chrom, start, end)] = motif
            except (ValueError, IndexError):
                continue

    logger.info("Loaded %d catalog entries from %s", len(catalog), bed_path)
    return catalog


def match_tier1_to_catalog(
    tier1_loci: list[Tier1Locus],
    catalog: dict[tuple[str, int, int], str],
    tolerance: int = 0,
) -> list[tuple[Tier1Locus, str]]:
    """Match Tier1 loci to adotto catalog to get motif sequences.

    First tries exact coordinate match, then tries overlap-based matching
    within the given tolerance using binary search for efficiency.

    Args:
        tier1_loci: List of Tier1 loci.
        catalog: Adotto catalog {(chrom, start, end): motif}.
        tolerance: Coordinate tolerance for matching (0 = exact only).

    Returns:
        List of (Tier1Locus, motif) tuples for matched loci.
    """
    import bisect
    from collections import defaultdict

    matched = []
    n_exact = 0
    n_approx = 0

    # Build sorted interval index for approximate matching
    chrom_starts: dict[str, list[int]] = defaultdict(list)
    chrom_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    if tolerance > 0:
        for (c, s, e), m in catalog.items():
            chrom_intervals[c].append((s, e, m))
        for c in chrom_intervals:
            chrom_intervals[c].sort()
            chrom_starts[c] = [iv[0] for iv in chrom_intervals[c]]

    for locus in tier1_loci:
        key = (locus.chrom, locus.start, locus.end)

        # Exact match
        if key in catalog:
            matched.append((locus, catalog[key]))
            n_exact += 1
            continue

        # Approximate match using binary search
        if tolerance > 0:
            starts = chrom_starts.get(locus.chrom)
            if starts is None:
                continue
            intervals = chrom_intervals[locus.chrom]

            # Find first interval that could overlap: start >= locus.start - tolerance - max_interval_size
            # Conservative: search from where start >= locus.start - tolerance - 10000
            search_start = locus.start - tolerance - 10000
            lo = bisect.bisect_left(starts, search_start)

            best_motif = None
            best_overlap = 0
            for j in range(lo, len(intervals)):
                s, e, m = intervals[j]
                if s > locus.end + tolerance:
                    break
                if e < locus.start - tolerance:
                    continue
                overlap = min(e, locus.end) - max(s, locus.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_motif = m
            if best_motif is not None:
                matched.append((locus, best_motif))
                n_approx += 1

    logger.info(
        "Matched %d/%d loci (exact=%d, approx=%d)",
        len(matched), len(tier1_loci), n_exact, n_approx,
    )
    return matched


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------

CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
CHROM_ORDER.update({"chrX": 23, "chrY": 24, "chrM": 25})

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 19)}       # chr1-18
VAL_CHROMS = {f"chr{i}" for i in range(19, 21)}         # chr19-20
TEST_CHROMS = {"chr21", "chr22", "chrX"}                 # chr21-22, chrX


def chrom_split(chrom: str) -> str:
    """Return 'train', 'val', or 'test' based on chromosome."""
    if chrom in TRAIN_CHROMS:
        return "train"
    elif chrom in VAL_CHROMS:
        return "val"
    elif chrom in TEST_CHROMS:
        return "test"
    else:
        return "other"


# ---------------------------------------------------------------------------
# Locus BED loader (4-column)
# ---------------------------------------------------------------------------

def load_loci_bed(
    bed_path: str,
    chroms: Optional[set[str]] = None,
) -> list[tuple[str, int, int, str]]:
    """Load TR loci from a 4-column BED file.

    Args:
        bed_path: Path to BED file (supports .gz).
        chroms: If provided, only load loci on these chromosomes.

    Returns:
        List of (chrom, start, end, motif) tuples.
    """
    loci = []
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            try:
                chrom = cols[0]
                if chroms is not None and chrom not in chroms:
                    continue
                start = int(cols[1])
                end = int(cols[2])
                motif = cols[3].strip()
                if motif and start < end:
                    loci.append((chrom, start, end, motif))
            except (ValueError, IndexError):
                continue
    return loci
