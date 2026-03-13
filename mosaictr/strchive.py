"""Pathogenic TR catalog for MosaicTR.

Built-in catalog of known pathogenic tandem repeat loci with clinical
annotations. Enables annotation of genotyping results with disease
relevance, allele status classification, and risk assessment without
requiring external catalog files.

Modeled after STRchive-style annotations used by TRGT and other tools,
but embedded directly in MosaicTR for zero-dependency clinical reporting.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in pathogenic TR catalog (GRCh38 coordinates)
# ---------------------------------------------------------------------------

PATHOGENIC_CATALOG: list[dict] = [
    {
        "gene": "HTT",
        "disease": "Huntington disease",
        "chrom": "chr4",
        "start": 3074876,
        "end": 3074933,
        "motif": "CAG",
        "normal_max": 35,
        "pathogenic_min": 40,
        "inheritance": "AD",
    },
    {
        "gene": "FMR1",
        "disease": "Fragile X syndrome",
        "chrom": "chrX",
        "start": 147912050,
        "end": 147912110,
        "motif": "CGG",
        "normal_max": 54,
        "pathogenic_min": 200,
        "inheritance": "X-linked",
    },
    {
        "gene": "DMPK",
        "disease": "Myotonic dystrophy type 1",
        "chrom": "chr19",
        "start": 45770204,
        "end": 45770264,
        "motif": "CTG",
        "normal_max": 34,
        "pathogenic_min": 50,
        "inheritance": "AD",
    },
    {
        "gene": "FXN",
        "disease": "Friedreich ataxia",
        "chrom": "chr9",
        "start": 69037286,
        "end": 69037304,
        "motif": "GAA",
        "normal_max": 33,
        "pathogenic_min": 66,
        "inheritance": "AR",
    },
    {
        "gene": "ATXN1",
        "disease": "Spinocerebellar ataxia type 1",
        "chrom": "chr6",
        "start": 16327633,
        "end": 16327723,
        "motif": "CAG",
        "normal_max": 38,
        "pathogenic_min": 39,
        "inheritance": "AD",
    },
    {
        "gene": "ATXN3",
        "disease": "Spinocerebellar ataxia type 3",
        "chrom": "chr14",
        "start": 92071009,
        "end": 92071042,
        "motif": "CAG",
        "normal_max": 44,
        "pathogenic_min": 55,
        "inheritance": "AD",
    },
    {
        "gene": "ATXN10",
        "disease": "Spinocerebellar ataxia type 10",
        "chrom": "chr22",
        "start": 45795354,
        "end": 45795424,
        "motif": "ATTCT",
        "normal_max": 32,
        "pathogenic_min": 800,
        "inheritance": "AD",
    },
    {
        "gene": "RFC1",
        "disease": "CANVAS",
        "chrom": "chr4",
        "start": 39348424,
        "end": 39348485,
        "motif": "AAGGG",
        "normal_max": 11,
        "pathogenic_min": 400,
        "inheritance": "AR",
    },
    {
        "gene": "C9ORF72",
        "disease": "ALS/FTD",
        "chrom": "chr9",
        "start": 27573528,
        "end": 27573546,
        "motif": "GGGGCC",
        "normal_max": 25,
        "pathogenic_min": 60,
        "inheritance": "AD",
    },
    {
        "gene": "AR",
        "disease": "Spinal and bulbar muscular atrophy",
        "chrom": "chrX",
        "start": 67545316,
        "end": 67545385,
        "motif": "CAG",
        "normal_max": 34,
        "pathogenic_min": 38,
        "inheritance": "X-linked",
    },
    {
        "gene": "ATXN2",
        "disease": "Spinocerebellar ataxia type 2",
        "chrom": "chr12",
        "start": 111598950,
        "end": 111599018,
        "motif": "CAG",
        "normal_max": 31,
        "pathogenic_min": 34,
        "inheritance": "AD",
    },
    {
        "gene": "TBP",
        "disease": "Spinocerebellar ataxia type 17",
        "chrom": "chr6",
        "start": 170561906,
        "end": 170562017,
        "motif": "CAG",
        "normal_max": 40,
        "pathogenic_min": 49,
        "inheritance": "AD",
    },
    {
        "gene": "ATXN7",
        "disease": "Spinocerebellar ataxia type 7",
        "chrom": "chr3",
        "start": 63912684,
        "end": 63912726,
        "motif": "CAG",
        "normal_max": 33,
        "pathogenic_min": 37,
        "inheritance": "AD",
    },
    {
        "gene": "PABPN1",
        "disease": "Oculopharyngeal muscular dystrophy",
        "chrom": "chr14",
        "start": 23321472,
        "end": 23321490,
        "motif": "GCN",
        "normal_max": 10,
        "pathogenic_min": 12,
        "inheritance": "AD",
    },
    {
        "gene": "CNBP",
        "disease": "Myotonic dystrophy type 2",
        "chrom": "chr3",
        "start": 129172576,
        "end": 129172656,
        "motif": "CCTG",
        "normal_max": 26,
        "pathogenic_min": 75,
        "inheritance": "AD",
    },
    {
        "gene": "JPH3",
        "disease": "Huntington disease-like 2",
        "chrom": "chr16",
        "start": 87637893,
        "end": 87637935,
        "motif": "CTG",
        "normal_max": 28,
        "pathogenic_min": 41,
        "inheritance": "AD",
    },
    {
        "gene": "ATN1",
        "disease": "Dentatorubral-pallidoluysian atrophy",
        "chrom": "chr12",
        "start": 6936716,
        "end": 6936773,
        "motif": "CAG",
        "normal_max": 35,
        "pathogenic_min": 48,
        "inheritance": "AD",
    },
    {
        "gene": "PPP2R2B",
        "disease": "Spinocerebellar ataxia type 12",
        "chrom": "chr5",
        "start": 146878727,
        "end": 146878757,
        "motif": "CAG",
        "normal_max": 32,
        "pathogenic_min": 51,
        "inheritance": "AD",
    },
    {
        "gene": "BEAN1",
        "disease": "Spinocerebellar ataxia type 31",
        "chrom": "chr16",
        "start": 66490398,
        "end": 66490453,
        "motif": "TGGAA",
        "normal_max": 0,
        "pathogenic_min": 110,
        "inheritance": "AD",
    },
    {
        "gene": "STARD7",
        "disease": "Familial adult myoclonic epilepsy type 2",
        "chrom": "chr2",
        "start": 96197066,
        "end": 96197121,
        "motif": "ATTTC",
        "normal_max": 0,
        "pathogenic_min": 661,
        "inheritance": "AD",
    },
    {
        "gene": "FGF14",
        "disease": "Spinocerebellar ataxia type 27B",
        "chrom": "chr13",
        "start": 102161724,
        "end": 102161734,
        "motif": "GAA",
        "normal_max": 250,
        "pathogenic_min": 300,
        "inheritance": "AD",
    },
]


# ---------------------------------------------------------------------------
# Catalog access helpers
# ---------------------------------------------------------------------------

def get_pathogenic_loci() -> list[dict]:
    """Return the built-in pathogenic TR catalog.

    Each entry is a dict with keys: gene, disease, chrom, start, end,
    motif, normal_max, pathogenic_min, inheritance.

    Returns:
        List of catalog entry dicts (defensive copy).
    """
    return [entry.copy() for entry in PATHOGENIC_CATALOG]


def get_pathogenic_bed() -> list[tuple[str, int, int, str]]:
    """Return pathogenic loci as BED-format tuples.

    Returns:
        List of (chrom, start, end, motif) tuples suitable for use
        as MosaicTR loci input.
    """
    return [
        (entry["chrom"], entry["start"], entry["end"], entry["motif"])
        for entry in PATHOGENIC_CATALOG
    ]


# ---------------------------------------------------------------------------
# Locus matching
# ---------------------------------------------------------------------------

def annotate_locus(
    chrom: str,
    start: int,
    end: int,
    motif: Optional[str] = None,
    tolerance: int = 50,
) -> Optional[dict]:
    """Match a genomic locus to the pathogenic catalog.

    Matching criteria:
    1. Same chromosome.
    2. Overlapping or within *tolerance* bp of a catalog entry.
    3. If *motif* is provided, it must match the catalog motif
       (case-insensitive).

    Args:
        chrom: Chromosome name (e.g. "chr4").
        start: 0-based start coordinate.
        end: 0-based end coordinate.
        motif: Repeat motif sequence (optional, used for stricter matching).
        tolerance: Maximum distance in bp between query and catalog
            boundaries to consider a match (default: 50).

    Returns:
        Copy of the matching catalog entry dict, or None if no match.
    """
    for entry in PATHOGENIC_CATALOG:
        if entry["chrom"] != chrom:
            continue

        # Check coordinate overlap within tolerance
        cat_start = entry["start"]
        cat_end = entry["end"]
        if start > cat_end + tolerance or end < cat_start - tolerance:
            continue

        # Optionally verify motif match
        if motif is not None:
            if motif.upper() != entry["motif"].upper():
                continue

        return entry.copy()

    return None


# ---------------------------------------------------------------------------
# Allele classification
# ---------------------------------------------------------------------------

def classify_allele(
    allele_size_bp: float,
    ref_size: float,
    motif_len: int,
    normal_max: int,
    pathogenic_min: int,
) -> str:
    """Classify an allele as normal, intermediate, or pathogenic.

    Converts the allele size from base pairs to approximate repeat units,
    then compares against the normal and pathogenic thresholds.

    The repeat unit count is computed as:
        repeat_units = allele_size_bp / motif_len

    Args:
        allele_size_bp: Measured allele size in base pairs.
        ref_size: Reference locus size in base pairs (end - start).
        motif_len: Length of the repeat motif in bp.
        normal_max: Maximum repeat units considered normal.
        pathogenic_min: Minimum repeat units considered pathogenic.

    Returns:
        One of 'normal', 'intermediate', or 'pathogenic'.
    """
    if motif_len <= 0:
        return "normal"

    repeat_units = allele_size_bp / motif_len

    if repeat_units <= normal_max:
        return "normal"
    elif repeat_units >= pathogenic_min:
        return "pathogenic"
    else:
        return "intermediate"


# ---------------------------------------------------------------------------
# Batch annotation of genotyping results
# ---------------------------------------------------------------------------

def _classify_status(
    allele1_size: float,
    allele2_size: float,
    ref_size: float,
    motif_len: int,
    normal_max: int,
    pathogenic_min: int,
) -> str:
    """Determine overall locus status from two allele classifications.

    Classification logic:
    - If either allele is pathogenic -> 'pathogenic'
    - If either allele is intermediate -> 'intermediate'
    - Otherwise -> 'normal'

    This reflects clinical practice where a single expanded allele
    can cause disease in dominant (AD) and X-linked conditions.
    """
    status1 = classify_allele(
        allele1_size, ref_size, motif_len, normal_max, pathogenic_min,
    )
    status2 = classify_allele(
        allele2_size, ref_size, motif_len, normal_max, pathogenic_min,
    )

    if status1 == "pathogenic" or status2 == "pathogenic":
        return "pathogenic"
    elif status1 == "intermediate" or status2 == "intermediate":
        return "intermediate"
    else:
        return "normal"


def annotate_results(
    loci: list[tuple[str, int, int, str]],
    results: list[Optional[dict]],
) -> list[dict]:
    """Annotate genotyping results with pathogenic TR information.

    For each locus that matches the pathogenic catalog, adds a
    'pathogenic_annotation' key to the result dict containing:
    - gene: gene symbol
    - disease: disease name
    - normal_max: max normal repeat units
    - pathogenic_min: min pathogenic repeat units
    - inheritance: inheritance pattern
    - allele1_status: classification for allele 1
    - allele2_status: classification for allele 2
    - status: overall locus status (normal/intermediate/pathogenic)

    Args:
        loci: List of (chrom, start, end, motif) tuples, one per locus.
        results: List of result dicts from genotyping (may contain None
            for failed loci). Each successful result dict has at least
            'allele1_size' and 'allele2_size' keys.

    Returns:
        List of result dicts with 'pathogenic_annotation' added where
        applicable. Failed loci (None results) are returned as empty
        dicts with only 'pathogenic_annotation' if the locus matches.
    """
    if len(loci) != len(results):
        raise ValueError(
            f"loci ({len(loci)}) and results ({len(results)}) must have "
            f"the same length"
        )

    annotated = []
    n_matched = 0

    for locus, result in zip(loci, results):
        chrom, start, end, motif = locus
        ref_size = end - start

        # Start with existing result or empty dict
        out = dict(result) if result is not None else {}

        # Try to match against pathogenic catalog
        entry = annotate_locus(chrom, start, end, motif)
        if entry is not None:
            n_matched += 1
            motif_len = len(entry["motif"])
            normal_max = entry["normal_max"]
            pathogenic_min = entry["pathogenic_min"]

            annotation = {
                "gene": entry["gene"],
                "disease": entry["disease"],
                "normal_max": normal_max,
                "pathogenic_min": pathogenic_min,
                "inheritance": entry["inheritance"],
            }

            # Classify alleles if genotyping succeeded
            if result is not None and "allele1_size" in result:
                a1 = result["allele1_size"]
                a2 = result["allele2_size"]

                annotation["allele1_status"] = classify_allele(
                    a1, ref_size, motif_len, normal_max, pathogenic_min,
                )
                annotation["allele2_status"] = classify_allele(
                    a2, ref_size, motif_len, normal_max, pathogenic_min,
                )
                annotation["status"] = _classify_status(
                    a1, a2, ref_size, motif_len, normal_max, pathogenic_min,
                )
            else:
                annotation["allele1_status"] = None
                annotation["allele2_status"] = None
                annotation["status"] = None

            out["pathogenic_annotation"] = annotation

        annotated.append(out)

    if n_matched > 0:
        logger.info(
            "Annotated %d/%d loci with pathogenic catalog entries",
            n_matched, len(loci),
        )

    return annotated
