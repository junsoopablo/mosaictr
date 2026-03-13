#!/usr/bin/env python3
"""Disease-associated loci analysis for MosaicTR.

Loads STRchive disease loci (76 loci, hg38), matches against MosaicTR,
LongTR, and TRGT genome-wide results, and produces a detailed per-locus
report including allele sizes, pathogenic thresholds, and accuracy.

Usage:
  python scripts/analyze_disease_loci.py \
    --mosaictr output/genome_wide/v4_hg002_genome_wide.bed \
    --longtr-vcf /path/to/HG002.longtr.vcf.gz \
    --trgt-vcf /path/to/HG002.trgt.sorted.phased.vcf.gz \
    --output output/genome_wide/disease_loci_report.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import LocusPrediction, load_predictions
from scripts.benchmark_genome_wide import (
    parse_longtr_vcf,
    parse_trgt_vcf,
    LONGTR_VCF,
    TRGT_VCF,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STRCHIVE_BED = "/vault/external-datasets/2026/STRchive_disease-loci/STRchive-disease-loci.hg38.general.bed"


def load_strchive_loci(bed_path: str) -> list[dict]:
    """Load STRchive disease loci from BED file."""
    loci = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 10:
                continue
            pathogenic_min = cols[7]
            try:
                pathogenic_min = int(pathogenic_min) if pathogenic_min != "None" else None
            except ValueError:
                pathogenic_min = None
            loci.append({
                "chrom": cols[0],
                "start": int(cols[1]),
                "end": int(cols[2]),
                "id": cols[3],
                "gene": cols[4],
                "ref_motif": cols[5],
                "pathogenic_motif": cols[6],
                "pathogenic_min": pathogenic_min,
                "inheritance": cols[8],
                "disease": cols[9],
            })
    return loci


def find_match_mosaictr(
    locus: dict,
    pred_lookup: dict[tuple[str, int], LocusPrediction],
    tolerance: int = 200,
) -> LocusPrediction | None:
    """Find MosaicTR prediction matching a disease locus."""
    for offset in range(0, tolerance + 1):
        for sign in [0, -1, 1]:
            key = (locus["chrom"], locus["start"] + sign * offset)
            if key in pred_lookup:
                return pred_lookup[key]
    return None


def find_match_tool(
    locus: dict,
    tool_dict: dict,
    tolerance: int = 200,
) -> tuple | None:
    """Find LongTR/TRGT result matching a disease locus.

    Supports both (chrom, start) and (chrom, start, end) key formats.
    """
    best = None
    best_dist = tolerance + 1
    for key, val in tool_dict.items():
        chrom = key[0]
        start = key[1]
        if chrom != locus["chrom"]:
            continue
        dist = abs(start - locus["start"])
        if dist < best_dist:
            best_dist = dist
            best = val
    return best if best_dist <= tolerance else None


def analyze_disease_loci(
    strchive_loci: list[dict],
    mosaictr_preds: list[LocusPrediction],
    longtr_dict: dict,
    trgt_dict: dict,
) -> list[dict]:
    """Match disease loci to tool results and compute per-locus report."""
    # Build MosaicTR lookup
    ht_lookup = {}
    for p in mosaictr_preds:
        ht_lookup[(p.chrom, p.start)] = p

    results = []
    for locus in strchive_loci:
        entry = {**locus}
        ref_size = locus["end"] - locus["start"]
        motif_len = len(locus["ref_motif"])

        # MosaicTR match
        ht_pred = find_match_mosaictr(locus, ht_lookup)
        if ht_pred:
            ht_ref = ht_pred.end - ht_pred.start
            ht_d1 = ht_pred.allele1_size - ht_ref
            ht_d2 = ht_pred.allele2_size - ht_ref
            # Convert to repeat units
            ht_ru1 = ht_pred.allele1_size / motif_len if motif_len > 0 else 0
            ht_ru2 = ht_pred.allele2_size / motif_len if motif_len > 0 else 0
            entry["ht_allele1"] = ht_pred.allele1_size
            entry["ht_allele2"] = ht_pred.allele2_size
            entry["ht_d1"] = ht_d1
            entry["ht_d2"] = ht_d2
            entry["ht_ru1"] = ht_ru1
            entry["ht_ru2"] = ht_ru2
            entry["ht_zyg"] = ht_pred.zygosity
            entry["ht_conf"] = ht_pred.confidence
            entry["ht_nreads"] = ht_pred.n_reads

        # LongTR match
        lt_match = find_match_tool(locus, longtr_dict)
        if lt_match:
            d1, d2, lt_ref = lt_match
            entry["lt_d1"] = d1
            entry["lt_d2"] = d2
            entry["lt_ru1"] = (lt_ref + d1) / motif_len if motif_len > 0 else 0
            entry["lt_ru2"] = (lt_ref + d2) / motif_len if motif_len > 0 else 0

        # TRGT match
        trgt_match = find_match_tool(locus, trgt_dict)
        if trgt_match:
            d1, d2, trgt_ref = trgt_match
            entry["trgt_d1"] = d1
            entry["trgt_d2"] = d2
            entry["trgt_ru1"] = (trgt_ref + d1) / motif_len if motif_len > 0 else 0
            entry["trgt_ru2"] = (trgt_ref + d2) / motif_len if motif_len > 0 else 0

        results.append(entry)

    return results


def format_report(results: list[dict]) -> str:
    lines = []
    w = lines.append

    w("=" * 120)
    w("  DISEASE-ASSOCIATED LOCI ANALYSIS")
    w("  STRchive 76 loci (hg38) — HG002 MosaicTR v4 / LongTR / TRGT")
    w("=" * 120)

    n_total = len(results)
    n_ht = sum(1 for r in results if "ht_d1" in r)
    n_lt = sum(1 for r in results if "lt_d1" in r)
    n_trgt = sum(1 for r in results if "trgt_d1" in r)

    w(f"\n  Total disease loci: {n_total}")
    w(f"  MosaicTR matched: {n_ht}")
    w(f"  LongTR matched: {n_lt}")
    w(f"  TRGT matched: {n_trgt}")

    # Detailed per-locus table
    w(f"\n{'='*120}")
    w("  PER-LOCUS RESULTS")
    w(f"{'='*120}")

    header = (f"  {'Gene':<12s} {'Motif':<8s} {'RefLen':>6s} {'PathMin':>7s} "
              f"{'HT_RU1':>7s} {'HT_RU2':>7s} {'HT_Zyg':<5s} {'HT_Conf':>7s} "
              f"{'LT_RU1':>7s} {'LT_RU2':>7s} "
              f"{'TR_RU1':>7s} {'TR_RU2':>7s} "
              f"{'Inherit':<5s}")
    w(header)
    w("  " + "-" * 115)

    for r in results:
        gene = r["gene"]
        motif = r["ref_motif"][:6]
        ref_len = r["end"] - r["start"]
        path_min = str(r["pathogenic_min"]) if r["pathogenic_min"] is not None else "."

        ht_ru1 = f"{r['ht_ru1']:>7.1f}" if "ht_ru1" in r else f"{'.'!s:>7s}"
        ht_ru2 = f"{r['ht_ru2']:>7.1f}" if "ht_ru2" in r else f"{'.'!s:>7s}"
        ht_zyg = r.get("ht_zyg", ".").ljust(5)
        ht_conf = f"{r['ht_conf']:>7.2f}" if "ht_conf" in r else f"{'.'!s:>7s}"

        lt_ru1 = f"{r['lt_ru1']:>7.1f}" if "lt_ru1" in r else f"{'.'!s:>7s}"
        lt_ru2 = f"{r['lt_ru2']:>7.1f}" if "lt_ru2" in r else f"{'.'!s:>7s}"

        tr_ru1 = f"{r['trgt_ru1']:>7.1f}" if "trgt_ru1" in r else f"{'.'!s:>7s}"
        tr_ru2 = f"{r['trgt_ru2']:>7.1f}" if "trgt_ru2" in r else f"{'.'!s:>7s}"

        inherit = r.get("inheritance", ".")[:5]

        w(f"  {gene:<12s} {motif:<8s} {ref_len:>6d} {path_min:>7s} "
          f"{ht_ru1} {ht_ru2} {ht_zyg} {ht_conf} "
          f"{lt_ru1} {lt_ru2} "
          f"{tr_ru1} {tr_ru2} "
          f"{inherit:<5s}")

    # Disease detail with descriptions
    w(f"\n{'='*120}")
    w("  DETAILED DISEASE ANNOTATIONS")
    w(f"{'='*120}")
    for r in results:
        status = "GENOTYPED" if "ht_d1" in r else "NOT FOUND"
        w(f"\n  [{status}] {r['gene']} — {r['disease'][:70]}")
        w(f"    Location: {r['chrom']}:{r['start']}-{r['end']}")
        w(f"    Ref motif: {r['ref_motif']}, Pathogenic motif: {r['pathogenic_motif']}")
        if r["pathogenic_min"] is not None:
            w(f"    Pathogenic threshold: >= {r['pathogenic_min']} repeats")
        w(f"    Inheritance: {r['inheritance']}")
        if "ht_ru1" in r:
            w(f"    MosaicTR: {r['ht_ru1']:.1f} / {r['ht_ru2']:.1f} RU, "
              f"{r['ht_zyg']}, conf={r['ht_conf']:.2f}, nreads={r['ht_nreads']}")

    # Summary: how many loci each tool found
    w(f"\n{'='*120}")
    w("  COVERAGE SUMMARY")
    w(f"{'='*120}")
    w(f"\n  {'Tool':<15s} {'Found':>6s} {'Total':>6s} {'Rate':>7s}")
    w("  " + "-" * 36)
    w(f"  {'MosaicTR v4':<15s} {n_ht:>6d} {n_total:>6d} {n_ht/n_total:>6.1%}")
    w(f"  {'LongTR':<15s} {n_lt:>6d} {n_total:>6d} {n_lt/n_total:>6.1%}")
    w(f"  {'TRGT':<15s} {n_trgt:>6d} {n_total:>6d} {n_trgt/n_total:>6.1%}")

    # All three agree check
    n_all3 = sum(1 for r in results if "ht_d1" in r and "lt_d1" in r and "trgt_d1" in r)
    w(f"  All three tools: {n_all3}/{n_total}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Disease loci analysis")
    parser.add_argument("--mosaictr", required=True, help="MosaicTR v4 BED output")
    parser.add_argument("--strchive", default=STRCHIVE_BED, help="STRchive disease loci BED")
    parser.add_argument("--longtr-vcf", default=LONGTR_VCF, help="LongTR VCF")
    parser.add_argument("--trgt-vcf", default=TRGT_VCF, help="TRGT VCF")
    parser.add_argument("--output", required=True, help="Output report file")
    args = parser.parse_args()

    logger.info("Loading STRchive disease loci: %s", args.strchive)
    strchive = load_strchive_loci(args.strchive)
    logger.info("Loaded %d disease loci", len(strchive))

    logger.info("Loading MosaicTR predictions...")
    ht_preds = load_predictions(args.mosaictr)
    logger.info("MosaicTR: %d predictions", len(ht_preds))

    logger.info("Parsing LongTR VCF...")
    longtr = parse_longtr_vcf(args.longtr_vcf)

    logger.info("Parsing TRGT VCF...")
    trgt = parse_trgt_vcf(args.trgt_vcf)

    results = analyze_disease_loci(strchive, ht_preds, longtr, trgt)
    report = format_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", output_path)
    print(report)


if __name__ == "__main__":
    main()
