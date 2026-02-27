"""Command-line interface for HaploTR."""

from __future__ import annotations

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.group()
@click.version_option(package_name="haplotr")
def main():
    """HaploTR: haplotype-aware tandem repeat genotyping from long-read sequencing."""


@main.command()
@click.option("--bam", required=True, help="Input BAM file (HP-tagged for best results)")
@click.option("--loci", required=True, help="TR loci BED (4-column: chrom, start, end, motif)")
@click.option("--output", required=True, help="Output BED file")
@click.option("--threads", default=8, help="Number of parallel worker processes")
@click.option("--min-mapq", default=5, help="Minimum mapping quality")
@click.option("--min-flank", default=50, help="Minimum flanking bases")
@click.option("--max-reads", default=200, help="Maximum reads per locus")
@click.option("--vntr-motif-cutoff", default=7, type=int,
              help="Motif length cutoff for STR vs VNTR (default: 7)")
@click.option("--ref", default=None, type=click.Path(exists=True),
              help="Reference FASTA for parasail realignment (optional)")
@click.option("--min-hp-reads", default=3, type=int,
              help="Minimum HP-tagged reads per haplotype (default: 3)")
@click.option("--min-hp-frac", default=0.15, type=float,
              help="Minimum fraction of HP-tagged reads (default: 0.15)")
@click.option("--concordance-threshold", default=0.7, type=float,
              help="HP concordance threshold for HET call (default: 0.7)")
@click.option("--min-confidence", default=0.0, type=float,
              help="Minimum confidence to report a call (default: 0, no filter)")
def genotype(bam, loci, output, threads, min_mapq, min_flank, max_reads,
             vntr_motif_cutoff, ref, min_hp_reads, min_hp_frac,
             concordance_threshold, min_confidence):
    """Genotype TR loci from a BAM file.

    Uses concordance-based zygosity with HP=0 read assignment,
    gap-based bimodality detection, and robust MAD-trimmed medians.
    """
    from .genotype import genotype as run_genotype

    run_genotype(
        bam_path=bam,
        loci_bed_path=loci,
        output_path=output,
        nprocs=threads,
        min_mapq=min_mapq,
        min_flank=min_flank,
        max_reads=max_reads,
        vntr_motif_cutoff=vntr_motif_cutoff,
        ref_path=ref,
        min_hp_reads=min_hp_reads,
        min_hp_frac=min_hp_frac,
        concordance_threshold=concordance_threshold,
        min_confidence=min_confidence,
    )


@main.command()
@click.option("--predictions", required=True, help="HaploTR predictions BED")
@click.option("--truth-bed", required=True, help="GIAB Tier1 BED")
@click.option("--catalog", required=True, help="Adotto catalog BED")
@click.option("--output", default=None, help="Output report file (default: stdout)")
@click.option("--chroms", default=None, help="Comma-separated chromosomes to evaluate")
def evaluate(predictions, truth_bed, catalog, output, chroms):
    """Evaluate predictions against GIAB truth."""
    from .benchmark import (
        evaluate as run_evaluate,
        format_results,
        load_predictions,
        prepare_truth,
    )
    from .utils import load_adotto_catalog, load_tier1_bed

    chrom_set = set(chroms.split(",")) if chroms else None

    preds = load_predictions(predictions)
    tier1 = load_tier1_bed(truth_bed, chroms=chrom_set)
    cat = load_adotto_catalog(catalog, chroms=chrom_set)

    truths = prepare_truth(tier1, cat)
    results = run_evaluate(preds, truths)
    report = format_results(results)

    if output:
        with open(output, "w") as f:
            f.write(report)
        click.echo(f"Report written to: {output}")
    else:
        click.echo(report)


@main.command()
@click.option("--bam", required=True, help="Input BAM file (HP-tagged for best results)")
@click.option("--loci", required=True, help="TR loci BED (4-column: chrom, start, end, motif)")
@click.option("--output", required=True, help="Output TSV file")
@click.option("--threads", default=1, help="Number of parallel worker processes (default: 1)")
@click.option("--min-mapq", default=5, help="Minimum mapping quality")
@click.option("--min-flank", default=50, help="Minimum flanking bases")
@click.option("--max-reads", default=200, help="Maximum reads per locus")
@click.option("--ref", default=None, type=click.Path(exists=True),
              help="Reference FASTA for parasail realignment (optional)")
@click.option("--min-hp-reads", default=3, type=int,
              help="Minimum HP-tagged reads per haplotype (default: 3)")
@click.option("--min-hp-frac", default=0.15, type=float,
              help="Minimum fraction of HP-tagged reads (default: 0.15)")
@click.option("--min-instability", default=0.0, type=float,
              help="Minimum AIS threshold for output filtering (default: 0, no filter)")
@click.option("--skip-hp-check", is_flag=True, default=False,
              help="Skip HP tag check (for non-HP-tagged BAMs; uses gap-split/pooled fallback)")
def instability(bam, loci, output, threads, min_mapq, min_flank, max_reads,
                ref, min_hp_reads, min_hp_frac, min_instability, skip_hp_check):
    """Compute per-haplotype somatic instability metrics for TR loci.

    Provides 6 novel metrics: HII (instability index), SER (expansion ratio),
    SCR (contraction ratio), ECB (expansion-contraction bias),
    IAS (instability asymmetry), and AIS (aggregate instability score).
    """
    from .instability import run_instability

    run_instability(
        bam_path=bam,
        loci_bed_path=loci,
        output_path=output,
        nprocs=threads,
        min_mapq=min_mapq,
        min_flank=min_flank,
        max_reads=max_reads,
        ref_path=ref,
        min_hp_reads=min_hp_reads,
        min_hp_frac=min_hp_frac,
        min_instability=min_instability,
        skip_hp_check=skip_hp_check,
    )


if __name__ == "__main__":
    main()
