"""Command-line interface for MosaicTR."""

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
@click.version_option(package_name="mosaictr")
def main():
    """MosaicTR: haplotype-aware tandem repeat genotyping and somatic instability analysis from long-read sequencing."""


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
@click.option("--vcf", "vcf_output", default=None, type=click.Path(),
              help="Additionally write VCF output to this path")
@click.option("--sample-name", default="SAMPLE", help="Sample name for VCF output")
def genotype(bam, loci, output, threads, min_mapq, min_flank, max_reads,
             vntr_motif_cutoff, ref, min_hp_reads, min_hp_frac,
             concordance_threshold, min_confidence, vcf_output, sample_name):
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

    if vcf_output:
        from .utils import load_loci_bed
        from .vcf_output import write_genotype_vcf
        # Re-read results from BED output to convert
        loci_list = load_loci_bed(loci)
        results = []
        with open(output) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.strip().split("\t")
                if cols[4] == ".":
                    results.append(None)
                else:
                    results.append({
                        "allele1_size": float(cols[4]),
                        "allele2_size": float(cols[5]),
                        "zygosity": cols[6],
                        "confidence": float(cols[7]) if cols[7] != "." else 0.0,
                        "n_reads": int(cols[8]),
                    })
        n = write_genotype_vcf(vcf_output, loci_list, results, sample_name, ref)
        click.echo(f"VCF written to: {vcf_output} ({n} loci)")


@main.command()
@click.option("--predictions", required=True, help="MosaicTR predictions BED")
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
              help="Minimum HII threshold for output filtering (default: 0, no filter)")
@click.option("--min-reads", default=0, type=int,
              help="Minimum total reads per locus to report (default: 0, no filter)")
@click.option("--noise-threshold", default=0.45, type=float,
              help="HII threshold for unstable_haplotype label (default: 0.45, 3σ noise floor)")
@click.option("--skip-hp-check", is_flag=True, default=False,
              help="Skip HP tag check (for non-HP-tagged BAMs; uses gap-split/pooled fallback)")
@click.option("--vcf", "vcf_output", default=None, type=click.Path(),
              help="Additionally write VCF output to this path")
@click.option("--sample-name", default="SAMPLE", help="Sample name for VCF output")
def instability(bam, loci, output, threads, min_mapq, min_flank, max_reads,
                ref, min_hp_reads, min_hp_frac, min_instability, min_reads,
                noise_threshold, skip_hp_check, vcf_output, sample_name):
    """Compute per-haplotype somatic instability metrics for TR loci.

    Provides 2 metrics: HII (haplotype instability index) and
    IAS (instability asymmetry score).
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
        min_reads=min_reads,
        noise_threshold=noise_threshold,
        skip_hp_check=skip_hp_check,
    )

    if vcf_output:
        from .vcf_output import write_instability_vcf
        loci_list = []
        results = []
        with open(output) as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    header = line.strip().lstrip("#").split("\t")
                    continue
                cols = line.strip().split("\t")
                if header is None:
                    continue
                col_map = dict(zip(header, cols))
                loci_list.append((
                    col_map["chrom"],
                    int(col_map["start"]),
                    int(col_map["end"]),
                    col_map["motif"],
                ))
                if col_map.get("median_h1", ".") == ".":
                    results.append(None)
                else:
                    float_keys = [
                        "median_h1", "median_h2", "hii_h1", "hii_h2",
                        "ias", "range_h1", "range_h2", "concordance",
                    ]
                    int_keys = ["n_h1", "n_h2", "n_total"]
                    row = {k: float(col_map[k]) for k in float_keys}
                    row.update({k: int(col_map[k]) for k in int_keys})
                    row["analysis_path"] = col_map["analysis_path"]
                    row["unstable_haplotype"] = col_map["unstable_haplotype"]
                    row["dropout_flag"] = col_map["dropout_flag"] == "1"
                    results.append(row)
        n = write_instability_vcf(vcf_output, loci_list, results, sample_name, ref)
        click.echo(f"VCF written to: {vcf_output} ({n} loci)")


@main.command()
@click.option("--bam", required=True, help="Input BAM file")
@click.option("--loci", required=True, help="TR loci BED (4-column)")
@click.option("--output", required=True, help="Output PNG file")
@click.option("--ref", default=None, type=click.Path(exists=True),
              help="Reference FASTA (optional)")
@click.option("--title", default=None, help="Plot title")
@click.option("--locus-index", default=0, type=int,
              help="Index of locus in BED to visualize (default: 0)")
def visualize(bam, loci, output, ref, title, locus_index):
    """Generate per-read waterfall and instability summary plots."""
    import pysam
    from .genotype import extract_reads_enhanced
    from .instability import compute_instability
    from .utils import load_loci_bed
    from .visualization import instability_summary_plot

    loci_list = load_loci_bed(loci)
    if locus_index >= len(loci_list):
        click.echo(f"Error: locus_index {locus_index} >= {len(loci_list)} loci", err=True)
        return
    chrom, start, end, motif = loci_list[locus_index]
    ref_size = end - start
    motif_len = len(motif)

    bam_file = pysam.AlignmentFile(bam, "rb")
    ref_fasta = pysam.FastaFile(ref) if ref else None
    try:
        reads = extract_reads_enhanced(
            bam_file, chrom, start, end, ref_fasta=ref_fasta, motif_len=motif_len,
        )
    finally:
        bam_file.close()
        if ref_fasta:
            ref_fasta.close()

    if not reads:
        click.echo(f"Error: no reads found at {chrom}:{start}-{end}", err=True)
        return

    result = compute_instability(reads, ref_size, motif_len)
    if result is None:
        click.echo(f"Error: could not compute instability at {chrom}:{start}-{end}", err=True)
        return

    plot_title = title or f"{chrom}:{start}-{end} ({motif})"
    instability_summary_plot(reads, ref_size, motif_len, result, output, title=plot_title)
    click.echo(f"Plot saved to: {output}")


@main.command()
@click.option("--bam", required=True, help="Input BAM file")
@click.option("--chrom", required=True, help="Chromosome")
@click.option("--start", required=True, type=int, help="Locus start")
@click.option("--end", required=True, type=int, help="Locus end")
@click.option("--motif", required=True, help="Repeat motif (e.g., CAG)")
@click.option("--ref", default=None, type=click.Path(exists=True),
              help="Reference FASTA (optional)")
@click.option("--max-reads", default=50, type=int, help="Maximum reads to analyze")
def interruptions(bam, chrom, start, end, motif, ref, max_reads):
    """Detect motif interruptions at a TR locus."""
    from .interruptions import analyze_reads_interruptions

    result = analyze_reads_interruptions(
        bam, chrom, start, end, motif,
        ref_fasta_path=ref, max_reads=max_reads,
    )
    click.echo(f"Reads analyzed: {result['n_reads']}")
    if result.get("common_variants"):
        click.echo("Common motif variants:")
        for variant, count in sorted(
            result["common_variants"].items(), key=lambda x: -x[1],
        ):
            click.echo(f"  {variant}: {count}")
    import math
    hp1_pur = result.get("hp1_purity")
    hp2_pur = result.get("hp2_purity")
    click.echo(f"HP1 mean purity: {hp1_pur:.3f}" if isinstance(hp1_pur, float) and not math.isnan(hp1_pur) else "HP1 mean purity: N/A")
    click.echo(f"HP2 mean purity: {hp2_pur:.3f}" if isinstance(hp2_pur, float) and not math.isnan(hp2_pur) else "HP2 mean purity: N/A")
    if result.get("consensus_interruptions"):
        click.echo("Consensus interruptions (>50% of reads):")
        for intr in result["consensus_interruptions"]:
            click.echo(f"  pos={intr['position']}: {intr['expected']}->{intr['observed']}")


@main.command("pathogenic-loci")
def pathogenic_loci():
    """List built-in pathogenic TR loci catalog."""
    from .strchive import get_pathogenic_loci

    loci = get_pathogenic_loci()
    click.echo(f"{'Gene':<10} {'Disease':<25} {'Chrom':<6} {'Motif':<8} "
               f"{'Normal':<8} {'Path':<8} {'Inherit':<10}")
    click.echo("-" * 80)
    for entry in loci:
        click.echo(
            f"{entry['gene']:<10} {entry['disease']:<25} {entry['chrom']:<6} "
            f"{entry['motif']:<8} <={entry['normal_max']:<5} >={entry['pathogenic_min']:<5} "
            f"{entry['inheritance']:<10}"
        )


@main.command()
@click.option("--baseline", required=True, help="Baseline tissue instability TSV (e.g., blood)")
@click.option("--target", required=True, help="Target tissue instability TSV (e.g., colon)")
@click.option("--output", required=True, help="Output comparison TSV")
@click.option("--noise-threshold", default=0.45, type=float,
              help="HII threshold for calling unstable (default: 0.45)")
@click.option("--min-delta", default=0.5, type=float,
              help="Minimum ΔHII to report (default: 0.5)")
@click.option("--baseline-label", default="baseline", help="Label for baseline tissue")
@click.option("--target-label", default="target", help="Label for target tissue")
def compare(baseline, target, output, noise_threshold, min_delta,
            baseline_label, target_label):
    """Compare instability between two tissues (e.g., blood vs colon).

    Identifies TR loci with tissue-specific somatic instability by comparing
    a baseline tissue (e.g., blood) against a target tissue. Designed for
    multi-tissue studies such as SMaHT.
    """
    from .compare import run_compare

    run_compare(
        baseline_path=baseline,
        target_path=target,
        output_path=output,
        noise_threshold=noise_threshold,
        min_delta=min_delta,
        baseline_label=baseline_label,
        target_label=target_label,
    )


@main.command()
@click.option("--inputs", required=True, multiple=True,
              help="Instability TSV files (repeat for each tissue)")
@click.option("--labels", required=True, multiple=True,
              help="Sample labels (same order as --inputs)")
@click.option("--output", required=True, help="Output matrix TSV")
@click.option("--noise-threshold", default=0.45, type=float,
              help="HII threshold for classifying loci (default: 0.45)")
def matrix(inputs, labels, output, noise_threshold):
    """Build multi-tissue HII matrix across N samples.

    Creates a loci × samples matrix of max HII values, with per-locus
    statistics (mean, SD, category). Classifies each locus as stable,
    tissue_variable, or constitutive.
    """
    if len(inputs) != len(labels):
        click.echo(f"Error: {len(inputs)} inputs but {len(labels)} labels", err=True)
        sys.exit(1)

    from .compare import run_matrix

    run_matrix(
        input_paths=list(inputs),
        sample_names=list(labels),
        output_path=output,
        noise_threshold=noise_threshold,
    )


if __name__ == "__main__":
    main()
