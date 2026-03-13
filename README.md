# MosaicTR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Tandem repeat somatic instability quantification from long-read sequencing.

MosaicTR uses HP (haplotype) tags in BAM files to decompose tandem repeat signals per haplotype, enabling:
- **Somatic instability** quantification with 2 per-haplotype metrics (HII, IAS)
- **Multi-sample comparison** for tissue-specific instability detection
- **Genotyping** with concordance-based zygosity calling

## Quick Start

```bash
pip install .

# Genotype TR loci
mosaictr genotype --bam tagged.bam --loci loci.bed --output genotypes.bed

# Compute somatic instability
mosaictr instability --bam tagged.bam --loci loci.bed --output instability.tsv

# Compare instability across tissues
mosaictr compare --baseline blood.tsv --target colon.tsv --output comparison.tsv
```

## Installation

```bash
pip install .
```

### Dependencies

- Python >= 3.10
- pysam >= 0.22
- numpy >= 1.24
- click >= 8.0
- tqdm >= 4.0

## Usage

### Genotype

```bash
mosaictr genotype \
  --bam aligned.bam \
  --loci loci.bed \
  --output genotypes.bed \
  --threads 1
```

Options:
- `--min-mapq 5` — Minimum mapping quality
- `--min-flank 50` — Minimum flanking bases
- `--max-reads 200` — Maximum reads per locus
- `--ref ref.fa` — Reference FASTA for parasail realignment (optional)
- `--min-hp-reads 3` — Minimum HP-tagged reads per haplotype
- `--min-hp-frac 0.15` — Minimum fraction of HP-tagged reads
- `--concordance-threshold 0.7` — HP concordance threshold for HET call
- `--min-confidence 0.0` — Minimum confidence to report a call (0 = no filter; 0.55 recommended for high-accuracy filtering)

### Instability

```bash
mosaictr instability \
  --bam aligned.bam \
  --loci loci.bed \
  --output instability.tsv \
  --threads 1
```

Options:
- `--min-instability 0.0` — Minimum HII threshold for output filtering
- `--min-hp-reads 3` / `--min-hp-frac 0.15` — HP sufficiency thresholds
- `--skip-hp-check` — Skip HP tag check (uses gap-split/pooled fallback)

### Compare (pairwise)

```bash
mosaictr compare \
  --baseline blood_instability.tsv \
  --target colon_instability.tsv \
  --output comparison.tsv
```

Options:
- `--noise-threshold 0.45` — HII threshold for calling unstable
- `--min-delta 0.5` — Minimum ΔHII to report
- `--baseline-label` / `--target-label` — Tissue labels

### Matrix (multi-sample)

```bash
mosaictr matrix \
  --inputs blood.tsv --inputs colon.tsv --inputs brain.tsv \
  --labels blood --labels colon --labels brain \
  --output hii_matrix.tsv
```

Builds a loci × samples HII matrix and classifies each locus as `stable`, `tissue_variable`, or `constitutive`.

### Evaluate

```bash
mosaictr evaluate \
  --predictions genotypes.bed \
  --truth-bed giab_tier1.bed \
  --catalog adotto_catalog.bed \
  --output report.txt
```

## HP Tagging (Recommended)

MosaicTR works best with **HP-tagged BAMs** for true per-haplotype analysis. Without HP tags, the tool falls back to gap-split (bimodal detection) or pooled analysis, which provides lower confidence results.

The instability module will log a warning if fewer than 50% of loci use HP-tagged analysis.

### How to add HP tags

You need a phased VCF (from variant calling + phasing) and your aligned BAM.

**WhatsHap** (most common):
```bash
whatshap haplotag --reference ref.fa input.bam phased.vcf.gz -o tagged.bam
samtools index tagged.bam
```

**HiPhase** (PacBio official):
```bash
hiphase --bam input.bam --vcf phased.vcf.gz --output-bam tagged.bam
samtools index tagged.bam
```

**LongPhase** (nanopore/HiFi):
```bash
longphase haplotag -b input.bam -s phased.vcf.gz -r ref.fa -o tagged
samtools index tagged.bam
```

## Input Format

### Loci BED (4-column)

```
chr4    3074876   3074933   CAG
chr9    27573528  27573546  AAGGG
chrX    147912050 147912110 CGG
```

Columns: chrom, start, end, motif sequence.

## Output Formats

### Genotype BED (9 columns)

| Column | Description |
|--------|-------------|
| chrom | Chromosome |
| start | Start position |
| end | End position |
| motif | Repeat motif sequence |
| allele1 | Allele 1 size (bp) |
| allele2 | Allele 2 size (bp) |
| zygosity | HOM or HET |
| confidence | Zygosity confidence score [0, 1] |
| n_reads | Total reads at locus |

### Instability TSV (13 columns)

| Column | Description |
|--------|-------------|
| chrom, start, end, motif | Locus coordinates and motif |
| median_h1, median_h2 | Median allele size per haplotype (bp) |
| hii_h1, hii_h2 | Haplotype Instability Index (motif-normalized MAD) |
| ias | Instability Asymmetry Score (inter-haplotype difference) |
| n_h1, n_h2, n_total | Read counts per haplotype and total |
| analysis_path | hp-tagged, gap-split, or pooled |

## License

MIT License. See [LICENSE](LICENSE) file.
