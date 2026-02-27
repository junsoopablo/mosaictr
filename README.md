# HaploTR

Haplotype-aware tandem repeat genotyping and somatic instability analysis from long-read sequencing.

HaploTR uses HP (haplotype) tags in BAM files to decompose tandem repeat signals per haplotype, enabling:
- **Genotyping** with concordance-based zygosity calling
- **Somatic instability** quantification with 6 per-haplotype metrics

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
haplotr genotype \
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
haplotr instability \
  --bam aligned.bam \
  --loci loci.bed \
  --output instability.tsv \
  --threads 1
```

Options:
- `--min-instability 0.0` — Minimum AIS threshold for output filtering
- `--min-hp-reads 3` / `--min-hp-frac 0.15` — HP sufficiency thresholds

### Evaluate

```bash
haplotr evaluate \
  --predictions genotypes.bed \
  --truth-bed giab_tier1.bed \
  --catalog adotto_catalog.bed \
  --output report.txt
```

## HP Tagging (Recommended)

HaploTR works best with **HP-tagged BAMs** for true per-haplotype analysis. Without HP tags, the tool falls back to gap-split (bimodal detection) or pooled analysis, which provides lower confidence results.

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
| allele1 | Allele 1 size (bp difference from reference) |
| allele2 | Allele 2 size (bp difference from reference) |
| zygosity | HOM or HET |
| confidence | Zygosity confidence score [0, 1] |
| n_reads | Total reads at locus |

### Instability TSV (25 columns)

| Column | Description |
|--------|-------------|
| chrom, start, end, motif | Locus coordinates and motif |
| modal_h1, modal_h2 | Modal allele size per haplotype (bp diff from ref) |
| hii_h1, hii_h2 | Haplotype Instability Index (motif-normalized dispersion) |
| ser_h1, ser_h2 | Somatic Expansion Ratio (fraction expanded > 1 motif unit) |
| scr_h1, scr_h2 | Somatic Contraction Ratio (fraction contracted > 1 motif unit) |
| ecb_h1, ecb_h2 | Expansion-Contraction Bias [-1, +1] |
| ias | Instability Asymmetry Score (inter-haplotype difference) |
| ais | Aggregate Instability Score (confidence-weighted summary) |
| range_h1, range_h2 | Read length range per haplotype |
| n_h1, n_h2, n_total | Read counts per haplotype and total |
| concordance | HP concordance score |
| analysis_path | hp-tagged, gap-split, or pooled |
| unstable_haplotype | Which haplotype is more unstable (h1/h2/both/none) |
| dropout_flag | 1 if possible allele dropout detected |

## License

See LICENSE file.
