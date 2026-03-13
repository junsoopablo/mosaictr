# MosaicTR Technical Status Report

**Date**: 2026-02-28
**Version**: 1.1.0
**Status**: Publication-ready (Bioinformatics target)

---

## 1. Architecture Overview

```
mosaictr/
  __init__.py          15 LOC   Version, module docstring
  genotype.py        1037 LOC   Core genotyping (v4 concordance-based)
  instability.py      880 LOC   Per-haplotype somatic instability metrics
  visualization.py    688 LOC   Waterfall, histogram, summary plots
  interruptions.py    513 LOC   Motif interruption detection
  strchive.py         507 LOC   21-disease pathogenic loci catalog
  vcf_output.py       388 LOC   VCF 4.2 genotype + instability output
  benchmark.py        365 LOC   GIAB Tier1 evaluation
  cli.py              327 LOC   Click CLI (6 commands)
  utils.py            272 LOC   BED/catalog loading utilities
  tests/            3,643 LOC   317 tests across 7 files
─────────────────────────────────
Total               7,947 LOC   (production 4,304 + tests 3,643)
```

### Module Dependencies

```
cli.py
  ├── genotype.py      (genotype command)
  │     └── utils.py
  ├── instability.py   (instability command)
  │     ├── genotype.py (extract_reads_enhanced, hp_cond_v4_genotype)
  │     └── utils.py
  ├── benchmark.py     (evaluate command)
  │     └── utils.py
  ├── visualization.py (visualize command)
  │     └── genotype.py
  ├── interruptions.py (interruptions command)
  ├── strchive.py      (pathogenic-loci command)
  └── vcf_output.py    (--vcf flag for genotype/instability)
```

### External Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| pysam | >= 0.22 | Yes | BAM/FASTA I/O |
| numpy | >= 1.24 | Yes | Weighted median, MAD statistics |
| click | >= 8.0 | Yes | CLI framework |
| tqdm | >= 4.0 | Yes | Progress bars |
| matplotlib | >= 3.7 | Optional | Visualization plots |
| seaborn | >= 0.12 | Optional | Visualization styling |
| parasail | >= 1.3 | Optional | Local realignment for VNTR |

---

## 2. CLI Commands

| Command | Description | Input | Output |
|---------|-------------|-------|--------|
| `mosaictr genotype` | v4 genotyping | BAM + BED | 9-col BED, VCF |
| `mosaictr instability` | Somatic instability | BAM + BED | 27-col TSV, VCF |
| `mosaictr evaluate` | GIAB benchmark | pred BED + truth BED + catalog | Report TXT |
| `mosaictr visualize` | Per-read plots | BAM + BED | PNG (waterfall + summary) |
| `mosaictr interruptions` | Motif interruption | BAM + BED | Console + per-read purity |
| `mosaictr pathogenic-loci` | Disease catalog | None | BED (21 loci) |

---

## 3. Genotyping Algorithm (v4)

### 3.1 Core Pipeline

```
Input: HP-tagged BAM + 4-col loci BED
  ↓
extract_reads_enhanced() → ReadInfo(allele_size, hp, mapq)
  ↓
hp_cond_v4_genotype() → (d1, d2, zygosity, confidence)
  │
  ├── STR path (motif_len <= 6):
  │     ├── HP sufficiency check (min_hp_reads=3, min_hp_frac=0.15)
  │     ├── MapQ-weighted median per HP
  │     ├── EM iterative HP=0 assignment (2 iterations)
  │     ├── Adaptive collapse threshold
  │     ├── Conditional motif-unit rounding
  │     └── Concordance-based zygosity
  │
  └── VNTR path (motif_len > 6):
        ├── Gap-based bimodality test
        ├── Two-pass MAD trimming
        └── Concordance-based zygosity
  ↓
Output: 9-col BED or VCF 4.2
```

### 3.2 Key Functions

| Function | Purpose |
|----------|---------|
| `_hp_concordance()` | HP reads가 자기 haplotype median에 가까운 비율 |
| `_assign_hp0_reads()` | HP=0 reads를 EM iterative로 가까운 cluster에 할당 |
| `_adaptive_collapse_threshold()` | `max(min_val, base * sqrt(ref_n/n))` |
| `_gap_bimodal_test()` | max_gap > gap_factor * median_gap AND > 1.0bp |
| `_v4_zygosity_decision()` | `abs(d1-d2) <= motif_len → HOM`, else concordance test |
| `_v4_str_genotype()` | STR 전용 경로 |
| `_v4_vntr_genotype()` | VNTR 전용 경로 (gap bimodal + MAD trimming) |

### 3.3 Zygosity Decision Logic

```
if abs(d1 - d2) <= motif_len:
    → HOM (confidence = 1.0 - concordance)
elif concordance >= threshold (0.7):
    → HET (confidence = concordance)
else:
    → HOM (confidence = 1.0 - concordance)
```

### 3.4 Fallback Hierarchy

```
HP-tagged (best) → gap-split (bimodal, no HP) → pooled (unimodal, no HP)
```

---

## 4. Instability Module

### 4.1 Six Metrics

| Metric | Formula | Range | Clinical Meaning |
|--------|---------|-------|------------------|
| HII | MAD / motif_len | [0, inf) | 전체 불안정성 (스크리닝) |
| SER | frac(reads > median + motif_len) | [0, 1] | 확장 방향성 (HD, DM1) |
| SCR | frac(reads < median - motif_len) | [0, 1] | 수축 방향성 (치료 반응) |
| ECB | (SER - SCR) / (SER + SCR) | [-1, +1] | 확장-수축 편향 |
| IAS | abs(HII_h1 - HII_h2) / max(HII_h1, HII_h2) | [0, 1] | haplotype 간 비대칭 |
| AIS | max(HII) * conc * sqrt(n/30) | [0, inf) | 신뢰 가중 종합 점수 |

### 4.2 v2 Improvements (2026-02-28)

- EM iterative HP=0 assignment (2 iterations)
- MAD-based outlier trimming (all 3 analysis paths)
- Dropout flag FP: 37% → 5.8% (84.4% 감소)
- Gap-split min_cluster_size=3 → 10,603 false splits 제거
- Noise-threshold-based unstable_haplotype labeling (default 0.45)
- Mononucleotide SER/SCR 2-unit noise threshold

### 4.3 Analysis Path Selection

```
HP reads >= min_hp_reads AND HP fraction >= min_hp_frac
  → hp-tagged (best, per-haplotype)

Else: bimodality test (gap-based)
  → gap-split (두 cluster 분리 가능 시)

Else:
  → pooled (전체 reads 하나로 분석)
```

---

## 5. Output Formats

### 5.1 Genotype BED (9 columns)

```
#chrom  start  end  motif  allele1_size  allele2_size  zygosity  confidence  n_reads
chr4    3074876 3074933 CAG   57   57   HOM   0.950   35
```

### 5.2 Instability TSV (27 columns)

```
#chrom  start  end  motif  median_h1  median_h2  hii_h1  hii_h2  ser_h1  ser_h2
scr_h1  scr_h2  ecb_h1  ecb_h2  ias  ais  range_h1  range_h2  n_h1  n_h2  n_total
concordance  analysis_path  unstable_haplotype  dropout_flag  n_trimmed_h1  n_trimmed_h2
```

### 5.3 VCF 4.2

**Genotype VCF**: FORMAT = `GT:AL:DP:CONF`
**Instability VCF**: FORMAT = `HII:SER:SCR:ECB:IAS:AIS:MEDIAN:DP:CONC:APATH:DROPOUT`

---

## 6. Performance Benchmarks

### 6.1 Genotyping Accuracy (HG002, 46,516 matched loci, chr21/22/X)

| Metric | v3 | v4 | Delta |
|--------|-----|-----|-------|
| Zygosity accuracy | 90.68% | **98.81%** | +8.13 pp |
| TP Zygosity | 74.05% | **97.97%** | +23.92 pp |
| Exact match | 95.65% | 95.65% | 0.00 |
| TP MAE | 3.6609 | 3.6597 | -0.0013 |

### 6.2 Genome-wide Benchmark (108K loci, corrected 2026-02-27)

| Metric | MosaicTR v4 | LongTR |
|--------|-----------|--------|
| Exact match | **97.2%** | 87.5% |
| Within 1bp | **99.3%** | 99.1% |
| MAE | **0.19** | 0.23 |
| Zygosity accuracy | 99.3% | **99.6%** |
| Genotype concordance | **99.7%** | 99.6% |

### 6.3 Confidence Filtering

| Threshold | Zyg Accuracy | Exact Match | Loci Retained |
|-----------|-------------|-------------|---------------|
| 0.00 (no filter) | 98.81% | 97.2% | 100% |
| **0.55 (recommended)** | **99.8%** | **99.1%** | **88.3%** |

### 6.4 Mendelian Consistency (HG002/003/004 trio, 1,769,186 loci)

| Metric | Value |
|--------|-------|
| Strict Mendelian | 89.6% |
| Off-by-one-unit | **99.3%** |

### 6.5 Competitor Comparison (Common 12K subset)

| Tool | Overall Zyg | TP Zyg | TP MAE |
|------|-------------|--------|--------|
| TRGT | 99.4% | 98.7% | **2.80** |
| LongTR | 99.5% | 98.5% | 3.11 |
| **MosaicTR v4** | 98.81% | 97.97% | 3.66 |

Note: MosaicTR는 전체 46.5K loci에서 평가 (다른 도구는 Common 12K subset).

### 6.6 Disease Loci Genotyping

- STRchive 75개 pathogenic loci 중 **69개 (92%) genotyping 성공**
- 실패 6개: coverage 부족 또는 reference에 포함 안된 loci

### 6.7 Runtime

| Dataset | Loci | Time | Throughput |
|---------|------|------|-----------|
| Genotyping (chr21/22/X) | 108,584 | ~45 min | ~40 loci/sec |
| Instability (HG002, genome-wide) | 107,230 | 915s | ~117 loci/sec |

---

## 7. Instability Validation

### 7.1 Noise Floor (HG002, 108K loci)

| Metric | Value |
|--------|-------|
| Mean HII | 0.023 |
| Std HII | 0.141 |
| P95 HII | 0.000 |
| 3-sigma threshold | **0.45** |
| HII = 0 fraction | 91.4% |

### 7.2 Trio Consistency (HG002/003/004)

| Comparison | HII Correlation |
|------------|----------------|
| HG002 - HG003 | 0.49 |
| HG002 - HG004 | 0.53 |
| HG003 - HG004 | 0.60 |
| All 3 low (HII < 0.45) | 95.8% |
| All 3 high | 0.59% (628 loci) |

### 7.3 Disease Carrier Detection

**PureTarget (Coriell, HiFi, gap-split path)**: 5/7 detected

| Sample | Gene | Disease | HII | Detection |
|--------|------|---------|-----|-----------|
| NA13509 | HTT | HD | 1.67 (11.6 sigma) | Detected |
| NA13515 | HTT | HD | 14.67 (103.6 sigma) | Detected |
| NA06905 | FMR1 | FXS | 1.0 (6.9 sigma) | Detected |
| NA06153 | ATXN3 | SCA3 | 0.67 (4.5 sigma) | Detected |
| NA13536 | ATXN1 | SCA1 | 4.0 (28.1 sigma) | Detected |
| NA03697 | DMPK | DM1 | 0 | Not detected (allele dropout) |
| NA15850 | FXN | FRDA | N/A | Not detected (3 reads only) |

**1000G ONT (ATXN10, HP-tagged via LongPhase)**: 3 confirmed carriers

| Sample | Alleles (RU) | HII (normal/expanded) | IAS |
|--------|-------------|----------------------|-----|
| HG01122 | 15 / 1041 | 0.20 / 61.0 | 0.997 |
| HG02252 | 512 / 944 | 4.0 / 10.0 | 0.600 |
| HG02345 | 14 / 321 | 0.20 / 2.6 | 0.923 |

**HTT Carrier (HG02275, 1KG-ONT-VIENNA, whatshap-tagged)**:
- HP=1: 22 CAG (normal), HP=2: 43 CAG (expanded)
- v2 (EM + trimming): HII=1 both haplotypes, IAS=0.00

### 7.4 Cross-Platform

| Platform | Mean HII | Noise Level |
|----------|----------|-------------|
| PacBio HiFi | 0.023 | Baseline |
| ONT | 0.36 | 16x higher (expected) |

---

## 8. Code Quality

### 8.1 Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| test_genotype.py | 124 | genotype.py (v4 algorithm, output writer) |
| test_instability.py | 76 | instability.py (6 metrics, TSV writer, analysis paths) |
| test_vcf_output.py | 31 | vcf_output.py (formatters, genotype VCF, instability VCF) |
| test_utils.py | 26 | utils.py (BED loading, Tier1, catalog, chrom_split) |
| test_strchive.py | 24 | strchive.py (catalog, annotation, classification) |
| test_interruptions.py | 24 | interruptions.py (motif units, detection, context) |
| test_visualization.py | 12 | visualization.py (smoke tests, all 3 plot types) |
| **Total** | **317** | |

### 8.2 Bug Fixes Applied (2026-02-28)

14 bugs fixed in this session, all with regression tests:

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | CRITICAL | vcf_output.py `_fmt_float` | `int(NaN)` → ValueError | `math.isnan`/`math.isinf` guard |
| 2 | CRITICAL | vcf_output.py `_fmt_size` | Same `int(NaN)` crash | Same guard |
| 3 | CRITICAL | genotype.py `_write_output_bed._fmt` | Same crash pattern | Same guard |
| 4 | CRITICAL | instability.py `_write_instability_tsv._fmt` | Same crash pattern | Same guard + `import math` |
| 5 | CRITICAL | vcf_output.py `write_genotype_vcf` | `{confidence:.3f}` produces "nan" | `_fmt_float()` |
| 6 | CRITICAL | cli.py genotype VCF | `float(".")` crash on "." confidence | `cols[7] != "."` guard |
| 7 | CRITICAL | cli.py instability VCF | loci/results misalignment with `--min-instability` | Extract loci from TSV rows |
| 8 | HIGH | vcf_output.py `_encode_gt` | NaN allele → wrong "1/1" GT | NaN check → "./." |
| 9 | HIGH | genotype.py `_write_output_bed` | NaN confidence bypasses `--min-confidence` | Explicit NaN/Inf check |
| 10 | HIGH | genotype.py `_genotype_chunk` | BAM/FASTA resource leak on exception | `try/finally` |
| 11 | HIGH | instability.py `_instability_chunk` | Same resource leak | `try/finally` |
| 12 | HIGH | cli.py `visualize` | BAM/FASTA resource leak | `try/finally` |
| 13 | MEDIUM | cli.py `interruptions` | NaN purity prints "nan" | `math.isnan` check |
| 14 | LOW | benchmark.py | `float(".")` on "." confidence | `cols[7] != "."` guard |

Additional: visualization.py docstring `modal_h1` → `median_h1`, removed unused import.

### 8.3 NaN/Inf Safety

All float-to-string conversion points now handle NaN/Inf:
- `_fmt_float()` → "." (vcf_output.py)
- `_fmt_size()` → "." (vcf_output.py)
- `_write_output_bed._fmt()` → "." (genotype.py)
- `_write_instability_tsv._fmt()` → "." (instability.py)
- `_encode_gt()` → "./." (vcf_output.py)
- Confidence fields → "." or 0.0 fallback (genotype.py, cli.py, benchmark.py)

### 8.4 Resource Management

All pysam file handles (BAM, FASTA) are wrapped in `try/finally`:
- `_genotype_chunk()` (genotype.py)
- `_instability_chunk()` (instability.py)
- `visualize` CLI command (cli.py)

---

## 9. Unique Contributions (vs Competitors)

| Feature | MosaicTR | TRGT | prancSTR | LongTR | searchSTR |
|---------|---------|------|----------|--------|-----------|
| Per-haplotype instability | **Yes (HP-tag)** | No | No (short read) | No | No |
| Dedicated instability metrics | **6 (HII/SER/SCR/ECB/IAS/AIS)** | ALLR only | C, f, p | None | SEI |
| Long-read native | Yes | Yes | No | Yes | No |
| STR + VNTR | Yes | Yes | STR only | Yes | STR only |
| Direct BAM input | Yes | Yes | HipSTR VCF | Yes | Short-read VCF |
| Fallback without HP tags | Yes (gap-split/pooled) | N/A | N/A | N/A | N/A |
| Motif interruption detection | Yes | No | No | No | No |
| Pathogenic loci catalog | Yes (21 diseases) | Yes | No | No | No |
| VCF 4.2 output | Yes | Yes | No | No | No |

**Core novelty**: No existing tool combines HP-tag haplotype resolution with dedicated per-locus somatic instability quantification from long-read data.

---

## 10. Known Limitations

### 10.1 Technical
- **HP-tagged BAM required** for best results; fallback paths have lower confidence
- **nprocs=1 recommended** due to pysam OOM in parallel (per-chrom split mitigates)
- **parasail disabled for VNTR** (motif_len >= 7) due to poor alignment quality
- **Large expansions** (>10kb) may not be captured by standard HiFi reads

### 10.2 Algorithmic
- **HOM→HET misclassification**: ~408/46,516 cases (0.9%); `--min-confidence 0.55` filter reduces to 0.2%
- **Allele dropout**: 5.8% FP rate after v2 improvements
- **ONT noise floor**: 16x higher than HiFi → stricter thresholds needed for ONT

### 10.3 Validation Gaps
- **CASTLE Phase 3** (cancer MSS specificity): audit complete, BAMs not yet aligned
- **Low coverage (<15x)**: not systematically tested
- **Population-scale (>1000 samples)**: not benchmarked (single-sample tool)

---

## 11. File Manifest

```
deeptr/
  README.md               Project documentation
  LICENSE                  MIT License
  pyproject.toml           Package metadata (v1.1.0)
  CLAUDE.md                Development instructions
  mosaictr/
    __init__.py            Package init (v1.1.0)
    genotype.py            Core genotyping module
    instability.py         Somatic instability module
    visualization.py       Plot generation
    interruptions.py       Motif interruption detection
    strchive.py            Pathogenic loci catalog
    vcf_output.py          VCF 4.2 output
    benchmark.py           GIAB evaluation
    cli.py                 Click CLI
    utils.py               Utilities
    tests/
      __init__.py
      test_genotype.py     124 tests
      test_instability.py   76 tests
      test_vcf_output.py    31 tests
      test_utils.py         26 tests
      test_strchive.py      24 tests
      test_interruptions.py 24 tests
      test_visualization.py 12 tests
  scripts/
    compare_v3_v4.py       v3 vs v4 comparison
    compare_v2_v3.py       v2 vs v3 comparison
    diagnose_tp_mae.py     TP MAE 3-way diagnosis
    benchmark_genome_wide.py  Genome-wide benchmark
    analyze_instability_hg002.py  HG002 instability analysis
    validate_instability.py       Phase 1A+1B validation
    validate_instability_phase2.py  Phase 2 PureTarget
    validate_instability_phase3.py  Phase 3 CASTLE
    validate_instability_1000g.py   1000G ONT carriers
  docs/
    technical_status.md    This document
  output/
    (benchmark results, figures, instability reports)
```
