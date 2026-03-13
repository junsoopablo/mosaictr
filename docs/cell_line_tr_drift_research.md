# Cell Line TR Drift Research

**Date**: 2026-03-06
**Status**: Feasibility study

---

## 1. Research Question

Cell line은 passage가 다르면 genome이 달라진다. TR loci는 mutation rate가
다른 영역보다 10-10,000x 높으므로, passage에 따른 TR drift가 genome-wide로
어떻게 나타나는지 long-read로 정량화할 수 있을까?

**핵심 gap**: Long-read genome-wide TR instability를 cell line passage/batch 간
정량 비교한 연구가 아직 없음.

---

## 2. 선행 연구

### 2.1 LCL STR Instability (Yoon et al. 2013, Genomics Data)
- EBV transformation 시 15 forensic STR loci의 9.2%에서 mutation
- FGA locus 가장 불안정 (mutation rate 0.032)
- **한계**: CE 기반, 15 loci만, genome-wide 아님

### 2.2 HeLa Lab-to-Lab Heterogeneity (Liu et al. 2018; Frattini et al. 2015)
- 13개 랩의 HeLa → concordance 89.3% (H9: 99.2%, HCT116: 98.5%)
- SNV/CNV 중심, TR은 체계적으로 분석 안 함

### 2.3 LCL Chromosomal Instability (Volleth et al. 2020)
- 21-49주 내 모든 LCL에서 trisomy 12 출현
- Karyotype/FISH 수준, TR resolution 아님

### 2.4 HG002 Batch Effect (NIST Mosaic Benchmark 2024)
- NIST RM 8391 vs Coriell NA24385: mosaic VAF에서 유의한 차이
- "differences likely result from changes in the cell line"
- **TR 차원은 미탐색**

### 2.5 TR Mutation Rates (Gymrek et al. 2017; Willems et al. 2017)
- TR은 다른 영역보다 10-10,000x 높은 mutation rate
- Lifelong somatic expansion이 질환 유발 가능

---

## 3. Available Data

### 3.1 HG008 Tumor-Normal (GIAB Cancer, 최적 데이터!)

| Sample | Type | Passage | Platform | Coverage | HP Tags | Local |
|--------|------|---------|----------|----------|---------|-------|
| HG008-T | Tumor | p23 | HiFi SPRQ | 124x | No | Yes (88GB) |
| HG008-T | Tumor | p23 | HiFi | 116x | No | No (FTP) |
| HG008-T | Tumor | p23 | HiFi | 106x | No | No (FTP) |
| HG008-T | Tumor | **p21** | HiFi | 43x | No | No (FTP) |
| HG008-T | Tumor | **p41** | HiFi | 43x | No | No (FTP) |
| HG008-N-P | Normal pancreas | NA | HiFi SPRQ | 38x | No | Yes (39GB) |
| HG008-N-D | Normal duodenum | NA | HiFi | 68x | No | No (FTP) |

**p21 vs p41 = 20 passages 차이!** 동일 tumor cell line의 passage별 TR drift를
genome-wide로 볼 수 있는 이상적인 데이터셋.

p21 FTP: `https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data_somatic/HG008/NIST/HG008-T_bulk/20240508p21/UMD-Revio-HG008Tp21-20241011/HG008-T_NIST-bulk-20240508p21_PacBio-HiFi-Revio_20241011_43x_GRCh38-GIABv3.bam`

p41 FTP: `https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data_somatic/HG008/NIST/HG008-T_bulk/20240508p41/UMD-Revio-HG008Tp41-20241011/HG008-T_NIST-bulk-20240508p41_PacBio-HiFi-Revio_20241011_43x_GRCh38-GIABv3.bam`

### 3.2 HG002 Multi-Platform (이미 로컬에 있음)
- HG002 PacBio HiFi Revio 48x (HP-tagged)
- HG002 vs HG003/HG004 trio (이미 instability 분석 완료)
- 다른 배치/sequencing run의 HG002 BAMs도 GIAB FTP에 있음

### 3.3 HG008 Tumor Clones
- Manifest에 clone ID별 karyotyping 데이터 존재 (2D6, 3E4, SC14 등)
- 일부 clone에 HiFi sequencing 있을 수 있음 (확인 필요)

---

## 4. Experimental Design

### Phase 1: HG008 Passage Comparison (p21 vs p41)

```
Step 1: Download p21 + p41 BAMs (~80GB total)
Step 2: Genotype with MosaicTR (--skip-hp-check, gap-split/pooled)
Step 3: Instability analysis on both
Step 4: mosaictr compare --baseline p21.tsv --target p41.tsv
Step 5: mosaictr matrix --inputs p21.tsv p23.tsv p41.tsv --labels p21 p23 p41
```

**Expected findings**:
- 대부분 loci (>95%): stable across passages
- 일부 loci: passage-dependent TR drift (특히 긴 repeat, mononucleotide)
- Tumor-specific MSI loci: p21 → p41 방향으로 점진적 확장
- Normal vs Tumor 비교: tumor에서 훨씬 높은 TR instability

### Phase 2: Normal vs Tumor Comparison

```
mosaictr compare --baseline HG008-N-P.tsv --target HG008-T-p23.tsv \
  --baseline-label "normal_pancreas" --target-label "tumor_p23"
```

**Expected findings**:
- Tumor에서 genome-wide TR instability 상승
- MSI loci 식별 가능
- 이건 cancer genomics에서 직접적 의미

### Phase 3: Multi-Passage Matrix

```
mosaictr matrix \
  --inputs normal.tsv p21.tsv p23.tsv p41.tsv \
  --labels normal p21 p23 p41
```

**Expected findings**:
- TR drift의 passage dose-response
- "Early drift" (p21에서 이미 변한 loci) vs "late drift" (p41에서만 변한 loci)
- Constitutive unstable loci (모든 passage에서 높은 HII)

---

## 5. Technical Considerations

### HP Tags
- HG008 BAMs에 HP tag 없음
- `--skip-hp-check` 사용 → gap-split/pooled fallback
- Per-haplotype 분석 불가하지만, 전체 instability는 측정 가능
- 원하면 whatshap haplotagging 가능 (phased VCF 필요)

### Coverage
- p21/p41: 43x (충분)
- p23: 106-124x (풍부)
- Normal: 38-68x (충분)

### Loci Catalog
- Adotto v1.2: 1,784,804 loci (genome-wide)
- 또는 테스트용 108K subset (chr21/22/X) 먼저 시도

### Storage
- p21 BAM: ~40GB, p41 BAM: ~40GB
- Total additional: ~80GB download

---

## 6. Novelty Assessment

| Aspect | 기존 | 이 연구 |
|--------|------|---------|
| TR 분석 규모 | 15 forensic loci | 1.7M genome-wide |
| 기술 | CE, short-read | PacBio HiFi long-read |
| 정량화 | Mutation yes/no | HII, SER, SCR, ECB metrics |
| Passage 비교 | 정성적 | Genome-wide quantitative (ΔHII) |
| 도구 | Ad hoc | MosaicTR compare/matrix |
| 데이터 | Custom cell lines | GIAB public reference (재현 가능) |

**Novelty**: 첫 번째 genome-wide long-read TR drift 정량화 (public GIAB data).

---

## 7. 논문 포함 여부 판단 기준

### 넣을 경우
- Application Note의 "use case" section으로 적합
- MosaicTR compare/matrix 기능의 실제 적용 사례
- GIAB public data → 재현 가능 → reviewer 친화적

### 별도 논문으로 할 경우
- 데이터가 풍부하면 독립 letter/brief communication 가능
- Cancer genomics 각도 (normal vs tumor TR landscape)
- Cell line QC 각도 (passage-aware TR profiling for reproducibility)

### 판단은 Phase 1 결과 후
- p21 vs p41에서 의미있는 차이 → 넣을 가치
- 차이가 미미 → skip 또는 supplementary
- 예상: pancreatic cancer cell line이므로 MSI 가능성 있음
