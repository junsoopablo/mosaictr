"""MosaicTR: haplotype-aware tandem repeat genotyping and somatic instability analysis from long-read sequencing.

Modules:
    genotype    - Core genotyping (v4 concordance-based zygosity)
    instability - Per-haplotype somatic instability metrics (HII, SER, SCR, ECB, IAS, AIS)
    vcf_output  - VCF 4.2 output for genotype and instability results
    strchive    - Built-in pathogenic TR loci catalog
    interruptions - Motif interruption detection
    visualization - Per-read waterfall and instability summary plots
    compare     - Cross-tissue instability comparison (paired + matrix)
    benchmark   - GIAB truth evaluation
    utils       - BED/catalog loading utilities
    cli         - Click command-line interface (8 commands)
"""

__version__ = "1.1.0"
