#!/usr/bin/env python3
"""Mendelian allele transmission analysis for trio.

Focuses on loci WHERE DIFFERENCES EXIST — not just overall concordance.
For each HET locus in child (HG002), checks if each allele can be traced
back to one parent (HG003=father, HG004=mother).

Key questions:
1. At HET loci, does child's allele pair match one-from-each-parent?
2. What are the "de novo" candidates, and are they real or measurement error?
3. How does accuracy vary by repeat size, motif length?
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter

# --- File paths ---
BED_DIR = "output/genome_wide"
HG002 = os.path.join(BED_DIR, "v4_hg002_genome_wide.bed")
HG003 = os.path.join(BED_DIR, "v4_hg003_genome_wide.bed")  # father
HG004 = os.path.join(BED_DIR, "v4_hg004_genome_wide.bed")  # mother

OUTDIR = "output/trio_transmission"
os.makedirs(OUTDIR, exist_ok=True)


def load_bed(path):
    """Load BED file into dict keyed by (chrom, start, end)."""
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            motif = parts[3]
            a1, a2 = parts[4], parts[5]
            zyg = parts[6]
            conf = parts[7]
            n_reads = int(parts[8]) if parts[8] != "." else 0

            # Skip failed loci
            if a1 == "." or a2 == ".":
                continue

            a1, a2 = float(a1), float(a2)
            conf = float(conf) if conf != "." else 0.0
            key = (chrom, start, end)
            data[key] = {
                "motif": motif,
                "a1": a1, "a2": a2,
                "zyg": zyg, "conf": conf,
                "n_reads": n_reads,
                "motif_len": len(motif),
                "ref_size": end - start,
            }
    return data


def allele_set(entry):
    """Return sorted allele pair."""
    return sorted([entry["a1"], entry["a2"]])


def check_mendelian_transmission(child, father, mother, tolerance=0):
    """Check if child alleles can be explained by one from each parent.

    Returns: (pass, best_assignment, min_error)
    best_assignment: (father_allele, mother_allele) that best matches child
    min_error: sum of |child_allele - parent_allele| for best assignment
    """
    c = sorted([child["a1"], child["a2"]])
    f = sorted([father["a1"], father["a2"]])
    m = sorted([mother["a1"], mother["a2"]])

    # Try all 4 combinations: (f_allele, m_allele) → (child_a1, child_a2)
    best_err = float("inf")
    best_assign = None

    for fi in range(2):
        for mi in range(2):
            # Assign: child gets f[fi] from father, m[mi] from mother
            pair = sorted([f[fi], m[mi]])
            err = abs(c[0] - pair[0]) + abs(c[1] - pair[1])
            if err < best_err:
                best_err = err
                best_assign = (f[fi], m[mi])

    passed = best_err <= tolerance
    return passed, best_assign, best_err


def main():
    print("Loading genotypes...")
    child = load_bed(HG002)
    father = load_bed(HG003)
    mother = load_bed(HG004)

    print(f"  HG002 (child):  {len(child):,} loci")
    print(f"  HG003 (father): {len(father):,} loci")
    print(f"  HG004 (mother): {len(mother):,} loci")

    # Common loci
    common = set(child.keys()) & set(father.keys()) & set(mother.keys())
    print(f"  Common: {len(common):,} loci")

    # --- Analysis ---
    results = {
        "all": {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0, "errors": []},
        "child_het": {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0, "errors": []},
        "child_hom": {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0, "errors": []},
        "parent_het": {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0, "errors": []},
        "both_parent_het": {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0, "errors": []},
    }

    # By motif length and ref size
    by_motif = defaultdict(lambda: {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0})
    by_refsize = defaultdict(lambda: {"strict": 0, "w1bp": 0, "w1unit": 0, "total": 0})
    error_distribution = []  # list of (min_error, motif_len, ref_size, child_zyg)

    # De novo candidates (child allele not explainable by parents within 1 unit)
    denovo_candidates = []

    for key in sorted(common):
        c = child[key]
        f = father[key]
        m = mother[key]
        motif_len = c["motif_len"]
        ref_size = c["ref_size"]

        strict_pass, assign, min_err = check_mendelian_transmission(c, f, m, tolerance=0)
        w1bp_pass, _, _ = check_mendelian_transmission(c, f, m, tolerance=2)  # ±1bp each allele
        w1unit_pass, _, _ = check_mendelian_transmission(c, f, m, tolerance=motif_len * 2)

        # Categorize
        child_het = c["zyg"] == "HET"
        father_het = f["zyg"] == "HET"
        mother_het = m["zyg"] == "HET"

        categories = ["all"]
        if child_het:
            categories.append("child_het")
        else:
            categories.append("child_hom")
        if father_het or mother_het:
            categories.append("parent_het")
        if father_het and mother_het:
            categories.append("both_parent_het")

        for cat in categories:
            results[cat]["total"] += 1
            if strict_pass:
                results[cat]["strict"] += 1
            if w1bp_pass:
                results[cat]["w1bp"] += 1
            if w1unit_pass:
                results[cat]["w1unit"] += 1

        # By motif
        if motif_len <= 1:
            mcat = "mono"
        elif motif_len <= 2:
            mcat = "di"
        elif motif_len <= 6:
            mcat = f"str_{motif_len}bp"
        else:
            mcat = "vntr"

        by_motif[mcat]["total"] += 1
        if strict_pass: by_motif[mcat]["strict"] += 1
        if w1bp_pass: by_motif[mcat]["w1bp"] += 1
        if w1unit_pass: by_motif[mcat]["w1unit"] += 1

        # By ref size bin
        if ref_size < 50:
            rcat = "<50bp"
        elif ref_size < 150:
            rcat = "50-150bp"
        elif ref_size < 300:
            rcat = "150-300bp"
        elif ref_size < 500:
            rcat = "300-500bp"
        else:
            rcat = ">=500bp"

        by_refsize[rcat]["total"] += 1
        if strict_pass: by_refsize[rcat]["strict"] += 1
        if w1bp_pass: by_refsize[rcat]["w1bp"] += 1
        if w1unit_pass: by_refsize[rcat]["w1unit"] += 1

        error_distribution.append((min_err, motif_len, ref_size, "HET" if child_het else "HOM"))

        # De novo candidates: child allele not within 1 motif unit of any parent allele
        if not w1unit_pass and child_het and c["n_reads"] >= 10:
            denovo_candidates.append({
                "key": key,
                "child": c, "father": f, "mother": m,
                "min_err": min_err,
            })

    # --- Report ---
    lines = []
    lines.append("=" * 80)
    lines.append("  MENDELIAN ALLELE TRANSMISSION ANALYSIS")
    lines.append("  Focus: loci WHERE DIFFERENCES EXIST")
    lines.append("=" * 80)
    lines.append("")

    # Overall
    lines.append("--- Overall Mendelian Concordance ---")
    for cat_name, cat_label in [
        ("all", "All loci"),
        ("child_het", "Child HET"),
        ("child_hom", "Child HOM"),
        ("parent_het", "Any parent HET"),
        ("both_parent_het", "Both parents HET"),
    ]:
        r = results[cat_name]
        t = r["total"]
        if t == 0:
            continue
        lines.append(f"  {cat_label:25s}  n={t:>10,}  strict={r['strict']/t*100:6.2f}%  "
                      f"±1bp={r['w1bp']/t*100:6.2f}%  ±1unit={r['w1unit']/t*100:6.2f}%")
    lines.append("")

    # By motif
    lines.append("--- By Motif Length ---")
    for mcat in ["mono", "di", "str_3bp", "str_4bp", "str_5bp", "str_6bp", "vntr"]:
        r = by_motif.get(mcat)
        if not r or r["total"] == 0:
            continue
        t = r["total"]
        lines.append(f"  {mcat:12s}  n={t:>10,}  strict={r['strict']/t*100:6.2f}%  "
                      f"±1bp={r['w1bp']/t*100:6.2f}%  ±1unit={r['w1unit']/t*100:6.2f}%")
    lines.append("")

    # By ref size
    lines.append("--- By Reference Size (long-read advantage) ---")
    for rcat in ["<50bp", "50-150bp", "150-300bp", "300-500bp", ">=500bp"]:
        r = by_refsize.get(rcat)
        if not r or r["total"] == 0:
            continue
        t = r["total"]
        lines.append(f"  {rcat:12s}  n={t:>10,}  strict={r['strict']/t*100:6.2f}%  "
                      f"±1bp={r['w1bp']/t*100:6.2f}%  ±1unit={r['w1unit']/t*100:6.2f}%")
    lines.append("")

    # Error distribution for child HET
    het_errors = [(e, ml, rs) for e, ml, rs, zyg in error_distribution if zyg == "HET"]
    if het_errors:
        errs = [e for e, _, _ in het_errors]
        lines.append("--- Error Distribution (Child HET loci) ---")
        lines.append(f"  n = {len(het_errors):,}")
        lines.append(f"  Mean error:   {np.mean(errs):.2f} bp")
        lines.append(f"  Median error: {np.median(errs):.2f} bp")
        lines.append(f"  P90 error:    {np.percentile(errs, 90):.2f} bp")
        lines.append(f"  P99 error:    {np.percentile(errs, 99):.2f} bp")
        lines.append(f"  Error = 0:    {sum(1 for e in errs if e == 0):,} ({sum(1 for e in errs if e == 0)/len(errs)*100:.1f}%)")
        lines.append(f"  Error <= 2:   {sum(1 for e in errs if e <= 2):,} ({sum(1 for e in errs if e <= 2)/len(errs)*100:.1f}%)")
        lines.append(f"  Error <= 5:   {sum(1 for e in errs if e <= 5):,} ({sum(1 for e in errs if e <= 5)/len(errs)*100:.1f}%)")
        lines.append(f"  Error > 10:   {sum(1 for e in errs if e > 10):,} ({sum(1 for e in errs if e > 10)/len(errs)*100:.1f}%)")
        lines.append(f"  Error > 50:   {sum(1 for e in errs if e > 50):,} ({sum(1 for e in errs if e > 50)/len(errs)*100:.1f}%)")
        lines.append("")

        # By ref size for HET only
        lines.append("--- Child HET Error by Reference Size ---")
        het_by_rs = defaultdict(list)
        for e, ml, rs in het_errors:
            if rs < 50: rcat = "<50bp"
            elif rs < 150: rcat = "50-150bp"
            elif rs < 300: rcat = "150-300bp"
            elif rs < 500: rcat = "300-500bp"
            else: rcat = ">=500bp"
            het_by_rs[rcat].append(e)
        for rcat in ["<50bp", "50-150bp", "150-300bp", "300-500bp", ">=500bp"]:
            errs_r = het_by_rs.get(rcat, [])
            if not errs_r:
                continue
            lines.append(f"  {rcat:12s}  n={len(errs_r):>6,}  mean_err={np.mean(errs_r):7.2f}  "
                          f"median={np.median(errs_r):5.1f}  exact={sum(1 for e in errs_r if e == 0)/len(errs_r)*100:5.1f}%  "
                          f"±1bp={sum(1 for e in errs_r if e <= 2)/len(errs_r)*100:5.1f}%")
        lines.append("")

    # De novo candidates
    lines.append(f"--- De Novo Candidates (HET, >=10 reads, >1 unit error) ---")
    lines.append(f"  Count: {len(denovo_candidates):,}")
    if denovo_candidates:
        # Sort by error
        denovo_candidates.sort(key=lambda x: -x["min_err"])
        lines.append(f"  Top 20:")
        lines.append(f"  {'Locus':35s} {'Motif':8s} {'RefSz':>6s}  {'Child':>15s} {'Father':>15s} {'Mother':>15s} {'MinErr':>7s} {'CReads':>6s}")
        for d in denovo_candidates[:20]:
            key = d["key"]
            c, f, m = d["child"], d["father"], d["mother"]
            locus = f"{key[0]}:{key[1]}-{key[2]}"
            motif = c["motif"][:6] + ("..." if len(c["motif"]) > 6 else "")
            lines.append(f"  {locus:35s} {motif:8s} {c['ref_size']:>6d}  "
                          f"({c['a1']:>6.0f},{c['a2']:>6.0f}) "
                          f"({f['a1']:>6.0f},{f['a2']:>6.0f}) "
                          f"({m['a1']:>6.0f},{m['a2']:>6.0f}) "
                          f"{d['min_err']:>7.0f} {c['n_reads']:>6d}")

    lines.append("")

    # Key takeaway
    lines.append("=" * 80)
    lines.append("  KEY FINDING")
    lines.append("=" * 80)
    r_het = results["child_het"]
    r_hom = results["child_hom"]
    t_het = r_het["total"]
    t_hom = r_hom["total"]
    lines.append(f"  Child HOM ({t_hom:,} loci): {r_hom['strict']/t_hom*100:.1f}% strict — most are trivially correct")
    lines.append(f"  Child HET ({t_het:,} loci): {r_het['strict']/t_het*100:.1f}% strict — this is where real validation happens")
    lines.append(f"  The overall 89.6% is inflated by {t_hom:,} easy HOM loci.")
    lines.append(f"  At HET loci, ±1 motif unit concordance = {r_het['w1unit']/t_het*100:.1f}%")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(OUTDIR, "trio_transmission_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
