#!/usr/bin/env python3
"""Find loci with progressive somatic expansion/contraction across HG008 passages.

v3: Balanced filtering — excludes pure step-function jumps but allows
semi-gradual changes. Requires at least 2 consecutive steps in the same direction.
"""

import csv
from pathlib import Path

BASE = Path("/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/passage_drift_hp")
PASSAGES = ["normal", "p21", "p23", "p41"]
PASSAGE_LABELS = ["Normal", "P21", "P23", "P41"]
FILES = {p: BASE / f"instability_{p}.tsv" for p in PASSAGES}

MIN_TOTAL_DELTA = 10   # bp, for the changing allele
MAX_STABLE_DELTA = 10  # bp, for the stable allele


def load_tsv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["#chrom"], row["start"], row["end"])
            data[key] = row
    return data


def safe_float(v):
    if v in (".", "", "nan", "inf", "-inf"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def evaluate_progression(vals):
    """Evaluate if vals show progressive change.

    Returns (direction, score) or (None, 0).
    Score rewards: (1) graduality, (2) monotonicity, (3) magnitude.
    Rejects: pure single-step jumps where 1 step accounts for >90% of change.
    """
    total_delta = vals[-1] - vals[0]
    abs_total = abs(total_delta)
    if abs_total < MIN_TOTAL_DELTA:
        return None, 0

    steps = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    max_step_frac = max(abs(s) for s in steps) / abs_total if abs_total > 0 else 1

    # Reject if a single step accounts for >90% (pure step function)
    if max_step_frac > 0.90:
        return None, 0

    direction = "expansion" if total_delta > 0 else "contraction"

    # Count steps in the correct direction
    if direction == "expansion":
        concordant = sum(1 for s in steps if s > 0)
    else:
        concordant = sum(1 for s in steps if s < 0)

    # Need at least 2 of 3 steps concordant
    if concordant < 2:
        return None, 0

    # Monotonicity score: fraction of concordant steps
    mono_score = concordant / len(steps)

    # Graduality score: 1 - max_step_fraction (higher = more even)
    grad_score = 1.0 - max_step_frac

    # Combined score: geometric mean of monotonicity and graduality, weighted by magnitude
    import math
    score = math.sqrt(mono_score * (0.3 + grad_score)) * abs_total

    return direction, score


def main():
    datasets = {p: load_tsv(f) for p, f in FILES.items()}

    common_keys = set(datasets["normal"].keys())
    for p in PASSAGES[1:]:
        common_keys &= set(datasets[p].keys())
    print(f"Common loci across all 4 passages: {len(common_keys)}")

    candidates = []

    for key in common_keys:
        rows = {p: datasets[p][key] for p in PASSAGES}

        h1_vals, h2_vals = [], []
        hii_h1_vals, hii_h2_vals = [], []
        valid = True

        for p in PASSAGES:
            m1 = safe_float(rows[p]["median_h1"])
            m2 = safe_float(rows[p]["median_h2"])
            hi1 = safe_float(rows[p]["hii_h1"])
            hi2 = safe_float(rows[p]["hii_h2"])
            if m1 is None or m2 is None or m1 == 0 or m2 == 0:
                valid = False
                break
            h1_vals.append(m1)
            h2_vals.append(m2)
            hii_h1_vals.append(hi1 if hi1 is not None else 0)
            hii_h2_vals.append(hi2 if hi2 is not None else 0)

        if not valid:
            continue

        motif = rows["normal"]["motif"]
        ref_size = int(rows["normal"]["end"]) - int(rows["normal"]["start"])

        for label, changing, stable, hii_changing, hii_stable in [
            ("h1", h1_vals, h2_vals, hii_h1_vals, hii_h2_vals),
            ("h2", h2_vals, h1_vals, hii_h2_vals, hii_h1_vals),
        ]:
            stable_range = max(stable) - min(stable)
            if stable_range >= MAX_STABLE_DELTA:
                continue

            direction, score = evaluate_progression(changing)
            if direction is None:
                continue

            total_delta = changing[-1] - changing[0]
            steps = [changing[i+1] - changing[i] for i in range(len(changing)-1)]

            candidates.append({
                "chrom": key[0],
                "start": key[1],
                "end": key[2],
                "ref_size": ref_size,
                "motif": motif,
                "motif_len": len(motif),
                "allele": label,
                "direction": direction,
                "changing_vals": changing,
                "stable_vals": stable,
                "hii_changing": hii_changing,
                "hii_stable": hii_stable,
                "total_delta": total_delta,
                "abs_delta": abs(total_delta),
                "stable_range": stable_range,
                "score": score,
                "steps": steps,
                "analysis_paths": [rows[p]["analysis_path"] for p in PASSAGES],
                "n_reads": [(int(rows[p].get("n_h1", 0)), int(rows[p].get("n_h2", 0)))
                            for p in PASSAGES],
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nProgressive candidates: {len(candidates)}")
    print(f"  (no single step > 90%, at least 2/3 concordant steps,")
    print(f"   total delta >= {MIN_TOTAL_DELTA}bp, other allele range < {MAX_STABLE_DELTA}bp)\n")

    # Pick top 10 with motif diversity
    seen_motif_lens = set()
    diverse_picks = []
    remaining = []
    for c in candidates:
        if c["motif_len"] not in seen_motif_lens and len(diverse_picks) < 6:
            diverse_picks.append(c)
            seen_motif_lens.add(c["motif_len"])
        else:
            remaining.append(c)

    top = diverse_picks + remaining[:max(0, 10 - len(diverse_picks))]
    top.sort(key=lambda x: x["score"], reverse=True)
    top = top[:10]

    print(f"{'='*140}")
    print(f"Top {len(top)} progressive drift loci")
    print(f"{'='*140}")

    for i, c in enumerate(top, 1):
        motif_display = c["motif"] if len(c["motif"]) <= 20 else c["motif"][:17] + "..."
        print(f"\n--- #{i} ---")
        print(f"Locus: {c['chrom']}:{c['start']}-{c['end']}  (ref={c['ref_size']}bp)")
        print(f"Motif: {motif_display} ({c['motif_len']}bp)")
        print(f"Direction: {c['direction']} on {c['allele']}")
        print(f"Total delta: {c['total_delta']:+.1f} bp  |  Steps: {' -> '.join(f'{s:+.0f}' for s in c['steps'])}")
        print(f"Score: {c['score']:.1f}  |  Stable allele range: {c['stable_range']:.1f} bp")
        print(f"Analysis: {' / '.join(c['analysis_paths'])}")
        print()
        header = f"  {'Passage':<10} {'Changing':>10} {'Stable':>10} {'HII(chg)':>10} {'HII(stb)':>10} {'n_reads':>12}"
        print(header)
        print(f"  {'-'*len(header)}")
        for j, p in enumerate(PASSAGE_LABELS):
            n_str = f"{c['n_reads'][j][0]}+{c['n_reads'][j][1]}"
            print(f"  {p:<10} {c['changing_vals'][j]:>10.1f} {c['stable_vals'][j]:>10.1f} "
                  f"{c['hii_changing'][j]:>10.3f} {c['hii_stable'][j]:>10.3f} {n_str:>12}")

    # Summary
    if candidates:
        print(f"\n{'='*140}")
        print(f"All {len(candidates)} progressive candidates:")
        exp = sum(1 for c in candidates if c["direction"] == "expansion")
        con = sum(1 for c in candidates if c["direction"] == "contraction")
        print(f"  Expansions: {exp}, Contractions: {con}")

        # Top 5 by motif length
        motif_counts = {}
        for c in candidates:
            ml = c["motif_len"]
            motif_counts[ml] = motif_counts.get(ml, 0) + 1
        top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top motif lengths: {', '.join(f'{m}bp({n})' for m, n in top_motifs)}")

        deltas = [c["abs_delta"] for c in candidates]
        print(f"  Delta range: {min(deltas):.0f} - {max(deltas):.0f} bp (median {sorted(deltas)[len(deltas)//2]:.0f})")


if __name__ == "__main__":
    main()
