#!/usr/bin/env python3
"""Diagnostic script: compare raw CIGAR vs parasail re-aligned features per-read.

Picks 5 specific loci from the validation report:
  - 2 improved  (large allele_size change under re-alignment)
  - 2 unchanged TN (reference loci where re-alignment had no effect)
  - 1 unchanged TP (variant locus where re-alignment had no effect)

For each read at each locus, prints:
  - Raw CIGAR allele_size vs re-aligned allele_size
  - The actual parasail CIGAR string (to inspect =, X, M format)
  - Full 7-feature vectors side-by-side
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import parasail
import pysam

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deeptr.features import (
    _REALIGN_FLANK,
    _REF_OPS,
    _QUERY_OPS,
    _M, _I, _D, _N, _S, _H, _P, _EQ, _X,
    _extract_query_range,
    _get_repeat_gap_open,
    _parse_cigar_features,
    _GAP_EXTEND,
    extract_read_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("diagnose_realign")

# ---- Hard-coded diagnostic loci ----
# From the validation report on chr21.
# Format: (chrom, start, end, motif, category)
DIAGNOSTIC_LOCI = [
    # 2 IMPROVED loci (re-alignment helped significantly)
    ("chr21", 42993626, 42994486, "TCTTTT", "improved"),   # -1970.8 bp improvement
    ("chr21", 41672566, 41672928, "CACAAC", "improved"),   # -1063.0 bp improvement
    # 2 UNCHANGED TN loci (reference, no variant, re-alignment identical)
    # Pick two small TN loci — typical "unchanged" cases
    ("chr21", 14520065, 14520081, "AATGG",  "unchanged_TN"),  # 16bp ref, short VNTR
    ("chr21", 14694680, 14694698, "AC",      "unchanged_TN"),  # 18bp ref, dinucleotide
    # 1 UNCHANGED TP (variant locus where re-alignment didn't help)
    ("chr21", 34926461, 34926657, "ACAGCA", "unchanged_TP"),  # 196bp ref, -0.5bp tiny change
]


def _raw_cigar_features(cigartuples, ref_start, locus_start, locus_end):
    """Extract the 7 CIGAR features from pysam's cigartuples (same logic as extract_read_features)."""
    ref_pos = ref_start
    allele_size = 0
    n_ins = n_del = total_ins = total_del = max_ins = max_del = 0
    entered = False
    passed = False

    for op, length in cigartuples:
        if passed:
            break
        if op == _S or op == _H:
            continue
        ref_advance = length if op in _REF_OPS else 0
        query_advance = length if op in _QUERY_OPS else 0

        if ref_advance > 0:
            op_end = ref_pos + ref_advance
            if op_end > locus_start and ref_pos < locus_end:
                entered = True
                overlap = min(op_end, locus_end) - max(ref_pos, locus_start)
                if op in (_M, _EQ, _X):
                    allele_size += overlap
                elif op in (_D, _N):
                    n_del += 1
                    total_del += overlap
                    max_del = max(max_del, overlap)
            if op_end >= locus_end:
                passed = True
            ref_pos += ref_advance
        elif query_advance > 0:
            if entered and not passed:
                allele_size += query_advance
                if op == _I:
                    n_ins += 1
                    total_ins += query_advance
                    max_ins = max(max_ins, query_advance)

    return {
        "allele_size": allele_size,
        "n_ins": n_ins,
        "n_del": n_del,
        "total_ins": total_ins,
        "total_del": total_del,
        "max_ins": max_ins,
        "max_del": max_del,
    }


def _do_realignment(aln, chrom, locus_start, locus_end, ref_fasta, motif_len):
    """Perform parasail re-alignment and return (cigar_string, features_dict, debug_info)."""
    if aln.query_sequence is None or aln.cigartuples is None:
        return None, None, "no query_sequence or cigartuples"

    ref_target_start = max(0, locus_start - _REALIGN_FLANK)
    ref_target_end = locus_end + _REALIGN_FLANK

    q_start, q_end = _extract_query_range(
        aln.cigartuples, aln.reference_start,
        ref_target_start, ref_target_end,
    )
    if q_start is None:
        return None, None, "q_start is None (read doesn't cover target region)"

    query_seq = aln.query_sequence[q_start:q_end]
    if not query_seq:
        return None, None, "empty query_seq"

    try:
        ref_seq = ref_fasta.fetch(chrom, ref_target_start, ref_target_end)
    except (ValueError, KeyError) as e:
        return None, None, f"ref fetch error: {e}"
    if not ref_seq:
        return None, None, "empty ref_seq"

    gap_open = _get_repeat_gap_open(motif_len)

    result = parasail.sg_trace_striped_16(
        query_seq, ref_seq,
        open=gap_open, extend=_GAP_EXTEND,
        matrix=parasail.dnafull,
    )
    cigar_str = result.cigar.decode
    if isinstance(cigar_str, bytes):
        cigar_str = cigar_str.decode("ascii")

    features = _parse_cigar_features(cigar_str, ref_target_start, locus_start, locus_end)

    debug_info = {
        "query_range": (q_start, q_end),
        "query_len": len(query_seq),
        "ref_range": (ref_target_start, ref_target_end),
        "ref_len": len(ref_seq),
        "gap_open": gap_open,
        "gap_extend": _GAP_EXTEND,
        "parasail_score": result.score,
        "cigar_str": cigar_str,
        "cigar_len": len(cigar_str),
    }

    feat_dict = None
    if features is not None:
        feat_dict = {
            "allele_size": features.allele_size_bp,
            "n_ins": features.n_insertions,
            "n_del": features.n_deletions,
            "total_ins": features.total_ins_bp,
            "total_del": features.total_del_bp,
            "max_ins": features.max_single_ins,
            "max_del": features.max_single_del,
        }

    return cigar_str, feat_dict, debug_info


def analyze_parasail_cigar(cigar_str: str) -> dict:
    """Analyze what operations a parasail CIGAR string contains."""
    ops = re.findall(r"(\d+)([MIDNSHPX=])", cigar_str)
    op_counts = {}
    op_total_bp = {}
    for length_str, op_char in ops:
        length = int(length_str)
        op_counts[op_char] = op_counts.get(op_char, 0) + 1
        op_total_bp[op_char] = op_total_bp.get(op_char, 0) + length
    return {
        "n_operations": len(ops),
        "op_counts": op_counts,
        "op_total_bp": op_total_bp,
        "unique_ops": sorted(op_counts.keys()),
        "first_5_ops": [(int(l), o) for l, o in ops[:5]],
        "last_5_ops": [(int(l), o) for l, o in ops[-5:]],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose re-alignment behavior per-read")
    parser.add_argument("--bam", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--min-mapq", type=int, default=5)
    parser.add_argument("--min-flank", type=int, default=50)
    parser.add_argument("--max-reads", type=int, default=10, help="Max reads per locus for detailed output")
    args = parser.parse_args()

    bam = pysam.AlignmentFile(args.bam, "rb", reference_filename=args.ref)
    ref_fasta = pysam.FastaFile(args.ref)

    print("=" * 100)
    print("  DIAGNOSTIC: Raw CIGAR vs Parasail Re-alignment — Per-Read Comparison")
    print("=" * 100)

    # Track parasail CIGAR format statistics across all reads
    all_parasail_ops = set()

    for chrom, start, end, motif, category in DIAGNOSTIC_LOCI:
        ref_size = end - start
        motif_len = len(motif)

        print(f"\n{'=' * 100}")
        print(f"  LOCUS: {chrom}:{start}-{end}  motif={motif}  ref_size={ref_size}bp  category={category}")
        print(f"{'=' * 100}")

        n_processed = 0
        raw_sizes = []
        realign_sizes = []

        try:
            for aln in bam.fetch(chrom, max(0, start - args.min_flank), end + args.min_flank):
                if n_processed >= args.max_reads:
                    break
                if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
                    continue
                if aln.mapping_quality < args.min_mapq:
                    continue

                # Check spanning
                if aln.reference_start > start - args.min_flank:
                    continue
                if aln.reference_end is None or aln.reference_end < end + args.min_flank:
                    continue

                n_processed += 1

                # ---- Raw CIGAR features ----
                raw_feats = _raw_cigar_features(aln.cigartuples, aln.reference_start, start, end)

                # ---- Re-aligned features ----
                cigar_str, realign_feats, debug_info = _do_realignment(
                    aln, chrom, start, end, ref_fasta, motif_len,
                )

                print(f"\n  --- Read {n_processed}: {aln.query_name} ---")
                print(f"  Read pos: {aln.reference_start}-{aln.reference_end}  MAPQ={aln.mapping_quality}  Strand={'rev' if aln.is_reverse else 'fwd'}")

                # Print raw minimap2 CIGAR (abbreviated)
                raw_cigar_str = aln.cigarstring
                if len(raw_cigar_str) > 200:
                    raw_cigar_abbreviated = raw_cigar_str[:100] + " ... " + raw_cigar_str[-100:]
                else:
                    raw_cigar_abbreviated = raw_cigar_str
                print(f"  Raw minimap2 CIGAR (len={len(raw_cigar_str)}): {raw_cigar_abbreviated}")

                # Check what ops minimap2 uses
                raw_ops = set(re.findall(r"[MIDNSHPX=]", raw_cigar_str))
                print(f"  Minimap2 CIGAR ops used: {sorted(raw_ops)}")

                print(f"\n  Raw CIGAR features:")
                print(f"    allele_size={raw_feats['allele_size']:.1f}  n_ins={raw_feats['n_ins']}  n_del={raw_feats['n_del']}  "
                      f"total_ins={raw_feats['total_ins']}  total_del={raw_feats['total_del']}  "
                      f"max_ins={raw_feats['max_ins']}  max_del={raw_feats['max_del']}")

                if isinstance(debug_info, str):
                    print(f"\n  Re-alignment FAILED: {debug_info}")
                    continue

                print(f"\n  Re-alignment debug info:")
                print(f"    query_range in read: [{debug_info['query_range'][0]}, {debug_info['query_range'][1]})  ({debug_info['query_len']} bp)")
                print(f"    ref_range: [{debug_info['ref_range'][0]}, {debug_info['ref_range'][1]})  ({debug_info['ref_len']} bp)")
                print(f"    gap_open={debug_info['gap_open']}  gap_extend={debug_info['gap_extend']}  parasail_score={debug_info['parasail_score']}")

                # Parasail CIGAR analysis
                p_cigar = debug_info['cigar_str']
                if len(p_cigar) > 300:
                    p_cigar_display = p_cigar[:150] + " ... " + p_cigar[-150:]
                else:
                    p_cigar_display = p_cigar
                print(f"\n  Parasail CIGAR (len={debug_info['cigar_len']}): {p_cigar_display}")

                cigar_analysis = analyze_parasail_cigar(debug_info['cigar_str'])
                all_parasail_ops.update(cigar_analysis['unique_ops'])
                print(f"  Parasail CIGAR ops used: {cigar_analysis['unique_ops']}")
                print(f"  Parasail op counts: {cigar_analysis['op_counts']}")
                print(f"  Parasail op total_bp: {cigar_analysis['op_total_bp']}")
                print(f"  First 5 ops: {cigar_analysis['first_5_ops']}")
                print(f"  Last 5 ops: {cigar_analysis['last_5_ops']}")

                if realign_feats is not None:
                    print(f"\n  Re-aligned CIGAR features:")
                    print(f"    allele_size={realign_feats['allele_size']:.1f}  n_ins={realign_feats['n_ins']}  n_del={realign_feats['n_del']}  "
                          f"total_ins={realign_feats['total_ins']}  total_del={realign_feats['total_del']}  "
                          f"max_ins={realign_feats['max_ins']}  max_del={realign_feats['max_del']}")

                    # Side-by-side comparison
                    delta_allele = realign_feats['allele_size'] - raw_feats['allele_size']
                    delta_ins = realign_feats['total_ins'] - raw_feats['total_ins']
                    delta_del = realign_feats['total_del'] - raw_feats['total_del']
                    print(f"\n  DELTA (realign - raw):")
                    print(f"    allele_size: {delta_allele:+.1f}bp  total_ins: {delta_ins:+d}bp  total_del: {delta_del:+d}bp")
                    print(f"    n_ins: {realign_feats['n_ins'] - raw_feats['n_ins']:+d}  "
                          f"n_del: {realign_feats['n_del'] - raw_feats['n_del']:+d}")

                    is_identical = (
                        abs(delta_allele) < 0.01
                        and raw_feats['n_ins'] == realign_feats['n_ins']
                        and raw_feats['n_del'] == realign_feats['n_del']
                        and raw_feats['total_ins'] == realign_feats['total_ins']
                        and raw_feats['total_del'] == realign_feats['total_del']
                    )
                    print(f"    Features IDENTICAL: {is_identical}")

                    raw_sizes.append(raw_feats['allele_size'])
                    realign_sizes.append(realign_feats['allele_size'])
                else:
                    print(f"\n  Re-aligned CIGAR features: NONE (parse returned None)")
                    raw_sizes.append(raw_feats['allele_size'])

        except ValueError as e:
            print(f"  ERROR fetching reads: {e}")

        if raw_sizes:
            raw_arr = np.array(raw_sizes)
            print(f"\n  LOCUS SUMMARY ({n_processed} reads processed, {len(raw_sizes)} spanning):")
            print(f"    Raw allele_sizes:     mean={np.mean(raw_arr):.1f}  std={np.std(raw_arr):.1f}  min={np.min(raw_arr):.1f}  max={np.max(raw_arr):.1f}")
            if realign_sizes:
                realign_arr = np.array(realign_sizes)
                print(f"    Realign allele_sizes: mean={np.mean(realign_arr):.1f}  std={np.std(realign_arr):.1f}  min={np.min(realign_arr):.1f}  max={np.max(realign_arr):.1f}")
                n_changed = np.sum(np.abs(realign_arr - raw_arr[:len(realign_arr)]) > 0.01)
                print(f"    Reads with changed allele_size: {n_changed}/{len(realign_arr)} ({100*n_changed/len(realign_arr):.1f}%)")
        else:
            print(f"\n  No spanning reads found!")

    # Global summary
    print(f"\n\n{'=' * 100}")
    print(f"  GLOBAL PARASAIL CIGAR FORMAT SUMMARY")
    print(f"{'=' * 100}")
    print(f"  All parasail CIGAR operations observed: {sorted(all_parasail_ops)}")
    print(f"  Uses '=' (sequence match): {'=' in all_parasail_ops}")
    print(f"  Uses 'X' (mismatch):       {'X' in all_parasail_ops}")
    print(f"  Uses 'M' (match/mismatch): {'M' in all_parasail_ops}")
    print()
    print("  KEY FINDINGS:")
    print("  If parasail uses '=' and 'X' (extended CIGAR) instead of 'M', our parser handles both.")
    print("  If all allele_sizes are identical between raw and realigned for unchanged loci,")
    print("  then the fundamental issue is that minimap2 was already near-optimal for those loci.")
    print()

    bam.close()
    ref_fasta.close()


if __name__ == "__main__":
    main()
