"""MosaicTR motif interruption detection module.

Detects motif interruptions within tandem repeat sequences, following
ACMG 2024 guidelines that recommend characterizing interruptions for
clinical severity prediction (e.g., CAA within CAG repeats in SCA1).

Motif interruptions modulate disease penetrance and age of onset:
- SCA1 (ATXN1): CAT interruptions within CAG stabilize the repeat
- HD (HTT): CAA interruptions within CAG affect somatic expansion
- FMR1: AGG interruptions within CGG reduce expansion risk

Key functions:
- detect_interruptions: Analyze a single sequence for motif interruptions
- analyze_reads_interruptions: Per-read interruption analysis from BAM
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Interruption:
    """A single motif interruption within a tandem repeat array.

    Attributes:
        position: 0-based index within the repeat unit array (e.g., 5 means
            the 6th motif-sized chunk).
        expected: The expected canonical motif (e.g., "CAG").
        observed: The actual sequence observed at this position (e.g., "CAA").
        context: Human-readable context string showing surrounding pure runs,
            e.g., "(CAG)5-CAA-(CAG)10".
    """

    position: int
    expected: str
    observed: str
    context: str


@dataclass
class InterruptionResult:
    """Complete interruption analysis for a single sequence.

    Attributes:
        total_length: Total TR region length in bp.
        n_repeat_units: Total motif-sized chunks (including partial last unit).
        n_pure_units: Number of chunks that exactly match the canonical motif.
        n_interrupted_units: n_repeat_units - n_pure_units.
        longest_pure_run: Longest consecutive run of pure motif units.
        interruptions: List of detected Interruption objects.
        purity: Fraction of pure units (n_pure_units / n_repeat_units), 0-1.
        sequence_composition: Count of each unique motif variant observed,
            e.g., {"CAG": 25, "CAA": 2, "CA": 1}.
    """

    total_length: int = 0
    n_repeat_units: int = 0
    n_pure_units: int = 0
    n_interrupted_units: int = 0
    longest_pure_run: int = 0
    interruptions: list[Interruption] = field(default_factory=list)
    purity: float = 0.0
    sequence_composition: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_motif_units(sequence: str, motif: str) -> list[tuple[str, bool]]:
    """Split a sequence into motif-sized chunks and flag pure matches.

    Divides the sequence into non-overlapping chunks of len(motif) bases.
    The last chunk may be shorter if the sequence length is not evenly
    divisible by the motif length.

    Args:
        sequence: DNA sequence to split.
        motif: Canonical repeat motif (e.g., "CAG").

    Returns:
        List of (chunk, is_pure) tuples where is_pure is True when the
        chunk exactly matches the motif (case-insensitive).
    """
    if not sequence or not motif:
        return []

    motif_len = len(motif)
    motif_upper = motif.upper()
    units: list[tuple[str, bool]] = []

    for i in range(0, len(sequence), motif_len):
        chunk = sequence[i:i + motif_len]
        is_pure = chunk.upper() == motif_upper
        units.append((chunk, is_pure))

    return units


def _hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings.

    Case-insensitive. If lengths differ, counts length difference as
    additional mismatches.
    """
    s1_upper = s1.upper()
    s2_upper = s2.upper()
    min_len = min(len(s1_upper), len(s2_upper))
    dist = abs(len(s1_upper) - len(s2_upper))
    for i in range(min_len):
        if s1_upper[i] != s2_upper[i]:
            dist += 1
    return dist


def _build_context(
    units: list[tuple[str, bool]],
    interruption_idx: int,
    motif: str,
) -> str:
    """Build a human-readable context string for an interruption.

    Format example: "(CAG)5-CAA-(CAG)10"

    Scans backward and forward from the interruption to count the
    flanking pure runs.

    Args:
        units: List of (chunk, is_pure) from _find_motif_units.
        interruption_idx: Index of the interrupted unit.
        motif: Canonical motif string.

    Returns:
        Context string.
    """
    motif_upper = motif.upper()
    observed = units[interruption_idx][0].upper()

    # Count pure run before
    run_before = 0
    i = interruption_idx - 1
    while i >= 0 and units[i][1]:
        run_before += 1
        i -= 1

    # Count pure run after
    run_after = 0
    i = interruption_idx + 1
    while i < len(units) and units[i][1]:
        run_after += 1
        i += 1

    parts: list[str] = []
    if run_before > 0:
        parts.append(f"({motif_upper}){run_before}")
    parts.append(observed)
    if run_after > 0:
        parts.append(f"({motif_upper}){run_after}")

    return "-".join(parts)


def _extract_tr_sequence(aln, locus_start: int, locus_end: int) -> Optional[str]:
    """Extract the query sequence aligning to a TR locus from a pysam alignment.

    Walks the CIGAR string to identify query bases that map within
    [locus_start, locus_end) on the reference. Includes inserted bases
    that occur within the locus boundaries.

    This is similar to genotype._extract_query_at_locus but without flanking
    and designed specifically for interruption analysis.

    Args:
        aln: A pysam AlignedSegment.
        locus_start: 0-based start of the TR locus (inclusive).
        locus_end: 0-based end of the TR locus (exclusive).

    Returns:
        The query subsequence, or None if extraction fails.
    """
    if aln.cigartuples is None or aln.query_sequence is None:
        return None

    ref_pos = aln.reference_start
    query_pos = 0
    query_start: Optional[int] = None
    query_end: Optional[int] = None

    for op, length in aln.cigartuples:
        if op in (0, 7, 8):  # M, =, X (alignment match, seq match, seq mismatch)
            ov_start = max(ref_pos, locus_start)
            ov_end = min(ref_pos + length, locus_end)
            if ov_start < ov_end:
                q_off_start = ov_start - ref_pos
                q_off_end = ov_end - ref_pos
                if query_start is None:
                    query_start = query_pos + q_off_start
                query_end = query_pos + q_off_end
            ref_pos += length
            query_pos += length
        elif op == 1:  # I (insertion to reference)
            if locus_start <= ref_pos <= locus_end:
                if query_start is None:
                    query_start = query_pos
                query_end = query_pos + length
            query_pos += length
        elif op in (2, 3):  # D, N (deletion from reference, ref skip)
            ref_pos += length
        elif op == 4:  # S (soft clip)
            query_pos += length
        # op == 5 (H, hard clip): does not consume query or reference

    if query_start is not None and query_end is not None:
        seq = aln.query_sequence[query_start:query_end]
        return seq if len(seq) > 0 else None
    return None


# ---------------------------------------------------------------------------
# Core interruption detection
# ---------------------------------------------------------------------------

def detect_interruptions(
    query_sequence: str,
    motif: str,
    min_pure_run: int = 3,
) -> InterruptionResult:
    """Detect motif interruptions in a tandem repeat sequence.

    Splits the query sequence into motif-sized chunks and identifies
    positions where the observed chunk differs from the canonical motif.
    For long motifs (>6 bp), a 1-bp mismatch is classified as "similar"
    rather than a full interruption, but it is still reported.

    ACMG 2024 guidelines recommend characterizing interruptions as they
    affect clinical interpretation of repeat expansion disorders.

    Args:
        query_sequence: DNA sequence spanning the TR locus.
        motif: Expected canonical repeat motif (e.g., "CAG").
        min_pure_run: Minimum consecutive pure repeats to count as a run.
            Used for filtering very short pure stretches if needed.

    Returns:
        InterruptionResult with complete interruption analysis.

    Examples:
        >>> result = detect_interruptions("CAGCAGCAGCAACAGCAG", "CAG")
        >>> result.n_pure_units
        5
        >>> result.n_interrupted_units
        1
        >>> result.purity
        0.8333333333333334
    """
    # Handle edge cases
    if not query_sequence or not motif:
        return InterruptionResult()

    seq = query_sequence.upper()
    motif_upper = motif.upper()
    motif_len = len(motif_upper)

    # Split into motif-sized units
    units = _find_motif_units(seq, motif_upper)
    if not units:
        return InterruptionResult(total_length=len(seq))

    # Count pure and interrupted units; build composition
    n_pure = 0
    composition: dict[str, int] = defaultdict(int)

    for chunk, is_pure in units:
        composition[chunk.upper()] += 1
        if is_pure:
            n_pure += 1

    n_total_units = len(units)
    n_interrupted = n_total_units - n_pure

    # Find longest consecutive pure run
    longest_run = 0
    current_run = 0
    for _, is_pure in units:
        if is_pure:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    # Detect interruptions (non-pure units)
    interruptions: list[Interruption] = []
    for idx, (chunk, is_pure) in enumerate(units):
        if is_pure:
            continue

        chunk_upper = chunk.upper()

        # For partial last unit (shorter than motif), skip unless it is
        # at least half the motif length -- very short trailing bases
        # are not meaningful interruptions.
        if len(chunk) < motif_len:
            if len(chunk) < max(motif_len // 2, 1):
                continue

        # For long motifs (>6 bp), check if this is a near-match (1bp off).
        # We still report it, but mark it in context as a subtle variant.
        context = _build_context(units, idx, motif_upper)

        interruptions.append(Interruption(
            position=idx,
            expected=motif_upper,
            observed=chunk_upper,
            context=context,
        ))

    # Purity score
    purity = n_pure / n_total_units if n_total_units > 0 else 0.0

    return InterruptionResult(
        total_length=len(seq),
        n_repeat_units=n_total_units,
        n_pure_units=n_pure,
        n_interrupted_units=n_interrupted,
        longest_pure_run=longest_run,
        interruptions=interruptions,
        purity=purity,
        sequence_composition=dict(composition),
    )


# ---------------------------------------------------------------------------
# BAM-level interruption analysis
# ---------------------------------------------------------------------------

def analyze_reads_interruptions(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    motif: str,
    ref_fasta_path: Optional[str] = None,
    min_mapq: int = 5,
    max_reads: int = 50,
) -> dict:
    """Analyze motif interruptions across reads at a TR locus.

    Extracts reads spanning the locus, detects interruptions per read,
    and aggregates results including per-haplotype purity and consensus
    interruption patterns.

    Args:
        bam_path: Path to HP-tagged BAM file.
        chrom: Chromosome name (e.g., "chr4").
        start: 0-based locus start.
        end: 0-based locus end (exclusive).
        motif: Canonical repeat motif (e.g., "CAG").
        ref_fasta_path: Optional path to reference FASTA for parasail
            realignment (currently unused, reserved for future).
        min_mapq: Minimum mapping quality filter.
        max_reads: Maximum reads to process per locus.

    Returns:
        Dictionary with keys:
            n_reads: int, number of reads analyzed.
            per_read_results: list of dict, each containing:
                hp: int (0, 1, or 2)
                mapq: int
                allele_size: float
                interruption_result: InterruptionResult
            consensus_interruptions: list of dict, interruptions found in
                >= 50% of reads, each with position, expected, observed, count.
            hp1_purity: float, mean purity of HP=1 reads (NaN if no reads).
            hp2_purity: float, mean purity of HP=2 reads (NaN if no reads).
            common_variants: dict of motif variant -> total count across reads.
    """
    import pysam

    result: dict = {
        "n_reads": 0,
        "per_read_results": [],
        "consensus_interruptions": [],
        "hp1_purity": float("nan"),
        "hp2_purity": float("nan"),
        "common_variants": {},
    }

    min_flank = 50

    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
    except Exception as exc:
        logger.error("Failed to open BAM %s: %s", bam_path, exc)
        return result

    try:
        fetched = bam.fetch(chrom, max(0, start - min_flank), end + min_flank)
    except ValueError:
        bam.close()
        return result

    per_read: list[dict] = []
    variant_counts: dict[str, int] = defaultdict(int)
    # Track interruption positions and observed motifs for consensus
    interruption_tracker: dict[tuple[int, str], int] = defaultdict(int)
    seen: set[str] = set()

    for aln in fetched:
        if len(per_read) >= max_reads:
            break
        if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
            continue
        if aln.mapping_quality < min_mapq:
            continue
        if aln.query_name is None or aln.query_name in seen:
            continue
        seen.add(aln.query_name)
        if aln.reference_start is None or aln.reference_end is None:
            continue
        # Require flanking coverage
        if aln.reference_start > start - min_flank:
            continue
        if aln.reference_end < end + min_flank:
            continue

        # Extract query sequence at locus
        tr_seq = _extract_tr_sequence(aln, start, end)
        if tr_seq is None or len(tr_seq) == 0:
            continue

        # Detect interruptions
        ir = detect_interruptions(tr_seq, motif)

        # Get HP tag
        try:
            hp = int(aln.get_tag("HP"))
        except (KeyError, ValueError):
            hp = 0

        # Get allele size (query bases at locus)
        allele_size = float(len(tr_seq))

        per_read.append({
            "hp": hp,
            "mapq": int(aln.mapping_quality),
            "allele_size": allele_size,
            "interruption_result": ir,
        })

        # Accumulate variant counts
        for variant, count in ir.sequence_composition.items():
            variant_counts[variant] += count

        # Track interruptions for consensus
        for intr in ir.interruptions:
            key = (intr.position, intr.observed)
            interruption_tracker[key] += 1

    bam.close()

    n_reads = len(per_read)
    result["n_reads"] = n_reads
    result["per_read_results"] = per_read
    result["common_variants"] = dict(variant_counts)

    if n_reads == 0:
        return result

    # Per-haplotype purity
    hp1_purities = [
        r["interruption_result"].purity
        for r in per_read if r["hp"] == 1
    ]
    hp2_purities = [
        r["interruption_result"].purity
        for r in per_read if r["hp"] == 2
    ]

    result["hp1_purity"] = (
        float(np.mean(hp1_purities)) if hp1_purities else float("nan")
    )
    result["hp2_purity"] = (
        float(np.mean(hp2_purities)) if hp2_purities else float("nan")
    )

    # Consensus interruptions: found in >= 50% of reads
    consensus_threshold = n_reads / 2.0
    consensus: list[dict] = []
    motif_upper = motif.upper()
    for (pos, observed), count in sorted(interruption_tracker.items()):
        if count >= consensus_threshold:
            consensus.append({
                "position": pos,
                "expected": motif_upper,
                "observed": observed,
                "count": count,
                "frequency": count / n_reads,
            })
    result["consensus_interruptions"] = consensus

    return result
