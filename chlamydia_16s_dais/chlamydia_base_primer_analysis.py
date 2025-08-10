#!/usr/bin/env python3
"""
Chlamydia trachomatis 16S rRNA Primer Analysis Tool

This script analyzes primer binding sites for Chlamydia trachomatis 16S rRNA sequences
using Bio.Align.PairwiseAligner for accurate local sequence alignment.

Features:
- Direct primer-to-sequence alignment using Bio.Align.PairwiseAligner
- Sequence limiter and CPU parallelization support
- RNA/DNA auto-conversion
- Proper local alignment scoring
- Canonical sequence comparison with exact match detection
- Binding site histogram analysis

Author: Claude (Anthropic)
Date: 2025
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import click
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

global GLOBAL_DEBUG

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Canonical CT 16S sequence (DNA format - converted to RNA during analysis)
TWIST_CT_16S = (
    """
    CTGAGAATTTGATCTTGGTTCAGATTGAACGCTGGCGGCGTGGATGAGGCATGCAAGTCGAACGGAGCAATTGT
    TTCGACGATTGTTTAGTGGCGGAAGGGTTAGTAATGCATAGATAATTTGTCCTTAACTTGGGAATAACGGTTGG
    AAACGGCCGCTAATACCGAATGTGGCGATATTTGGGCATCCGAGTAACGTTAAAGAAGGGGATCTTAGGACCTT
    TCGGTTAAGGGAGAGTCTATGTGATATCAGCTAGTTGGTGGGGTAAAGGCCTACCAAGGCTATGACGTCTAGGC
    GGATTGAGAGATTGGCCGCCAACACTGGGACTGAGACACTGCCCAGACTCCTACGGGAGGCTGCAGTCGAGAAT
    CTTTCGCAATGGACGGAAGTCTGACGAAGCGACGCCGCGTGTGTGATGAAGGCTCTAGGGTTGTAAAGCACTTT
    CGCTTGGGAATAAGAGAAGACGGTTAATACCCGCTGGATTTGAGCGTACCAGGTAAAGAAGCACCGGCTAACTC
    CGTGCCAGCAGCTGCGGTAATACGGAGGGTGCTAGCGTTAATCGGATTTATTGGGCGTAAAGGGCGTGTAGGCG
    GAAAGGTAAGTTAGTTGTCAAAGATCGGGGCTCAACCCCGAGTCGGCATCTAATACTATTTTTCTAGAGGATAG
    ATGGAGAAAAGGGAATTTCACGTGTAGCGGTGAAATGCGTAGATATGTGGAAGAACACCAGTGGCGAAGGCGCT
    TTTCTAATTTATACCTGACGCTAAGGCGCGAAAGCAAGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCTTG
    CCGTAAACGATGCATACTTGATGTGGATGGTCTCAACCCCATCCGTGTCGGAGCTAACGCGTTAAGTATGCCGC
    CTGAGGAGTACACTCGCAAGGGTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCAGTGGAGCATGTGGTT
    TAATTCGATGCAACGCGAAGGACCTTACCTGGGTTTGACATGTATATGACCGCGGCAGAAATGTCGTTTTCCGC
    AAGGACATATACACAGGTGCTGCATGGCTGTCGTCAGCTCGTGCCGTGAGGTGTTGGGTTAAGTCCCGCAACGA
    GCGCAACCCTTATCGTTAGTTGCCAGCACTTAGGGTGGGAACTCTAACGAGACTGCCTGGGTTAACCAGGAGGA
    AGGCGAGGATGACGTCAAGTCAGCATGGCCCTTATGCCCAGGGCGACACACGTGCTACAATGGCCAGTACAGAA
    GGTGGCAAGATCGCGAGATGGAGCAAATCCTCAAAGCTGGCCCCAGTTCGGATTGTAGTCTGCAACTCGACTAC
    ATGAAGTCGGAATTGCTAGTAATGGCGTGTCAGCCATAACGCCGTGAATACGTTCCCGGGCCTTGTACACACCG
    CCCGTCACATCATGGGAGTTGGTTTTACCTTAAGTCGTTGACTCAACCCGCAAGGGAGAGAGGCGCCCAAGGTG
    AGGCTGATGACTAGGATGAAGTCGTAACAAGGTAGCCCTACCGGAAGGTGGGGCTGGATCACCTCCTTT
""".replace(
        '\n', ''
    )
    .replace(' ', '')
    .replace('\t', '')
)  # Remove whitespace for easier processing

# Alignment configurations
STANDARD_CONFIG = {
    'match_score': 2,
    'mismatch_score': -1,
    'open_gap_score': -2,
    'extend_gap_score': -0.5,
    'min_score_ratio': 0.6,
}

RELAXED_CONFIG = {
    'match_score': 2,
    'mismatch_score': -1,
    'open_gap_score': -2,
    'extend_gap_score': -0.5,
    'min_score_ratio': 0.4,
}


# ============================================================================
# DATA CLASSES
# ============================================================================


class Primer:
    """Represents a PCR primer with name and sequence."""

    def __init__(self, name: str, sequence: str):
        self.name = name
        self.sequence = sequence.upper()

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_rna(self) -> str:
        """Convert primer sequence to RNA format (T -> U)."""
        return self.sequence.replace('T', 'U')

    def __str__(self) -> str:
        return f"{self.name} ({self.length}bp): {self.sequence}"


class PrimerPair:
    """Represents a pair of forward and reverse primers with alignment configuration."""

    def __init__(
        self,
        forward_primer: Primer,
        reverse_primer: Primer,
        name: str,
        config: Dict = None,
    ):
        self.forward = forward_primer
        self.reverse = reverse_primer
        self.name = name
        self.config = config or STANDARD_CONFIG

    def __str__(self) -> str:
        return f"{self.name}: {self.forward.name} + {self.reverse.name}"


class BindingSite:
    """Represents a primer binding site with alignment information."""

    def __init__(
        self,
        position: int,
        end_position: int,
        score: float,
        binding_sequence: str,
        score_ratio: float,
    ):
        self.position = position
        self.end_position = end_position
        self.score = score
        self.binding_sequence = binding_sequence
        self.score_ratio = score_ratio
        self.total_errors = 0  # Can be calculated from score

    def __str__(self) -> str:
        return f"Site at {self.position}-{self.end_position} (score: {self.score:.1f}): {self.binding_sequence}"


# ============================================================================
# PRIMER DEFINITIONS
# ============================================================================


def get_primer_pairs() -> Dict[str, PrimerPair]:
    """Define and return all primer pairs with their configurations."""

    # TETR primer pair
    # tetr_forward = Primer("CTR 70", "GGCGTATTTGGGCATCCGAGTAACG") # noqa: typo
    # modified by a single insertion to match the "canonical" sequence
    tetr_forward = Primer("CTR 70", "GGCG" "A" "TATTTGGGCATCCGAGTAACG")  # noqa: typo
    # tetr_reverse = Primer("CTR 71", "TCAAATCCAGCGGGTATTAACCGCCT") # noqa: typo
    # modified by a single mutation to match the "canonical" sequence
    tetr_reverse = Primer("CTR 71", "TCAAATCCAGCGGGTATTAACCGTCT")  # noqa: typo

    # S11 primer pair
    s11_forward = Primer("S11-F", "CATGCAAGTCGAACGGAGCAATTGTTTCGACGATT")  # noqa: typo
    s11_reverse = Primer("S11-R", "CCAACTAGCTGATATCACATAGACTCTCCCTTAA")  # noqa: typo

    # IMRS primer pair (requires relaxed parameters)
    # imrs_forward = Primer("IMRS-F",              "TGCTGC"     "TG"     "CTG"      "ATTACGA" "GCCG" "A") # noqa: typo
    imrs_forward = Primer(
        "IMRS-F",
        "TGCTGC" "A" "TG" "G" "CTG" "TCGTCAGCTCGT" "GCCG",  # noqa: typo
    )  # noqa: typo
    # imrs_reverse = Primer("IMRS-R",              "TG"      "TA" "GGAGGA" "GC"       "CTC"     "TTAGAG" "AA") # noqa: typo
    imrs_reverse = Primer(
        "IMRS-R",
        "TG" "GT" "TA" "ACCCAG" "GC" "AGT" "CTC" "G" "TTAGAG",  # noqa: typo
    )  # noqa: typo

    return {
        'TETR': PrimerPair(tetr_forward, tetr_reverse, "TETR", STANDARD_CONFIG),
        'S11': PrimerPair(s11_forward, s11_reverse, "S11", STANDARD_CONFIG),
        'IMRS': PrimerPair(imrs_forward, imrs_reverse, "IMRS", RELAXED_CONFIG),
    }


# ============================================================================
# SEQUENCE ALIGNMENT FUNCTIONS
# ============================================================================


def align_primer_to_sequence(
    primer_seq: str, target_seq: str, config: Dict, debug: bool = False
) -> List[BindingSite]:
    """
    Align primer to sequence using Bio.Align.PairwiseAligner.

    Args:
        primer_seq: Primer sequence (DNA format)
        target_seq: Target sequence (RNA format expected)
        config: Alignment configuration dictionary
        debug: Enable debug output

    Returns:
        List of BindingSite objects sorted by score (descending)
    """
    primer_str = primer_seq.upper()
    target_str = target_seq.upper()

    # Convert T to U if target is RNA
    if 'U' in target_str and 'T' in primer_str:
        primer_str = primer_str.replace('T', 'U')
        if debug:
            print(f"    T→U conversion: {primer_seq} → {primer_str}")

    # Configure aligner
    aligner = PairwiseAligner()
    aligner.match_score = config['match_score']
    aligner.mismatch_score = config['mismatch_score']
    aligner.open_gap_score = config['open_gap_score']
    aligner.extend_gap_score = config['extend_gap_score']
    aligner.mode = 'local'

    # Perform alignment
    alignments = aligner.align(primer_str, target_str)
    perfect_score = len(primer_str) * config['match_score']
    min_score = perfect_score * config['min_score_ratio']

    if debug:
        print(
            f"    Found {len(alignments)} alignments, perfect: {perfect_score}, min: {min_score:.1f}"
        )

    binding_sites = []

    for i, alignment in enumerate(alignments):
        if alignment.score >= min_score:
            try:
                if len(alignment.aligned) >= 2 and len(alignment.aligned[1]) > 0:
                    target_offsets = alignment.aligned[1]
                    target_start = int(target_offsets[0][0])
                    target_end = int(target_offsets[-1][1])

                    # Extract binding site sequence
                    binding_sequence = target_str[target_start:target_end]

                    # Calculate total errors from score difference
                    total_errors = int(
                        (perfect_score - alignment.score)
                        / abs(config['mismatch_score'])
                    )

                    site = BindingSite(
                        position=target_start,
                        end_position=target_end,
                        score=alignment.score,
                        binding_sequence=binding_sequence,
                        score_ratio=alignment.score / perfect_score,
                    )
                    site.total_errors = total_errors

                    binding_sites.append(site)

                    if debug:
                        print(f"    ✅ Site {i+1}: {site}")

            except Exception as e:
                if debug:
                    print(f"    ❌ Exception: {e}")
                continue

    # Sort by score (descending) and limit results
    binding_sites.sort(key=lambda x: x.score, reverse=True)
    return binding_sites[:10]


def get_best_binding_sites(
    primer: Primer, sequences: List[SeqRecord], config: Dict
) -> List[str]:
    """
    Get the best binding site sequence for each input sequence.

    Args:
        primer: Primer object
        sequences: List of sequence records
        config: Alignment configuration

    Returns:
        List of binding site sequences (one per input sequence)
    """
    print(
        f"Analyzing {primer.name} binding sites (min_score_ratio={config['min_score_ratio']})..."
    )

    best_binding_sequences = []

    for seq_idx, record in enumerate(sequences):
        seq_str = str(record.seq)
        sites = align_primer_to_sequence(primer.sequence, seq_str, config)

        if sites:
            # Get sites with the best score and pick the first one
            best_score = max(site.score for site in sites)
            best_sites = [site for site in sites if site.score == best_score]
            best_site = best_sites[0]
            best_binding_sequences.append(best_site.binding_sequence)

    return best_binding_sequences


# ============================================================================
# CANONICAL SEQUENCE ANALYSIS
# ============================================================================


def get_canonical_binding_sites(
    primer: Primer, canonical_seq: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find primer binding sites in the canonical sequence.

    Args:
        primer: Primer object
        canonical_seq: Canonical sequence (DNA format)

    Returns:
        Tuple of (forward_site, reverse_site) or (None, None) if not found
    """
    primer_rna = primer.to_rna()
    canonical_rna = canonical_seq.replace('T', 'U')

    # Check forward strand binding
    forward_site = None
    if primer_rna in canonical_rna:
        pos = canonical_rna.find(primer_rna)
        forward_site = canonical_rna[pos : pos + len(primer_rna)]

    # Check reverse complement binding
    canonical_rc = str(Seq(canonical_seq).reverse_complement()).replace('T', 'U')
    reverse_site = None
    if primer_rna in canonical_rc:
        pos = canonical_rc.find(primer_rna)
        reverse_site = canonical_rc[pos : pos + len(primer_rna)]
    else:
        pass

    return forward_site, reverse_site


# ============================================================================
# ANALYSIS AND REPORTING FUNCTIONS
# ============================================================================


def create_binding_site_histogram(
    binding_sequences: List[str],
    primer_name: str,
    canonical_site: Optional[str] = None,
    top_n: int = 10,
) -> Counter:
    """
    Create and display histogram of binding site sequences.

    Args:
        binding_sequences: List of binding site sequences
        primer_name: Name for display
        canonical_site: Canonical binding site for comparison
        top_n: Number of top sequences to display

    Returns:
        Counter object with sequence frequencies
    """
    print(f"\n=== {primer_name} BINDING SITE HISTOGRAM ===")

    if not binding_sequences:
        print("No binding sites found")
        return Counter()

    sequence_counts = Counter(binding_sequences)

    print(f"Total binding sites found: {len(binding_sequences)}")
    print(f"Unique binding sequences: {len(sequence_counts)}")
    if canonical_site:
        print(f"Canonical sequence (twist_ct_16s): {canonical_site}")
    print()

    print(f"Top {top_n} most frequent binding site sequences:")
    for i, (sequence, count) in enumerate(sequence_counts.most_common(top_n), 1):
        frequency = count / len(binding_sequences) * 100

        # Check for canonical match (with T/U conversion)
        canonical_match = ""
        if canonical_site:
            sequence_rna = sequence.replace('T', 'U')
            canonical_rna = canonical_site.replace('T', 'U')
            if sequence_rna == canonical_rna:
                canonical_match = " (!)"

        print(
            f"  {i}. {sequence}{canonical_match} (count: {count}, frequency: {frequency:.1f}%)"
        )

    # Show frequency distribution
    print(f"\nFrequency distribution:")
    freq_dist = Counter(sequence_counts.values())
    for count, num_sequences in sorted(freq_dist.items(), reverse=True):
        print(f"  {num_sequences} sequences appear {count} time(s)")

    return sequence_counts


def analyze_primer_pair(
    pair: PrimerPair,
    sequences: List[SeqRecord],
    canonical_sites: Dict[str, Dict[str, Optional[str]]],
) -> Dict:
    """
    Perform complete analysis of a primer pair.

    Args:
        pair: PrimerPair object
        sequences: List of sequence records
        canonical_sites: Dictionary of canonical binding sites

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {pair.name} PRIMER PAIR")
    if pair.name == 'IMRS':
        print("(Using relaxed parameters: min_score_ratio=0.4)")
    print(f"{'='*60}")

    results = {}

    # Analyze forward primer
    forward_sites = get_best_binding_sites(pair.forward, sequences, pair.config)
    forward_canonical = canonical_sites[pair.name]['forward']
    forward_counts = create_binding_site_histogram(
        forward_sites, f"{pair.name} Forward ({pair.forward.name})", forward_canonical
    )
    results['forward'] = forward_counts

    # Analyze reverse primer (on reverse complement sequences)
    print(
        f"\nAnalyzing {pair.reverse.name} on reverse complement sequences (min_score_ratio={pair.config['min_score_ratio']})..."
    )
    reverse_sites = []

    for seq_idx, record in enumerate(sequences):
        rc_seq_str = str(record.seq.reverse_complement())
        sites = align_primer_to_sequence(pair.reverse.sequence, rc_seq_str, pair.config)

        if sites:
            best_score = max(site.score for site in sites)
            best_sites = [site for site in sites if site.score == best_score]
            best_site = best_sites[0]
            reverse_sites.append(best_site.binding_sequence)

    reverse_canonical = canonical_sites[pair.name]['reverse']
    reverse_counts = create_binding_site_histogram(
        reverse_sites, f"{pair.name} Reverse ({pair.reverse.name})", reverse_canonical
    )
    results['reverse'] = reverse_counts

    return results


def print_summary_report(all_results: Dict, canonical_sites: Dict):
    """Print final summary report with canonical matches."""

    print(f"\n{'='*60}")
    print("FINAL SUMMARY - CANONICAL MATCHES")
    print(f"{'='*60}")

    for pair_name, results in all_results.items():
        canonical_forward = canonical_sites[pair_name]['forward']
        canonical_reverse = canonical_sites[pair_name]['reverse']

        print(f"\n{pair_name}:")

        # Forward matches
        if canonical_forward and results['forward']:
            # Convert for comparison
            canonical_forward_rna = canonical_forward.replace('T', 'U')

            match_found = False
            for sequence, count in results['forward'].items():
                sequence_rna = sequence.replace('T', 'U')
                if sequence_rna == canonical_forward_rna:
                    total = sum(results['forward'].values())
                    percent = count / total * 100
                    print(
                        f"  Forward canonical match: {count}/{total} sequences ({percent:.1f}%)"
                    )
                    match_found = True
                    break

            if not match_found:
                print(f"  Forward canonical match: Not found in top sequences")
        else:
            print(f"  Forward canonical match: Not found in canonical sequence")

        # Reverse matches
        if canonical_reverse and results['reverse']:
            # Convert for comparison
            canonical_reverse_rna = canonical_reverse.replace('T', 'U')

            match_found = False
            for sequence, count in results['reverse'].items():
                sequence_rna = sequence.replace('T', 'U')
                if sequence_rna == canonical_reverse_rna:
                    total = sum(results['reverse'].values())
                    percent = count / total * 100
                    print(
                        f"  Reverse canonical match: {count}/{total} sequences ({percent:.1f}%)"
                    )
                    match_found = True
                    break

            if not match_found:
                print(f"  Reverse canonical match: Not found in top sequences")
        else:
            print(f"  Reverse canonical match: Not found in canonical sequence")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================


def run_primer_analysis(
    fasta_file: str,
    include_canonical: bool = True,
    max_sequences: Optional[int] = None,
) -> Dict:
    """
    Run complete primer binding site analysis.

    Args:
        fasta_file: Path to FASTA file with 16S sequences
        include_canonical: Whether to include canonical sequence in analysis
        max_sequences: Maximum number of sequences to analyze (None for all)

    Returns:
        Dictionary with complete analysis results
    """
    print("=" * 80)
    print("CHLAMYDIA TRACHOMATIS 16S rRNA PRIMER ANALYSIS")
    print("=" * 80)

    # Load sequences
    print(f"\nLoading sequences from: {fasta_file}")
    try:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if max_sequences:
            sequences = sequences[:max_sequences]
        print(f"Loaded {len(sequences)} sequences from FASTA")
    except FileNotFoundError:
        print(f"Error: Could not find file at {fasta_file}")
        return {}

    # Add canonical sequence if requested
    if include_canonical:
        canonical_rna_seq = TWIST_CT_16S
        canonical_record = SeqRecord(
            Seq(canonical_rna_seq),
            id="twist_ct_16s",
            description="Canonical CT 16S sequence (RNA)",
        )
        sequences.append(canonical_record)
        print(f"Added 1 canonical twist_ct_16s sequence: {len(TWIST_CT_16S)} bp")

    # convert sequences to RNA format
    for record in sequences:
        if 'T' in str(record.seq):
            record.seq = record.seq.replace('T', 'U')

    print(f"Total sequences to analyze: {len(sequences)}")

    # Get primer pairs
    primer_pairs = get_primer_pairs()
    print(f"\nAnalyzing {len(primer_pairs)} primer pairs:")
    for name, pair in primer_pairs.items():
        print(f"  {pair}")

    # Extract canonical binding sites
    print(f"\nExtracting canonical binding sites from twist_ct_16s...")
    canonical_sites = {}
    for pair_name, pair in primer_pairs.items():
        forward_canonical, _ = get_canonical_binding_sites(pair.forward, TWIST_CT_16S)
        _, reverse_canonical = get_canonical_binding_sites(pair.reverse, TWIST_CT_16S)

        canonical_sites[pair_name] = {
            'forward': forward_canonical,
            'reverse': reverse_canonical,
        }

        print(f"{pair_name}:")
        print(f"  Forward canonical: {forward_canonical}")
        print(f"  Reverse canonical: {reverse_canonical}")

    # Run analysis for each primer pair
    all_results = {}
    for pair_name, pair in primer_pairs.items():
        all_results[pair_name] = analyze_primer_pair(pair, sequences, canonical_sites)

    # Print summary report
    print_summary_report(all_results, canonical_sites)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    return {
        'results': all_results,
        'canonical_sites': canonical_sites,
        'sequences_analyzed': len(sequences),
        'primer_pairs': primer_pairs,
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('fasta_file', type=click.Path(exists=True))
@click.option(
    '--max-sequences',
    type=int,
    default=None,
    help='Maximum number of sequences to analyze (default: all)',
)
@click.option(
    '--top-n',
    type=int,
    default=10,
    help='Number of top binding sites to display (default: 10)',
)
@click.option(
    '--no-canonical', is_flag=True, help='Exclude canonical sequence from analysis'
)
@click.option('--debug', is_flag=True, help='Enable debug output')
def main(fasta_file, max_sequences, no_canonical, debug):
    """Analyze Chlamydia trachomatis 16S rRNA primer binding sites."""
    global GLOBAL_DEBUG
    GLOBAL_DEBUG = debug

    results = run_primer_analysis(
        fasta_file=fasta_file,
        include_canonical=not no_canonical,
        max_sequences=max_sequences,
    )

    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Sequences analyzed: {results['sequences_analyzed']}")
        print(f"Primer pairs: {len(results['primer_pairs'])}")
        exit(0)
    else:
        print(f"\nAnalysis failed!")
        exit(1)


if __name__ == "__main__":
    main()
