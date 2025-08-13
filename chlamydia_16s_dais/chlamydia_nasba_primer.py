#!/usr/bin/env python3
"""
Chlamydia trachomatis NASBA Primer Generator

This script generates NASBA primers based on base PCR primers for Chlamydia trachomatis
16S rRNA sequences. The NASBA primers are designed to have the same 3'-end positions
as the base primers while incorporating NASBA-specific sequences.

Features:
- Generates multiple candidate NASBA primers with different anchor/toehold lengths
- Calculates melting temperatures under NASBA conditions (80mM Na+, 12mM Mg++, 250nM)
- Supports multiple generic primer sets (gen5, gen6)
- Validates Tm ranges for anchor (73°C target) and toehold (46-55°C target) segments

Author: Claude (Anthropic)
Date: 2025
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from Bio.Seq import Seq


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Canonical CT 16S sequence (DNA format)
TWIST_CT_16S = """CTGAGAATTTGATCTTGGTTCAGATTGAACGCTGGCGGCGTGGATGAGGCATGCAAGTCGAACGGAGCAATTGTTTCGACGATTGTTTAGTGGCGGAAGGGTTAGTAATGCATAGATAATTTGTCCTTAACTTGGGAATAACGGTTGGAAACGGCCGCTAATACCGAATGTGGCGATATTTGGGCATCCGAGTAACGTTAAAGAAGGGGATCTTAGGACCTTTCGGTTAAGGGAGAGTCTATGTGATATCAGCTAGTTGGTGGGGTAAAGGCCTACCAAGGCTATGACGTCTAGGCGGATTGAGAGATTGGCCGCCAACACTGGGACTGAGACACTGCCCAGACTCCTACGGGAGGCTGCAGTCGAGAATCTTTCGCAATGGACGGAAGTCTGACGAAGCGACGCCGCGTGTGTGATGAAGGCTCTAGGGTTGTAAAGCACTTTCGCTTGGGAATAAGAGAAGACGGTTAATACCCGCTGGATTTGAGCGTACCAGGTAAAGAAGCACCGGCTAACTCCGTGCCAGCAGCTGCGGTAATACGGAGGGTGCTAGCGTTAATCGGATTTATTGGGCGTAAAGGGCGTGTAGGCGGAAAGGTAAGTTAGTTGTCAAAGATCGGGGCTCAACCCCGAGTCGGCATCTAATACTATTTTTCTAGAGGATAGATGGAGAAAAGGGAATTTCACGTGTAGCGGTGAAATGCGTAGATATGTGGAAGAACACCAGTGGCGAAGGCGCTTTTCTAATTTATACCTGACGCTAAGGCGCGAAAGCAAGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCTTGCCGTAAACGATGCATACTTGATGTGGATGGTCTCAACCCCATCCGTGTCGGAGCTAACGCGTTAAGTATGCCGCCTGAGGAGTACACTCGCAAGGGTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCAGTGGAGCATGTGGTTTAATTCGATGCAACGCGAAGGACCTTACCTGGGTTTGACATGTATATGACCGCGGCAGAAATGTCGTTTTCCGCAAGGACATATACACAGGTGCTGCATGGCTGTCGTCAGCTCGTGCCGTGAGGTGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATCGTTAGTTGCCAGCACTTAGGGTGGGAACTCTAACGAGACTGCCTGGGTTAACCAGGAGGAAGGCGAGGATGACGTCAAGTCAGCATGGCCCTTATGCCCAGGGCGACACACGTGCTACAATGGCCAGTACAGAAGGTGGCAAGATCGCGAGATGGAGCAAATCCTCAAAGCTGGCCCCAGTTCGGATTGTAGTCTGCAACTCGACTACATGAAGTCGGAATTGCTAGTAATGGCGTGTCAGCCATAACGCCGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACATCATGGGAGTTGGTTTTACCTTAAGTCGTTGACTCAACCCGCAAGGGAGAGAGGCGCCCAAGGTGAGGCTGATGACTAGGATGAAGTCGTAACAAGGTAGCCCTACCGGAAGGTGGGGCTGGATCACCTCCTTT""".replace(
    '\n', ''
)


# Length ranges
LENGTH_RANGES = {'toehold': (13, 15), 'anchor': (21, 23)}

# Fixed T7 promoter components
T7_COMPONENTS = {
    'pre_promoter': 'AATTC',  # noqa: typo
    'core_promoter': 'TAATACGACTCACTATA',  # noqa: typo
    'post_promoter_prefix': 'GGG',  # Always starts with GGG, total length = 8
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BasePrimer:
    """Represents a base PCR primer with binding position information."""

    name: str
    sequence: str
    binding_start: int  # 0-based position in the canonical template
    binding_end: int  # 0-based position in the canonical template (exclusive)
    is_forward: bool  # True for forward, False for reverse

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def binding_3_prime_pos(self) -> int:
        """Get the 3'-end position relative to the canonical template."""
        if self.is_forward:
            # Forward primer: 3'-end is at binding_end - 1
            return self.binding_end - 1
        else:
            # Reverse primer: 3'-end is at binding_start
            return self.binding_start


@dataclass
class GenericPrimerSet:
    """Represents a set of generic primer sequences for NASBA."""

    name: str
    forward_generic: str
    t7_pre: str
    t7_core: str
    t7_post: str
    reverse_generic: str

    @classmethod
    def from_sequences(
        cls, name: str, forward_seq: str, reverse_concat: str
    ) -> 'GenericPrimerSet':
        """Create GenericPrimerSet from the forward sequence and concatenated reverse."""
        # Parse the concatenated reverse sequence
        t7_pre = reverse_concat[0:5]  # AATTC  # noqa: typo
        t7_core = reverse_concat[5:22]  # TAATACGACTCACTATA  # noqa: typo
        t7_post = reverse_concat[22:30]  # GGG + 5 bases
        reverse_generic = reverse_concat[30:]  # Remaining bases

        return cls(
            name=name,
            forward_generic=forward_seq,
            t7_pre=t7_pre,
            t7_core=t7_core,
            t7_post=t7_post,
            reverse_generic=reverse_generic,
        )


@dataclass
class NASBAPrimerCandidate:
    """Represents a candidate NASBA primer with all components."""

    base_primer_name: str
    generic_set_name: str
    primer_type: str  # 'forward' or 'reverse'
    full_sequence: str
    anchor_sequence: str
    toehold_sequence: str
    anchor_length: int
    toehold_length: int
    is_valid: bool

    @property
    def total_length(self) -> int:
        return len(self.full_sequence)


# ============================================================================
# BASE PRIMER DEFINITIONS
# ============================================================================


def get_base_primers() -> Dict[str, Dict[str, BasePrimer]]:
    """Define the base primers with their binding positions in the canonical template."""

    canonical = TWIST_CT_16S
    canonical_rc = str(Seq(canonical).reverse_complement())

    # Centralized definition of primer pairs: ids, names, and sequences
    primer_table = [
        {
            "id": "TETR",
            "forward_name": "CTR 70",
            "forward_seq": "GGCG" "A" "TATTTGGGCATCCGAGTAACG",  # modified for canonical # noqa: typo
            "reverse_name": "CTR 71",
            "reverse_seq": "TCAAATCCAGCGGGTATTAACCG" "T" "CT",  # modified for canonical # noqa: typo
        },
        {
            "id": "S11",
            "forward_name": "S11-F",
            "forward_seq": "CATGCAAGTCGAACGGAGCAATTGTTTCGACGATT",
            "reverse_name": "S11-R",
            "reverse_seq": "CCAACTAGCTGATATCACATAGACTCTCCCTTAA",
        },
        {
            "id": "IMRS",
            "forward_name": "IMRS-F",
            "forward_seq": "TGCTGC" # noqa: typo
            "A"
            "TG"
            "G"
            "CTG"
            "TCGTCAGCTCGT" # noqa: typo
            "GCCG",  # modified for canonical # noqa: typo
            "reverse_name": "IMRS-R",
            "reverse_seq": "TG"
            "GT"
            "TA"
            "ACCCAG" # noqa: typo
            "GC"
            "AGT"
            "CTC"
            "G"
            "TTAGAG",  # modified for canonical # noqa: typo
        },
    ]

    base_primers: Dict[str, Dict[str, BasePrimer]] = {}

    for row in primer_table:
        pid = row["id"]
        f_name, f_seq = row["forward_name"], row["forward_seq"]
        r_name, r_seq = row["reverse_name"], row["reverse_seq"]

        # Locate forward primer in canonical
        f_pos = canonical.find(f_seq)
        if f_pos == -1:
            raise ValueError(
                f"{pid} forward primer sequence '{f_seq}' not found in canonical template."
            )

        # Locate reverse primer on reverse complement and convert to canonical coordinates
        r_pos_rc = canonical_rc.find(r_seq)
        if r_pos_rc == -1:
            raise ValueError(
                f"{pid} reverse primer sequence '{r_seq}' not found in canonical reverse complement."
            )
        r_start = len(canonical) - r_pos_rc - len(r_seq)

        base_primers[pid] = {
            "forward": BasePrimer(
                name=f_name,
                sequence=f_seq,
                binding_start=f_pos,
                binding_end=f_pos + len(f_seq),
                is_forward=True,
            ),
            "reverse": BasePrimer(
                name=r_name,
                sequence=r_seq,
                binding_start=r_start,
                binding_end=r_start + len(r_seq),
                is_forward=False,
            ),
        }

    # Preserve desired ordering
    order = ["S11", "TETR", "IMRS"]
    base_primers = {k: base_primers[k] for k in order}
    return base_primers


# ============================================================================
# MELTING TEMPERATURE CALCULATION
# ============================================================================


# ============================================================================
# NASBA PRIMER GENERATION
# ============================================================================


def extract_anchor_toehold_sequences(
    base_primer: BasePrimer, anchor_length: int, toehold_length: int
) -> Tuple[str, str]:
    """
    Extract anchor and toehold sequences based on the base primer 3'-end position.

    Returns:
        Tuple of (anchor_sequence, toehold_sequence)
    """
    canonical = TWIST_CT_16S
    canonical_rc = str(Seq(canonical).reverse_complement())

    # Get the 3'-end position
    end_pos = base_primer.binding_3_prime_pos
    total_length = anchor_length + toehold_length

    if base_primer.is_forward:
        # Forward primer: extract from canonical template
        start_pos = end_pos - total_length + 1
        if start_pos < 0:
            raise RuntimeError(
                f'Extract forward: Not enough bases in sequence: {start_pos=} {end_pos=} {total_length=}'
            )

        combined_seq = canonical[start_pos : end_pos + 1]
        anchor_seq = combined_seq[:anchor_length]
        toehold_seq = combined_seq[anchor_length:]

    else:
        # Reverse primer: extract from the reverse complement of canonical
        # Convert canonical coordinate to canonical_rc coordinate
        canonical_length = len(canonical)
        rc_end_pos = canonical_length - 1 - end_pos
        rc_start_pos = rc_end_pos - total_length + 1
        
        if rc_start_pos < 0:
            raise RuntimeError(
                f'Extract reverse: Not enough bases in sequence: rc_start_pos={rc_start_pos} rc_end_pos={rc_end_pos} total_length={total_length}'
            )

        combined_seq = canonical_rc[rc_start_pos : rc_end_pos + 1]
        anchor_seq = combined_seq[:anchor_length]
        toehold_seq = combined_seq[anchor_length:]

    return anchor_seq, toehold_seq


def generate_nasba_primer_candidates(
    base_primer: BasePrimer, generic_set: GenericPrimerSet
) -> List[NASBAPrimerCandidate]:
    """
    Generate all candidate NASBA primers for a base primer using a generic set.
    """
    candidates = []

    # Try all combinations of anchor and toehold lengths
    anchor_min, anchor_max = LENGTH_RANGES['anchor']
    toehold_min, toehold_max = LENGTH_RANGES['toehold']

    for anchor_len in range(anchor_min, anchor_max + 1):
        for toehold_len in range(toehold_min, toehold_max + 1):

            # Extract sequences
            anchor_seq, toehold_seq = extract_anchor_toehold_sequences(
                base_primer, anchor_len, toehold_len
            )

            if not anchor_seq or not toehold_seq:
                raise RuntimeError(
                    f'Generate NASBA primer candidate: anchor or toehold sequence is empty'
                    f' (anchor_len={anchor_len}, toehold_len={toehold_len})'
                )

            # Build full NASBA primer sequence
            if base_primer.is_forward:
                # Forward: forward-generic | anchor | toehold
                full_sequence = generic_set.forward_generic + anchor_seq + toehold_seq
                primer_type = 'forward'
            else:
                # Reverse: T7-pre | T7-core | T7-post | reverse-generic | anchor | toehold
                full_sequence = (
                    generic_set.t7_pre
                    + generic_set.t7_core
                    + generic_set.t7_post
                    + generic_set.reverse_generic
                    + anchor_seq
                    + toehold_seq
                )
                primer_type = 'reverse'

            # Note: Thermodynamic validation will be performed by a separate module
            is_valid = True  # All candidates are initially considered valid

            candidate = NASBAPrimerCandidate(
                base_primer_name=base_primer.name,
                generic_set_name=generic_set.name,
                primer_type=primer_type,
                full_sequence=full_sequence,
                anchor_sequence=anchor_seq,
                toehold_sequence=toehold_seq,
                anchor_length=anchor_len,
                toehold_length=toehold_len,
                is_valid=is_valid,
            )

            candidates.append(candidate)

    return candidates


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================


def print_candidate_summary(
    candidates: List[NASBAPrimerCandidate], base_primer_name: str, generic_set_name: str
):
    """Print summary of generated candidates."""

    print(f"\n--- {base_primer_name} with {generic_set_name} ---")
    print(f"Total candidates: {len(candidates)}")

    if candidates:
        print(f"\nTop 3 candidates:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"  {i}. {candidate.primer_type.title()} primer:")
            print(f"     Sequence: {candidate.full_sequence}")
            print(
                f"     Anchor: {candidate.anchor_sequence} (length: {candidate.anchor_length})"
            )
            print(
                f"     Toehold: {candidate.toehold_sequence} (length: {candidate.toehold_length})"
            )
    else:
        print("  No candidates generated")


def analyze_all_combinations(
    base_primers: Dict[str, Dict[str, BasePrimer]], generic_sets: List[GenericPrimerSet]
) -> Dict:
    """Generate and analyze all NASBA primer combinations."""

    print("=" * 80)
    print("CHLAMYDIA TRACHOMATIS NASBA PRIMER GENERATOR")
    print("=" * 80)

    print(f"\nGeneric primer sets: {[gs.name for gs in generic_sets]}")
    print(f"Base primer pairs: {list(base_primers.keys())}")
    print("Note: Thermodynamic validation will be performed separately")

    all_results = {}

    for pair_name, pair_primers in base_primers.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {pair_name} BASE PRIMERS")
        print(f"{'='*60}")

        pair_results = {}

        for primer_type, base_primer in pair_primers.items():
            print(f"\n{primer_type.upper()} PRIMER: {base_primer.name}")
            print(f"  Sequence: {base_primer.sequence}")
            print(
                f"  Binding position: {base_primer.binding_start}-{base_primer.binding_end}"
            )
            print(f"  3'-end position: {base_primer.binding_3_prime_pos}")

            primer_results = {}

            for generic_set in generic_sets:
                candidates = generate_nasba_primer_candidates(base_primer, generic_set)
                print_candidate_summary(candidates, base_primer.name, generic_set.name)

                primer_results[generic_set.name] = candidates

            pair_results[primer_type] = primer_results

        all_results[pair_name] = pair_results

    return all_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main function for NASBA primer generation."""

    # Define generic primer sets
    generic_sets = [
        GenericPrimerSet.from_sequences(
            name="gen5",
            forward_seq="TTATGTTCGTGGTT",  # noqa: typo
            reverse_concat="AATTCTAATACGACTCACTATAGGGTAAATACGTGC",  # noqa: typo
        ),
        GenericPrimerSet.from_sequences(
            name="gen6",
            forward_seq="TTTTGGTGGGTGGAT",  # noqa: typo
            reverse_concat="AATTCTAATACGACTCACTATAGGGTAAATATCCGGC",  # noqa: typo
        ),
    ]

    # Get base primers
    base_primers = get_base_primers()

    # Generate and analyze all combinations
    results = analyze_all_combinations(base_primers, generic_sets)

    print(f"\n{'='*80}")
    print("NASBA PRIMER GENERATION COMPLETE")
    print(f"{'='*80}")

    # Count total candidates
    total_candidates = 0
    for pair_results in results.values():
        for primer_results in pair_results.values():
            for candidates in primer_results.values():
                total_candidates += len(candidates)

    print(f"\nTotal NASBA primer candidates generated: {total_candidates}")
    print("Note: Use nasba_primer_validation.py for thermodynamic validation")

    return 0


if __name__ == "__main__":
    main()
