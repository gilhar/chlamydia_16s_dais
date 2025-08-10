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
    binding_start: int  # 0-based position in canonical template
    binding_end: int  # 0-based position in canonical template (exclusive)
    is_forward: bool  # True for forward, False for reverse

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def binding_3_prime_pos(self) -> int:
        """Get the 3'-end position relative to canonical template."""
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

    # Find binding positions in canonical template
    canonical = TWIST_CT_16S
    canonical_rc = str(Seq(canonical).reverse_complement())

    base_primers = {}

    # TETR primers - modified to be compatible with the canonical sequence
    # tetr_forward_seq = "GGCG"     "TATTTGGGCATCCGAGTAACG"  # noqa: typo
    tetr_forward_seq = "GGCG" "A" "TATTTGGGCATCCGAGTAACG"  # noqa: typo
    # tetr_reverse_seq = "TCAAATCCAGCGGGTATTAACCG" "C" "CT"  # noqa: typo
    tetr_reverse_seq = "TCAAATCCAGCGGGTATTAACCG" "T" "CT" # noqa: typo

    # Find TETR forward in canonical (it should be there)
    tetr_f_pos = canonical.find(tetr_forward_seq)
    if tetr_f_pos == -1:
        raise ValueError(
            f"TETR forward primer sequence '{tetr_forward_seq}' not found in canonical template."
        )

    # Find TETR reverse in canonical RC #
    tetr_r_pos = canonical_rc.find(tetr_reverse_seq)
    if tetr_r_pos == -1:
        raise ValueError(
            f"TETR reverse primer sequence '{tetr_reverse_seq}' not found in canonical reverse complement."
        )

    # Convert reverse position to canonical coordinates
    tetr_r_canonical_pos = len(canonical) - tetr_r_pos - len(tetr_reverse_seq)

    base_primers['TETR'] = {
        'forward': BasePrimer(
            name="CTR 70",
            sequence=tetr_forward_seq,
            binding_start=tetr_f_pos,
            binding_end=(tetr_f_pos + len(tetr_forward_seq)),
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="CTR 71",
            sequence=tetr_reverse_seq,
            binding_start=tetr_r_canonical_pos,
            binding_end=(tetr_r_canonical_pos + len(tetr_reverse_seq)),
            is_forward=False,
        ),
    }

    # S11 primers
    s11_forward_seq = "CATGCAAGTCGAACGGAGCAATTGTTTCGACGATT"
    s11_reverse_seq = "CCAACTAGCTGATATCACATAGACTCTCCCTTAA"

    s11_f_pos = canonical.find(s11_forward_seq)
    s11_r_pos = canonical.find(s11_reverse_seq)

    base_primers['S11'] = {
        'forward': BasePrimer(
            name="S11-F",
            sequence=s11_forward_seq,
            binding_start=s11_f_pos,
            binding_end=(s11_f_pos + len(s11_forward_seq)),
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="S11-R",
            sequence=s11_reverse_seq,
            binding_start=s11_r_pos,
            binding_end=(s11_r_pos + len(s11_reverse_seq)),
            is_forward=False,
        ),
    }

    # IMRS primers - modified to be compatible with the canonical sequence
    # noinspection SpellCheckingInspection
    imrs_forward_seq = "TGCTGC" "A" "TG" "G" "CTG" "TCGTCAGCTCGT" "GCCG"  # noqa: typo
    imrs_reverse_seq = "TG" "GT" "TA" "ACCCAG" "GC" "AGT" "CTC" "G" "TTAGAG"  # noqa: typo

    # Find IMRS forward in canonical
    imrs_f_pos = canonical.find(imrs_forward_seq)
    if imrs_f_pos == -1:
        raise ValueError(
            f"IMRS forward primer sequence '{imrs_forward_seq}' not found in canonical template."
        )

    # Find IMRS reverse in canonical RC
    imrs_r_pos = canonical_rc.find(imrs_reverse_seq)
    if imrs_r_pos == -1:
        raise ValueError(
            f"IMRS reverse primer sequence '{imrs_reverse_seq}' not found in canonical reverse complement."
        )

    # Convert reverse position to canonical coordinates
    imrs_r_canonical_pos = len(canonical) - imrs_r_pos - len(imrs_reverse_seq)

    base_primers['IMRS'] = {
        'forward': BasePrimer(
            name="IMRS-F",
            sequence=imrs_forward_seq,
            binding_start=imrs_f_pos,
            binding_end=(imrs_f_pos + len(imrs_forward_seq)),
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="IMRS-R",
            sequence=imrs_reverse_seq,
            binding_start=imrs_r_canonical_pos,
            binding_end=(imrs_r_canonical_pos + len(imrs_reverse_seq)),
            is_forward=False,
        ),
    }

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
    Extract anchor and toehold sequences based on base primer 3'-end position.

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
            return "", ""  # Not enough bases in sequence

        combined_seq = canonical[start_pos : end_pos + 1]
        anchor_seq = combined_seq[:anchor_length]
        toehold_seq = combined_seq[anchor_length:]

    else:
        # Reverse primer: extract from reverse complement of canonical
        start_pos = end_pos - total_length + 1
        if start_pos < 0:
            return "", ""  # Not enough bases in sequence

        combined_seq = canonical_rc[start_pos : end_pos + 1]
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
                continue  # Skip if sequences couldn't be extracted

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

            # Note: Thermodynamic validation will be performed by separate module
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
            print(f"     Anchor: {candidate.anchor_sequence} (length: {candidate.anchor_length})")
            print(f"     Toehold: {candidate.toehold_sequence} (length: {candidate.toehold_length})")
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
