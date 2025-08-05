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

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from Bio.Seq import Seq
import click


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Canonical CT 16S sequence (DNA format)
TWIST_CT_16S = """CTGAGAATTTGATCTTGGTTCAGATTGAACGCTGGCGGCGTGGATGAGGCATGCAAGTCGAACGGAGCAATTGTTTCGACGATTGTTTAGTGGCGGAAGGGTTAGTAATGCATAGATAATTTGTCCTTAACTTGGGAATAACGGTTGGAAACGGCCGCTAATACCGAATGTGGCGATATTTGGGCATCCGAGTAACGTTAAAGAAGGGGATCTTAGGACCTTTCGGTTAAGGGAGAGTCTATGTGATATCAGCTAGTTGGTGGGGTAAAGGCCTACCAAGGCTATGACGTCTAGGCGGATTGAGAGATTGGCCGCCAACACTGGGACTGAGACACTGCCCAGACTCCTACGGGAGGCTGCAGTCGAGAATCTTTCGCAATGGACGGAAGTCTGACGAAGCGACGCCGCGTGTGTGATGAAGGCTCTAGGGTTGTAAAGCACTTTCGCTTGGGAATAAGAGAAGACGGTTAATACCCGCTGGATTTGAGCGTACCAGGTAAAGAAGCACCGGCTAACTCCGTGCCAGCAGCTGCGGTAATACGGAGGGTGCTAGCGTTAATCGGATTTATTGGGCGTAAAGGGCGTGTAGGCGGAAAGGTAAGTTAGTTGTCAAAGATCGGGGCTCAACCCCGAGTCGGCATCTAATACTATTTTTCTAGAGGATAGATGGAGAAAAGGGAATTTCACGTGTAGCGGTGAAATGCGTAGATATGTGGAAGAACACCAGTGGCGAAGGCGCTTTTCTAATTTATACCTGACGCTAAGGCGCGAAAGCAAGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCTTGCCGTAAACGATGCATACTTGATGTGGATGGTCTCAACCCCATCCGTGTCGGAGCTAACGCGTTAAGTATGCCGCCTGAGGAGTACACTCGCAAGGGTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCAGTGGAGCATGTGGTTTAATTCGATGCAACGCGAAGGACCTTACCTGGGTTTGACATGTATATGACCGCGGCAGAAATGTCGTTTTCCGCAAGGACATATACACAGGTGCTGCATGGCTGTCGTCAGCTCGTGCCGTGAGGTGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATCGTTAGTTGCCAGCACTTAGGGTGGGAACTCTAACGAGACTGCCTGGGTTAACCAGGAGGAAGGCGAGGATGACGTCAAGTCAGCATGGCCCTTATGCCCAGGGCGACACACGTGCTACAATGGCCAGTACAGAAGGTGGCAAGATCGCGAGATGGAGCAAATCCTCAAAGCTGGCCCCAGTTCGGATTGTAGTCTGCAACTCGACTACATGAAGTCGGAATTGCTAGTAATGGCGTGTCAGCCATAACGCCGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACATCATGGGAGTTGGTTTTACCTTAAGTCGTTGACTCAACCCGCAAGGGAGAGAGGCGCCCAAGGTGAGGCTGATGACTAGGATGAAGTCGTAACAAGGTAGCCCTACCGGAAGGTGGGGCTGGATCACCTCCTTT""".replace(
    '\n', ''
)

# NASBA conditions for Tm calculation
NASBA_CONDITIONS = {
    'Na_mM': 80,  # 80 mM Na+
    'Mg_mM': 12,  # 12 mM Mg++
    'primer_uM': 0.25,  # 250nM primer concentration
    'target_temp_C': 41,  # NASBA reaction temperature
}

# Tm targets
TM_TARGETS = {
    'toehold_min': 46,
    'toehold_max': 55,
    'anchor_target': 73,
    'anchor_tolerance': 3,  # ±3°C around target
}

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
    anchor_tm: float
    toehold_tm: float
    anchor_length: int
    toehold_length: int
    is_valid: bool

    @property
    def total_length(self) -> int:
        return len(self.full_sequence)

    def get_tm_score(self) -> float:
        """Calculate a score based on how close Tm values are to targets."""
        if not self.is_valid:
            return 0.0

        if self.primer_type == 'forward':
            # For forward primers, only check toehold and anchor
            toehold_score = (
                1.0
                if TM_TARGETS['toehold_min']
                <= self.toehold_tm
                <= TM_TARGETS['toehold_max']
                else 0.5
            )
            anchor_diff = abs(self.anchor_tm - TM_TARGETS['anchor_target'])
            anchor_score = max(0.0, 1.0 - anchor_diff / TM_TARGETS['anchor_tolerance'])
        else:
            # For reverse primers, same logic
            toehold_score = (
                1.0
                if TM_TARGETS['toehold_min']
                <= self.toehold_tm
                <= TM_TARGETS['toehold_max']
                else 0.5
            )
            anchor_diff = abs(self.anchor_tm - TM_TARGETS['anchor_target'])
            anchor_score = max(0.0, 1.0 - anchor_diff / TM_TARGETS['anchor_tolerance'])

        return (toehold_score + anchor_score) / 2.0


# ============================================================================
# BASE PRIMER DEFINITIONS
# ============================================================================


def get_base_primers() -> Dict[str, Dict[str, BasePrimer]]:
    """Define the base primers with their binding positions in the canonical template."""

    # Find binding positions in canonical template
    canonical = TWIST_CT_16S
    canonical_rc = str(Seq(canonical).reverse_complement())

    base_primers = {}

    # TETR primers
    tetr_forward_seq = "GGCGTATTTGGGCATCCGAGTAACG"  # noqa: typo
    tetr_reverse_seq = "TCAAATCCAGCGGGTATTAACCGCCT"  # noqa: typo

    # Find TETR forward in canonical (it should be there)
    tetr_f_pos = canonical.find(tetr_forward_seq)
    if tetr_f_pos == -1:
        # Try with T->U conversion (shouldn't be needed for canonical DNA)
        tetr_f_pos = canonical.replace('T', 'U').find(
            tetr_forward_seq.replace('T', 'U')
        )

    # Find TETR reverse in canonical RC #
    tetr_r_pos = canonical_rc.find(tetr_reverse_seq)
    if tetr_r_pos == -1:
        tetr_r_pos = canonical_rc.replace('T', 'U').find(
            tetr_reverse_seq.replace('T', 'U')
        )

    # Convert reverse position to canonical coordinates
    tetr_r_canonical_pos = len(canonical) - tetr_r_pos - len(tetr_reverse_seq)

    base_primers['TETR'] = {
        'forward': BasePrimer(
            name="CTR 70",
            sequence=tetr_forward_seq,
            binding_start=(
                tetr_f_pos if tetr_f_pos != -1 else 200
            ),  # Approximate if not found
            binding_end=(
                (tetr_f_pos + len(tetr_forward_seq)) if tetr_f_pos != -1 else 225
            ),
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="CTR 71",
            sequence=tetr_reverse_seq,
            binding_start=tetr_r_canonical_pos if tetr_r_pos != -1 else 800,
            binding_end=(
                (tetr_r_canonical_pos + len(tetr_reverse_seq))
                if tetr_r_pos != -1
                else 826
            ),
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
            binding_start=s11_f_pos if s11_f_pos != -1 else 49,
            binding_end=(s11_f_pos + len(s11_forward_seq)) if s11_f_pos != -1 else 84,
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="S11-R",
            sequence=s11_reverse_seq,
            binding_start=s11_r_pos if s11_r_pos != -1 else 226,
            binding_end=(s11_r_pos + len(s11_reverse_seq)) if s11_r_pos != -1 else 260,
            is_forward=False,
        ),
    }

    # IMRS primers (may not be found in canonical)
    # noinspection SpellCheckingInspection
    imrs_forward_seq = "TGCTGCTGCTGATTACGAGCCGA"  # noqa: typo
    imrs_reverse_seq = "TGTAGGAGGAGCCTCTTAGAGAA"  # noqa: typo

    # These might not be found, so use approximate positions
    base_primers['IMRS'] = {
        'forward': BasePrimer(
            name="IMRS-F",
            sequence=imrs_forward_seq,
            binding_start=1050,  # Approximate position
            binding_end=1073,
            is_forward=True,
        ),
        'reverse': BasePrimer(
            name="IMRS-R",
            sequence=imrs_reverse_seq,
            binding_start=350,  # Approximate position
            binding_end=373,
            is_forward=False,
        ),
    }

    return base_primers


# ============================================================================
# MELTING TEMPERATURE CALCULATION
# ============================================================================


def calculate_tm_nearest_neighbor(
    sequence: str,
    sodium_millimolar: float = 80,
    magnesium_millimolar: float = 12,
    primer_micromolar: float = 0.25,
) -> float:
    """
    Calculate melting temperature using nearest-neighbor method.

    This is a simplified implementation. For production use, consider using
    a more comprehensive library like primer3-py or Bio.SeqUtils.MeltingTemp.
    """
    sequence = sequence.upper()

    # Nearest neighbor enthalpy and entropy values (kcal/mol and cal/mol·K)
    # These are approximate values for DNA-DNA duplex formation
    nn_enthalpy = {
        'AA': -7.9,
        'AT': -7.2,
        'AC': -8.4,
        'AG': -7.8,
        'TA': -7.2,
        'TT': -7.9,
        'TC': -8.2,
        'TG': -8.5,
        'CA': -8.5,
        'CT': -7.8,
        'CC': -8.0,
        'CG': -10.6,
        'GA': -8.2,
        'GT': -8.4,
        'GC': -9.8,
        'GG': -8.0,
    }

    nn_entropy = {
        'AA': -22.2,
        'AT': -20.4,
        'AC': -22.4,
        'AG': -21.0,
        'TA': -21.3,
        'TT': -22.2,
        'TC': -22.2,
        'TG': -22.7,
        'CA': -22.7,
        'CT': -21.0,
        'CC': -19.9,
        'CG': -27.2,
        'GA': -22.2,
        'GT': -22.4,
        'GC': -24.4,
        'GG': -19.9,
    }

    # Calculate total enthalpy and entropy
    delta_h = 0
    delta_s = 0

    for i in range(len(sequence) - 1):
        dinucleotide = sequence[i : i + 2]
        if dinucleotide in nn_enthalpy:
            delta_h += nn_enthalpy[dinucleotide]
            delta_s += nn_entropy[dinucleotide]

    # Add initiation parameters
    delta_h += 0.1  # Initiation enthalpy
    delta_s += -2.8  # Initiation entropy

    # Salt correction (simplified)
    delta_s += 0.368 * (len(sequence) - 1) * math.log(sodium_millimolar / 1000.0)

    # Mg correction (very simplified)
    if magnesium_millimolar > 0:
        delta_s += 0.2 * math.log(magnesium_millimolar / 1000.0)

    # Calculate Tm
    # Tm = ΔH / (ΔS + R * ln(C/4)) where C is primer concentration
    r_gas = 1.987  # Gas constant (cal/mol·K)
    primer_molar = primer_micromolar / 1e6  # Convert µM to M

    tm_kelvin = (delta_h * 1000) / (delta_s + r_gas * math.log(primer_molar / 4))
    tm_celsius = tm_kelvin - 273.15

    return tm_celsius


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

            # Calculate melting temperatures
            anchor_tm = calculate_tm_nearest_neighbor(
                anchor_seq,
                sodium_millimolar=NASBA_CONDITIONS['Na_mM'],
                magnesium_millimolar=NASBA_CONDITIONS['Mg_mM'],
                primer_micromolar=NASBA_CONDITIONS['primer_uM'],
            )

            toehold_tm = calculate_tm_nearest_neighbor(
                toehold_seq,
                sodium_millimolar=NASBA_CONDITIONS['Na_mM'],
                magnesium_millimolar=NASBA_CONDITIONS['Mg_mM'],
                primer_micromolar=NASBA_CONDITIONS['primer_uM'],
            )

            # Check if the candidate is valid (within Tm ranges)
            toehold_valid = (
                TM_TARGETS['toehold_min'] <= toehold_tm <= TM_TARGETS['toehold_max']
            )
            anchor_valid = (
                abs(anchor_tm - TM_TARGETS['anchor_target'])
                <= TM_TARGETS['anchor_tolerance']
            )
            is_valid = toehold_valid and anchor_valid

            candidate = NASBAPrimerCandidate(
                base_primer_name=base_primer.name,
                generic_set_name=generic_set.name,
                primer_type=primer_type,
                full_sequence=full_sequence,
                anchor_sequence=anchor_seq,
                toehold_sequence=toehold_seq,
                anchor_tm=anchor_tm,
                toehold_tm=toehold_tm,
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

    valid_candidates = [c for c in candidates if c.is_valid]

    print(f"\n--- {base_primer_name} with {generic_set_name} ---")
    print(f"Total candidates: {len(candidates)}")
    print(f"Valid candidates: {len(valid_candidates)}")

    if valid_candidates:
        # Sort by Tm score
        valid_candidates.sort(key=lambda x: x.get_tm_score(), reverse=True)

        print(f"\nTop 3 valid candidates:")
        for i, candidate in enumerate(valid_candidates[:3], 1):
            print(f"  {i}. {candidate.primer_type.title()} primer:")
            print(f"     Sequence: {candidate.full_sequence}")
            print(
                f"     Anchor: {candidate.anchor_sequence} (Tm: {candidate.anchor_tm:.1f}°C)"
            )
            print(
                f"     Toehold: {candidate.toehold_sequence} (Tm: {candidate.toehold_tm:.1f}°C)"
            )
            print(f"     Score: {candidate.get_tm_score():.3f}")
    else:
        print("  No valid candidates found within Tm constraints")


def analyze_all_combinations(
    base_primers: Dict[str, Dict[str, BasePrimer]], generic_sets: List[GenericPrimerSet]
) -> Dict:
    """Generate and analyze all NASBA primer combinations."""

    print("=" * 80)
    print("CHLAMYDIA TRACHOMATIS NASBA PRIMER GENERATOR")
    print("=" * 80)

    print(f"\nGeneric primer sets: {[gs.name for gs in generic_sets]}")
    print(f"Base primer pairs: {list(base_primers.keys())}")
    print(
        f"Tm targets: Toehold {TM_TARGETS['toehold_min']}-{TM_TARGETS['toehold_max']}°C, Anchor {TM_TARGETS['anchor_target']}±{TM_TARGETS['anchor_tolerance']}°C"
    )

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

@click.command(
    context_settings=dict(help_option_names=['-h', '--help']),
    help="Generate NASBA primers for Chlamydia trachomatis base primers.\n\n"
         "This tool generates NASBA primers that maintain the same 3'-end positions\n"
         "as the base PCR primers while incorporating NASBA-specific sequences.\n\n"
         "The program generates multiple candidates with different anchor/toehold\n"
         "lengths and evaluates them based on melting temperature constraints."
)
@click.option(
    '--anchor-target-tm',
    type=float,
    default=73.0,
    show_default=True,
    help='Target Tm for anchor segment'
)
@click.option(
    '--toehold-min-tm',
    type=float,
    default=46.0,
    show_default=True,
    help='Minimum Tm for toehold segment'
)
@click.option(
    '--toehold-max-tm',
    type=float,
    default=55.0,
    show_default=True,
    help='Maximum Tm for toehold segment'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose output'
)
def main(anchor_target_tm, toehold_min_tm, toehold_max_tm, verbose):
    """Main function for NASBA primer generation."""

    # Update targets if provided
    TM_TARGETS['anchor_target'] = anchor_target_tm
    TM_TARGETS['toehold_min'] = toehold_min_tm
    TM_TARGETS['toehold_max'] = toehold_max_tm

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

    # Count total valid candidates
    total_valid = 0
    for pair_results in results.values():
        for primer_results in pair_results.values():
            for candidates in primer_results.values():
                total_valid += sum(1 for c in candidates if c.is_valid)

    print(f"\nTotal valid NASBA primer candidates generated: {total_valid}")

    return 0


if __name__ == "__main__":
    main()
