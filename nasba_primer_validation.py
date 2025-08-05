#!/usr/bin/env python3
"""
NASBA Primer Validation Framework

This module implements comprehensive testing for NASBA primer validation, including
1. Hetero-dimer fraction measurements
2. Four-primer cross-reactivity analysis
3. Individual primer-dais binding specificity tests

Based on specifications for validating NASBA primers against daises (reverse complements
of anchor portions) to ensure proper binding specificity and prevent cross-reactivity.

Author: Claude (Anthropic)
Date: 2025
"""

import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

import click
from Bio.Seq import Seq
from Bio import SeqIO

# Import from the main NASBA primer module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chlamydia_nasba_primer import (
    get_base_primers,
    GenericPrimerSet,
    generate_nasba_primer_candidates,
)
from nupack_complex_analysis import (
    SequenceInput,
    analyze_sequence_complexes,
)


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================


# Load canonical sequences
def load_canonical_sequences() -> Dict[str, str]:
    """Load canonical sequences from FASTA files."""
    sequences = {}

    # Load C.T canonical
    ct_fasta_path = os.path.join(
        os.path.dirname(__file__), "twist_ct_16s_canonical.fasta"
    )
    if os.path.exists(ct_fasta_path):
        with open(ct_fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences['CT'] = str(record.seq)
                break

    # Load N.G canonical
    ng_fasta_path = os.path.join(
        os.path.dirname(__file__), "twist_ng_16s_canonical.fasta"
    )
    if os.path.exists(ng_fasta_path):
        with open(ng_fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences['NG'] = str(record.seq)
                break

    return sequences


CANONICAL_SEQUENCES = load_canonical_sequences()

# Test thresholds
VALIDATION_THRESHOLDS = {
    'correct_dais_primer_dimer_min': 0.90,  # >90% hetero-dimer formation expected
    'primer_monomer_vs_other_primers': 0.90,  # >90% monomer in presence of other primers
    'primer_monomer_vs_wrong_daises': 0.90,  # >90% monomer with incorrect daises
    'primer_unpaired_3_prime_min': 0.998,  # >99.8% probability of 3'-end unpaired
    'primer_signal_binding_min': 0.80,  # >80% primer-signal binding expected
    'primer_3_prime_binding_min': 0.90,  # >90% 3'-end binding in primer-signal dimer
    'generic_primer_amplicon_binding_min': 0.80,  # >80% generic primer-amplicon binding expected
    'generic_primer_3_prime_binding_min': 0.90,  # >90% 3'-end binding in generic primer-amplicon dimer
    'generic_primer_low_cross_binding_max': 0.20,  # <20% binding to unintended signals
    'generic_primer_cross_monomer_min': 0.90,  # >90% monomer with unintended signals
    'generic_primer_cross_unpaired_min': 0.998,  # >99.8% 3'-end unpaired with unintended signals
    'primer_set_low_interaction_max': 0.20,  # <20% interaction between primers in same set
    'primer_set_monomer_min': 0.90,  # >90% monomer for all primers in mixed set
    'primer_set_generic_unpaired_min': 0.998,  # >99.8% 3'-end unpaired for generic primers in set
}

# NASBA primer testing conditions
TESTING_CONDITIONS = {
    'temperature_C': 41,  # NASBA reaction temperature
    'primer_concentration_nM': 250,  # 250nM
    'dais_concentration_nM': 250,  # 250nM
    'signal_concentration_pM': 10,  # 10pM for signal binding tests
    'generic_primer_concentration_nM': 25,  # 25nM for generic primer tests
    'generic_signal_concentration_nM': 1,  # 1nM for generic primer signal tests
}


# ============================================================================
# DATA CLASSES FOR VALIDATION
# ============================================================================


@dataclass
class ValidationPrimer:
    """Represents a NASBA primer for validation testing."""

    name: str
    sequence: str
    anchor_sequence: str
    toehold_sequence: str
    species: str  # 'CT' or 'NG'
    primer_type: str  # 'forward' or 'reverse'
    generic_set: str  # 'gen5' or 'gen6'


@dataclass
class Dais:
    """Represents a dais (reverse complement of anchor)."""

    name: str
    sequence: str
    species: str  # 'CT' or 'NG'
    primer_type: str  # 'forward' or 'reverse'
    generic_set: str  # 'gen5' or 'gen6'


@dataclass
class ValidationResult:
    """Results from primer validation tests."""

    test_name: str
    primer_name: str
    target_dais: List[str]
    hetero_dimer_fraction: Optional[float] = None
    monomer_fraction: Optional[float] = None
    unpaired_3_prime_prob: Optional[float] = None
    passed: bool = False
    details: str = ""


# ============================================================================
# N.G PRIMER GENERATION
# ============================================================================


def generate_ng_primers(generic_sets: List[GenericPrimerSet]) -> List[ValidationPrimer]:
    """
    Generate N.G primers with explicitly defined anchors and toeholds.
    
    Verifies that d4 daises are reverse complements of anchors and raises
    exceptions if any expected conditions are not met.
    """
    ng_primers = []

    if 'NG' not in CANONICAL_SEQUENCES:
        raise ValueError("N.G canonical sequence not found in loaded sequences")

    ng_canonical = CANONICAL_SEQUENCES['NG']
    
    # Explicitly define N.G primer components
    # These are the actual sequences used in the frozen N.G primers
    ng_components = {
        'forward': {
            'anchor': "CGGGCTCAACCCGGGAACTGC",  # 21 bp anchor
            'toehold': "AAGCCTGATCCAGC"  # 14 bp toehold
        },
        'reverse': {
            'anchor': "TTGCGACCGTACTCCCCAGGC",  # 20 bp anchor  
            'toehold': "CATGCCGCGTGTCT"  # 14 bp toehold
        }
    }
    
    # Expected d4 dais sequences (should be reverse complements of anchors)
    expected_d4_daises = {
        'forward': "GCAGTTCCCGGGTTGAGCCCG",   # Forward dais (21 bp)
        'reverse': "GCCTGGGGAGTACGGTCGCAA"    # Reverse dais (20 bp)
    }
    
    # Verify that daises are reverse complements of anchors
    for primer_type in ['forward', 'reverse']:
        anchor = ng_components[primer_type]['anchor']
        expected_dais = expected_d4_daises[primer_type]
        calculated_dais = str(Seq(anchor).reverse_complement())
        
        if calculated_dais != expected_dais:
            raise ValueError(
                f"N.G {primer_type} dais mismatch: expected {expected_dais}, "
                f"but anchor RC gives {calculated_dais}"
            )
    
    print(f"✓ Verified N.G anchors match expected d4 daises")
    print(f"N.G Forward anchor: {ng_components['forward']['anchor']}")
    print(f"N.G Forward toehold: {ng_components['forward']['toehold']}")
    print(f"N.G Reverse anchor: {ng_components['reverse']['anchor']}")  
    print(f"N.G Reverse toehold: {ng_components['reverse']['toehold']}")
    
    # Verify anchors and toeholds are found in canonical sequence
    for primer_type in ['forward', 'reverse']:
        anchor = ng_components[primer_type]['anchor']
        toehold = ng_components[primer_type]['toehold']
        
        if primer_type == 'forward':
            # Forward anchor should be found directly in canonical
            anchor_pos = ng_canonical.find(anchor)
            if anchor_pos == -1:
                raise ValueError(f"N.G forward anchor {anchor} not found in canonical sequence")
            
            # Forward toehold should be 3' to anchor
            expected_toehold_pos = anchor_pos + len(anchor)
            if expected_toehold_pos + len(toehold) > len(ng_canonical):
                raise ValueError(f"N.G forward toehold extends beyond canonical sequence")
            
            canonical_toehold = ng_canonical[expected_toehold_pos:expected_toehold_pos + len(toehold)]
            if canonical_toehold != toehold:
                raise ValueError(
                    f"N.G forward toehold mismatch: expected {toehold}, "
                    f"found {canonical_toehold} at position {expected_toehold_pos}"
                )
        
        else:  # reverse
            # Reverse anchor RC should be found in canonical
            anchor_rc = str(Seq(anchor).reverse_complement())
            anchor_rc_pos = ng_canonical.find(anchor_rc)
            if anchor_rc_pos == -1:
                raise ValueError(f"N.G reverse anchor RC {anchor_rc} not found in canonical sequence")
            
            # Reverse toehold should be 5' to anchor RC position (reverse binds at first base)
            expected_toehold_pos = anchor_rc_pos - len(toehold)
            if expected_toehold_pos < 0:
                raise ValueError(f"N.G reverse toehold extends before start of canonical sequence")
            
            canonical_toehold = ng_canonical[expected_toehold_pos:anchor_rc_pos]
            if canonical_toehold != toehold:
                raise ValueError(
                    f"N.G reverse toehold mismatch: expected {toehold}, "
                    f"found {canonical_toehold} at position {expected_toehold_pos}"
                )
    
    print(f"✓ Verified all N.G components found in canonical sequence")
    
    # Generate NASBA primers for each generic set
    for generic_set in generic_sets:
        # Forward primer
        forward_anchor = ng_components['forward']['anchor']
        forward_toehold = ng_components['forward']['toehold']
        forward_sequence = generic_set.forward_generic + forward_anchor + forward_toehold

        ng_primers.append(
            ValidationPrimer(
                name=f"NG-F-{generic_set.name}",
                sequence=forward_sequence,
                anchor_sequence=forward_anchor,
                toehold_sequence=forward_toehold,
                species="NG",
                primer_type="forward",
                generic_set=generic_set.name,
            )
        )

        # Reverse primer
        reverse_anchor = ng_components['reverse']['anchor']
        reverse_toehold = ng_components['reverse']['toehold']
        reverse_sequence = (
            generic_set.t7_pre
            + generic_set.t7_core
            + generic_set.t7_post
            + generic_set.reverse_generic
            + reverse_anchor
            + reverse_toehold
        )

        ng_primers.append(
            ValidationPrimer(
                name=f"NG-R-{generic_set.name}",
                sequence=reverse_sequence,
                anchor_sequence=reverse_anchor,
                toehold_sequence=reverse_toehold,
                species="NG",
                primer_type="reverse",
                generic_set=generic_set.name,
            )
        )

    return ng_primers


# ============================================================================
# DAIS GENERATION
# ============================================================================


def generate_daises(primers: List[ValidationPrimer]) -> List[Dais]:
    """Generate daises (reverse complements of anchor portions) for all primers."""
    daises = []

    for primer in primers:
        # Dais is the reverse complement of the anchor sequence
        dais_sequence = str(Seq(primer.anchor_sequence).reverse_complement())

        dais = Dais(
            name=f"dais-{primer.name}",
            sequence=dais_sequence,
            species=primer.species,
            primer_type=primer.primer_type,
            generic_set=primer.generic_set,
        )

        daises.append(dais)

    return daises


# ============================================================================
# THERMODYNAMIC CALCULATIONS
# ============================================================================


def calculate_dimer_formation_probability(
    seq1: str, seq2: str, temperature_celsius: float = 41
) -> float:
    """
    Calculate the probability of dimer formation between two sequences.

    This is a simplified calculation. For production use, consider using
    more sophisticated tools like NUPACK, ViennaRNA, or primer3.
    """
    # For primer-dais pairs that should bind (perfect complementarity),
    # check if seq2 is the reverse complement of a substring of seq1
    seq1_rc = str(Seq(seq1).reverse_complement())

    # Check if these are designed to be complementary (the primer-dais pair)
    max_complementary_length = 0

    # Check all possible alignments for complementarity
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            complement_length = 0
            for k in range(min(len(seq1) - i, len(seq2) - j)):
                if (
                    (seq1[i + k] == 'A' and seq2[j + k] == 'T')
                    or (seq1[i + k] == 'T' and seq2[j + k] == 'A')
                    or (seq1[i + k] == 'G' and seq2[j + k] == 'C')
                    or (seq1[i + k] == 'C' and seq2[j + k] == 'G')
                ):
                    complement_length += 1
                else:
                    break

            max_complementary_length = max(max_complementary_length, complement_length)

    # Also check reverse complement alignment
    for i in range(len(seq1_rc)):
        for j in range(len(seq2)):
            complement_length = 0
            for k in range(min(len(seq1_rc) - i, len(seq2) - j)):
                if (
                    (seq1_rc[i + k] == 'A' and seq2[j + k] == 'T')
                    or (seq1_rc[i + k] == 'T' and seq2[j + k] == 'A')
                    or (seq1_rc[i + k] == 'G' and seq2[j + k] == 'C')
                    or (seq1_rc[i + k] == 'C' and seq2[j + k] == 'G')
                ):
                    complement_length += 1
                else:
                    break

            max_complementary_length = max(max_complementary_length, complement_length)

    # Convert complementary length to binding probability
    if max_complementary_length >= 20:  # Strong binding for 20+ bp complementarity
        return 0.95
    elif max_complementary_length >= 15:  # Good binding for 15-19 bp
        return 0.80
    elif max_complementary_length >= 10:  # Moderate binding for 10-14 bp
        return 0.60
    elif max_complementary_length >= 6:  # Weak binding for 6-9 bp
        return 0.30
    else:  # Very weak binding for <6 bp
        return 0.05


def calculate_3_prime_unpaired_probability(
    primer_seq: str, competing_sequences: List[str]
) -> float:
    """
    Calculate the probability that the 3'-end most two bases of primer remain unpaired.

    This checks the last two bases of the primer sequence.
    """
    if len(primer_seq) < 2:
        return 0.0

    three_prime_end = primer_seq[-2:]  # Last two bases

    # Check each competing sequence for complementarity to 3'-end
    total_binding_prob = 0.0

    for comp_seq in competing_sequences:
        # Find the best match for 3'-end in the competing sequence
        best_match_prob = 0.0

        for i in range(len(comp_seq) - 1):
            two_bases = comp_seq[i : i + 2]

            # Check complementarity
            matches = 0
            if (
                (three_prime_end[0] == 'A' and two_bases[1] == 'T')
                or (three_prime_end[0] == 'T' and two_bases[1] == 'A')
                or (three_prime_end[0] == 'G' and two_bases[1] == 'C')
                or (three_prime_end[0] == 'C' and two_bases[1] == 'G')
            ):
                matches += 1

            if (
                (three_prime_end[1] == 'A' and two_bases[0] == 'T')
                or (three_prime_end[1] == 'T' and two_bases[0] == 'A')
                or (three_prime_end[1] == 'G' and two_bases[0] == 'C')
                or (three_prime_end[1] == 'C' and two_bases[0] == 'G')
            ):
                matches += 1

            # Convert matches to binding probability
            if matches == 2:
                match_prob = 0.8  # Strong binding
            elif matches == 1:
                match_prob = 0.3  # Weak binding
            else:
                match_prob = 0.05  # Very weak binding

            best_match_prob = max(best_match_prob, match_prob)

        total_binding_prob = max(total_binding_prob, best_match_prob)

    # Probability of remaining unpaired is 1 - binding probability
    unpaired_prob = 1.0 - total_binding_prob
    return max(0.0, min(1.0, unpaired_prob))


# ============================================================================
# VALIDATION TESTS
# ============================================================================

# TODO: Additional validation tests to be implemented:
#
# TEST 4: NASBA primer-signal binding specificity
# - Each NASBA primer should bind to its intended signal with >80% binding
# - Use 250nM primer concentration and 10pM signal concentration
# - Signal for forward primers: reverse complement of canonical sequence
# - Signal for reverse primers: canonical sequence (forward strand)
# - Within the primer-signal dimer, the 2 3'-end bases of primer should be >90% bound
#
# TEST 5: NASBA primer cross-reactivity with unintended signals
# - Each primer should NOT bind significantly to three unintended signals:
#   1. Forward signal of other pathogen (C.T vs N.G)  
#   2. Reverse signal of other pathogen (C.T vs N.G)
#   3. Wrong orientation signal of same pathogen:
#      - Forward primers: should not bind to canonical (forward) signal
#      - Reverse primers: should not bind to reverse complement of canonical
# - Requires multiple NUPACK runs (cannot place signal and its RC in same tube)
# - Permissible to place multiple primers with non-intended signal in single tube
# - Use same concentrations: 250nM primers, 10pM signal
#
# TEST 6: DNA amplicon construction and validation
# - Define complete DNA amplicon structure:
#   1. 5'-end: Forward primer generic parts (5' to 3')
#   2. Forward primer anchor + toehold (which match canonical sequence)
#   3. Canonical sequence segment between primer binding sites
#   4. Reverse complement of (reverse primer anchor + toehold)
#   5. 3'-end: Reverse complement of reverse primer non-complementary parts
#      (T7 parts + generic, in reverse order)
# - Construct both N.G and C.T amplicons correctly
# - Verify reverse primer orientation is properly reverse-complemented
#
# TEST 7: Inter-pathogen amplicon cross-reactivity
# - Test four amplicon pairs for excessive dimer formation:
#   1. (N.G-amplicon, C.T-amplicon) - both sense strand
#   2. (N.G-RC-amplicon, C.T-RC-amplicon) - both antisense strand  
#   3. (N.G-amplicon, C.T-RC-amplicon) - sense vs antisense
#   4. (N.G-RC-amplicon, C.T-amplicon) - antisense vs sense
# - Ensure amplicons from different pathogens don't form significant dimers
# - Critical for multiplex assay specificity
#
# TEST 8: Generic primer-signal binding specificity
# - Generic forward primer: 5' portion of forward NASBA primer up to anchor
# - Generic reverse primer: T7 parts + generic part of reverse NASBA primer (up to anchor)
# - Generic primers are identical for N.G and C.T within same generic set (gen5/gen6)
# - Each generic primer should bind its designated signal >80%
#   * Generic forward primer binds to sense amplicon signal
#   * Generic reverse primer binds to antisense amplicon signal
# - Within primer-signal dimer, 2 3'-end bases of generic primer should bind >90%
# - Use 250nM primer concentration, 10pM signal concentration
#
# TEST 9: Generic primer cross-reactivity with unintended signals
# - Each generic primer should NOT bind significantly to 3 non-designated signals:
#   1. Wrong pathogen amplicon (same orientation)
#   2. Wrong pathogen amplicon (opposite orientation) 
#   3. Same pathogen amplicon (wrong orientation)
# - Cannot place sense and antisense of same amplicon in single tube
# - Use same concentrations: 250nM primers, 10pM signal
#
# TEST 10: Generic primer cross-reactivity within primer set
# - Test all 6 primers from same generic set in single tube:
#   * 2 generic primers (forward + reverse)
#   * 2 N.G NASBA primers (forward + reverse)
#   * 2 C.T NASBA primers (forward + reverse)
# - Generic primers should not bind to each other significantly
# - Generic primers should not bind to NASBA primers significantly
# - All primers should remain predominantly monomeric
# - Use 250nM concentration for all primers
# - Separate tests for gen5 and gen6 (never mix generic sets)


def test_1_hetero_dimer_measurements(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
    ct_daises: List[Dais],
    ng_daises: List[Dais],
) -> List[ValidationResult]:
    """
    Test 1: Hetero-dimer fraction measurements.

    Each C.T primer should bind strongly (>90%) to its corresponding dais.
    Each N.G primer should bind strongly (>90%) to its corresponding dais.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 1: HETERO-DIMER FRACTION MEASUREMENTS")
    print("=" * 80)

    # Test C.T primers with their daises
    for primer in ct_primers:
        # Find corresponding dais
        target_dais = None
        for dais in ct_daises:
            if (
                dais.species == primer.species
                and dais.primer_type == primer.primer_type
                and dais.generic_set == primer.generic_set
            ):
                target_dais = dais
                break

        if target_dais:
            hetero_dimer_fraction = calculate_dimer_formation_probability(
                primer.sequence,
                target_dais.sequence,
                TESTING_CONDITIONS['temperature_C'],
            )

            passed = hetero_dimer_fraction >= VALIDATION_THRESHOLDS['correct_dais_primer_dimer_min']

            result = ValidationResult(
                test_name="Test_1_Hetero_Dimer",
                primer_name=primer.name,
                target_dais=[target_dais.name],
                hetero_dimer_fraction=hetero_dimer_fraction,
                passed=passed,
                details=f"C.T primer vs its dais: {hetero_dimer_fraction:.3f} ({'PASS' if passed else 'FAIL'})",
            )
            results.append(result)

            print(
                f"{primer.name:20s} -> {target_dais.name:20s}: {hetero_dimer_fraction:.3f} ({'✓' if passed else '✗'})"
            )

    # Test N.G primers with their daises
    for primer in ng_primers:
        # Find corresponding dais
        target_dais = None
        for dais in ng_daises:
            if (
                dais.species == primer.species
                and dais.primer_type == primer.primer_type
                and dais.generic_set == primer.generic_set
            ):
                target_dais = dais
                break

        if target_dais:
            hetero_dimer_fraction = calculate_dimer_formation_probability(
                primer.sequence,
                target_dais.sequence,
                TESTING_CONDITIONS['temperature_C'],
            )

            passed = hetero_dimer_fraction >= VALIDATION_THRESHOLDS['correct_dais_primer_dimer_min']

            result = ValidationResult(
                test_name="Test_1_Hetero_Dimer",
                primer_name=primer.name,
                target_dais=[target_dais.name],
                hetero_dimer_fraction=hetero_dimer_fraction,
                passed=passed,
                details=f"N.G primer vs its dais: {hetero_dimer_fraction:.3f} ({'PASS' if passed else 'FAIL'})",
            )
            results.append(result)

            print(
                f"{primer.name:20s} -> {target_dais.name:20s}: {hetero_dimer_fraction:.3f} ({'✓' if passed else '✗'})"
            )

    return results


def test_2_four_primer_cross_reactivity(
    ct_primers: List[ValidationPrimer], ng_primers: List[ValidationPrimer]
) -> List[ValidationResult]:
    """
    Test 2: Four-primer cross-reactivity analysis.

    Place all four primers (C.T forward, C.T reverse, N.G forward, N.G reverse)
    for each generic set in a single tube. All primers should remain as monomers (> 90%),
    and 3'-end should be unpaired (>99.8%).
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 2: FOUR-PRIMER CROSS-REACTIVITY ANALYSIS")
    print("=" * 80)

    # Group primers by generic set
    for generic_set in ['gen5', 'gen6']:
        primers_in_tube = []

        # Collect all 4 primers for this generic set
        for primer in ct_primers + ng_primers:
            if primer.generic_set == generic_set:
                primers_in_tube.append(primer)

        if len(primers_in_tube) != 4:
            print(
                f"Warning: Expected 4 primers for {generic_set}, found {len(primers_in_tube)}"
            )
            continue

        print(f"\nTesting {generic_set} primer set:")

        # Test each primer in the context of the other three
        for target_primer in primers_in_tube:
            other_primers = [p for p in primers_in_tube if p != target_primer]
            other_sequences = [p.sequence for p in other_primers]

            # Calculate monomer fraction (1 - max binding probability with others)
            max_binding_prob = 0.0
            for other_seq in other_sequences:
                binding_prob = calculate_dimer_formation_probability(
                    target_primer.sequence,
                    other_seq,
                    TESTING_CONDITIONS['temperature_C'],
                )
                max_binding_prob = max(max_binding_prob, binding_prob)

            monomer_fraction = 1.0 - max_binding_prob

            # Calculate 3'-end unpaired probability
            unpaired_3_prime_prob = calculate_3_prime_unpaired_probability(
                target_primer.sequence, other_sequences
            )

            # Check if both criteria are met
            monomer_pass = monomer_fraction >= VALIDATION_THRESHOLDS['primer_monomer_vs_other_primers']
            unpaired_pass = (
                unpaired_3_prime_prob >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']
            )
            overall_pass = monomer_pass and unpaired_pass

            result = ValidationResult(
                test_name="Test_2_Cross_Reactivity",
                primer_name=target_primer.name,
                target_dais=[p.name for p in other_primers],
                monomer_fraction=monomer_fraction,
                unpaired_3_prime_prob=unpaired_3_prime_prob,
                passed=overall_pass,
                details=f"Monomer: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired: {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)

            print(
                f"  {target_primer.name:15s}: Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
            )

    return results


def test_3_individual_primer_dais_binding(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
    ct_daises: List[Dais],
    ng_daises: List[Dais],
) -> List[ValidationResult]:
    """
    Test 3: Individual primer-dais binding specificity.

    8 tests total: each primer (4 primers) X each generic set (2 sets) tested against
    the three daises it should NOT bind to. Primer should remain monomer (>90%), and
    3'-end should be unpaired (>99.8%).
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 3: INDIVIDUAL PRIMER-DAIS BINDING SPECIFICITY")
    print("=" * 80)

    all_primers = ct_primers + ng_primers
    all_daises = ct_daises + ng_daises

    for primer in all_primers:
        # Find the three daises this primer should NOT bind to
        # Each primer should be tested against 3 specific daises
        non_target_daises = []

        # Logic: Each primer should NOT bind to:
        # 1. The dais from the other species with the same primer_type and generic_set
        # 2. The dais from the same species with opposite primer_type and the same generic_set
        # 3. The dais from the same species, same primer_type but opposite generic_set

        for dais in all_daises:
            should_not_bind = False

            # Case 1: Different species, same primer_type and generic_set
            if (
                dais.species != primer.species
                and dais.primer_type == primer.primer_type
                and dais.generic_set == primer.generic_set
            ):
                should_not_bind = True

            # Case 2: Same species, different primer_type, same generic_set
            elif (
                dais.species == primer.species
                and dais.primer_type != primer.primer_type
                and dais.generic_set == primer.generic_set
            ):
                should_not_bind = True

            # Case 3: Same species, same primer_type, different generic_set
            elif (
                dais.species == primer.species
                and dais.primer_type == primer.primer_type
                and dais.generic_set != primer.generic_set
            ):
                should_not_bind = True

            if should_not_bind:
                non_target_daises.append(dais)

        # Should have exactly 3 non-target daises for each primer
        if len(non_target_daises) != 3:
            print(
                f"Debug: {primer.name} has {len(non_target_daises)} non-target daises"
            )
            for dais in non_target_daises:
                print(
                    f"  - {dais.name} ({dais.species}, {dais.primer_type}, {dais.generic_set})"
                )

        if non_target_daises:
            non_target_sequences = [d.sequence for d in non_target_daises]
            non_target_names = [d.name for d in non_target_daises]

            # Calculate the monomer fraction (1 - max binding probability with non-targets)
            max_binding_prob = 0.0
            for dais_seq in non_target_sequences:
                binding_prob = calculate_dimer_formation_probability(
                    primer.sequence, dais_seq, TESTING_CONDITIONS['temperature_C']
                )
                max_binding_prob = max(max_binding_prob, binding_prob)

            monomer_fraction = 1.0 - max_binding_prob

            # Calculate 3'-end unpaired probability
            unpaired_3_prime_prob = calculate_3_prime_unpaired_probability(
                primer.sequence, non_target_sequences
            )

            # Check if both criteria are met
            monomer_pass = monomer_fraction >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']
            unpaired_pass = (
                unpaired_3_prime_prob >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']
            )
            overall_pass = monomer_pass and unpaired_pass

            result = ValidationResult(
                test_name="Test_3_Individual_Binding",
                primer_name=primer.name,
                target_dais=non_target_names,
                monomer_fraction=monomer_fraction,
                unpaired_3_prime_prob=unpaired_3_prime_prob,
                passed=overall_pass,
                details=f"Monomer: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired: {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)

            print(
                f"{primer.name:15s} vs non-targets: Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
            )

    return results


def test_4_primer_signal_binding_specificity(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
) -> List[ValidationResult]:
    """
    Test 4: NASBA primer-signal binding specificity.

    Each NASBA primer should bind to its intended signal with >80% binding.
    Within the primer-signal dimer, the 2 3'-end bases of primer should be >90% bound.
    
    Signal construction:
    - Forward primers: Optimal signal starts with reverse primer (generic + anchor + toehold) 
      as RC, then continues with RC of canonical sequence to the end
    - Reverse primers: Complete canonical sequence (forward strand)
    
    Uses 250nM primer concentration and 10pM signal concentration.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 4: NASBA PRIMER-SIGNAL BINDING SPECIFICITY")
    print("=" * 80)

    # Get canonical sequences
    if 'CT' not in CANONICAL_SEQUENCES or 'NG' not in CANONICAL_SEQUENCES:
        raise ValueError("Required canonical sequences not found")

    ct_canonical = CANONICAL_SEQUENCES['CT']
    ng_canonical = CANONICAL_SEQUENCES['NG']

    # Helper function to construct optimal forward signal
    def construct_forward_signal(species: str, primers: List[ValidationPrimer]) -> str:
        """
        Construct optimal forward signal: RC of (reverse primer generic + anchor + toehold) + RC of canonical
        """
        canonical = ct_canonical if species == 'CT' else ng_canonical
        
        # Find a reverse primer for this species to get the reverse primer structure
        reverse_primer = None
        for p in primers:
            if p.species == species and p.primer_type == 'reverse':
                reverse_primer = p
                break
        
        if not reverse_primer:
            raise ValueError(f"No reverse primer found for species {species}")
        
        # Extract the generic + anchor + toehold part from reverse primer
        # Reverse primer structure: T7_parts + generic + anchor + toehold
        # We need: generic + anchor + toehold (the part that binds to canonical)
        full_reverse_seq = reverse_primer.sequence
        binding_part = reverse_primer.anchor_sequence + reverse_primer.toehold_sequence
        
        # Find where the binding part starts in the full reverse primer
        binding_start = full_reverse_seq.find(binding_part)
        if binding_start == -1:
            raise ValueError(f"Could not locate binding part in reverse primer {reverse_primer.name}")
        
        # Extract everything from the binding start (generic + anchor + toehold)
        reverse_binding_portion = full_reverse_seq[binding_start:]
        
        # Construct optimal forward signal
        optimal_signal = str(Seq(reverse_binding_portion).reverse_complement()) + str(Seq(canonical).reverse_complement())
        
        return optimal_signal

    all_primers = ct_primers + ng_primers
    
    # Pre-construct forward signals for both species
    forward_signals = {
        'CT': construct_forward_signal('CT', all_primers),
        'NG': construct_forward_signal('NG', all_primers)
    }
    
    # Define all signals
    signals = {
        'CT': {
            'forward': forward_signals['CT'],
            'reverse': ct_canonical  # Forward strand for reverse primers
        },
        'NG': {
            'forward': forward_signals['NG'],  
            'reverse': ng_canonical  # Forward strand for reverse primers
        }
    }

    for primer in all_primers:
        # Get the intended signal for this primer
        intended_signal = signals[primer.species][primer.primer_type]
        
        print(f"\nTesting {primer.name} binding to its intended signal...")
        print(f"  Signal length: {len(intended_signal)} bp")
        
        # Set up NUPACK analysis with appropriate concentrations
        primer_concentration_M = TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9  # Convert nM to M
        signal_concentration_M = TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12  # Convert pM to M
        
        sequences = [
            SequenceInput(f"primer_{primer.name}", primer.sequence, primer_concentration_M),
            SequenceInput(f"signal_{primer.species}_{primer.primer_type}", intended_signal, signal_concentration_M),
        ]

        # Run NUPACK analysis with base-pairing to get 3'-end binding info
        nupack_results = analyze_sequence_complexes(
            temperature_celsius=TESTING_CONDITIONS['temperature_C'],
            sequences=sequences,
            sodium_millimolar=80.0,  # NASBA conditions
            magnesium_millimolar=12.0,  # NASBA conditions
            max_complex_size=2,
            base_pairing_analysis=True,
        )

        # Calculate primer-signal hetero-dimer fraction
        primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
            f"primer_{primer.name}", 
            f"signal_{primer.species}_{primer.primer_type}"
        )

        # Calculate 3'-end binding probability
        three_prime_binding_prob = 0.0
        primer_length = len(primer.sequence)
        
        # Find the primer-signal hetero-dimer complex
        for complex_result in nupack_results.complexes:
            if (complex_result.size == 2 and 
                complex_result.unpaired_probability and
                f"primer_{primer.name}" in complex_result.sequence_id_map.values() and
                f"signal_{primer.species}_{primer.primer_type}" in complex_result.sequence_id_map.values()):
                
                # Find sequence ID for primer in this complex
                primer_seq_id = None
                for seq_id, name in complex_result.sequence_id_map.items():
                    if name == f"primer_{primer.name}":
                        primer_seq_id = seq_id
                        break
                
                if primer_seq_id is not None:
                    # Get unpaired probability for the last two bases (3'-end)
                    base1_key = (primer_seq_id, primer_length - 1)  # Second to last base
                    base2_key = (primer_seq_id, primer_length)      # Last base
                    
                    # Get unpaired probabilities (lower unpaired = higher paired)
                    unpaired_prob1 = complex_result.unpaired_probability.get(base1_key, 1.0)
                    unpaired_prob2 = complex_result.unpaired_probability.get(base2_key, 1.0)
                    
                    # 3'-end binding probability = 1 - unpaired probability
                    base1_binding = 1.0 - unpaired_prob1
                    base2_binding = 1.0 - unpaired_prob2
                    
                    # Both bases should be bound (probability both are bound)
                    three_prime_binding_prob = base1_binding * base2_binding
                    break

        # Check validation criteria
        binding_pass = primer_signal_binding >= VALIDATION_THRESHOLDS['primer_signal_binding_min']
        three_prime_pass = three_prime_binding_prob >= VALIDATION_THRESHOLDS['primer_3_prime_binding_min']
        overall_pass = binding_pass and three_prime_pass

        result = ValidationResult(
            test_name="Test_4_Primer_Signal_Binding",
            primer_name=primer.name,
            target_dais=[f"signal_{primer.species}_{primer.primer_type}"],
            hetero_dimer_fraction=primer_signal_binding,
            unpaired_3_prime_prob=1.0 - three_prime_binding_prob,  # Store as unpaired for consistency
            passed=overall_pass,
            details=f"Signal binding: {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
                   f"3'-end binding: {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})",
        )
        results.append(result)

        print(f"  {primer.name:20s}: Signal binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
              f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})")

    return results


def test_5_primer_cross_reactivity_with_unintended_signals(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
) -> List[ValidationResult]:
    """
    Test 5: NASBA primer cross-reactivity with unintended signals.

    Each primer should NOT bind significantly to three unintended signals:
    1. Forward signal of other pathogen (C.T vs N.G)  
    2. Reverse signal of other pathogen (C.T vs N.G)
    3. Wrong orientation signal of same pathogen:
       - Forward primers: should not bind to canonical (forward) signal
       - Reverse primers: should not bind to reverse complement of canonical

    Requires multiple NUPACK runs (cannot place signal and its RC in same tube).
    Uses 250nM primers, 10pM signal concentrations.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 5: NASBA PRIMER CROSS-REACTIVITY WITH UNINTENDED SIGNALS")
    print("=" * 80)

    # Get canonical sequences
    if 'CT' not in CANONICAL_SEQUENCES or 'NG' not in CANONICAL_SEQUENCES:
        raise ValueError("Required canonical sequences not found")

    ct_canonical = CANONICAL_SEQUENCES['CT']
    ng_canonical = CANONICAL_SEQUENCES['NG']

    all_primers = ct_primers + ng_primers
    
    # Helper function to construct optimal forward signal (reused from Test 4)
    def construct_forward_signal(species: str, primers: List[ValidationPrimer]) -> str:
        canonical = ct_canonical if species == 'CT' else ng_canonical
        
        reverse_primer = None
        for p in primers:
            if p.species == species and p.primer_type == 'reverse':
                reverse_primer = p
                break
        
        if not reverse_primer:
            raise ValueError(f"No reverse primer found for species {species}")
        
        full_reverse_seq = reverse_primer.sequence
        binding_part = reverse_primer.anchor_sequence + reverse_primer.toehold_sequence
        
        binding_start = full_reverse_seq.find(binding_part)
        if binding_start == -1:
            raise ValueError(f"Could not locate binding part in reverse primer {reverse_primer.name}")
        
        reverse_binding_portion = full_reverse_seq[binding_start:]
        optimal_signal = str(Seq(reverse_binding_portion).reverse_complement()) + str(Seq(canonical).reverse_complement())
        
        return optimal_signal

    # Pre-construct all signals
    signals = {
        'CT': {
            'forward': construct_forward_signal('CT', all_primers),
            'reverse': ct_canonical
        },
        'NG': {
            'forward': construct_forward_signal('NG', all_primers),  
            'reverse': ng_canonical
        }
    }

    # Define wrong orientation signals for same pathogen
    wrong_orientation_signals = {
        'CT': {
            'forward': ct_canonical,  # Forward primers should not bind to canonical (forward)
            'reverse': str(Seq(ct_canonical).reverse_complement())  # Reverse primers should not bind to RC
        },
        'NG': {
            'forward': ng_canonical,  # Forward primers should not bind to canonical (forward)
            'reverse': str(Seq(ng_canonical).reverse_complement())  # Reverse primers should not bind to RC
        }
    }

    for primer in all_primers:
        print(f"\nTesting {primer.name} cross-reactivity with unintended signals...")
        
        # Define the three unintended signals for this primer
        other_species = 'NG' if primer.species == 'CT' else 'CT'
        
        unintended_signals = [
            # 1. Forward signal of other pathogen
            (f"other_pathogen_forward", signals[other_species]['forward']),
            # 2. Reverse signal of other pathogen  
            (f"other_pathogen_reverse", signals[other_species]['reverse']),
            # 3. Wrong orientation signal of same pathogen
            (f"wrong_orientation", wrong_orientation_signals[primer.species][primer.primer_type])
        ]

        # Test each unintended signal separately (cannot mix sense/antisense in same tube)
        for signal_name, signal_sequence in unintended_signals:
            print(f"  Testing against {signal_name} (length: {len(signal_sequence)} bp)")
            
            # Set up NUPACK analysis
            primer_concentration_M = TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9
            signal_concentration_M = TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12
            
            sequences = [
                SequenceInput(f"primer_{primer.name}", primer.sequence, primer_concentration_M),
                SequenceInput(signal_name, signal_sequence, signal_concentration_M),
            ]

            # Run NUPACK analysis
            nupack_results = analyze_sequence_complexes(
                temperature_celsius=TESTING_CONDITIONS['temperature_C'],
                sequences=sequences,
                sodium_millimolar=80.0,
                magnesium_millimolar=12.0,
                max_complex_size=2,
                base_pairing_analysis=True,
            )

            # Calculate primer-signal binding (should be low for unintended signals)
            primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
                f"primer_{primer.name}", 
                signal_name
            )

            # Calculate monomer fraction (should be high - primer stays as monomer)
            monomer_fraction = nupack_results.get_monomer_fraction(f"primer_{primer.name}")

            # Calculate 3'-end unpaired probability (should be high - no binding at 3'-end)
            three_prime_unpaired_prob = 1.0  # Default to unpaired
            primer_length = len(primer.sequence)
            
            # Look for any complex containing this primer to get 3'-end info
            for complex_result in nupack_results.complexes:
                if (complex_result.unpaired_probability and
                    f"primer_{primer.name}" in complex_result.sequence_id_map.values()):
                    
                    primer_seq_id = None
                    for seq_id, name in complex_result.sequence_id_map.items():
                        if name == f"primer_{primer.name}":
                            primer_seq_id = seq_id
                            break
                    
                    if primer_seq_id is not None:
                        # Get unpaired probability for the last two bases
                        base1_key = (primer_seq_id, primer_length - 1)
                        base2_key = (primer_seq_id, primer_length)
                        
                        unpaired_prob1 = complex_result.unpaired_probability.get(base1_key, 1.0)
                        unpaired_prob2 = complex_result.unpaired_probability.get(base2_key, 1.0)
                        
                        # Weight by complex concentration and take the maximum unpaired probability
                        complex_weight = complex_result.concentration_M
                        weighted_unpaired = (unpaired_prob1 * unpaired_prob2) * complex_weight
                        three_prime_unpaired_prob = max(three_prime_unpaired_prob, weighted_unpaired)

            # Validation criteria (primers should NOT bind significantly to unintended signals)
            low_binding_pass = primer_signal_binding <= (1.0 - VALIDATION_THRESHOLDS['primer_signal_binding_min'])  # <20% binding
            monomer_pass = monomer_fraction >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']  # >90% monomer
            unpaired_pass = three_prime_unpaired_prob >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']  # >99.8% unpaired
            
            overall_pass = low_binding_pass and monomer_pass and unpaired_pass

            result = ValidationResult(
                test_name="Test_5_Unintended_Signal_Cross_Reactivity",
                primer_name=primer.name,
                target_dais=[signal_name],
                hetero_dimer_fraction=primer_signal_binding,
                monomer_fraction=monomer_fraction,
                unpaired_3_prime_prob=three_prime_unpaired_prob,
                passed=overall_pass,
                details=f"{signal_name}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
                       f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                       f"3'-unpaired {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)

            print(f"    {signal_name:20s}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
                  f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                  f"3'-unpaired {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})")

    return results


# ============================================================================
# DNA AMPLICON CONSTRUCTION
# ============================================================================


@dataclass
class DNAAmplicon:
    """Represents a complete DNA amplicon with construction details."""
    
    name: str
    species: str  # 'CT' or 'NG'
    generic_set: str  # 'gen5' or 'gen6'
    sequence: str
    length: int
    
    # Construction components for verification
    forward_generic_part: str
    forward_anchor_toehold: str
    canonical_segment: str
    reverse_anchor_toehold_rc: str
    reverse_noncomplementary_rc: str


def construct_dna_amplicons(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
) -> Dict[str, List[DNAAmplicon]]:
    """
    Construct complete DNA amplicons with internal verification.
    
    Amplicon structure (5' to 3'):
    1. Forward primer generic parts (5' to 3')
    2. Forward primer anchor + toehold (which match canonical sequence)
    3. Canonical sequence segment between primer binding sites
    4. Reverse complement of (reverse primer anchor + toehold)
    5. Reverse complement of reverse primer non-complementary parts
       (T7 parts + generic, in reverse order)
    
    Returns dict with keys 'CT' and 'NG', each containing list of amplicons per generic set.
    """
    print("\n" + "=" * 60)
    print("DNA AMPLICON CONSTRUCTION")
    print("=" * 60)
    
    # Get canonical sequences
    if 'CT' not in CANONICAL_SEQUENCES or 'NG' not in CANONICAL_SEQUENCES:
        raise ValueError("Required canonical sequences not found")

    ct_canonical = CANONICAL_SEQUENCES['CT']
    ng_canonical = CANONICAL_SEQUENCES['NG']
    
    amplicons = {'CT': [], 'NG': []}
    
    # Process each species
    for species in ['CT', 'NG']:
        canonical = ct_canonical if species == 'CT' else ng_canonical
        species_primers = [p for p in (ct_primers + ng_primers) if p.species == species]
        
        print(f"\nConstructing {species} amplicons...")
        
        # Group primers by generic set
        for generic_set in ['gen5', 'gen6']:
            forward_primer = None
            reverse_primer = None
            
            for primer in species_primers:
                if primer.generic_set == generic_set:
                    if primer.primer_type == 'forward':
                        forward_primer = primer
                    elif primer.primer_type == 'reverse':
                        reverse_primer = primer
            
            if not forward_primer or not reverse_primer:
                print(f"  Warning: Missing primers for {species} {generic_set}")
                continue
            
            print(f"  Constructing {species} {generic_set} amplicon...")
            
            # 1. Forward primer generic parts
            forward_full_seq = forward_primer.sequence
            forward_anchor_toehold = forward_primer.anchor_sequence + forward_primer.toehold_sequence
            forward_generic_start = forward_full_seq.find(forward_anchor_toehold)
            
            if forward_generic_start == -1:
                raise ValueError(f"Cannot locate anchor+toehold in forward primer {forward_primer.name}")
            
            forward_generic_part = forward_full_seq[:forward_generic_start]
            
            # 2. Forward primer anchor + toehold (already have this)
            
            # 3. Canonical sequence segment between primer binding sites
            # Find where forward primer binds (anchor + toehold should be in canonical)
            forward_bind_pos = canonical.find(forward_anchor_toehold)
            if forward_bind_pos == -1:
                raise ValueError(f"Forward primer anchor+toehold not found in {species} canonical sequence")
            
            # Find where reverse primer binds (reverse complement of anchor + toehold should be in canonical)
            reverse_anchor_toehold = reverse_primer.anchor_sequence + reverse_primer.toehold_sequence
            reverse_anchor_toehold_rc = str(Seq(reverse_anchor_toehold).reverse_complement())
            reverse_bind_pos = canonical.find(reverse_anchor_toehold_rc)
            if reverse_bind_pos == -1:
                raise ValueError(f"Reverse primer anchor+toehold RC not found in {species} canonical sequence")
            
            # Extract canonical segment between primer binding sites
            forward_end = forward_bind_pos + len(forward_anchor_toehold)
            canonical_segment = canonical[forward_end:reverse_bind_pos]
            
            # 4. Reverse complement of (reverse primer anchor + toehold) - already have this
            
            # 5. Reverse complement of reverse primer non-complementary parts
            reverse_full_seq = reverse_primer.sequence
            reverse_binding_start = reverse_full_seq.find(reverse_anchor_toehold)
            if reverse_binding_start == -1:
                raise ValueError(f"Cannot locate anchor+toehold in reverse primer {reverse_primer.name}")
            
            reverse_noncomplementary_part = reverse_full_seq[:reverse_binding_start]
            reverse_noncomplementary_rc = str(Seq(reverse_noncomplementary_part).reverse_complement())
            
            # Construct complete amplicon
            amplicon_sequence = (
                forward_generic_part +
                forward_anchor_toehold +
                canonical_segment +
                reverse_anchor_toehold_rc +
                reverse_noncomplementary_rc
            )
            
            # Create amplicon object
            amplicon = DNAAmplicon(
                name=f"{species}_amplicon_{generic_set}",
                species=species,
                generic_set=generic_set,
                sequence=amplicon_sequence,
                length=len(amplicon_sequence),
                forward_generic_part=forward_generic_part,
                forward_anchor_toehold=forward_anchor_toehold,
                canonical_segment=canonical_segment,
                reverse_anchor_toehold_rc=reverse_anchor_toehold_rc,
                reverse_noncomplementary_rc=reverse_noncomplementary_rc,
            )
            
            amplicons[species].append(amplicon)
            
            # Internal verification
            print(f"    ✓ Amplicon constructed: {amplicon.length} bp")
            print(f"      Forward generic: {len(forward_generic_part)} bp")
            print(f"      Forward anchor+toehold: {len(forward_anchor_toehold)} bp")
            print(f"      Canonical segment: {len(canonical_segment)} bp")
            print(f"      Reverse anchor+toehold RC: {len(reverse_anchor_toehold_rc)} bp")
            print(f"      Reverse non-complementary RC: {len(reverse_noncomplementary_rc)} bp")
            
            # Verify construction logic
            _verify_amplicon_construction(amplicon, forward_primer, reverse_primer, canonical)
    
    print(f"\n✓ Constructed {sum(len(amps) for amps in amplicons.values())} amplicons total")
    return amplicons


def _verify_amplicon_construction(
    amplicon: DNAAmplicon, 
    forward_primer: ValidationPrimer, 
    reverse_primer: ValidationPrimer,
    canonical: str
) -> None:
    """Internal verification of amplicon construction correctness."""
    
    # Verify forward primer can bind to start of amplicon
    forward_binding_region = amplicon.forward_anchor_toehold
    if forward_binding_region not in canonical:
        raise ValueError(f"Forward binding region not found in canonical for {amplicon.name}")
    
    # Verify reverse primer can bind to end of amplicon (as reverse complement)
    reverse_binding_region = amplicon.reverse_anchor_toehold_rc
    expected_reverse_binding = str(Seq(reverse_primer.anchor_sequence + reverse_primer.toehold_sequence).reverse_complement())
    if reverse_binding_region != expected_reverse_binding:
        raise ValueError(f"Reverse binding region mismatch for {amplicon.name}")
    
    # Verify total length is reasonable (should be substantial portion of canonical + primer parts)
    min_expected_length = len(amplicon.canonical_segment) + 50  # At least canonical segment + some primer parts
    if amplicon.length < min_expected_length:
        raise ValueError(f"Amplicon {amplicon.name} too short: {amplicon.length} bp")
    
    # Verify all parts sum to total length
    total_parts = (len(amplicon.forward_generic_part) + 
                   len(amplicon.forward_anchor_toehold) +
                   len(amplicon.canonical_segment) +
                   len(amplicon.reverse_anchor_toehold_rc) +
                   len(amplicon.reverse_noncomplementary_rc))
    
    if total_parts != amplicon.length:
        raise ValueError(f"Amplicon {amplicon.name} part lengths don't sum correctly: {total_parts} != {amplicon.length}")
    
    print(f"    ✓ Construction verified for {amplicon.name}")


def test_6_inter_pathogen_amplicon_cross_reactivity(
    amplicons: Dict[str, List[DNAAmplicon]],
) -> List[ValidationResult]:
    """
    Test 6: Inter-pathogen amplicon cross-reactivity.

    Test four amplicon pairs for excessive dimer formation:
    1. (N.G-amplicon, C.T-amplicon) - both sense strand
    2. (N.G-RC-amplicon, C.T-RC-amplicon) - both antisense strand  
    3. (N.G-amplicon, C.T-RC-amplicon) - sense vs antisense
    4. (N.G-RC-amplicon, C.T-amplicon) - antisense vs sense

    Ensure amplicons from different pathogens don't form significant dimers.
    Critical for multiplex assay specificity.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 6: INTER-PATHOGEN AMPLICON CROSS-REACTIVITY")
    print("=" * 80)

    ct_amplicons = amplicons.get('CT', [])
    ng_amplicons = amplicons.get('NG', [])

    if not ct_amplicons or not ng_amplicons:
        raise ValueError("Both C.T and N.G amplicons required for cross-reactivity testing")

    # Test each generic set combination
    for ct_amplicon in ct_amplicons:
        for ng_amplicon in ng_amplicons:
            if ct_amplicon.generic_set != ng_amplicon.generic_set:
                continue  # Only test within same generic set
            
            print(f"\nTesting {ct_amplicon.name} vs {ng_amplicon.name} cross-reactivity...")
            
            # Define the four test pairs
            test_pairs = [
                # 1. Both sense strand
                ("sense_vs_sense", ct_amplicon.sequence, ng_amplicon.sequence),
                # 2. Both antisense strand
                ("antisense_vs_antisense", 
                 str(Seq(ct_amplicon.sequence).reverse_complement()),
                 str(Seq(ng_amplicon.sequence).reverse_complement())),
                # 3. Sense vs antisense
                ("sense_vs_antisense",
                 ct_amplicon.sequence,
                 str(Seq(ng_amplicon.sequence).reverse_complement())),
                # 4. Antisense vs sense  
                ("antisense_vs_sense",
                 str(Seq(ct_amplicon.sequence).reverse_complement()),
                 ng_amplicon.sequence),
            ]

            for pair_name, ct_seq, ng_seq in test_pairs:
                print(f"  Testing {pair_name}...")
                print(f"    C.T sequence: {len(ct_seq)} bp")
                print(f"    N.G sequence: {len(ng_seq)} bp")
                
                # Set up NUPACK analysis with equal concentrations
                # Using signal concentrations (10pM) since these are amplicons
                amplicon_concentration_M = TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12
                
                sequences = [
                    SequenceInput(f"CT_{pair_name}", ct_seq, amplicon_concentration_M),
                    SequenceInput(f"NG_{pair_name}", ng_seq, amplicon_concentration_M),
                ]

                # Run NUPACK analysis
                nupack_results = analyze_sequence_complexes(
                    temperature_celsius=TESTING_CONDITIONS['temperature_C'],
                    sequences=sequences,
                    sodium_millimolar=80.0,
                    magnesium_millimolar=12.0,
                    max_complex_size=2,
                    base_pairing_analysis=False,  # Not needed for this test
                )

                # Calculate hetero-dimer formation (should be low)
                hetero_dimer_fraction = nupack_results.get_hetero_dimer_fraction(
                    f"CT_{pair_name}", 
                    f"NG_{pair_name}"
                )

                # Calculate monomer fractions (should be high)
                ct_monomer_fraction = nupack_results.get_monomer_fraction(f"CT_{pair_name}")
                ng_monomer_fraction = nupack_results.get_monomer_fraction(f"NG_{pair_name}")
                min_monomer_fraction = min(ct_monomer_fraction, ng_monomer_fraction)

                # Validation criteria - amplicons should NOT form significant dimers
                low_dimer_pass = hetero_dimer_fraction <= (1.0 - VALIDATION_THRESHOLDS['primer_signal_binding_min'])  # <20% dimer
                high_monomer_pass = min_monomer_fraction >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']  # >90% monomer
                
                overall_pass = low_dimer_pass and high_monomer_pass

                result = ValidationResult(
                    test_name="Test_6_Inter_Pathogen_Amplicon_Cross_Reactivity",
                    primer_name=f"{ct_amplicon.name}_vs_{ng_amplicon.name}",
                    target_dais=[pair_name],
                    hetero_dimer_fraction=hetero_dimer_fraction,
                    monomer_fraction=min_monomer_fraction,
                    passed=overall_pass,
                    details=f"{pair_name}: Dimer {hetero_dimer_fraction:.3f} ({'✓' if low_dimer_pass else '✗'}), "
                           f"Min monomer {min_monomer_fraction:.3f} ({'✓' if high_monomer_pass else '✗'})",
                )
                results.append(result)

                print(f"    {pair_name:20s}: Dimer {hetero_dimer_fraction:.3f} ({'✓' if low_dimer_pass else '✗'}), "
                      f"Min monomer {min_monomer_fraction:.3f} ({'✓' if high_monomer_pass else '✗'})")

    return results


def test_7_generic_primer_amplicon_binding_specificity(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
    amplicons: Dict[str, List[DNAAmplicon]],
) -> List[ValidationResult]:
    """
    Test 7: Generic primer-amplicon binding specificity.

    Generic primers should bind their designated amplicon signals >80%
    with >90% 3'-end binding within the dimer.
    
    Generic forward primer binds to sense amplicon signal.
    Generic reverse primer binds to antisense amplicon signal.
    
    Uses 25nM primer concentration, 1nM signal concentration.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 7: GENERIC PRIMER-AMPLICON BINDING SPECIFICITY")
    print("=" * 80)

    # Define explicit generic primers
    generic_primers = {
        'gen5': {
            'forward': "TTATGTTCGTGGTT",
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC"
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT", 
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC"
        }
    }

    print("\nVerifying NASBA primers match expected generic primers...")
    
    # Verify that C.T and N.G NASBA primers start with correct generic primers
    all_primers = ct_primers + ng_primers
    for primer in all_primers:
        expected_generic = generic_primers[primer.generic_set][primer.primer_type]
        
        if primer.primer_type == 'forward':
            # Forward primers should start with generic sequence
            if not primer.sequence.startswith(expected_generic):
                raise ValueError(
                    f"Primer {primer.name} does not start with expected generic sequence. "
                    f"Expected: {expected_generic}, Got: {primer.sequence[:len(expected_generic)]}"
                )
        else:  # reverse
            # Reverse primers should contain generic sequence after T7 parts
            if expected_generic not in primer.sequence:
                raise ValueError(
                    f"Primer {primer.name} does not contain expected generic sequence: {expected_generic}"
                )
            
            # Verify the generic part is positioned correctly (should be before anchor)
            generic_pos = primer.sequence.find(expected_generic)
            anchor_pos = primer.sequence.find(primer.anchor_sequence)
            if generic_pos == -1 or anchor_pos == -1 or generic_pos >= anchor_pos:
                raise ValueError(
                    f"Generic sequence not positioned correctly in reverse primer {primer.name}"
                )
    
    print("✓ All NASBA primers verified to match expected generic sequences")

    # Test each generic set
    for generic_set in ['gen5', 'gen6']:
        print(f"\nTesting {generic_set} generic primers...")
        
        forward_generic = generic_primers[generic_set]['forward']
        reverse_generic = generic_primers[generic_set]['reverse']
        
        # Get amplicons for this generic set
        ct_amplicons_set = [a for a in amplicons.get('CT', []) if a.generic_set == generic_set]
        ng_amplicons_set = [a for a in amplicons.get('NG', []) if a.generic_set == generic_set]
        
        if not ct_amplicons_set or not ng_amplicons_set:
            print(f"  Warning: Missing amplicons for {generic_set}")
            continue

        # Test generic forward primer with sense amplicons (both C.T and N.G)
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(f"  Testing forward generic vs {amplicon.name} (sense)...")
            
            # Forward generic binds to sense amplicon
            _test_generic_primer_binding(
                results, 
                f"generic_forward_{generic_set}",
                forward_generic,
                f"{amplicon.name}_sense", 
                amplicon.sequence,
                "forward_vs_sense"
            )

        # Test generic reverse primer with antisense amplicons (both C.T and N.G)  
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(f"  Testing reverse generic vs {amplicon.name} (antisense)...")
            
            # Reverse generic binds to antisense amplicon
            antisense_amplicon = str(Seq(amplicon.sequence).reverse_complement())
            _test_generic_primer_binding(
                results,
                f"generic_reverse_{generic_set}",
                reverse_generic,
                f"{amplicon.name}_antisense",
                antisense_amplicon, 
                "reverse_vs_antisense"
            )

    return results


def _test_generic_primer_binding(
    results: List[ValidationResult],
    primer_name: str,
    primer_sequence: str, 
    signal_name: str,
    signal_sequence: str,
    test_type: str
) -> None:
    """Helper function to test generic primer binding to amplicon signal."""
    
    # Set up NUPACK analysis with generic primer concentrations
    primer_concentration_M = TESTING_CONDITIONS['generic_primer_concentration_nM'] * 1e-9  # Convert nM to M
    signal_concentration_M = TESTING_CONDITIONS['generic_signal_concentration_nM'] * 1e-9  # Convert nM to M
    
    sequences = [
        SequenceInput(primer_name, primer_sequence, primer_concentration_M),
        SequenceInput(signal_name, signal_sequence, signal_concentration_M),
    ]

    # Run NUPACK analysis with base-pairing to get 3'-end binding info
    nupack_results = analyze_sequence_complexes(
        temperature_celsius=TESTING_CONDITIONS['temperature_C'],
        sequences=sequences,
        sodium_millimolar=80.0,  # NASBA conditions
        magnesium_millimolar=12.0,  # NASBA conditions
        max_complex_size=2,
        base_pairing_analysis=True,
    )

    # Calculate primer-signal hetero-dimer fraction
    primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
        primer_name, 
        signal_name
    )

    # Calculate 3'-end binding probability
    three_prime_binding_prob = 0.0
    primer_length = len(primer_sequence)
    
    # Find the primer-signal hetero-dimer complex
    for complex_result in nupack_results.complexes:
        if (complex_result.size == 2 and 
            complex_result.unpaired_probability and
            primer_name in complex_result.sequence_id_map.values() and
            signal_name in complex_result.sequence_id_map.values()):
            
            # Find sequence ID for primer in this complex
            primer_seq_id = None
            for seq_id, name in complex_result.sequence_id_map.items():
                if name == primer_name:
                    primer_seq_id = seq_id
                    break
            
            if primer_seq_id is not None:
                # Get unpaired probability for the last two bases (3'-end)
                base1_key = (primer_seq_id, primer_length - 1)  # Second to last base
                base2_key = (primer_seq_id, primer_length)      # Last base
                
                # Get unpaired probabilities (lower unpaired = higher paired)
                unpaired_prob1 = complex_result.unpaired_probability.get(base1_key, 1.0)
                unpaired_prob2 = complex_result.unpaired_probability.get(base2_key, 1.0)
                
                # 3'-end binding probability = 1 - unpaired probability
                base1_binding = 1.0 - unpaired_prob1
                base2_binding = 1.0 - unpaired_prob2
                
                # Both bases should be bound (probability both are bound)
                three_prime_binding_prob = base1_binding * base2_binding
                break

    # Check validation criteria
    binding_pass = primer_signal_binding >= VALIDATION_THRESHOLDS['generic_primer_amplicon_binding_min']
    three_prime_pass = three_prime_binding_prob >= VALIDATION_THRESHOLDS['generic_primer_3_prime_binding_min']
    overall_pass = binding_pass and three_prime_pass

    result = ValidationResult(
        test_name="Test_7_Generic_Primer_Amplicon_Binding",
        primer_name=primer_name,
        target_dais=[signal_name],
        hetero_dimer_fraction=primer_signal_binding,
        unpaired_3_prime_prob=1.0 - three_prime_binding_prob,  # Store as unpaired for consistency
        passed=overall_pass,
        details=f"{test_type}: Binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
               f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})",
    )
    results.append(result)

    print(f"    {test_type:20s}: Binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
          f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})")


def test_8_generic_primer_cross_reactivity_with_unintended_signals(
    amplicons: Dict[str, List[DNAAmplicon]],
) -> List[ValidationResult]:
    """
    Test 8: Generic primer cross-reactivity with unintended signals.

    Each generic primer should NOT bind significantly to unintended signals:
    - Forward generic primers should not bind to sense amplicons (both C.T and N.G)
    - Reverse generic primers should not bind to antisense amplicons (both C.T and N.G)
    
    Cannot place sense and antisense of same amplicon in single tube.
    Uses 25nM primers, 1nM signal concentrations.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 8: GENERIC PRIMER CROSS-REACTIVITY WITH UNINTENDED SIGNALS")
    print("=" * 80)

    # Define explicit generic primers (same as Test 7)
    generic_primers = {
        'gen5': {
            'forward': "TTATGTTCGTGGTT",
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC"
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT", 
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC"
        }
    }

    # Test each generic set
    for generic_set in ['gen5', 'gen6']:
        print(f"\nTesting {generic_set} generic primer cross-reactivity...")
        
        forward_generic = generic_primers[generic_set]['forward']
        reverse_generic = generic_primers[generic_set]['reverse']
        
        # Get amplicons for this generic set
        ct_amplicons_set = [a for a in amplicons.get('CT', []) if a.generic_set == generic_set]
        ng_amplicons_set = [a for a in amplicons.get('NG', []) if a.generic_set == generic_set]
        
        if not ct_amplicons_set or not ng_amplicons_set:
            print(f"  Warning: Missing amplicons for {generic_set}")
            continue

        # Test forward generic primer against UNINTENDED signals (sense amplicons)
        print(f"  Testing forward generic primer against unintended sense amplicons...")
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(f"    Testing forward generic vs {amplicon.name} (sense - unintended)...")
            
            _test_generic_primer_cross_reactivity(
                results,
                f"generic_forward_{generic_set}",
                forward_generic,
                f"{amplicon.name}_sense",
                amplicon.sequence,  # Sense amplicon is unintended for forward generic
                f"forward_vs_unintended_{amplicon.species}_sense"
            )

        # Test reverse generic primer against UNINTENDED signals (antisense amplicons)  
        print(f"  Testing reverse generic primer against unintended antisense amplicons...")
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(f"    Testing reverse generic vs {amplicon.name} (antisense - unintended)...")
            
            antisense_amplicon = str(Seq(amplicon.sequence).reverse_complement())
            _test_generic_primer_cross_reactivity(
                results,
                f"generic_reverse_{generic_set}",
                reverse_generic,
                f"{amplicon.name}_antisense",
                antisense_amplicon,  # Antisense amplicon is unintended for reverse generic
                f"reverse_vs_unintended_{amplicon.species}_antisense"
            )

    return results


def _test_generic_primer_cross_reactivity(
    results: List[ValidationResult],
    primer_name: str,
    primer_sequence: str, 
    signal_name: str,
    signal_sequence: str,
    test_type: str
) -> None:
    """Helper function to test generic primer cross-reactivity with unintended signals."""
    
    # Set up NUPACK analysis with generic primer concentrations
    primer_concentration_M = TESTING_CONDITIONS['generic_primer_concentration_nM'] * 1e-9  # Convert nM to M
    signal_concentration_M = TESTING_CONDITIONS['generic_signal_concentration_nM'] * 1e-9  # Convert nM to M
    
    sequences = [
        SequenceInput(primer_name, primer_sequence, primer_concentration_M),
        SequenceInput(signal_name, signal_sequence, signal_concentration_M),
    ]

    # Run NUPACK analysis
    nupack_results = analyze_sequence_complexes(
        temperature_celsius=TESTING_CONDITIONS['temperature_C'],
        sequences=sequences,
        sodium_millimolar=80.0,
        magnesium_millimolar=12.0,
        max_complex_size=2,
        base_pairing_analysis=True,
    )

    # Calculate primer-signal binding (should be low for unintended signals)
    primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
        primer_name, 
        signal_name
    )

    # Calculate monomer fraction (should be high - primer stays as monomer)
    monomer_fraction = nupack_results.get_monomer_fraction(primer_name)

    # Calculate 3'-end unpaired probability (should be high - no binding at 3'-end)
    three_prime_unpaired_prob = 1.0  # Default to unpaired
    primer_length = len(primer_sequence)
    
    # Look for any complex containing this primer to get 3'-end info
    for complex_result in nupack_results.complexes:
        if (complex_result.unpaired_probability and
            primer_name in complex_result.sequence_id_map.values()):
            
            primer_seq_id = None
            for seq_id, name in complex_result.sequence_id_map.items():
                if name == primer_name:
                    primer_seq_id = seq_id
                    break
            
            if primer_seq_id is not None:
                # Get unpaired probability for the last two bases
                base1_key = (primer_seq_id, primer_length - 1)
                base2_key = (primer_seq_id, primer_length)
                
                unpaired_prob1 = complex_result.unpaired_probability.get(base1_key, 1.0)
                unpaired_prob2 = complex_result.unpaired_probability.get(base2_key, 1.0)
                
                # Weight by complex concentration and take the maximum unpaired probability
                complex_weight = complex_result.concentration_M
                weighted_unpaired = (unpaired_prob1 * unpaired_prob2) * complex_weight
                three_prime_unpaired_prob = max(three_prime_unpaired_prob, weighted_unpaired)

    # Validation criteria (generic primers should NOT bind significantly to unintended signals)
    low_binding_pass = primer_signal_binding <= VALIDATION_THRESHOLDS['generic_primer_low_cross_binding_max']  # <20% binding
    monomer_pass = monomer_fraction >= VALIDATION_THRESHOLDS['generic_primer_cross_monomer_min']  # >90% monomer
    unpaired_pass = three_prime_unpaired_prob >= VALIDATION_THRESHOLDS['generic_primer_cross_unpaired_min']  # >99.8% unpaired
    
    overall_pass = low_binding_pass and monomer_pass and unpaired_pass

    result = ValidationResult(
        test_name="Test_8_Generic_Primer_Cross_Reactivity",
        primer_name=primer_name,
        target_dais=[signal_name],
        hetero_dimer_fraction=primer_signal_binding,
        monomer_fraction=monomer_fraction,
        unpaired_3_prime_prob=three_prime_unpaired_prob,
        passed=overall_pass,
        details=f"{test_type}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
               f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
               f"3'-unpaired {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
    )
    results.append(result)

    print(f"      {test_type:25s}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
          f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
          f"3'-unpaired {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})")


def test_9_generic_primer_cross_reactivity_within_primer_set(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
) -> List[ValidationResult]:
    """
    Test 9: Generic primer cross-reactivity within primer set.

    Test all 6 primers from same generic set in single tube:
    - 2 generic primers (forward + reverse)
    - 2 N.G NASBA primers (forward + reverse)
    - 2 C.T NASBA primers (forward + reverse)

    Generic primers should not bind to each other significantly.
    Generic primers should not bind to NASBA primers significantly.
    All primers should remain predominantly monomeric.
    
    Uses 250nM concentration for all primers.
    Separate tests for gen5 and gen6 (never mix generic sets).
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 9: GENERIC PRIMER CROSS-REACTIVITY WITHIN PRIMER SET")
    print("=" * 80)

    # Define explicit generic primers
    generic_primers = {
        'gen5': {
            'forward': "TTATGTTCGTGGTT",
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC"
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT", 
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC"
        }
    }

    all_primers = ct_primers + ng_primers

    # Test each generic set separately
    for generic_set in ['gen5', 'gen6']:
        print(f"\nTesting {generic_set} primer set interactions...")
        
        # Collect all 6 primers for this generic set
        nasba_primers_in_set = [p for p in all_primers if p.generic_set == generic_set]
        
        if len(nasba_primers_in_set) != 4:
            print(f"  Warning: Expected 4 NASBA primers for {generic_set}, found {len(nasba_primers_in_set)}")
            continue
        
        # Add the 2 generic primers to make 6 total
        forward_generic = generic_primers[generic_set]['forward']
        reverse_generic = generic_primers[generic_set]['reverse']
        
        print(f"  Testing 6 primers in {generic_set} set:")
        print(f"    - 2 generic primers: forward ({len(forward_generic)} bp), reverse ({len(reverse_generic)} bp)")
        print(f"    - 4 NASBA primers: {[p.name for p in nasba_primers_in_set]}")
        
        # Set up all 6 primers in single NUPACK tube
        primer_concentration_M = TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9  # 250nM
        
        sequences = [
            # Generic primers
            SequenceInput(f"generic_forward_{generic_set}", forward_generic, primer_concentration_M),
            SequenceInput(f"generic_reverse_{generic_set}", reverse_generic, primer_concentration_M),
        ]
        
        # Add NASBA primers
        for primer in nasba_primers_in_set:
            sequences.append(SequenceInput(primer.name, primer.sequence, primer_concentration_M))

        print(f"    Running NUPACK analysis with {len(sequences)} primers...")

        # Run NUPACK analysis with all 6 primers
        nupack_results = analyze_sequence_complexes(
            temperature_celsius=TESTING_CONDITIONS['temperature_C'],
            sequences=sequences,
            sodium_millimolar=80.0,
            magnesium_millimolar=12.0,
            max_complex_size=6,  # Allow up to 6-mer complexes
            base_pairing_analysis=True,
        )

        # Test 1: Generic primers should not bind to each other
        generic_generic_binding = nupack_results.get_hetero_dimer_fraction(
            f"generic_forward_{generic_set}",
            f"generic_reverse_{generic_set}"
        )
        
        generic_generic_pass = generic_generic_binding <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
        
        result = ValidationResult(
            test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
            primer_name=f"{generic_set}_generic_primers",
            target_dais=["generic_generic_interaction"],
            hetero_dimer_fraction=generic_generic_binding,
            passed=generic_generic_pass,
            details=f"Generic-generic binding: {generic_generic_binding:.3f} ({'✓' if generic_generic_pass else '✗'})",
        )
        results.append(result)
        
        print(f"    Generic-generic binding: {generic_generic_binding:.3f} ({'✓' if generic_generic_pass else '✗'})")

        # Test 2: Generic primers should not bind to NASBA primers significantly
        for nasba_primer in nasba_primers_in_set:
            # Test forward generic vs NASBA primer
            forward_nasba_binding = nupack_results.get_hetero_dimer_fraction(
                f"generic_forward_{generic_set}",
                nasba_primer.name
            )
            
            # Test reverse generic vs NASBA primer  
            reverse_nasba_binding = nupack_results.get_hetero_dimer_fraction(
                f"generic_reverse_{generic_set}",
                nasba_primer.name
            )
            
            forward_nasba_pass = forward_nasba_binding <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
            reverse_nasba_pass = reverse_nasba_binding <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
            
            # Forward generic vs NASBA primer result
            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=f"generic_forward_{generic_set}",
                target_dais=[nasba_primer.name],
                hetero_dimer_fraction=forward_nasba_binding,
                passed=forward_nasba_pass,
                details=f"Forward generic vs {nasba_primer.name}: {forward_nasba_binding:.3f} ({'✓' if forward_nasba_pass else '✗'})",
            )
            results.append(result)
            
            # Reverse generic vs NASBA primer result
            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=f"generic_reverse_{generic_set}",
                target_dais=[nasba_primer.name],
                hetero_dimer_fraction=reverse_nasba_binding,
                passed=reverse_nasba_pass,
                details=f"Reverse generic vs {nasba_primer.name}: {reverse_nasba_binding:.3f} ({'✓' if reverse_nasba_pass else '✗'})",
            )
            results.append(result)
            
            print(f"    Forward generic vs {nasba_primer.name}: {forward_nasba_binding:.3f} ({'✓' if forward_nasba_pass else '✗'})")
            print(f"    Reverse generic vs {nasba_primer.name}: {reverse_nasba_binding:.3f} ({'✓' if reverse_nasba_pass else '✗'})")

        # Test 3: All primers should remain predominantly monomeric
        all_primer_names = [f"generic_forward_{generic_set}", f"generic_reverse_{generic_set}"] + [p.name for p in nasba_primers_in_set]
        
        print(f"    Monomer fractions:")
        for primer_name in all_primer_names:
            monomer_fraction = nupack_results.get_monomer_fraction(primer_name)
            monomer_pass = monomer_fraction >= VALIDATION_THRESHOLDS['primer_set_monomer_min']
            
            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=primer_name,
                target_dais=["monomer_fraction"],
                monomer_fraction=monomer_fraction,
                passed=monomer_pass,
                details=f"Monomer fraction: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'})",
            )
            results.append(result)
            
            print(f"      {primer_name:20s}: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'})")

        # Test 4: Generic primers should have high 3'-end unpaired probability
        for generic_type in ['forward', 'reverse']:
            generic_name = f"generic_{generic_type}_{generic_set}"
            generic_sequence = generic_primers[generic_set][generic_type]
            
            # Calculate 3'-end unpaired probability
            three_prime_unpaired_prob = _calculate_generic_3_prime_unpaired(
                nupack_results, generic_name, generic_sequence
            )
            
            unpaired_pass = three_prime_unpaired_prob >= VALIDATION_THRESHOLDS['primer_set_generic_unpaired_min']
            
            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=generic_name,
                target_dais=["3_prime_unpaired"],
                unpaired_3_prime_prob=three_prime_unpaired_prob,
                passed=unpaired_pass,
                details=f"3'-end unpaired: {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)
            
            print(f"    {generic_name} 3'-end unpaired: {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})")

    return results


def _calculate_generic_3_prime_unpaired(
    nupack_results, 
    primer_name: str, 
    primer_sequence: str
) -> float:
    """Helper to calculate 3'-end unpaired probability for generic primer in complex mixture."""
    
    three_prime_unpaired_prob = 1.0  # Default to unpaired
    primer_length = len(primer_sequence)
    
    # Look through all complexes containing this primer
    for complex_result in nupack_results.complexes:
        if (complex_result.unpaired_probability and
            primer_name in complex_result.sequence_id_map.values()):
            
            primer_seq_id = None
            for seq_id, name in complex_result.sequence_id_map.items():
                if name == primer_name:
                    primer_seq_id = seq_id
                    break
            
            if primer_seq_id is not None:
                # Get unpaired probability for the last two bases
                base1_key = (primer_seq_id, primer_length - 1)
                base2_key = (primer_seq_id, primer_length)
                
                unpaired_prob1 = complex_result.unpaired_probability.get(base1_key, 1.0)
                unpaired_prob2 = complex_result.unpaired_probability.get(base2_key, 1.0)
                
                # Weight by complex concentration and take minimum unpaired probability
                complex_weight = complex_result.concentration_M
                weighted_unpaired = (unpaired_prob1 * unpaired_prob2) * complex_weight
                three_prime_unpaired_prob = min(three_prime_unpaired_prob, weighted_unpaired)
    
    return three_prime_unpaired_prob


# ============================================================================
# MAIN VALIDATION WORKFLOW
# ============================================================================


def run_comprehensive_validation() -> Dict:
    """Run all validation tests and return comprehensive results."""

    print("=" * 80)
    print("NASBA PRIMER COMPREHENSIVE VALIDATION FRAMEWORK")
    print("=" * 80)

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

    # Generate base primers and create the best NASBA candidates
    base_primers = get_base_primers()
    
    print(f"\nTesting all three C.T primer pairs: {list(base_primers.keys())}")
    
    # Test each primer pair separately
    all_validation_results = {}
    
    for pair_name, primer_pair in base_primers.items():
        print(f"\n{'='*60}")
        print(f"TESTING {pair_name} PRIMER PAIR")
        print(f"{'='*60}")
        
        ct_primers = []
        
        print(f"Generating C.T NASBA primers for {pair_name}...")

        for primer_type, base_primer in primer_pair.items():
            for generic_set in generic_sets:
                candidates = generate_nasba_primer_candidates(base_primer, generic_set)
                valid_candidates = [c for c in candidates if c.is_valid]

                if valid_candidates:
                    # Take the best candidate (highest Tm score)
                    best_candidate = max(valid_candidates, key=lambda x: x.get_tm_score())
                else:
                    # If no valid candidates, take the best invalid one for testing purposes
                    if candidates:
                        best_candidate = max(candidates, key=lambda x: x.get_tm_score())
                    else:
                        continue

                ct_primer = ValidationPrimer(
                    name=f"{pair_name}-{primer_type[0].upper()}-{generic_set.name}",
                    sequence=best_candidate.full_sequence,
                    anchor_sequence=best_candidate.anchor_sequence,
                    toehold_sequence=best_candidate.toehold_sequence,
                    species="CT",
                    primer_type=primer_type,
                    generic_set=generic_set.name,
                )
                ct_primers.append(ct_primer)

        print(f"Generated {len(ct_primers)} C.T primers for {pair_name}")

        # Generate N.G primers (same for all pairs)
        print(f"Generating N.G primers...")
        ng_primers = generate_ng_primers(generic_sets)
        print(f"Generated {len(ng_primers)} N.G primers")

        # Generate daises
        print(f"Generating daises...")
        ct_daises = generate_daises(ct_primers)
        ng_daises = generate_daises(ng_primers)
        print(f"Generated {len(ct_daises)} C.T daises and {len(ng_daises)} N.G daises")

        # Run validation tests for this primer pair
        pair_results = {}

        # Test 1: Hetero-dimer measurements
        test1_results = test_1_hetero_dimer_measurements(
            ct_primers, ng_primers, ct_daises, ng_daises
        )
        pair_results['test_1'] = test1_results

        # Test 2: Four-primer cross-reactivity
        test2_results = test_2_four_primer_cross_reactivity(ct_primers, ng_primers)
        pair_results['test_2'] = test2_results

        # Test 3: Individual primer-dais binding
        test3_results = test_3_individual_primer_dais_binding(
            ct_primers, ng_primers, ct_daises, ng_daises
        )
        pair_results['test_3'] = test3_results

        # Test 4: Primer-signal binding specificity
        test4_results = test_4_primer_signal_binding_specificity(ct_primers, ng_primers)
        pair_results['test_4'] = test4_results

        # Test 5: Primer cross-reactivity with unintended signals
        test5_results = test_5_primer_cross_reactivity_with_unintended_signals(ct_primers, ng_primers)
        pair_results['test_5'] = test5_results

        # Construct DNA amplicons for this primer pair
        amplicons = construct_dna_amplicons(ct_primers, ng_primers)

        # Test 6: Inter-pathogen amplicon cross-reactivity
        test6_results = test_6_inter_pathogen_amplicon_cross_reactivity(amplicons)
        pair_results['test_6'] = test6_results

        # Test 7: Generic primer-amplicon binding specificity
        test7_results = test_7_generic_primer_amplicon_binding_specificity(ct_primers, ng_primers, amplicons)
        pair_results['test_7'] = test7_results

        # Test 8: Generic primer cross-reactivity with unintended signals
        test8_results = test_8_generic_primer_cross_reactivity_with_unintended_signals(amplicons)
        pair_results['test_8'] = test8_results

        # Test 9: Generic primer cross-reactivity within primer set
        test9_results = test_9_generic_primer_cross_reactivity_within_primer_set(ct_primers, ng_primers)
        pair_results['test_9'] = test9_results

        # Summary for this primer pair
        print(f"\n{'-'*60}")
        print(f"VALIDATION SUMMARY FOR {pair_name}")
        print(f"{'-'*60}")

        for test_name, results in pair_results.items():
            total_tests = len(results)
            passed_tests = sum(1 for r in results if r.passed)

            if total_tests > 0:
                print(
                    f"{test_name.upper().replace('_', ' ')}: {passed_tests}/{total_tests} tests passed "
                    f"({100*passed_tests/total_tests:.1f}%)"
                )
            else:
                print(f"{test_name.upper().replace('_', ' ')}: No tests performed")

        # Store results for this pair
        all_validation_results[pair_name] = {
            'primers': {'CT': ct_primers, 'NG': ng_primers},
            'daises': {'CT': ct_daises, 'NG': ng_daises},
            'results': pair_results
        }

    # Overall summary across all primer pairs
    print("\n" + "=" * 80)
    print("OVERALL VALIDATION SUMMARY ACROSS ALL PRIMER PAIRS")
    print("=" * 80)
    
    total_all_tests = 0
    total_passed = 0
    
    for pair_name, pair_data in all_validation_results.items():
        pair_total_tests = sum(len(results) for results in pair_data['results'].values())
        pair_passed = sum(
            sum(1 for r in results if r.passed) for results in pair_data['results'].values()
        )
        
        total_all_tests += pair_total_tests
        total_passed += pair_passed
        
        print(f"{pair_name}: {pair_passed}/{pair_total_tests} tests passed "
              f"({100*pair_passed/pair_total_tests:.1f}%)" if pair_total_tests > 0 else f"{pair_name}: No tests")

    print(f"\nGRAND TOTAL: {total_passed}/{total_all_tests} tests passed "
          f"({100*total_passed/total_all_tests:.1f}%)" if total_all_tests > 0 else "\nGRAND TOTAL: No tests")

    return {
        'all_pairs': all_validation_results,
        'summary': {
            'total_tests': total_all_tests,
            'passed_tests': total_passed,
            'pass_rate': total_passed / total_all_tests if total_all_tests > 0 else 0.0,
        },
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================


@click.command()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--export-results', type=str, help='Export detailed results to CSV file')
def main(verbose: bool, export_results: Optional[str] = None):
    """
    Main function for NASBA primer validation.

    This tool implements three comprehensive validation tests:

    Test 1: Hetero-dimer fraction measurements
    - Each C.T and N.G primer should bind strongly to its corresponding dais

    Test 2: Four-primer cross-reactivity analysis
    - All four primers (CT-F, CT-R, NG-F, NG-R) per generic set should remain monomers (>90%)
    - 3'-end should remain unpaired (>99.8%) in presence of other primers

    Test 3: Individual primer-dais binding specificity
    - Each primer tested against three non-target daises should remain monomer (>90%)
    - 3'-end should remain unpaired (>99.8%) with non-target daises

    Total: 8 individual tests per generic set (2 tests) = 16 tests for Test 3
    """

    # Run comprehensive validation
    validation_data = run_comprehensive_validation()

    # Export results if requested
    if export_results:
        print(f"\nExporting results to {export_results}...")
        # Implementation would write CSV with detailed results
        print("CSV export not yet implemented")

    return 0


if __name__ == "__main__":
    exit(main())
