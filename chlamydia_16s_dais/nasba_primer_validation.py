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
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import click
from Bio.Seq import Seq
from Bio import SeqIO
import pandas as pd


# Import from the main NASBA primer module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chlamydia_nasba_primer import (
    get_base_primers,
    GenericPrimerSet,
    generate_nasba_primer_candidates,
)
from nasba_primer_thermodynamics import (
    calculate_primer_bound_fractions,
    validate_bound_fractions,
    analyze_multi_primer_solution,
    analyze_sequence_comprehensive,
    DEFAULT_ASSAY_CONCENTRATIONS,
    NASBA_TEMPERATURE_CELSIUS,
    NASBA_SODIUM_MOLAR,
    NASBA_MAGNESIUM_MOLAR,
    NASBA_PRIMER_CONCENTRATION_MOLAR, calculate_weighted_three_prime_end_paired_probabilities, NASBA_CONDITIONS,
)
from nupack_complex_analysis import (
    SequenceInput,
    ComplexAnalysisResult,
    ComplexResult,
)
from nupack_subprocess import (
    analyze_sequence_complexes_subprocess,
    SequenceParam,
)


# Subprocess-backed NUPACK analysis wrapper used by this module
def analyze_sequence_complexes(
    temperature_celsius: float,
    sequences: List[SequenceInput],
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
    max_complex_size: int = 4,
    base_pairing_analysis: bool = False,
) -> ComplexAnalysisResult:
    # Build worker payload sequences
    seq_params = [
        SequenceParam(
            name=s.name,
            sequence=s.sequence,
            concentration_M=s.concentration_M,
        )
        for s in sequences
    ]

    # Run analysis in a fresh subprocess
    result = analyze_sequence_complexes_subprocess(
        temperature_celsius=temperature_celsius,
        sequences=seq_params,
        sodium_millimolar=sodium_millimolar,
        magnesium_millimolar=magnesium_millimolar,
        max_complex_size=max_complex_size,
        base_pairing_analysis=base_pairing_analysis,
    )

    # Rehydrate ComplexResult objects
    complexes: List[ComplexResult] = []
    for c in result.get("complexes", []):
        seq_id_map = {int(k): v for k, v in (c.get("sequence_id_map") or {}).items()}

        unpaired = None
        if c.get("unpaired_probability") is not None:
            unpaired = {}
            for k, v in c["unpaired_probability"].items():
                seq_str, base_str = k.split(":")
                unpaired[(int(seq_str), int(base_str))] = float(v)

        pairing = None
        if c.get("pairing_probability") is not None:
            pairing = {}
            for k, v in c["pairing_probability"].items():
                left, right = k.split("|")
                s1, b1 = left.split(":")
                s2, b2 = right.split(":")
                pairing[(int(s1), int(b1), int(s2), int(b2))] = float(v)

        complexes.append(
            ComplexResult(
                complex_id=str(c["complex_id"]),
                size=int(c["size"]),
                concentration_molar=float(c["concentration_molar"]),
                sequence_id_map=seq_id_map,
                unpaired_probability=unpaired,
                pairing_probability=pairing,
            )
        )

    return ComplexAnalysisResult(
        temperature_celsius=float(result["temperature_celsius"]),
        ionic_conditions={
            "sodium_mM": float(result["ionic_conditions"]["sodium_mM"]),
            "magnesium_mM": float(result["ionic_conditions"]["magnesium_mM"]),
        },
        max_complex_size=int(result["max_complex_size"]),
        total_sequences=int(result["total_sequences"]),
        complexes=complexes,
    )


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================


def filter_candidates_by_bound_fraction(candidates: List) -> tuple[List, List]:
    """
    Filter NASBA primer candidates based on bound-fraction criteria.

    Args:
        candidates: List of NASBAPrimerCandidate objects

    Returns:
        Tuple of (valid_candidates, all_candidates_with_scores)
    """
    valid_candidates = []
    all_candidates_with_scores = []

    print(f"\nEvaluating {len(candidates)} candidates with bound-fraction criteria...")

    for candidate in candidates:
        # Calculate bound fractions for anchor and toehold
        anchor_bf, toehold_bf = calculate_primer_bound_fractions(
            anchor_sequence=candidate.anchor_sequence,
            toehold_sequence=candidate.toehold_sequence,
        )

        # Validate bound fractions
        bf_result = validate_bound_fractions(anchor_bf, toehold_bf)

        # Add bound fraction info to the candidate
        candidate.anchor_bound_fraction = anchor_bf
        candidate.toehold_bound_fraction = toehold_bf
        candidate.bound_fraction_score = bf_result.score
        candidate.is_valid = bf_result.is_valid

        all_candidates_with_scores.append(candidate)

        if bf_result.is_valid:
            valid_candidates.append(candidate)
            print(
                f"  ✓ Valid: Anchor={anchor_bf:.3f}, Toehold={toehold_bf:.3f}, Score={bf_result.score:.3f}"
            )
        else:
            print(
                f"  ✗ Invalid: Anchor={anchor_bf:.3f}, Toehold={toehold_bf:.3f}, Score={bf_result.score:.3f}"
            )

    print(
        f"Found {len(valid_candidates)} valid candidates out of {len(candidates)} total"
    )

    return valid_candidates, all_candidates_with_scores


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================


# Load canonical sequences
def load_canonical_sequences() -> Dict[str, str]:
    """Load canonical sequences from FASTA files."""
    sequences = {}

    # Load C.T canonical
    ct_fasta_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', "twist_ct_16s_canonical.fasta"
    )
    if os.path.exists(ct_fasta_path):
        with open(ct_fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences['CT'] = str(record.seq)
                break
    else:
        raise FileNotFoundError(
            f"C.T canonical sequence file not found: {ct_fasta_path}"
        )

    # Load N.G canonical
    ng_fasta_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', "twist_ng_16s_canonical.fasta"
    )
    if os.path.exists(ng_fasta_path):
        with open(ng_fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences['NG'] = str(record.seq)
                break
    else:
        raise FileNotFoundError(
            f"N.G canonical sequence file not found: {ng_fasta_path}"
        )

    return sequences


CANONICAL_SEQUENCES = load_canonical_sequences()

# Test thresholds
VALIDATION_THRESHOLDS = {
    'correct_dais_primer_dimer_min': 0.90,  # >90% hetero-dimer formation expected
    'primer_monomer_vs_other_primers': 0.90,  # >90% monomer in presence of other primers
    'primer_monomer_vs_wrong_daises': 0.90,  # >90% monomer with incorrect daises
    'primer_unpaired_3_prime_min': 0.5,  # Minimum probability of 3'-end unpaired
    'primer_signal_binding_min': 0.80,  # Minimum primer-signal binding expected
    'primer_3_prime_binding_min': 0.90,  # >90% 3'-end binding in primer-signal dimer
    'generic_primer_amplicon_binding_min': 0.80,  # Minimum generic primer-amplicon binding expected
    'generic_primer_3_prime_binding_min': 0.90,  # >90% 3'-end binding in generic primer-amplicon dimer
    'generic_primer_low_cross_binding_max': 0.20,  # Maximum binding to unintended signals
    'generic_primer_cross_monomer_min': 0.90,  # >90% monomer with unintended signals
    'generic_primer_cross_unpaired_min': 0.998,  # Minimum 3'-end unpaired with unintended signals
    'primer_set_low_interaction_max': 0.20,  # Maximum interaction between primers in the same set
    'primer_set_monomer_min': 0.90,  # >90% monomer for all primers in a mixed set
    'primer_set_generic_unpaired_min': 0.998,  # Minimum 3'-end unpaired for generic primers in a set
}

# NASBA primer testing conditions - using centralized constants
TESTING_CONDITIONS = {
    'temperature_C': NASBA_TEMPERATURE_CELSIUS,
    'primer_concentration_nM': NASBA_PRIMER_CONCENTRATION_MOLAR * 1e9,  # Convert M to nM
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


@dataclass
class ValidationResult:
    """Results from primer validation tests."""

    test_name: str
    primer_name: str
    target_dais: List[str]
    hetero_dimer_fraction: Optional[float] = None
    monomer_fraction: Optional[float] = None
    unpaired_3_prime_prob: Optional[float] = None
    unpaired_3_prime_probs: Optional[Tuple[float, ...]] = None
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
            'anchor': "CGGGCTCAACCCGGGAACTGC",  # 21 bp anchor # noqa: typo
            'toehold': "GTTCTGAACTGGG",  # 14 bp toehold # noqa: typo
        },
        'reverse': {
            'anchor': "TTGCGACCGTACTCCCCAGGC",  # 20 bp anchor # noqa: typo
            'toehold': "GGTCAATTTCACGC",  # 14 bp toehold # noqa: typo
        },
    }

    # Expected d4 dais sequences (should be reverse complements of anchors)
    expected_d4_daises = {
        'forward': "GCAGTTCCCGGGTTGAGCCCG",  # Forward dais (21 bp) # noqa: typo
        'reverse': "GCCTGGGGAGTACGGTCGCAA",  # Reverse dais (20 bp) # noqa: typo
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

    # Verify anchors and toeholds are found in the canonical sequence
    for primer_type in ['forward', 'reverse']:
        anchor = ng_components[primer_type]['anchor']
        toehold = ng_components[primer_type]['toehold']

        if primer_type == 'forward':
            # Forward anchor should be found directly in canonical
            anchor_pos = ng_canonical.find(anchor)
            if anchor_pos == -1:
                raise ValueError(
                    f"N.G forward anchor {anchor} not found in canonical sequence"
                )

            # Forward toehold should be 3' to anchor
            expected_toehold_pos = anchor_pos + len(anchor)
            if expected_toehold_pos + len(toehold) > len(ng_canonical):
                raise ValueError(
                    f"N.G forward toehold extends beyond canonical sequence"
                )

            canonical_toehold = ng_canonical[
                expected_toehold_pos : expected_toehold_pos + len(toehold)
            ]
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
                raise ValueError(
                    f"N.G reverse anchor RC {anchor_rc} not found in canonical sequence"
                )

            # Reverse toehold RC should be 5' to anchor RC position (reverse binds at first base)
            toehold_rc = str(Seq(toehold).reverse_complement())
            expected_toehold_pos = anchor_rc_pos - len(toehold_rc)
            if expected_toehold_pos < 0:
                raise ValueError(
                    f"N.G reverse toehold extends before start of canonical sequence"
                )

            canonical_toehold = ng_canonical[expected_toehold_pos:anchor_rc_pos]
            if canonical_toehold != toehold_rc:
                raise ValueError(
                    f"N.G reverse toehold RC mismatch: expected {toehold_rc}, "
                    f"found {canonical_toehold} at position {expected_toehold_pos}"
                )

    print(f"✓ Verified all N.G components found in canonical sequence")

    # Generate NASBA primers for each generic set
    for generic_set in generic_sets:
        # Forward primer
        forward_anchor = ng_components['forward']['anchor']
        forward_toehold = ng_components['forward']['toehold']
        forward_sequence = (
            generic_set.forward_generic + forward_anchor + forward_toehold
        )

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
    """Generate unique daises from primers (reverse complement of anchor sequence)."""
    # Use a dictionary to deduplicate by (species, primer_type)
    # Since DAIS are generic set independent, we only need one per species+primer_type
    unique_daises = {}

    for primer in primers:
        # Create a key for deduplication
        key = (primer.species, primer.primer_type)

        if key not in unique_daises:
            # Dais is the reverse complement of the anchor sequence
            dais_sequence = str(Seq(primer.anchor_sequence).reverse_complement())

            # Create a generic name without a generic set identifier
            dais_name = f"dais-{primer.species}-{primer.primer_type}"

            dais = Dais(
                name=dais_name,
                sequence=dais_sequence,
                species=primer.species,
                primer_type=primer.primer_type,
            )

            unique_daises[key] = dais

    return list(unique_daises.values())


# ============================================================================
# THERMODYNAMIC CALCULATIONS
# ============================================================================


def calculate_dimer_formation_probability_with_assay_concentrations(
    seq1: str,
    seq2: str,
    temperature_celsius: float = NASBA_CONDITIONS['target_temp_C'],
    assay_type: str = "primer_dais_binding",
) -> float:
    """
    Calculate the probability of dimer formation between two sequences using NUPACK.

    This function determines the appropriate concentrations based on assay type and
    calls the NUPACK-based calculation from nasba_primer_thermodynamics.

    Args:
        seq1: First sequence (typically primer)
        seq2: Second sequence (typically target/dais/other)
        temperature_celsius: Temperature for calculation
        assay_type: Type of assay to determine concentrations:
            - "primer_dais_binding": 250nM both (intended binding)
            - "cross_reactivity": 250nM primer, 10pM other sequence
            - "off_target_dais": 250nM primer, 250nM non-intended dais
            - "amplicon_binding": 100nM both amplicon strands
            - "generic_amplicon": 250nM generic primer, 50nM amplicon sense

    Returns:
        Dimer formation probability (0.0 to 1.0)
    """
    # Get appropriate concentrations for assay type
    concentrations = DEFAULT_ASSAY_CONCENTRATIONS

    if assay_type == "primer_dais_binding":
        seq1_conc = concentrations.primer_dais_binding['primer_concentration_M']
        seq2_conc = concentrations.primer_dais_binding['target_dais_concentration_M']
    elif assay_type == "cross_reactivity":
        seq1_conc = concentrations.primer_non_signal_cross_reactivity['primer_concentration_M']
        seq2_conc = concentrations.primer_non_signal_cross_reactivity['other_sequence_concentration_M']
    elif assay_type == "off_target_dais":
        seq1_conc = concentrations.off_target_dais['primer_concentration_M']
        seq2_conc = concentrations.off_target_dais['non_intended_dais_concentration_M']
    elif assay_type == "amplicon_binding":
        seq1_conc = concentrations.amplicon_strand_binding[
            'amplicon_strand1_concentration_M'
        ]
        seq2_conc = concentrations.amplicon_strand_binding[
            'amplicon_strand2_concentration_M'
        ]
    elif assay_type == "generic_amplicon":
        seq1_conc = concentrations.generic_primer_amplicon[
            'generic_primer_concentration_M'
        ]
        seq2_conc = concentrations.generic_primer_amplicon[
            'amplicon_sense_concentration_M'
        ]
    else:
        # Default to primer-dais binding
        seq1_conc = concentrations.primer_dais_binding['primer_concentration_M']
        seq2_conc = concentrations.primer_dais_binding['target_dais_concentration_M']

    # Use NUPACK-based calculation
    from nasba_primer_thermodynamics import calculate_dimer_formation_probability

    return calculate_dimer_formation_probability(
        sequence1=seq1,
        sequence2=seq2,
        sequence1_concentration_molar=seq1_conc,
        sequence2_concentration_molar=seq2_conc,
        temp_celsius=temperature_celsius,
        sodium_molar=NASBA_SODIUM_MOLAR,
        magnesium_molar=NASBA_MAGNESIUM_MOLAR,
    )


def calculate_3_prime_unpaired_probability(
    primer_seq: str, competing_sequences: List[str]
) -> tuple[float, float, tuple[float, ...]]:
    """
    Calculate the concentration-weighted probability that the 3'-end bases of primer remain unpaired.

    This function uses NUPACK multi-strand analysis to determine the thermodynamically correct
    unpaired probability of the last 2 bases of the primer sequence in the presence of competing sequences.
    All sequences are analyzed together in a single NUPACK calculation, and probabilities are weighted
    by actual complex concentrations.

    Args:
        primer_seq: Primer sequence
        competing_sequences: List of competing sequences

    Returns:
        Concentration-weighted average unpaired probability for the last 2 bases (0.0 to 1.0)
    """
    # Validate inputs - be strict rather than using fallbacks
    if len(primer_seq) < 2:
        raise ValueError(
            f"Primer sequence too short ({len(primer_seq)} bases). Need at least 2 bases for 3'-end analysis."
        )

    if not competing_sequences:
        raise ValueError(
            "No competing sequences provided. Cannot calculate unpaired probability without competition."
        )

    comprehensive_result = analyze_sequence_comprehensive(
        primary_sequence=primer_seq,
        primary_sequence_name='target_primer',
        primary_sequence_concentration=TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9,
        other_sequences={
            f'competing_{i}': competing_sequences[i]
            for i in range(len(competing_sequences))
        },
        other_sequence_concentrations={
            f'competing{i}': TESTING_CONDITIONS['competition_concentration_nM'] * 1e-9
            for i in range(len(competing_sequences))
        },
        temp_celsius=NASBA_CONDITIONS['target_temp_C'],
        n_bases=3,
    )
    return (
        comprehensive_result.primary_monomer_fraction,
        comprehensive_result.weighted_three_prime_unpaired_prob,
        comprehensive_result.weighted_three_prime_unpaired_probs,
    )

# ============================================================================
# VALIDATION TESTS
# ============================================================================

# TODO: Additional validation tests to be implemented:
#
# TEST 4: NASBA primer-signal binding specificity
# - Each NASBA primer should meet minimum binding thresholds with its intended signal
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
# - Each generic primer should meet minimum binding thresholds with its designated signal
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

    Each C.T primer should meet minimum binding thresholds with its corresponding dais.
    Each N.G primer should meet minimum binding thresholds with its corresponding dais.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 1: HETERO-DIMER FRACTION MEASUREMENTS")
    print("=" * 80)

    # Test all primers with their corresponding daises
    all_primers = ct_primers + ng_primers
    all_daises = ct_daises + ng_daises

    for primer in all_primers:
        # Find corresponding dais
        target_dais = None
        for dais in all_daises:
            if (
                dais.species == primer.species
                and dais.primer_type == primer.primer_type
            ):
                target_dais = dais
                break

        if not target_dais:
            raise ValueError(
                f"No matching DAIS found for primer {primer.name} "
                f"(species: {primer.species}, primer_type: {primer.primer_type})"
            )

        hetero_dimer_fraction = (
            calculate_dimer_formation_probability_with_assay_concentrations(
                primer.sequence,
                target_dais.sequence,
                NASBA_CONDITIONS['target_temp_C'],
                assay_type="primer_dais_binding",
            )
        )

        passed = (
            hetero_dimer_fraction
            >= VALIDATION_THRESHOLDS['correct_dais_primer_dimer_min']
        )

        result = ValidationResult(
            test_name="Test_1_Hetero_Dimer",
            primer_name=primer.name,
            target_dais=[target_dais.name],
            hetero_dimer_fraction=hetero_dimer_fraction,
            passed=passed,
            details=f"{primer.species} primer vs its dais: {hetero_dimer_fraction:.3f} ({'PASS' if passed else 'FAIL'})",
        )
        results.append(result)

        if not passed:
            print(
                f"Warning: {primer.name} -> {target_dais.name} hetero-dimer fraction too low: "
                f"{hetero_dimer_fraction:.3f} < {VALIDATION_THRESHOLDS['correct_dais_primer_dimer_min']:.2f}"
            )

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
    for each generic set in a single tube. All primers should meet monomer and
    3'-end unpaired probability thresholds.
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
                f"Error: Expected 4 primers for {generic_set}, found {len(primers_in_tube)}"
            )
            raise ValueError(
                f"Expected 4 primers for {generic_set}, found {len(primers_in_tube)}"
            )

        print(f"\nTesting {generic_set} primer set:")

        # Efficient analysis: Run NUPACK once for all 4 primers together
        primer_sequences = [p.sequence for p in primers_in_tube]
        primer_names = [p.name for p in primers_in_tube]
        primer_concentrations = [NASBA_PRIMER_CONCENTRATION_MOLAR] * len(
            primers_in_tube
        )

        print(
            f"    Running single NUPACK analysis for all {len(primers_in_tube)} primers..."
        )

        try:
            # Single efficient analysis for all primers
            analysis_results = analyze_multi_primer_solution(
                primer_sequences=primer_sequences,
                primer_names=primer_names,
                primer_concentrations=primer_concentrations,
                n_bases=2,
                temp_celsius=NASBA_CONDITIONS['target_temp_C'],
                sodium_molar=NASBA_CONDITIONS['Na_mM'] / 1e3,
                magnesium_molar=NASBA_CONDITIONS['Mg_mM'] / 1e3,
            )

            # Process results for each primer
            for target_primer in primers_in_tube:
                primer_result = analysis_results[target_primer.name]

                monomer_fraction = primer_result["monomer_fraction"]
                unpaired_3_prime_prob = primer_result["weighted_unpaired_prob"]
                unpaired_3_prime_probs = primer_result["weighted_unpaired_probs"]

                # Check if both criteria are met
                monomer_pass = (
                    monomer_fraction
                    >= VALIDATION_THRESHOLDS['primer_monomer_vs_other_primers']
                )
                unpaired_pass = (
                    unpaired_3_prime_prob
                    >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']
                )
                overall_pass = monomer_pass and unpaired_pass

                other_primers = [p for p in primers_in_tube if p != target_primer]

                result = ValidationResult(
                    test_name="Test_2_Cross_Reactivity",
                    primer_name=target_primer.name,
                    target_dais=[p.name for p in other_primers],
                    monomer_fraction=monomer_fraction,
                    unpaired_3_prime_prob=unpaired_3_prime_prob,
                    unpaired_3_prime_probs=unpaired_3_prime_probs,
                    passed=overall_pass,
                    details=f"Monomer: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                    f"3'-unpaired: {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
                )
                results.append(result)

                print(
                    f"  {target_primer.name:15s}: Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                    f"3'-unpaired {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
                )

        except Exception as e:
            print(f"Error analyzing {generic_set} primer set: {e}")
            raise RuntimeError(
                f"Failed to analyze {generic_set} primer set due to: {e}"
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
    the three daises it should NOT bind to. Primer should meet monomer and
    3'-end unpaired probability thresholds.
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
        # 3. The dais from the same species, the same primer_type but opposite generic_set
        for dais in all_daises:
            should_not_bind = False

            # Case 1: Different species
            if dais.species != primer.species:
                should_not_bind = True

            # Case 2: Same species, different primer_type
            elif (
                dais.species == primer.species
                and dais.primer_type != primer.primer_type
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
                print(f"  - {dais.name} ({dais.species}, {dais.primer_type})")

        if non_target_daises:
            non_target_sequences = [d.sequence for d in non_target_daises]
            non_target_names = [d.name for d in non_target_daises]

            # Calculate both monomer fraction and 3'-end unpaired probability
            # from a single multi-sequence NUPACK analysis
            monomer_fraction, unpaired_3_prime_prob, unpaired_3_prime_probs = (
                analyze_sequence_comprehensive(
                    primary_sequence=primer.sequence,
                    primary_sequence_name=f'primer_{primer.name}',
                    primary_sequence_concentration=TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9,
                    other_sequences={
                        dais.name: dais.sequence for dais in non_target_sequences
                    },
                    other_sequence_concentrations={
                        dais.name: TESTING_CONDITIONS['dais_concentration_nM'] * 1e-9
                        for dais in non_target_sequences
                    },
                    temp_celsius=NASBA_CONDITIONS['target_temp_C'],
                    n_bases=3,
                )
            )


            # Check if both criteria are met
            monomer_pass = (
                monomer_fraction
                >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']
            )
            unpaired_pass = (
                unpaired_3_prime_prob
                >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']
            )
            overall_pass = monomer_pass and unpaired_pass

            result = ValidationResult(
                test_name="Test_3_Individual_Binding",
                primer_name=primer.name,
                target_dais=non_target_names,
                monomer_fraction=monomer_fraction,
                unpaired_3_prime_prob=unpaired_3_prime_prob,
                unpaired_3_prime_probs=unpaired_3_prime_probs,
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

    Each NASBA primer should meet minimum binding thresholds with its intended signal.
    Within the primer-signal dimer, the 2 3'-end bases of primer should be >90% bound.

    Signal construction:
    - Forward primers: Optimal signal starts with reverse primer (generic + anchor + toehold)
      as RC, then continues with RC of the canonical sequence to the end
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

        # Extract the generic + anchor + toehold part from reverse primer.
        # Reverse primer structure: T7_parts + generic + anchor + toehold
        # We need: generic + anchor + toehold (the part that binds to canonical)
        full_reverse_seq = reverse_primer.sequence
        binding_part = reverse_primer.anchor_sequence + reverse_primer.toehold_sequence

        # Find where the binding part starts in the full reverse primer
        binding_start = full_reverse_seq.find(binding_part)
        if binding_start == -1:
            raise ValueError(
                f"Could not locate binding part in reverse primer {reverse_primer.name}"
            )

        # Extract everything from the binding start (generic + anchor + toehold)
        reverse_binding_portion = full_reverse_seq[binding_start:]

        # Construct optimal forward signal
        optimal_signal = str(Seq(reverse_binding_portion).reverse_complement()) + str(
            Seq(canonical).reverse_complement()
        )

        return optimal_signal

    all_primers = ct_primers + ng_primers

    # Pre-construct forward signals for both species
    forward_signals = {
        'CT': construct_forward_signal('CT', all_primers),
        'NG': construct_forward_signal('NG', all_primers),
    }

    # Define all signals
    signals = {
        'CT': {
            'forward': forward_signals['CT'],
            'reverse': ct_canonical,  # Forward strand for reverse primers
        },
        'NG': {
            'forward': forward_signals['NG'],
            'reverse': ng_canonical,  # Forward strand for reverse primers
        },
    }

    for primer in all_primers:
        # Get the intended signal for this primer
        intended_signal = signals[primer.species][primer.primer_type]

        print(f"\nTesting {primer.name} binding to its intended signal...")
        print(f"  Signal length: {len(intended_signal)} bp")

        # Set up NUPACK analysis with appropriate concentrations
        primer_concentration_molar = (
            TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9
        )  # Convert nM to M
        signal_concentration_molar = (
            TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12
        )  # Convert pM to M

        primer_seq_name = f'primer_{primer.name}'
        signal_seq_name = f'signal_{primer.species}_{primer.primer_type}'

        sequences = [
            SequenceInput(
                primer_seq_name, primer.sequence, primer_concentration_molar
            ),
            SequenceInput(
                signal_seq_name,
                intended_signal,
                signal_concentration_molar,
            ),
        ]

        # Run NUPACK analysis with base-pairing to get 3'-end binding info
        nupack_results = analyze_sequence_complexes(
            temperature_celsius=NASBA_CONDITIONS['target_temp_C'],
            sequences=sequences,
            sodium_millimolar=80.0,  # NASBA conditions
            magnesium_millimolar=12.0,  # NASBA conditions
            max_complex_size=2,
            base_pairing_analysis=True,
        )

        # Calculate primer-signal hetero-dimer fraction
        primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
            seq1_name=primer_seq_name,
            seq2_name=signal_seq_name,
            seq1_input_conc_molar=primer_concentration_molar,
            seq2_input_conc_molar=signal_concentration_molar,
        )

        # find the dimer complex
        dimer_complex = None
        for complex_result in nupack_results.complexes:
            complex_strands = complex_result.sequence_id_map.values()
            if len(complex_strands) != 2:
                continue
            if primer_seq_name not in complex_strands:
                continue
            if signal_seq_name not in complex_strands:
                continue
            dimer_complex = complex_result
            break

        if not dimer_complex:
            raise ValueError(f"Could not find dimer complex for {primer.name}")

        three_prime_binding_prob, three_prime_binding_probs = calculate_weighted_three_prime_end_paired_probabilities(
            sequence_name=f"primer_{primer.name}",
            other_sequence_name=f"signal_{primer.species}_{primer.primer_type}",
            sequence=primer.sequence,
            other_sequence=intended_signal,
            sequence_concentration_molar=primer_concentration_molar,
            other_sequence_concentration_molar=signal_concentration_molar,
            dimer_concentration=dimer_complex.concentration_molar,
            dimer_unpaired_probabilities=dimer_complex.unpaired_probability,
            dimer_id_map=dimer_complex.sequence_id_map,
            n_bases=3,
        )

        # Check validation criteria
        binding_pass = (
            primer_signal_binding >= VALIDATION_THRESHOLDS['primer_signal_binding_min']
        )
        three_prime_pass = (
            three_prime_binding_prob
            >= VALIDATION_THRESHOLDS['primer_3_prime_binding_min']
        )
        overall_pass = binding_pass and three_prime_pass

        result = ValidationResult(
            test_name="Test_4_Primer_Signal_Binding",
            primer_name=primer.name,
            target_dais=[f"signal_{primer.species}_{primer.primer_type}"],
            hetero_dimer_fraction=primer_signal_binding,
            unpaired_3_prime_prob=(
                1.0 - three_prime_binding_prob
            ),  # Store as unpaired for consistency
            unpaired_3_prime_probs=tuple(1 - p for p in three_prime_binding_probs),
            passed=overall_pass,
            details=f"Signal binding: {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
            f"3'-end binding: {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})",
        )
        results.append(result)

        print(
            f"  {primer.name:20s}: Signal binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
            f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})"
        )

    return results


def test_5_primer_cross_reactivity_with_unintended_signals(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
) -> List[ValidationResult]:
    """
    Test 5: NASBA primer cross-reactivity with unintended signals.

    Each primer should NOT bind significantly to three unintended signals:
    1. Forward signal of the other pathogen (C.T vs. N.G)
    2. Reverse signal of the other pathogen (C.T vs. N.G)
    3. Wrong orientation signal of the same pathogen:
       - Forward primers: should not bind to canonical (forward) signal
       - Reverse primers: should not bind to reverse complement of canonical

    Requires multiple NUPACK runs (cannot place signal and its RC in the same tube).
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
            raise ValueError(
                f"Could not locate binding part in reverse primer {reverse_primer.name}"
            )

        reverse_binding_portion = full_reverse_seq[binding_start:]
        optimal_signal = str(Seq(reverse_binding_portion).reverse_complement()) + str(
            Seq(canonical).reverse_complement()
        )

        return optimal_signal

    # Pre-construct all signals
    signals = {
        'CT': {
            'forward': construct_forward_signal('CT', all_primers),
            'reverse': ct_canonical,
        },
        'NG': {
            'forward': construct_forward_signal('NG', all_primers),
            'reverse': ng_canonical,
        },
    }

    # Define wrong orientation signals for same pathogen
    wrong_orientation_signals = {
        'CT': {
            'forward': ct_canonical,  # Forward primers should not bind to canonical (forward)
            'reverse': str(
                Seq(ct_canonical).reverse_complement()
            ),  # Reverse primers should not bind to RC
        },
        'NG': {
            'forward': ng_canonical,  # Forward primers should not bind to canonical (forward)
            'reverse': str(
                Seq(ng_canonical).reverse_complement()
            ),  # Reverse primers should not bind to RC
        },
    }

    for primer in all_primers:
        print(f"\nTesting {primer.name} cross-reactivity with unintended signals...")

        # Define the three unintended signals for this primer
        other_species = 'NG' if primer.species == 'CT' else 'CT'

        unintended_signals = [
            # 1. Forward signal of the other pathogen
            (f"other_pathogen_forward", signals[other_species]['forward']),
            # 2. Reverse signal of the other pathogen
            (f"other_pathogen_reverse", signals[other_species]['reverse']),
            # 3. Wrong orientation signal of the same pathogen
            (
                f"wrong_orientation",
                wrong_orientation_signals[primer.species][primer.primer_type],
            ),
        ]

        # Test each unintended signal separately (cannot mix sense/antisense in same tube)
        for signal_name, signal_sequence in unintended_signals:
            print(
                f"  Testing against {signal_name} (length: {len(signal_sequence)} bp)"
            )

            primer_seq_name = f'primer_{primer.name}'

            comprehensive_result = analyze_sequence_comprehensive(
                primary_sequence=primer.sequence,
                primary_sequence_name=primer_seq_name,
                primary_sequence_concentration=TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9,
                other_sequences={ signal_name: signal_sequence},
                other_sequence_concentrations={ signal_name: TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12 },
                temp_celsius=NASBA_CONDITIONS['target_temp_C'],
                n_bases=3,
            )

            monomer_fraction = comprehensive_result.primary_monomer_fraction
            primer_signal_binding = comprehensive_result.dimer_fraction[signal_name]
            unpaired_3_prime_prob = comprehensive_result.weighted_three_prime_unpaired_prob
            unpaired_3_prime_probs = comprehensive_result.weighted_three_prime_unpaired_probs

            # Validation criteria (primers should NOT bind significantly to unintended signals)
            low_binding_pass = primer_signal_binding <= (
                1.0 - VALIDATION_THRESHOLDS['primer_signal_binding_min']
            )
            monomer_pass = (
                monomer_fraction
                >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']
            )  # >90% monomer
            unpaired_pass = (
                unpaired_3_prime_prob
                >= VALIDATION_THRESHOLDS['primer_unpaired_3_prime_min']
            )

            overall_pass = low_binding_pass and monomer_pass and unpaired_pass

            result = ValidationResult(
                test_name="Test_5_Unintended_Signal_Cross_Reactivity",
                primer_name=primer.name,
                target_dais=[signal_name],
                hetero_dimer_fraction=primer_signal_binding,
                monomer_fraction=monomer_fraction,
                unpaired_3_prime_prob=unpaired_3_prime_prob,
                unpaired_3_prime_probs=unpaired_3_prime_probs,
                passed=overall_pass,
                details=f"{signal_name}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
                f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)

            print(
                f"    {signal_name:20s}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
                f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
                f"3'-unpaired {unpaired_3_prime_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
            )

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
            forward_anchor_toehold = (
                forward_primer.anchor_sequence + forward_primer.toehold_sequence
            )
            forward_generic_start = forward_full_seq.find(forward_anchor_toehold)

            if forward_generic_start == -1:
                raise ValueError(
                    f"Cannot locate anchor+toehold in forward primer {forward_primer.name}"
                )

            forward_generic_part = forward_full_seq[:forward_generic_start]

            # 2. Forward primer anchor + toehold (already have this)

            # 3. Canonical sequence segment between primer binding sites
            # Find where forward primer binds (anchor + toehold should be in canonical)
            forward_bind_pos = canonical.find(forward_anchor_toehold)
            if forward_bind_pos == -1:
                raise ValueError(
                    f"Forward primer anchor+toehold not found in {species} canonical sequence"
                )

            # Find where reverse primer binds (reverse complement of anchor + toehold should be in canonical)
            reverse_anchor_toehold = (
                reverse_primer.anchor_sequence + reverse_primer.toehold_sequence
            )
            reverse_anchor_toehold_rc = str(
                Seq(reverse_anchor_toehold).reverse_complement()
            )
            reverse_bind_pos = canonical.find(reverse_anchor_toehold_rc)
            if reverse_bind_pos == -1:
                raise ValueError(
                    f"Reverse primer anchor+toehold RC not found in {species} canonical sequence"
                )

            # Extract canonical segment between primer binding sites
            forward_end = forward_bind_pos + len(forward_anchor_toehold)
            canonical_segment = canonical[forward_end:reverse_bind_pos]

            # 4. Reverse complement of (reverse primer anchor + toehold) - already have this

            # 5. Reverse complement of reverse primer non-complementary parts
            reverse_full_seq = reverse_primer.sequence
            reverse_binding_start = reverse_full_seq.find(reverse_anchor_toehold)
            if reverse_binding_start == -1:
                raise ValueError(
                    f"Cannot locate anchor+toehold in reverse primer {reverse_primer.name}"
                )

            reverse_noncomplementary_part = reverse_full_seq[:reverse_binding_start]
            reverse_noncomplementary_rc = str(
                Seq(reverse_noncomplementary_part).reverse_complement()
            )

            # Construct complete amplicon
            amplicon_sequence = (
                forward_generic_part
                + forward_anchor_toehold
                + canonical_segment
                + reverse_anchor_toehold_rc
                + reverse_noncomplementary_rc
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
            print(
                f"      Reverse anchor+toehold RC: {len(reverse_anchor_toehold_rc)} bp"
            )
            print(
                f"      Reverse non-complementary RC: {len(reverse_noncomplementary_rc)} bp"
            )

            # Verify construction logic
            _verify_amplicon_construction(
                amplicon=amplicon, reverse_primer=reverse_primer, canonical=canonical
            )

    print(
        f"\n✓ Constructed {sum(len(amps) for amps in amplicons.values())} amplicons total"
    )
    return amplicons


def _verify_amplicon_construction(
    amplicon: DNAAmplicon, reverse_primer: ValidationPrimer, canonical: str
) -> None:
    """Internal verification of amplicon construction correctness."""

    # Verify forward primer can bind to start of amplicon
    forward_binding_region = amplicon.forward_anchor_toehold
    if forward_binding_region not in canonical:
        raise ValueError(
            f"Forward binding region not found in canonical for {amplicon.name}"
        )

    # Verify reverse primer can bind to end of amplicon (as reverse complement)
    reverse_binding_region = amplicon.reverse_anchor_toehold_rc
    expected_reverse_binding = str(
        Seq(
            reverse_primer.anchor_sequence + reverse_primer.toehold_sequence
        ).reverse_complement()
    )
    if reverse_binding_region != expected_reverse_binding:
        raise ValueError(f"Reverse binding region mismatch for {amplicon.name}")

    # Verify total length is reasonable (should be substantial portion of canonical + primer parts)
    min_expected_length = (
        len(amplicon.canonical_segment) + 50
    )  # At least canonical segment + some primer parts
    if amplicon.length < min_expected_length:
        raise ValueError(f"Amplicon {amplicon.name} too short: {amplicon.length} bp")

    # Verify all parts sum to total length
    total_parts = (
        len(amplicon.forward_generic_part)
        + len(amplicon.forward_anchor_toehold)
        + len(amplicon.canonical_segment)
        + len(amplicon.reverse_anchor_toehold_rc)
        + len(amplicon.reverse_noncomplementary_rc)
    )

    if total_parts != amplicon.length:
        raise ValueError(
            f"Amplicon {amplicon.name} part lengths don't sum correctly: {total_parts} != {amplicon.length}"
        )

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
        raise ValueError(
            "Both C.T and N.G amplicons required for cross-reactivity testing"
        )

    # Test each generic set combination
    for ct_amplicon in ct_amplicons:
        for ng_amplicon in ng_amplicons:
            if ct_amplicon.generic_set != ng_amplicon.generic_set:
                continue  # Only test within same generic set

            print(
                f"\nTesting {ct_amplicon.name} vs {ng_amplicon.name} cross-reactivity..."
            )

            # Define the four test pairs
            test_pairs = [
                # 1. Both sense strand
                ("sense_vs_sense", ct_amplicon.sequence, ng_amplicon.sequence),
                # 2. Both antisense strand
                (
                    "antisense_vs_antisense",
                    str(Seq(ct_amplicon.sequence).reverse_complement()),
                    str(Seq(ng_amplicon.sequence).reverse_complement()),
                ),
                # 3. Sense vs antisense
                (
                    "sense_vs_antisense",
                    ct_amplicon.sequence,
                    str(Seq(ng_amplicon.sequence).reverse_complement()),
                ),
                # 4. Antisense vs sense
                (
                    "antisense_vs_sense",
                    str(Seq(ct_amplicon.sequence).reverse_complement()),
                    ng_amplicon.sequence,
                ),
            ]

            for pair_name, ct_seq, ng_seq in test_pairs:
                print(f"  Testing {pair_name}...")
                print(f"    C.T sequence: {len(ct_seq)} bp")
                print(f"    N.G sequence: {len(ng_seq)} bp")

                # Set up NUPACK analysis with equal concentrations
                # Using signal concentrations (10pM) since these are amplicons
                amplicon_concentration_molar = (
                    TESTING_CONDITIONS['signal_concentration_pM'] * 1e-12
                )

                sequences = [
                    SequenceInput(
                        f"CT_{pair_name}", ct_seq, amplicon_concentration_molar
                    ),
                    SequenceInput(
                        f"NG_{pair_name}", ng_seq, amplicon_concentration_molar
                    ),
                ]

                # Run NUPACK analysis
                nupack_results = analyze_sequence_complexes(
                    temperature_celsius=NASBA_CONDITIONS['target_temp_C'],
                    sequences=sequences,
                    sodium_millimolar=80.0,
                    magnesium_millimolar=12.0,
                    max_complex_size=2,
                    base_pairing_analysis=False,  # Not needed for this test
                )

                # Calculate hetero-dimer formation (should be low)
                hetero_dimer_fraction = nupack_results.get_hetero_dimer_fraction(
                    seq1_name=f"CT_{pair_name}",
                    seq2_name=f"NG_{pair_name}",
                    seq1_input_conc_molar=amplicon_concentration_molar,
                    seq2_input_conc_molar=amplicon_concentration_molar,
                )

                # Calculate monomer fractions (should be high)
                ct_monomer_fraction = nupack_results.get_monomer_fraction(
                    sequence_name=f"CT_{pair_name}",
                    sequence_input_conc_molar=amplicon_concentration_molar,
                )
                ng_monomer_fraction = nupack_results.get_monomer_fraction(
                    sequence_name="NG_{pair_name}",
                    sequence_input_conc_molar=amplicon_concentration_molar,
                )
                min_monomer_fraction = min(ct_monomer_fraction, ng_monomer_fraction)

                # Validation criteria - amplicons should NOT form significant dimers
                low_dimer_pass = hetero_dimer_fraction <= (
                    1.0 - VALIDATION_THRESHOLDS['primer_signal_binding_min']
                )
                high_monomer_pass = (
                    min_monomer_fraction
                    >= VALIDATION_THRESHOLDS['primer_monomer_vs_wrong_daises']
                )  # >90% monomer

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

                print(
                    f"    {pair_name:20s}: Dimer {hetero_dimer_fraction:.3f} ({'✓' if low_dimer_pass else '✗'}), "
                    f"Min monomer {min_monomer_fraction:.3f} ({'✓' if high_monomer_pass else '✗'})"
                )

    return results


def test_7_generic_primer_amplicon_binding_specificity(
    ct_primers: List[ValidationPrimer],
    ng_primers: List[ValidationPrimer],
    amplicons: Dict[str, List[DNAAmplicon]],
) -> List[ValidationResult]:
    """
    Test 7: Generic primer-amplicon binding specificity.

    Generic primers should meet minimum binding thresholds with their designated amplicon signals
    with >90% 3'-end binding within the dimer.

    Generic forward primer binds to sense amplicon signal.
    Generic reverse primer binds to antisense amplicon signal.
    """
    results = []

    print("\n" + "=" * 80)
    print("TEST 7: GENERIC PRIMER-AMPLICON BINDING SPECIFICITY")
    print("=" * 80)

    # Define explicit generic primers
    generic_primers = {
        'gen5': {
            'forward': "TTATGTTCGTGGTT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC",  # noqa: typo
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC",  # noqa: typo
        },
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
        ct_amplicons_set = [
            a for a in amplicons.get('CT', []) if a.generic_set == generic_set
        ]
        ng_amplicons_set = [
            a for a in amplicons.get('NG', []) if a.generic_set == generic_set
        ]

        if not ct_amplicons_set or not ng_amplicons_set:
            raise RuntimeError(
                f'No amplicons found for generic set "{generic_set}"'
            )

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
                "forward_vs_sense",
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
                "reverse_vs_antisense",
            )

    return results


def _test_generic_primer_binding(
    results: List[ValidationResult],
    primer_name: str,
    primer_sequence: str,
    signal_name: str,
    signal_sequence: str,
    test_type: str,
) -> None:
    """Helper function to test generic primer binding to amplicon signal."""

    # Set up NUPACK analysis with generic primer concentrations
    primer_concentration_molar = (
        TESTING_CONDITIONS['generic_primer_concentration_nM'] * 1e-9
    )  # Convert nM to M
    signal_concentration_molar = (
        TESTING_CONDITIONS['generic_signal_concentration_nM'] * 1e-9
    )  # Convert nM to M

    sequences = [
        SequenceInput(primer_name, primer_sequence, primer_concentration_molar),
        SequenceInput(signal_name, signal_sequence, signal_concentration_molar),
    ]

    comprehensive_result = analyze_sequence_comprehensive(
        primary_sequence=primer_sequence,
        primary_sequence_name=primer_name,
        primary_sequence_concentration=primer_concentration_molar,
        other_sequences={
            signal_name: signal_sequence,
        },
        other_sequence_concentrations={
            signal_name: signal_concentration_molar,
        },
        temp_celsius=NASBA_CONDITIONS['target_temp_C'],
        n_bases=3,
    )

    primer_signal_binding = comprehensive_result.dimer_fraction[signal_name]
    three_prime_binding_prob = comprehensive_result.weighted_dimer_three_prime_paired_prob[signal_name]
    three_prime_binding_probs = comprehensive_result.weighted_dimer_three_prime_paired_probs[signal_name]

    # Check validation criteria
    binding_pass = (
        primer_signal_binding
        >= VALIDATION_THRESHOLDS['generic_primer_amplicon_binding_min']
    )
    three_prime_pass = (
        three_prime_binding_prob
        >= VALIDATION_THRESHOLDS['generic_primer_3_prime_binding_min']
    )
    overall_pass = binding_pass and three_prime_pass

    result = ValidationResult(
        test_name="Test_7_Generic_Primer_Amplicon_Binding",
        primer_name=primer_name,
        target_dais=[signal_name],
        hetero_dimer_fraction=primer_signal_binding,
        unpaired_3_prime_prob=(
            1.0 - three_prime_binding_prob
        ),  # Store as unpaired for consistency
        unpaired_3_prime_probs=tuple((1.0 - p) for p in three_prime_binding_probs),
        passed=overall_pass,
        details=f"{test_type}: Binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
        f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})",
    )
    results.append(result)

    print(
        f"    {test_type:20s}: Binding {primer_signal_binding:.3f} ({'✓' if binding_pass else '✗'}), "
        f"3'-end binding {three_prime_binding_prob:.3f} ({'✓' if three_prime_pass else '✗'})"
    )


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
            'forward': "TTATGTTCGTGGTT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC",  # noqa: typo
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC",  # noqa: typo
        },
    }

    # Test each generic set
    for generic_set in ['gen5', 'gen6']:
        print(f"\nTesting {generic_set} generic primer cross-reactivity...")

        forward_generic = generic_primers[generic_set]['forward']
        reverse_generic = generic_primers[generic_set]['reverse']

        # Get amplicons for this generic set
        ct_amplicons_set = [
            a for a in amplicons.get('CT', []) if a.generic_set == generic_set
        ]
        ng_amplicons_set = [
            a for a in amplicons.get('NG', []) if a.generic_set == generic_set
        ]

        if not ct_amplicons_set or not ng_amplicons_set:
            print(f"  Warning: Missing amplicons for {generic_set}")
            continue

        # Test forward generic primer against UNINTENDED signals (sense amplicons)
        print(f"  Testing forward generic primer against unintended sense amplicons...")
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(
                f"    Testing forward generic vs {amplicon.name} (sense - unintended)..."
            )

            _test_generic_primer_cross_reactivity(
                results,
                f"generic_forward_{generic_set}",
                forward_generic,
                f"{amplicon.name}_sense",
                amplicon.sequence,  # Sense amplicon is unintended for forward generic
                f"forward_vs_unintended_{amplicon.species}_sense",
            )

        # Test reverse generic primer against UNINTENDED signals (antisense amplicons)
        print(
            f"  Testing reverse generic primer against unintended antisense amplicons..."
        )
        for amplicon in ct_amplicons_set + ng_amplicons_set:
            print(
                f"    Testing reverse generic vs {amplicon.name} (antisense - unintended)..."
            )

            antisense_amplicon = str(Seq(amplicon.sequence).reverse_complement())
            _test_generic_primer_cross_reactivity(
                results,
                f"generic_reverse_{generic_set}",
                reverse_generic,
                f"{amplicon.name}_antisense",
                antisense_amplicon,  # Antisense amplicon is unintended for reverse generic
                f"reverse_vs_unintended_{amplicon.species}_antisense",
            )

    return results


def _test_generic_primer_cross_reactivity(
    results: List[ValidationResult],
    primer_name: str,
    primer_sequence: str,
    signal_name: str,
    signal_sequence: str,
    test_type: str,
) -> None:
    """Helper function to test generic primer cross-reactivity with unintended signals."""

    # Set up NUPACK analysis with generic primer concentrations
    primer_concentration_molar = (
        TESTING_CONDITIONS['generic_primer_concentration_nM'] * 1e-9
    )  # Convert nM to M
    signal_concentration_molar = (
        TESTING_CONDITIONS['generic_signal_concentration_nM'] * 1e-9
    )  # Convert nM to M

    sequences = [
        SequenceInput(primer_name, primer_sequence, primer_concentration_molar),
        SequenceInput(signal_name, signal_sequence, signal_concentration_molar),
    ]

    # Run NUPACK analysis
    nupack_results = analyze_sequence_complexes(
        temperature_celsius=NASBA_CONDITIONS['target_temp_C'],
        sequences=sequences,
        sodium_millimolar=80.0,
        magnesium_millimolar=12.0,
        max_complex_size=2,
        base_pairing_analysis=True,
    )

    # Calculate primer-signal binding (should be low for unintended signals)
    primer_signal_binding = nupack_results.get_hetero_dimer_fraction(
        seq1_name=primer_name,
        seq2_name=signal_name,
        seq1_input_conc_molar=primer_concentration_molar,
        seq2_input_conc_molar=signal_concentration_molar,
    )

    # Calculate monomer fraction (should be high - primer stays as monomer)
    monomer_fraction = nupack_results.get_monomer_fraction(
        sequence_name=primer_name,
        sequence_input_conc_molar=primer_concentration_molar,
    )

    # Calculate 3'-end unpaired probability (should be high - no binding at 3'-end)
    three_prime_unpaired_prob = 1.0  # Default to unpaired
    primer_length = len(primer_sequence)

    # Look for any complex containing this primer to get 3'-end info
    for complex_result in nupack_results.complexes:
        if (
            complex_result.unpaired_probability
            and primer_name in complex_result.sequence_id_map.values()
        ):

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
                complex_weight = complex_result.concentration_molar
                weighted_unpaired = (unpaired_prob1 * unpaired_prob2) * complex_weight
                three_prime_unpaired_prob = max(
                    three_prime_unpaired_prob, weighted_unpaired
                )

    # Validation criteria (generic primers should NOT bind significantly to unintended signals)
    low_binding_pass = (
        primer_signal_binding
        <= VALIDATION_THRESHOLDS['generic_primer_low_cross_binding_max']
    )
    monomer_pass = (
        monomer_fraction >= VALIDATION_THRESHOLDS['generic_primer_cross_monomer_min']
    )  # >90% monomer
    unpaired_pass = (
        three_prime_unpaired_prob
        >= VALIDATION_THRESHOLDS['generic_primer_cross_unpaired_min']
    )

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

    print(
        f"      {test_type:25s}: Binding {primer_signal_binding:.3f} ({'✓' if low_binding_pass else '✗'}), "
        f"Monomer {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'}), "
        f"3'-unpaired {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
    )


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
            'forward': "TTATGTTCGTGGTT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATACGTGC",  # noqa: typo
        },
        'gen6': {
            'forward': "TTTTGGTGGGTGGAT",  # noqa: typo
            'reverse': "AATTCTAATACGACTCACTATAGGGTAAATATCCGGC",  # noqa: typo
        },
    }

    all_primers = ct_primers + ng_primers

    # Test each generic set separately
    for generic_set in ['gen5', 'gen6']:
        print(f"\nTesting {generic_set} primer set interactions...")

        # Collect all 6 primers for this generic set
        nasba_primers_in_set = [p for p in all_primers if p.generic_set == generic_set]

        if len(nasba_primers_in_set) != 4:
            print(
                f"  Warning: Expected 4 NASBA primers for {generic_set}, found {len(nasba_primers_in_set)}"
            )
            continue

        # Add the 2 generic primers to make 6 total
        forward_generic = generic_primers[generic_set]['forward']
        reverse_generic = generic_primers[generic_set]['reverse']

        print(f"  Testing 6 primers in {generic_set} set:")
        print(
            f"    - 2 generic primers: forward ({len(forward_generic)} bp), reverse ({len(reverse_generic)} bp)"
        )
        print(f"    - 4 NASBA primers: {[p.name for p in nasba_primers_in_set]}")

        # Set up all 6 primers in single NUPACK tube
        primer_concentration_molar = (
            TESTING_CONDITIONS['primer_concentration_nM'] * 1e-9
        )  # 250nM

        sequences = [
            # Generic primers
            SequenceInput(
                f"generic_forward_{generic_set}",
                forward_generic,
                primer_concentration_molar,
            ),
            SequenceInput(
                f"generic_reverse_{generic_set}",
                reverse_generic,
                primer_concentration_molar,
            ),
        ]

        # Add NASBA primers
        for primer in nasba_primers_in_set:
            sequences.append(
                SequenceInput(primer.name, primer.sequence, primer_concentration_molar)
            )

        print(f"    Running NUPACK analysis with {len(sequences)} primers...")

        # Run NUPACK analysis with all 6 primers
        nupack_results = analyze_sequence_complexes(
            temperature_celsius=NASBA_CONDITIONS['target_temp_C'],
            sequences=sequences,
            sodium_millimolar=80.0,
            magnesium_millimolar=12.0,
            max_complex_size=6,  # Allow up to 6-mer complexes
            base_pairing_analysis=True,
        )

        # Test 1: Generic primers should not bind to each other
        generic_generic_binding = nupack_results.get_hetero_dimer_fraction(
            seq1_name=f"generic_forward_{generic_set}",
            seq2_name=f"generic_reverse_{generic_set}",
            seq1_input_conc_molar=primer_concentration_molar,
            seq2_input_conc_molar=primer_concentration_molar,
        )

        generic_generic_pass = (
            generic_generic_binding
            <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
        )

        result = ValidationResult(
            test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
            primer_name=f"{generic_set}_generic_primers",
            target_dais=["generic_generic_interaction"],
            hetero_dimer_fraction=generic_generic_binding,
            passed=generic_generic_pass,
            details=f"Generic-generic binding: {generic_generic_binding:.3f} ({'✓' if generic_generic_pass else '✗'})",
        )
        results.append(result)

        print(
            f"    Generic-generic binding: {generic_generic_binding:.3f} ({'✓' if generic_generic_pass else '✗'})"
        )

        # Test 2: Generic primers should not bind to NASBA primers significantly
        for nasba_primer in nasba_primers_in_set:
            # Test forward generic vs NASBA primer
            forward_nasba_binding = nupack_results.get_hetero_dimer_fraction(
                seq1_name="generic_forward_{generic_set}",
                seq2_name=nasba_primer.name,
                seq1_input_conc_molar=primer_concentration_molar,
                seq2_input_conc_molar=primer_concentration_molar,
            )

            # Test reverse generic vs NASBA primer
            reverse_nasba_binding = nupack_results.get_hetero_dimer_fraction(
                seq1_name=f"generic_reverse_{generic_set}",
                seq2_name=nasba_primer.name,
                seq1_input_conc_molar=primer_concentration_molar,
                seq2_input_conc_molar=primer_concentration_molar,
            )

            forward_nasba_pass = (
                forward_nasba_binding
                <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
            )
            reverse_nasba_pass = (
                reverse_nasba_binding
                <= VALIDATION_THRESHOLDS['primer_set_low_interaction_max']
            )

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

            print(
                f"    Forward generic vs {nasba_primer.name}: {forward_nasba_binding:.3f} ({'✓' if forward_nasba_pass else '✗'})"
            )
            print(
                f"    Reverse generic vs {nasba_primer.name}: {reverse_nasba_binding:.3f} ({'✓' if reverse_nasba_pass else '✗'})"
            )

        # Test 3: All primers should remain predominantly monomeric
        all_primer_names = [
            f"generic_forward_{generic_set}",
            f"generic_reverse_{generic_set}",
        ] + [p.name for p in nasba_primers_in_set]

        print(f"    Monomer fractions:")
        for primer_name in all_primer_names:
            monomer_fraction = nupack_results.get_monomer_fraction(
                sequence_name=primer_name,
                sequence_input_conc_molar=primer_concentration_molar,
            )
            monomer_pass = (
                monomer_fraction >= VALIDATION_THRESHOLDS['primer_set_monomer_min']
            )

            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=primer_name,
                target_dais=["monomer_fraction"],
                monomer_fraction=monomer_fraction,
                passed=monomer_pass,
                details=f"Monomer fraction: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'})",
            )
            results.append(result)

            print(
                f"      {primer_name:20s}: {monomer_fraction:.3f} ({'✓' if monomer_pass else '✗'})"
            )

        # Test 4: Generic primers should have high 3'-end unpaired probability
        for generic_type in ['forward', 'reverse']:
            generic_name = f"generic_{generic_type}_{generic_set}"
            generic_sequence = generic_primers[generic_set][generic_type]

            # Calculate 3'-end unpaired probability
            three_prime_unpaired_prob = _calculate_generic_3_prime_unpaired(
                nupack_results, generic_name, generic_sequence
            )

            unpaired_pass = (
                three_prime_unpaired_prob
                >= VALIDATION_THRESHOLDS['primer_set_generic_unpaired_min']
            )

            result = ValidationResult(
                test_name="Test_9_Generic_Primer_Set_Cross_Reactivity",
                primer_name=generic_name,
                target_dais=["3_prime_unpaired"],
                unpaired_3_prime_prob=three_prime_unpaired_prob,
                passed=unpaired_pass,
                details=f"3'-end unpaired: {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})",
            )
            results.append(result)

            print(
                f"    {generic_name} 3'-end unpaired: {three_prime_unpaired_prob:.4f} ({'✓' if unpaired_pass else '✗'})"
            )

    return results


def _calculate_generic_3_prime_unpaired(
    nupack_results, primer_name: str, primer_sequence: str
) -> float:
    """Helper to calculate 3'-end unpaired probability for generic primer in complex mixture."""

    three_prime_unpaired_prob = 1.0  # Default to unpaired
    primer_length = len(primer_sequence)

    # Look through all complexes containing this primer
    for complex_result in nupack_results.complexes:
        if (
            complex_result.unpaired_probability
            and primer_name in complex_result.sequence_id_map.values()
        ):

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
                complex_weight = complex_result.concentration_molar
                weighted_unpaired = (unpaired_prob1 * unpaired_prob2) * complex_weight
                three_prime_unpaired_prob = min(
                    three_prime_unpaired_prob, weighted_unpaired
                )

    return three_prime_unpaired_prob


# ============================================================================
# STANDARDIZED TEST INTERFACE
# ============================================================================


@dataclass
class ValidationContext:
    """Unified input object for all validation tests."""

    ct_primers: List[ValidationPrimer]
    ng_primers: List[ValidationPrimer]
    ct_daises: List[Dais]
    ng_daises: List[Dais]
    amplicons: Optional[Dict[str, List[DNAAmplicon]]] = None

    # Useful shared configuration
    canonical_sequences: Dict[str, str] = field(
        default_factory=lambda: CANONICAL_SEQUENCES
    )
    testing_conditions: Dict[str, float] = field(
        default_factory=lambda: TESTING_CONDITIONS
    )
    thresholds: Dict[str, float] = field(default_factory=lambda: VALIDATION_THRESHOLDS)


def get_tests_registry() -> (
    Dict[str, Callable[[ValidationContext], List[ValidationResult]]]
):
    """Return a registry mapping test names to callables taking a ValidationContext."""
    return {
        'test_1': lambda ctx: test_1_hetero_dimer_measurements(
            ctx.ct_primers, ctx.ng_primers, ctx.ct_daises, ctx.ng_daises
        ),
        'test_2': lambda ctx: test_2_four_primer_cross_reactivity(
            ctx.ct_primers, ctx.ng_primers
        ),
        'test_3': lambda ctx: test_3_individual_primer_dais_binding(
            ctx.ct_primers, ctx.ng_primers, ctx.ct_daises, ctx.ng_daises
        ),
        'test_4': lambda ctx: test_4_primer_signal_binding_specificity(
            ctx.ct_primers, ctx.ng_primers
        ),
        'test_5': lambda ctx: test_5_primer_cross_reactivity_with_unintended_signals(
            ctx.ct_primers, ctx.ng_primers
        ),
        'test_6': lambda ctx: test_6_inter_pathogen_amplicon_cross_reactivity(
            ctx.amplicons or {}
        ),
        'test_7': lambda ctx: test_7_generic_primer_amplicon_binding_specificity(
            ctx.ct_primers, ctx.ng_primers, ctx.amplicons or {}
        ),
        'test_8': lambda ctx: test_8_generic_primer_cross_reactivity_with_unintended_signals(
            ctx.amplicons or {}
        ),
        'test_9': lambda ctx: test_9_generic_primer_cross_reactivity_within_primer_set(
            ctx.ct_primers, ctx.ng_primers
        ),
    }


def run_selected_tests(
    context: ValidationContext, selected_tests: Optional[List[str]] = None
) -> Dict[str, List[ValidationResult]]:
    """
    Execute a subset (or all) validation tests using a standardized interface.

    Args:
        context: ValidationContext with all necessary inputs populated
        selected_tests: Optional list of test names (keys in registry) to run.
                        If None, all tests in the registry are run.

    Returns:
        Dict[str, List[ValidationResult]] keyed by test name.
    """
    registry = get_tests_registry()
    tests_to_run = selected_tests or list(registry.keys())

    results: Dict[str, List[ValidationResult]] = {}
    for test_name in tests_to_run:
        if test_name not in registry:
            raise ValueError(
                f"Unknown test '{test_name}'. Available: {list(registry.keys())}"
            )
        results[test_name] = registry[test_name](context)

    return results


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

                # Filter candidates using bound-fraction criteria
                valid_candidates, all_scored_candidates = (
                    filter_candidates_by_bound_fraction(
                        candidates,
                    )
                )

                if valid_candidates:
                    # Take the best candidate (highest bound-fraction score)
                    best_candidate = max(
                        valid_candidates, key=lambda x: x.bound_fraction_score
                    )
                else:
                    # If no valid candidates, take the best invalid one for testing purposes
                    if all_scored_candidates:
                        best_candidate = max(
                            all_scored_candidates, key=lambda x: x.bound_fraction_score
                        )
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

        # Construct DNA amplicons once (used by several tests)
        amplicons = construct_dna_amplicons(ct_primers, ng_primers)

        # Build a unified context and run a selectable set of tests
        ctx = ValidationContext(
            ct_primers=ct_primers,
            ng_primers=ng_primers,
            ct_daises=ct_daises,
            ng_daises=ng_daises,
            amplicons=amplicons,
        )

        # To run a partial set, pass e.g. selected = ['test_4', 'test_5']
        # selected: Optional[List[str]] = None
        selected: Optional[List[str]] = [
            'test_5',
            'test_6',
            'test_7',
            'test_8',
        ]
        pair_results = run_selected_tests(ctx, selected_tests=selected)

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
            'results': pair_results,
        }

    # Overall summary across all primer pairs
    print("\n" + "=" * 80)
    print("OVERALL VALIDATION SUMMARY ACROSS ALL PRIMER PAIRS")
    print("=" * 80)

    total_all_tests = 0
    total_passed = 0

    for pair_name, pair_data in all_validation_results.items():
        pair_total_tests = sum(
            len(results) for results in pair_data['results'].values()
        )
        pair_passed = sum(
            sum(1 for r in results if r.passed)
            for results in pair_data['results'].values()
        )

        total_all_tests += pair_total_tests
        total_passed += pair_passed

        print(
            f"{pair_name}: {pair_passed}/{pair_total_tests} tests passed "
            f"({100*pair_passed/pair_total_tests:.1f}%)"
            if pair_total_tests > 0
            else f"{pair_name}: No tests"
        )

    print(
        f"\nGRAND TOTAL: {total_passed}/{total_all_tests} tests passed "
        f"({100*total_passed/total_all_tests:.1f}%)"
        if total_all_tests > 0
        else "\nGRAND TOTAL: No tests"
    )

    return {
        'all_pairs': all_validation_results,
        'summary': {
            'total_tests': total_all_tests,
            'passed_tests': total_passed,
            'pass_rate': total_passed / total_all_tests if total_all_tests > 0 else 0.0,
        },
    }


# ============================================================================
# DATA AGGREGATION TO DATAFRAME
# ============================================================================


def build_validation_dataframe(all_pairs: Dict) -> pd.DataFrame:
    """
    Flatten the full validation results into a long-form pandas DataFrame.

    Rows:
      - One row per ValidationResult per target entry (target_dais list expanded)

    Columns:
      - pair: Primer pair label (e.g., from get_base_primers keys)
      - test: Test identifier (e.g., 'test_4')
      - primer: Primer name (e.g., 'CT-F-gen5')
      - target: Target/dais/signal name for this measurement (expanded from target_dais)
      - species: Primer species ('CT' or 'NG') if available
      - primer_type: 'forward' or 'reverse' if available
      - generic_set: 'gen5' or 'gen6' if available
      - hetero_dimer_fraction: Numeric hetero-dimer fraction (if measured)
      - monomer_fraction: Numeric monomer fraction (if measured)
      - unpaired_3_prime_prob: Numeric 3'-end unpaired probability (if measured)
      - passed: Boolean pass/fail as computed originally (retained for reference)
      - details: Free-text details string
      - temperature_C: Temperature used for the test (from TESTING_CONDITIONS)
    """
    rows: List[Dict] = []

    for pair_name, pair_data in (all_pairs or {}).items():
        # Build a lookup of primer metadata by primer name for enrichment
        primer_meta: Dict[str, Dict[str, str]] = {}

        primers_section = pair_data.get('primers', {})
        for species_key in ('CT', 'NG'):
            for p in primers_section.get(species_key, []) or []:
                primer_meta[p.name] = {
                    'species': p.species,
                    'primer_type': p.primer_type,
                    'generic_set': p.generic_set,
                }

        # Flatten ValidationResult entries
        for test_name, results in (pair_data.get('results') or {}).items():
            for vr in results:
                targets = vr.target_dais if vr.target_dais else [None]
                for tgt in targets:
                    meta = primer_meta.get(vr.primer_name, {})
                    rows.append(
                        {
                            'pair': pair_name,
                            'test': test_name,
                            'primer': vr.primer_name,
                            'target': tgt,
                            'species': meta.get('species'),
                            'primer_type': meta.get('primer_type'),
                            'generic_set': meta.get('generic_set'),
                            'hetero_dimer_fraction': vr.hetero_dimer_fraction,
                            'monomer_fraction': vr.monomer_fraction,
                            'unpaired_3_prime_prob': vr.unpaired_3_prime_prob,
                            'passed': vr.passed,
                            'details': vr.details,
                            'temperature_C': NASBA_CONDITIONS['target_temp_C'],
                        }
                    )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


@click.command()
def main():
    """
    Main function for NASBA primer validation.

    This tool implements three comprehensive validation tests:

    Test 1: Hetero-dimer fraction measurements
    - Each C.T and N.G primer should bind strongly to its corresponding dais

    Test 2: Four-primer cross-reactivity analysis
    - All four primers (CT-F, CT-R, NG-F, NG-R) per generic set should meet monomer thresholds
    - 3'-end should meet unpaired probability thresholds in presence of other primers

    Test 3: Individual primer-dais binding specificity
    - Each primer tested against non-target daises should meet monomer thresholds
    - 3'-end should meet unpaired probability thresholds with non-target daises

    Total: 8 individual tests per generic set (2 tests) = 16 tests for Test 3
    """

    # Run comprehensive validation
    run_comprehensive_validation()
    return 0


if __name__ == "__main__":
    exit(main())
