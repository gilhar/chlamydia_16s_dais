#!/usr/bin/env python3
"""
NASBA Primer Thermodynamics Module

This module provides NUPACK-based thermodynamic calculations for NASBA primer validation,
focusing on bound-fraction calculations at the NASBA reaction temperature (41°C).

Key functions:
- Calculate bound fractions using NUPACK
- Validate primers based on bound-fraction criteria
- Score and rank primer candidates

Author: Claude (Anthropic)
Date: 2025
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import nupack
from Bio.PDB.ic_data import primary_angles
from nupack import SetSpec

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# NASBA thermodynamic conditions - centralized constants
NASBA_TEMPERATURE_CELSIUS = 41.0  # NASBA reaction temperature
NASBA_SODIUM_MOLAR = 0.08  # 80 mM Na+ converted to M
NASBA_MAGNESIUM_MOLAR = 0.012  # 12 mM Mg++ converted to M
NASBA_PRIMER_CONCENTRATION_MOLAR = 250e-9  # 250nM primer concentration

# Legacy NASBA conditions dict (for backward compatibility)
NASBA_CONDITIONS = {
    'Na_mM': NASBA_SODIUM_MOLAR * 1000,  # Convert back to mM for display
    'Mg_mM': NASBA_MAGNESIUM_MOLAR * 1000,  # Convert back to mM for display
    'primer_uM': (
        NASBA_PRIMER_CONCENTRATION_MOLAR * 1e6
    ),  # Convert back to µM for display
    'target_temp_C': NASBA_TEMPERATURE_CELSIUS,
}

# Bound-fraction targets at 41°C (NASBA reaction temperature)
BOUND_FRACTION_TARGETS = {
    'toehold_min': 0.6,
    'toehold_max': 0.99,
    'anchor_min': 0.97,
    'anchor_max': 1.01,
}


# Concentration settings for different binding assays
@dataclass
class AssayConcentrations:
    """Concentrations for different types of binding assays."""

    primer_dais_binding: Dict[str, float] = None
    primer_primer_cross_reactivity: Dict[str, float] = None
    primer_non_signal_cross_reactivity: Dict[str, float] = None
    off_target_dais: Dict[str, float] = None
    amplicon_strand_binding: Dict[str, float] = None
    generic_primer_amplicon: Dict[str, float] = None

    def __post_init__(self):
        if self.primer_dais_binding is None:
            self.primer_dais_binding = {
                'primer_concentration_M': 250e-9,  # 250nM primer
                'target_dais_concentration_M': 250e-9,  # 250nM target dais
            }

        if self.primer_primer_cross_reactivity is None:
            self.primer_primer_cross_reactivity = {
                'primer_concentration_M': 250e-9,  # 250nM primer
                'other_primer_concentration_M': 250e-9,
            }
        if self.primer_non_signal_cross_reactivity is None:
            self.primer_non_signal_cross_reactivity = {
                'primer_concentration_M': 250e-9,  # 250nM primer
                'other_sequence_concentration_M': 10e-12,  # 10pM other sequences
            }

        if self.off_target_dais is None:
            self.off_target_dais = {
                'primer_concentration_M': 250e-9,  # 250nM primer
                'non_intended_dais_concentration_M': 250e-9,  # 250nM non-intended dais
            }

        if self.amplicon_strand_binding is None:
            self.amplicon_strand_binding = {
                'amplicon_strand1_concentration_M': 100e-9,  # 100nM amplicon strand 1
                'amplicon_strand2_concentration_M': 100e-9,  # 100nM amplicon strand 2
            }

        if self.generic_primer_amplicon is None:
            self.generic_primer_amplicon = {
                'generic_primer_concentration_M': 250e-9,  # 250nM generic primer
                'amplicon_sense_concentration_M': 50e-9,  # 50nM amplicon (non-intended sense)
            }


# Default assay concentrations
DEFAULT_ASSAY_CONCENTRATIONS = AssayConcentrations()


# ============================================================================
# INTERNAL NUPACK ANALYSIS
# ============================================================================


@dataclass
class NupackAnalysisResult:
    """Complete NUPACK analysis result with optional base-pairing information."""

    complex_concentrations: Dict
    unpaired_probabilities: Dict = (
        None  # {complex: {(strand_idx, base_idx): probability}}
    )
    pairing_probabilities: Dict = (
        None  # {complex: {(strand1_idx, base1_idx, strand2_idx, base2_idx): probability}}
    )
    strand_id_map: Dict[nupack.Complex, dict[int, str]] = (
        None  # {complex: {strand_idx: sequence_name}}
    )


def _analyze_sequences_with_nupack(
    sequences: List[str],
    sequence_names: List[str],
    concentrations_molar: List[float],
    temp_celsius: float = NASBA_TEMPERATURE_CELSIUS,
    sodium_molar: float = NASBA_SODIUM_MOLAR,
    magnesium_molar: float = NASBA_MAGNESIUM_MOLAR,
    max_complex_size: int = 2,
    include_base_pairing: bool = False,
) -> NupackAnalysisResult:
    """
    Internal function that performs complete NUPACK analysis.

    Args:
        sequences: List of DNA sequences
        sequence_names: List of names for sequences
        concentrations_molar: List of concentrations in M
        temp_celsius: Temperature in Celsius
        sodium_molar: Sodium concentration in M
        magnesium_molar: Magnesium concentration in M
        max_complex_size: Maximum complex size to analyze
        include_base_pairing: Whether to include base-pairing analysis

    Returns:
        NupackAnalysisResult with complete analysis data
    """
    # Validate inputs
    if not sequences or not sequence_names or not concentrations_molar:
        raise ValueError(
            "sequences, sequence_names, and concentrations_molar cannot be empty"
        )

    if len(sequences) != len(sequence_names) != len(concentrations_molar):
        raise ValueError(
            f"Length mismatch: sequences={len(sequences)}, names={len(sequence_names)}, concentrations={len(concentrations_molar)}"
        )

    for i, (seq, name, conc) in enumerate(
        zip(sequences, sequence_names, concentrations_molar)
    ):
        if not seq:
            raise ValueError(f"Sequence {i} ({name}) is empty")
        if not name:
            raise ValueError(f"Sequence name {i} is empty")
        if conc <= 0:
            raise ValueError(f"Concentration for {name} must be positive: {conc}")

    if temp_celsius < -273.15:
        raise ValueError(f"Temperature {temp_celsius}°C is below absolute zero")

    if sodium_molar < 0 or magnesium_molar < 0:
        raise ValueError(
            f"Salt concentrations must be non-negative: Na={sodium_molar}, Mg={magnesium_molar}"
        )

    if max_complex_size < 1:
        raise ValueError(f"max_complex_size must be at least 1: {max_complex_size}")

    try:
        # Create NUPACK strands
        strands = []
        strand_concentrations = {}

        for i, (seq, name, conc) in enumerate(
            zip(sequences, sequence_names, concentrations_molar)
        ):
            strand = nupack.Strand(seq, name=name, material='dna')
            strands.append(strand)
            strand_concentrations[strand] = conc

        # Create thermodynamic model
        model = nupack.Model(
            material='dna',
            celsius=temp_celsius,
            sodium=sodium_molar,
            magnesium=magnesium_molar,
        )

        # Create tube with specified concentrations
        tube = nupack.Tube(
            name='analysis_tube',
            strands=strand_concentrations,
            complexes=SetSpec(max_size=max_complex_size),
        )

        # Set up compute parameters
        compute_params = ['pfunc']
        if include_base_pairing:
            compute_params.append('pairs')

        # Calculate equilibrium concentrations
        result = nupack.tube_analysis(tubes=[tube], model=model, compute=compute_params)

        # Extract complex concentrations
        complex_concentrations = result.tubes[tube].complex_concentrations

        # Complex concentrations are now the primary result
        # Specific complex concentrations can be extracted using utility functions

        # Extract base-pairing information if requested
        unpaired_probs = None
        pairing_probs = None
        strand_id_map = None

        if include_base_pairing:
            # Validate that NUPACK result contains the expected data
            if not hasattr(result, 'complexes'):
                raise ValueError(
                    "NUPACK result does not contain complex data, but 'pairs' computation was requested. "
                    "This indicates a problem with the NUPACK analysis setup."
                )

            unpaired_probs = {}
            pairing_probs = {}
            strand_id_map = {}

            # Process all complexes - they must all have pairs data since we requested it
            for complex_obj, complex_result in result.complexes.items():
                if not hasattr(complex_result, 'pairs'):
                    raise ValueError(
                        f"Complex {complex_obj} does not contain pairs data, but 'pairs' computation was requested. "
                        "This indicates a problem with the NUPACK analysis."
                    )

                # Get the pair probability matrix for this complex
                pairs_matrix = complex_result.pairs.to_array()

                # Initialize dictionaries for this complex
                complex_unpaired_probs = {}
                complex_pairing_probs = {}
                strand_names = {}

                # Get strands in this complex to understand structure
                complex_strands = complex_obj.strands

                # Extract unpaired and pairing probabilities
                total_bases = 0
                strand_base_offsets = []  # Track where each strand starts in the matrix

                for strand_idx, strand in enumerate(complex_strands):
                    strand_names[strand_idx] = strand.name
                    strand_base_offsets.append(total_bases)
                    seq_len = len(str(strand))

                    for base_idx in range(seq_len):
                        matrix_idx = total_bases + base_idx

                        # Unpaired probability is on the diagonal
                        unpaired_prob = pairs_matrix[matrix_idx, matrix_idx]

                        # Validate unpaired probability is in the valid range
                        if not (0.0 <= unpaired_prob <= 1.0):
                            raise ValueError(
                                f"Unpaired probability {unpaired_prob} for base ({strand_idx}, {base_idx}) "
                                f"at matrix position [{matrix_idx}, {matrix_idx}] is outside valid range [0, 1]."
                            )

                        complex_unpaired_probs[(strand_idx, base_idx)] = unpaired_prob

                        # Store pairing probabilities with other bases (off-diagonal elements)
                        for other_matrix_idx in range(pairs_matrix.shape[0]):
                            if other_matrix_idx != matrix_idx:
                                pair_prob = pairs_matrix[matrix_idx, other_matrix_idx]

                                if (
                                    pair_prob > 1e-6
                                ):  # Only store significant pairing probabilities
                                    # Convert back to strand/base coordinates
                                    other_strand_idx = None
                                    other_base_idx = None

                                    running_total = 0
                                    for s_idx, s_strand in enumerate(complex_strands):
                                        if (
                                            running_total
                                            <= other_matrix_idx
                                            < running_total + len(str(s_strand))
                                        ):
                                            other_strand_idx = s_idx
                                            other_base_idx = (
                                                other_matrix_idx - running_total
                                            )
                                            break
                                        running_total += len(str(s_strand))

                                    if other_strand_idx is not None:
                                        complex_pairing_probs[
                                            (
                                                strand_idx,
                                                base_idx,
                                                other_strand_idx,
                                                other_base_idx,
                                            )
                                        ] = pair_prob

                    total_bases += seq_len

                # Store the probabilities for this complex
                unpaired_probs[complex_obj] = complex_unpaired_probs
                pairing_probs[complex_obj] = complex_pairing_probs
                strand_id_map[complex_obj] = strand_names

        return NupackAnalysisResult(
            complex_concentrations=complex_concentrations,
            unpaired_probabilities=unpaired_probs,
            pairing_probabilities=pairing_probs,
            strand_id_map=strand_id_map,
        )

    except Exception as e:
        raise ValueError(f"Failed to analyze sequences {sequence_names}: {e}") from e


# ============================================================================
# THERMODYNAMIC CALCULATIONS
# ============================================================================


def calculate_dimer_formation_probability(
    sequence1: str,
    sequence2: str,
    sequence1_concentration_molar: float,
    sequence2_concentration_molar: float,
    temp_celsius: float = NASBA_TEMPERATURE_CELSIUS,
    sodium_molar: float = NASBA_SODIUM_MOLAR,
    magnesium_molar: float = NASBA_MAGNESIUM_MOLAR,
) -> float:
    """
    Calculate the dimer formation probability between two sequences using NUPACK.

    The dimer concentration is measured as a fraction of the component with lower concentration.

    Args:
        sequence1: First sequence (DNA)
        sequence2: Second sequence (DNA)
        sequence1_concentration_molar: Concentration of sequence1 in M
        sequence2_concentration_molar: Concentration of sequence2 in M
        temp_celsius: Temperature in Celsius
        sodium_molar: Sodium concentration in M
        magnesium_molar: Magnesium concentration in M

    Returns:
        Dimer formation probability (0.0 to 1.0) - dimer concentration / min(conc1, conc2)
    """
    # Validate inputs
    if not sequence1 or not sequence2:
        raise ValueError("Both sequences must be non-empty strings")

    if sequence1_concentration_molar <= 0 or sequence2_concentration_molar <= 0:
        raise ValueError(
            f"Concentrations must be positive: seq1={sequence1_concentration_molar}, seq2={sequence2_concentration_molar}"
        )

    if temp_celsius < -273.15:
        raise ValueError(f"Temperature {temp_celsius}°C is below absolute zero")

    if sodium_molar < 0 or magnesium_molar < 0:
        raise ValueError(
            f"Salt concentrations must be non-negative: Na={sodium_molar}, Mg={magnesium_molar}"
        )

    # Use internal analysis function
    result = _analyze_sequences_with_nupack(
        sequences=[sequence1, sequence2],
        sequence_names=['seq1', 'seq2'],
        concentrations_molar=[
            sequence1_concentration_molar,
            sequence2_concentration_molar,
        ],
        temp_celsius=temp_celsius,
        sodium_molar=sodium_molar,
        magnesium_molar=magnesium_molar,
        max_complex_size=2,
        include_base_pairing=False,
    )

    # Extract hetero-dimer concentration using utility function
    hetero_dimer_concentration = extract_heterodimer_concentration(
        result.complex_concentrations,
        sequence1,
        sequence2,
        strand1_name='seq1',
        strand2_name='seq2',
    )

    # Calculate dimer formation probability as fraction of limiting component
    limiting_concentration = min(
        sequence1_concentration_molar, sequence2_concentration_molar
    )

    if limiting_concentration <= 0:
        raise ValueError(
            f"Invalid concentration: limiting concentration is {limiting_concentration}"
        )

    dimer_probability = hetero_dimer_concentration / limiting_concentration

    # Validate that result is in expected range
    if not (0.0 <= dimer_probability <= 1.0):
        raise ValueError(
            f"Dimer probability {dimer_probability} is outside valid range [0, 1]. "
            f"Hetero-dimer concentration: {hetero_dimer_concentration}, limiting concentration: {limiting_concentration}"
        )

    return dimer_probability


def extract_heterodimer_concentration(
    complex_concentrations: Dict,
    strand1_sequence: str,
    strand2_sequence: str,
    strand1_name: str = "strand1",
    strand2_name: str = "strand2",
) -> float:
    """
    Extract the hetero-dimer concentration from complex concentrations dict.

    This utility function safely extracts the concentration of a hetero-dimer
    complex formed between two different strands, with validation.

    Args:
        complex_concentrations: Dict from NUPACK analysis results
        strand1_sequence: First strand sequence
        strand2_sequence: Second strand sequence
        strand1_name: First strand name (must match name used in analysis)
        strand2_name: Second strand name (must match name used in analysis)

    Returns:
        Concentration of the hetero-dimer complex

    Raises:
        ValueError: If validation fails or hetero-dimer not found
    """
    if not complex_concentrations:
        raise ValueError("complex_concentrations dict is empty")

    if not strand1_sequence or not strand2_sequence:
        raise ValueError("Both strand sequences must be non-empty")

    if strand1_sequence == strand2_sequence:
        raise ValueError("Sequences are identical - cannot form hetero-dimer")

    # Create strands to match those used in analysis
    import nupack

    strand1 = nupack.Strand(strand1_sequence, name=strand1_name, material='dna')
    strand2 = nupack.Strand(strand2_sequence, name=strand2_name, material='dna')

    # Create the expected hetero-dimer complex
    heterodimer_complex = nupack.Complex([strand1, strand2])

    # Look for this complex in the concentration dict
    heterodimer_concentration = None

    for complex_obj, concentration in complex_concentrations.items():
        if complex_obj == heterodimer_complex:
            heterodimer_concentration = concentration
            break

    if heterodimer_concentration is None:
        # More detailed error with available complexes
        available_complexes = [str(c) for c in complex_concentrations.keys()]
        raise ValueError(
            f"Hetero-dimer complex {heterodimer_complex} not found in results. "
            f"Available complexes: {available_complexes}"
        )

    if heterodimer_concentration < 0:
        raise ValueError(f"Invalid negative concentration: {heterodimer_concentration}")

    return heterodimer_concentration


# calculate the pairing probabilities of the last n_bases of the sequence,
#  when in a tube with other sequences, where the complex concentrations and
#  pairing probabilities were already computed and are provided in
#  complex_concentrations and pairing_probabilities
#
#  sequence_name: Name of the sequence
#  other_sequence_name: Name of the other sequence
#  sequence: Sequence
#  other_sequence: Sequence
#  sequence_concentration_molar: Concentration of the sequence in M
#  other_sequence_concentration_molar: Concentration of the other sequence in M
#  complex_concentrations: Dict from NUPACK analysis results
#  other_sequence_name: Name of the other sequence
#  n_bases: Number of 3'-end bases to analyze
#
# We take the probabilities of the last n_bases of the sequence, and weigh
#  by the concentration of the dimer relative to the limiting concentration
#  ( min(sequence_concentration_molar, other_sequence_concentration_molar) )
# We use (1-unpaired_prob) as the assumption here is that we do not
#   really care (or know, or care enough to know) where precisely the 3'-end of
#   sequence should bind to other_sequence, so we use the aggregate
def calculate_weighted_three_prime_end_paired_probabilities(
    sequence_name: str,
    other_sequence_name: str,
    sequence: str,
    other_sequence: str,
    sequence_concentration_molar: float,
    other_sequence_concentration_molar: float,
    dimer_concentration: float,
    dimer_unpaired_probabilities: Dict[tuple[int, int], float],
    dimer_id_map: dict[int, str],
    n_bases: int = 3,
) -> tuple[float, tuple[float, ...]]:
    # Calculate weighted unpaired probability
    weighted_unpaired_probs = []  # One value per base position

    limiting_sequence_concentration_molar = min(
        sequence_concentration_molar, other_sequence_concentration_molar
    )
    contribution_to_primer_conc = dimer_concentration
    if sequence == other_sequence:
        contribution_to_primer_conc *= 2

    # Weight by the fraction of primer that's in this complex
    weight = (
        contribution_to_primer_conc / limiting_sequence_concentration_molar
    )

    if len(dimer_id_map) != 2 or sequence_name not in dimer_id_map or other_sequence_name not in dimer_id_map:
        raise ValueError(
            f"Invalid dimer_id_map: {dimer_id_map} for sequence {sequence_name} and other sequence {other_sequence_name}"
        )

    sequence_strand_idx = 0 if sequence_name == dimer_id_map[0] else 1
    other_sequence_strand_idx = 0 if other_sequence_name == dimer_id_map[1] else 1
    if dimer_id_map[sequence_strand_idx] != sequence_name or dimer_id_map[other_sequence_strand_idx] != other_sequence_name:
        raise ValueError(
            f"Strand indices do not match names: {dimer_id_map} for sequence {sequence_name} and other sequence {other_sequence_name}"
        )
    for base_offset in range(n_bases):
        base_idx = len(sequence) - 1 - base_offset  # Count from 3'-end

        # Get unpaired probability for this base in this complex
        unpaired_prob_in_complex = dimer_unpaired_probabilities.get(
            (sequence_strand_idx, base_idx)
        )

        if unpaired_prob_in_complex is None:
            raise ValueError(
                f"Missing unpaired probability for primer base {base_idx} "
                f"(offset {base_offset}) in dimer <{dimer_id_map[0]},{dimer_id_map[1]}>"
            )
        weighted_prob_for_base = unpaired_prob_in_complex * weight

        if weighted_prob_for_base is None:
            raise ValueError(
                f"Missing unpaired probability for primer base {base_idx} "
                f"(offset {base_offset})"
            )
        weighted_unpaired_probs.append(weighted_prob_for_base)

    # Average weighted probability across the n_bases
    weighted_unpaired_prob = sum(weighted_unpaired_probs) / len(weighted_unpaired_probs)

    return 1 - weighted_unpaired_prob, tuple(1 - p for p in weighted_unpaired_probs)


# compute the weighted three prime end unpaired probabilities of the last
#  n_bases of the sequence, when in a tube with other sequences, where the complex
#  concentrations and unpairing probabilities were already computed and
#  are provided in complex_concentrations and unpaired_probabilities
#
#  sequence_name: Name of the sequence
#  sequence: Sequence
#  sequence_concentration_molar: Concentration of the sequence in M
#  complex_concentrations: Dict from NUPACK analysis results
#  unpaired_probabilities: Dict from NUPACK analysis results
#  strand_id_map: Dict from NUPACK analysis results, mapping strand_id to strand name
#  n_bases: Number of 3'-end bases to analyze
#
# The unpaired probabilities in each complex which includes the given sequence-name
#  are summed, weighted by the concentration of the complex
def calculate_weighted_three_prime_end_unpaired_probabilities(
    sequence_name: str,
    sequence: str,
    sequence_concentration_molar: float,
    complex_concentrations: Dict[nupack.Complex, float],
    unpaired_probabilities: Dict[nupack.Complex, Dict[Tuple[int, int], float]],
    strand_id_map: Dict[nupack.Complex, Dict[int, str]],
    n_bases: int = 3,
) -> tuple[float, tuple[float, ...]]:
    # Calculate weighted unpaired probability
    weighted_unpaired_probs = []  # One value per base position

    for base_offset in range(n_bases):
        base_idx = len(sequence) - 1 - base_offset  # Count from 3'-end
        weighted_prob_for_base = 0.0

        # Sum over all complexes containing this primer
        for (
            complex_obj,
            complex_unpaired_probs,
        ) in unpaired_probabilities.items():
            complex_strand_map = strand_id_map[complex_obj]

            # Find this primer in the complex
            sequence_strand_idx = None
            assert complex_strand_map is not None
            for strand_idx, strand_name in complex_strand_map.items():
                if strand_name == sequence_name:
                    sequence_strand_idx = strand_idx
                    break

            if sequence_strand_idx is not None:
                # This sequence is in this complex
                complex_concentration = complex_concentrations[complex_obj]

                # Calculate how much of this primer's concentration is in this complex
                primer_strands_in_complex = sum(
                    1 for name in complex_strand_map.values() if name == sequence_name
                )
                contribution_to_primer_conc = (
                    complex_concentration * primer_strands_in_complex
                )

                # Get unpaired probability for this base in this complex
                unpaired_prob_in_complex = complex_unpaired_probs.get(
                    (sequence_strand_idx, base_idx)
                )

                if unpaired_prob_in_complex is None:
                    raise ValueError(
                        f"Missing unpaired probability for primer base {base_idx} "
                        f"(offset {base_offset}) in complex {complex_obj}"
                    )
                # Weight by the fraction of primer that's in this complex
                weight = contribution_to_primer_conc / sequence_concentration_molar
                weighted_prob_for_base += unpaired_prob_in_complex * weight

        weighted_unpaired_probs.append(weighted_prob_for_base)

    # Average weighted probability across the n_bases
    weighted_unpaired_prob = sum(weighted_unpaired_probs) / len(weighted_unpaired_probs)

    return weighted_unpaired_prob, tuple(weighted_unpaired_probs)


def analyze_multi_primer_solution(
    primer_sequences: List[str],
    primer_names: List[str],
    primer_concentrations: List[float],
    n_bases: int = 3,
    temp_celsius: float = NASBA_TEMPERATURE_CELSIUS,
    sodium_molar: float = NASBA_SODIUM_MOLAR,
    magnesium_molar: float = NASBA_MAGNESIUM_MOLAR,
) -> Dict[str, Dict[str, float | tuple[float, ...]]]:
    """
    Efficiently analyze all primers in a multi-primer solution with a single NUPACK run.

    This function performs one NUPACK analysis for all primers together, then extracts
    results for each primer. This is much more efficient than separate pairwise analyzes.

    Args:
        primer_sequences: List of all primer sequences
        primer_names: List of primer names (must match sequences)
        primer_concentrations: List of primer concentrations in M
        n_bases: Number of 3'-end bases to analyze
        temp_celsius: Temperature in Celsius
        sodium_molar: Sodium concentration in M
        magnesium_molar: Magnesium concentration in M

    Returns:
        Dict mapping primer names to their analysis results:
        {
            "primer1": {
                "monomer_fraction": 0.95,
                "weighted_unpaired_prob": 0.88,
                "total_concentration_check": 250e-9
            },
            ...
        }

    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    if len(primer_sequences) != len(primer_names) != len(primer_concentrations):
        raise ValueError(
            f"Length mismatch: sequences={len(primer_sequences)}, names={len(primer_names)}, concentrations={len(primer_concentrations)}"
        )

    if len(primer_sequences) < 2:
        raise ValueError("Need at least 2 primers for multi-primer analysis")

    # Perform single NUPACK analysis with all primers
    result = _analyze_sequences_with_nupack(
        sequences=primer_sequences,
        sequence_names=primer_names,
        concentrations_molar=primer_concentrations,
        temp_celsius=temp_celsius,
        sodium_molar=sodium_molar,
        magnesium_molar=magnesium_molar,
        max_complex_size=2,
        include_base_pairing=True,
    )

    if not result.unpaired_probabilities or not result.strand_id_map:
        raise ValueError("NUPACK analysis failed to return base-pairing data")

    # Analyze each primer in the context of the full solution
    primer_results = {}

    for i, (primer_seq, primer_name, primer_conc) in enumerate(
        zip(primer_sequences, primer_names, primer_concentrations)
    ):
        weighted_unpaired_prob, weighted_unpaired_probs = (
            calculate_weighted_three_prime_end_unpaired_probabilities(
                sequence_name=primer_name,
                sequence=primer_seq,
                sequence_concentration_molar=primer_conc,
                complex_concentrations=result.complex_concentrations,
                unpaired_probabilities=result.unpaired_probabilities,
                strand_id_map=result.strand_id_map,
                n_bases=n_bases,
            )
        )

        # Calculate monomer fraction
        monomer_concentration = 0.0
        total_primer_concentration_check = 0.0

        for complex_obj, complex_conc in result.complex_concentrations.items():
            complex_strand_map = result.strand_id_map[complex_obj]

            total_primer_concentration_check += (
                sum(1 for name in complex_strand_map.values() if name == primer_name)
                * complex_conc
            )

            # Skip complexes whose size is not 1
            if len(complex_strand_map) != 1:
                continue

            # Skip complexes that do not contain this primer
            if primer_name not in complex_strand_map.values():
                continue

            monomer_concentration = complex_conc
            break

        # Calculate monomer fraction using input concentration as denominator
        monomer_fraction = (
            (monomer_concentration / primer_conc) if primer_conc > 0 else 0.0
        )

        # Store results
        primer_results[primer_name] = {
            "monomer_fraction": monomer_fraction,
            "weighted_unpaired_prob": weighted_unpaired_prob,
            "weighted_unpaired_probs": weighted_unpaired_probs,
        }

        # Check total concentration conservation
        concentration_error = (
            abs(total_primer_concentration_check - primer_conc) / primer_conc
        )
        if concentration_error > 0.01:  # 1% tolerance
            raise ValueError(
                f"Concentration conservation failed for {primer_name}: "
                f"expected={primer_conc:.2e}, found={total_primer_concentration_check:.2e}, "
                f"error={concentration_error:.1%} > 1%"
            )

    return primer_results


def calculate_bound_fraction_nupack(
    primer_sequence: str,
    target_sequence: str,
    temp_celsius: float = NASBA_TEMPERATURE_CELSIUS,
    sodium_molar: float = NASBA_SODIUM_MOLAR,
    magnesium_molar: float = NASBA_MAGNESIUM_MOLAR,
    primer_concentration_molar: float = NASBA_PRIMER_CONCENTRATION_MOLAR,
) -> float:
    """
    Calculate the bound fraction of primer-target duplex using NUPACK.

    Args:
        primer_sequence: Primer sequence (DNA)
        target_sequence: Target sequence (DNA, should be reverse complement of primer for perfect match)
        temp_celsius: Temperature in Celsius
        sodium_molar: Sodium concentration in M
        magnesium_molar: Magnesium concentration in M
        primer_concentration_molar: Primer concentration in M

    Returns:
        Bound fraction (0.0 to 1.0)
    """
    try:
        # Create NUPACK strands
        primer = nupack.Strand(primer_sequence, name='primer', material='dna')
        target = nupack.Strand(target_sequence, name='target', material='dna')

        # Create thermodynamic model
        model = nupack.Model(
            material='dna',
            celsius=temp_celsius,
            sodium=sodium_molar,
            magnesium=magnesium_molar,
        )

        # Create tube with specified concentrations
        tube = nupack.Tube(
            name='nasba_primer_tube',
            strands={
                primer: primer_concentration_molar,
                target: primer_concentration_molar,
            },
            complexes=SetSpec(max_size=2),
        )

        # Calculate equilibrium concentrations
        result = nupack.tube_analysis(tubes=[tube], model=model, compute=['pfunc'])

        duplex_complex = nupack.Complex([primer, target])

        # Get duplex concentration
        duplex_concentration = result.tubes[tube].complex_concentrations[duplex_complex]

        # Calculate bound fraction (duplex concentration / initial primer concentration)
        bound_fraction = duplex_concentration / primer_concentration_molar

        return min(1.0, max(0.0, bound_fraction))  # Clamp to [0, 1]

    except Exception as e:
        print(
            f"Warning: NUPACK calculation failed for sequences {primer_sequence[:10]}.../{target_sequence[:10]}...: {e}"
        )
        raise ValueError(
            f"Failed to calculate bound fraction for primer '{primer_sequence}' and target '{target_sequence}'. "
            "Ensure sequences are valid and NUPACK is properly configured."
        ) from e


def calculate_primer_bound_fractions(
    anchor_sequence: str,
    toehold_sequence: str,
) -> Tuple[float, float]:
    """
    Calculate bound fractions for anchor and toehold segments of a NASBA primer.

    Args:
        anchor_sequence: Anchor segment sequence
        toehold_sequence: Toehold segment sequence

    Returns:
        Tuple of (anchor_bound_fraction, toehold_bound_fraction)
    """
    from Bio.Seq import Seq

    # For binding calculation, we want the reverse complement of the primer segments
    # because we're calculating primer-target binding
    anchor_target = str(Seq(anchor_sequence).reverse_complement())
    toehold_target = str(Seq(toehold_sequence).reverse_complement())

    # Calculate bound fractions using NASBA conditions
    anchor_bound_fraction = calculate_bound_fraction_nupack(
        primer_sequence=anchor_sequence,
        target_sequence=anchor_target,
        temp_celsius=NASBA_CONDITIONS['target_temp_C'],
        sodium_molar=NASBA_CONDITIONS['Na_mM'] / 1000.0,
        magnesium_molar=NASBA_CONDITIONS['Mg_mM'] / 1000.0,
        primer_concentration_molar=NASBA_CONDITIONS['primer_uM'] / 1e6,
    )

    toehold_bound_fraction = calculate_bound_fraction_nupack(
        primer_sequence=toehold_sequence,
        target_sequence=toehold_target,
        temp_celsius=NASBA_CONDITIONS['target_temp_C'],
        sodium_molar=NASBA_CONDITIONS['Na_mM'] / 1000.0,
        magnesium_molar=NASBA_CONDITIONS['Mg_mM'] / 1000.0,
        primer_concentration_molar=NASBA_CONDITIONS['primer_uM'] / 1e6,
    )

    return anchor_bound_fraction, toehold_bound_fraction


# ============================================================================
# VALIDATION AND SCORING
# ============================================================================


@dataclass
class BoundFractionResult:
    """Results of bound fraction analysis for a primer candidate."""

    anchor_bound_fraction: float
    toehold_bound_fraction: float
    anchor_valid: bool
    toehold_valid: bool
    is_valid: bool
    score: float


def validate_bound_fractions(
    anchor_bound_fraction: float,
    toehold_bound_fraction: float,
) -> BoundFractionResult:
    """
    Validate bound fractions against target ranges.

    Args:
        anchor_bound_fraction: Calculated anchor bound fraction
        toehold_bound_fraction: Calculated toehold bound fraction

    Returns:
        BoundFractionResult with validation status and score
    """
    # Check if bound fractions are within target ranges
    anchor_valid = (
        BOUND_FRACTION_TARGETS['anchor_min']
        <= anchor_bound_fraction
        <= BOUND_FRACTION_TARGETS['anchor_max']
    )

    toehold_valid = (
        BOUND_FRACTION_TARGETS['toehold_min']
        <= toehold_bound_fraction
        <= BOUND_FRACTION_TARGETS['toehold_max']
    )

    is_valid = anchor_valid and toehold_valid

    # Calculate score based on proximity to ideal values
    # Ideal anchor: 0.985 (middle of 0.97-1.0 range)
    # Ideal toehold: 0.75 (middle of 0.6-0.9 range)
    ideal_anchor = (
        BOUND_FRACTION_TARGETS['anchor_min'] + BOUND_FRACTION_TARGETS['anchor_max']
    ) / 2
    ideal_toehold = (
        BOUND_FRACTION_TARGETS['toehold_min'] + BOUND_FRACTION_TARGETS['toehold_max']
    ) / 2

    anchor_score = 1.0 - min(1.0, abs(anchor_bound_fraction - ideal_anchor) / 0.1)
    toehold_score = 1.0 - min(1.0, abs(toehold_bound_fraction - ideal_toehold) / 0.15)

    # Overall score is average of both scores
    score = (anchor_score + toehold_score) / 2.0

    if not is_valid:
        # If invalid, explain why
        if not anchor_valid:
            print(
                f"Anchor bound fraction {anchor_bound_fraction:.3f} is out of range "
                f"[{BOUND_FRACTION_TARGETS['anchor_min']:.2f}, {BOUND_FRACTION_TARGETS['anchor_max']:.2f}]"
            )
        if not toehold_valid:
            print(
                f"Toehold bound fraction {toehold_bound_fraction:.3f} is out of range "
                f"[{BOUND_FRACTION_TARGETS['toehold_min']:.2f}, {BOUND_FRACTION_TARGETS['toehold_max']:.2f}]"
            )

    return BoundFractionResult(
        anchor_bound_fraction=anchor_bound_fraction,
        toehold_bound_fraction=toehold_bound_fraction,
        anchor_valid=anchor_valid,
        toehold_valid=toehold_valid,
        is_valid=is_valid,
        score=score,
    )


def score_candidates_by_bound_fraction(candidates: List) -> List:
    """
    Score and sort primer candidates by bound fraction criteria.

    Args:
        candidates: List of primer candidates with bound fraction results

    Returns:
        Sorted list of candidates (best to worst)
    """
    # Sort by bound fraction score (highest first)
    return sorted(
        candidates, key=lambda x: getattr(x, 'bound_fraction_score', 0.0), reverse=True
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def find_strand_index_by_name(
    strand_id_map: Dict,
    complex_obj,
    target_strand_name: str,
) -> int:
    """
    Find the strand index for a given strand name within a specific complex.

    Args:
        strand_id_map: The strand_id_map from NupackAnalysisResult
        complex_obj: The complex to search within
        target_strand_name: The name of the strand to find

    Returns:
        The strand index for the target strand name

    Raises:
        ValueError: If complex or strand name not found
    """
    if not strand_id_map:
        raise ValueError("strand_id_map is None or empty")

    complex_strand_map = strand_id_map.get(complex_obj)
    if complex_strand_map is None:
        available_complexes = list(strand_id_map.keys())
        raise ValueError(
            f"Complex {complex_obj} not found in strand_id_map. "
            f"Available complexes: {available_complexes}"
        )

    # Find the strand index that corresponds to the target name
    for strand_idx, strand_name in complex_strand_map.items():
        if strand_name == target_strand_name:
            return strand_idx

    # If not found, raise descriptive error
    available_names = list(complex_strand_map.values())
    raise ValueError(
        f"Strand '{target_strand_name}' not found in complex {complex_obj}. "
        f"Available strand names: {available_names}"
    )


def calculate_monomer_fractions(
    sequence1: str, sequence2: str, temp_celsius: float, assay_type: str
) -> Tuple[float, float]:
    """
    Calculate monomer fractions for two sequences.

    Args:
        sequence1: First DNA sequence
        sequence2: Second DNA sequence
        temp_celsius: Temperature in Celsius
        assay_type: Type of assay (determines concentrations)

    Returns:
        Tuple of (seq1_monomer_fraction, seq2_monomer_fraction)
        where fractions are monomer_concentration / input_concentration
    """
    # Get concentrations for assay type
    concentrations = DEFAULT_ASSAY_CONCENTRATIONS

    if assay_type == "primer_dais_binding":
        conc_dict = concentrations.primer_dais_binding
        seq1_conc = conc_dict['primer_concentration_M']
        seq2_conc = conc_dict['target_dais_concentration_M']
    elif assay_type == "cross_reactivity":
        conc_dict = concentrations.primer_non_signal_cross_reactivity
        seq1_conc = conc_dict['primer_concentration_M']
        seq2_conc = conc_dict['other_sequence_concentration_M']
    elif assay_type == "off_target_dais":
        conc_dict = concentrations.off_target_dais
        seq1_conc = conc_dict['primer_concentration_M']
        seq2_conc = conc_dict['non_intended_dais_concentration_M']
    elif assay_type == "amplicon_strand_binding":
        conc_dict = concentrations.amplicon_strand_binding
        seq1_conc = conc_dict['amplicon_strand1_concentration_M']
        seq2_conc = conc_dict['amplicon_strand2_concentration_M']
    elif assay_type == "generic_primer_amplicon":
        conc_dict = concentrations.generic_primer_amplicon
        seq1_conc = conc_dict['generic_primer_concentration_M']
        seq2_conc = conc_dict['amplicon_sense_concentration_M']
    else:
        raise ValueError(f"Unknown assay type: {assay_type}")

    # Analyze with NUPACK
    result = _analyze_sequences_with_nupack(
        sequences=[sequence1, sequence2],
        sequence_names=["seq1", "seq2"],
        concentrations_molar=[seq1_conc, seq2_conc],
        temp_celsius=temp_celsius,
        max_complex_size=2,
    )

    # Find monomer concentrations
    seq1_monomer_conc = 0.0
    seq2_monomer_conc = 0.0

    for complex_obj, concentration in result.complex_concentrations.items():
        # Get strand count for this complex
        strand_count = len(complex_obj.strands)

        if strand_count == 1:
            # This is a monomer complex
            strand = complex_obj.strands[0]
            strand_seq = str(strand)

            if strand_seq == sequence1:
                seq1_monomer_conc += concentration
            elif strand_seq == sequence2:
                seq2_monomer_conc += concentration

    # Calculate fractions
    seq1_fraction = seq1_monomer_conc / seq1_conc if seq1_conc > 0 else 0.0
    seq2_fraction = seq2_monomer_conc / seq2_conc if seq2_conc > 0 else 0.0

    return seq1_fraction, seq2_fraction

# define a dataclass to hold the results of the calculation
@dataclass
class ComprehensiveAnalysisResult:
    """Results of NASBA calculation for a primer candidate."""
    primary_monomer_fraction: float
    dimer_fraction: dict[str, float] # sequence name to primer+sequence fraction
    weighted_three_prime_unpaired_prob: float # primer 3'-end average unpaired probability
    weighted_three_prime_unpaired_probs: tuple[float, ...] # primer 3'-end unpaired probabilities
    weighted_dimer_three_prime_paired_prob: dict[str, float] # average probability of
    weighted_dimer_three_prime_paired_probs: dict[str, tuple[float, ...]]


def analyze_sequence_comprehensive(
    primary_sequence: str,
    primary_sequence_name: str,
    primary_sequence_concentration: float,
    other_sequences: dict[str, str], # from sequence name to sequence
    other_sequence_concentrations: dict[str, float],
    temp_celsius: float,
    n_bases: int = 3,
) -> ComprehensiveAnalysisResult:

    # Create lists for multi-strand analysis
    all_sequences = [primary_sequence] + list(other_sequences.values())
    all_names = [f'seq_{i}_{name}' for i, name in enumerate([primary_sequence_name] + list(other_sequences.keys()))]
    primary_seq_name = all_names[0]
    all_concentrations = [primary_sequence_concentration] + list(other_sequence_concentrations)

    # Perform single NUPACK analysis
    result = _analyze_sequences_with_nupack(
        sequences=all_sequences,
        sequence_names=all_names,
        concentrations_molar=all_concentrations,
        temp_celsius=temp_celsius,
        max_complex_size=2,
        include_base_pairing=True,
    )

    # Calculate the monomer fraction for the primer
    primary_monomer_concentration = 0.0

    for complex_obj, complex_conc in result.complex_concentrations.items():
        complex_strand_map = result.strand_id_map[complex_obj]

        # Check if this is a monomer complex containing only our primer
        if (
            len(complex_strand_map) == 1
            and primary_seq_name in complex_strand_map.values()
        ):
            primary_monomer_concentration = complex_conc
            break

    monomer_fraction = (
        primary_monomer_concentration / primary_sequence_concentration if primary_sequence_concentration > 0 else 0.0
    )

    weighted_unpaired_prob, weighted_unpaired_probs = (
        calculate_weighted_three_prime_end_unpaired_probabilities(
            sequence_name=primary_seq_name,
            sequence=primary_sequence,
            sequence_concentration_molar=primary_sequence_concentration,
            complex_concentrations=result.complex_concentrations,
            unpaired_probabilities=result.unpaired_probabilities,
            strand_id_map=result.strand_id_map,
            n_bases=n_bases,
        )
    )

    dimer_fraction_dict = {}
    weighted_dimer_three_prime_paired_prob = {}
    weighted_dimer_three_prime_paired_probs = {}
    for i, (other_sequence_name, other_seq) in enumerate(other_sequences.items()):
        other_seq_name = f'seq_{i+1}_{other_sequence_name}'
        for complex_obj, complex_conc in result.complex_concentrations.items():
            complex_strand_map = result.strand_id_map[complex_obj]
            if (len(complex_strand_map) != 2
                    or primary_seq_name not in complex_strand_map.values()
                    or other_seq_name not in complex_strand_map.values()
            ):
                continue
            # This is a dimer complex containing our primer and the other sequence
            dimer_fraction_dict[other_sequence_name] = (
                complex_conc / min(primary_sequence_concentration, other_sequence_concentrations[other_seq_name])
            )
            weighted_dimer_three_prime_paired_prob[other_sequence_name], weighted_dimer_three_prime_paired_probs[other_sequence_name] = (
                calculate_weighted_three_prime_end_paired_probabilities(
                    sequence_name=primary_seq_name,
                    other_sequence_name=other_seq_name,
                    sequence=primary_sequence,
                    other_sequence=other_sequences[other_sequence_name],
                    sequence_concentration_molar=primary_sequence_concentration,
                    other_sequence_concentration_molar=other_sequence_concentrations[other_sequence_name],
                    dimer_concentration=complex_conc,
                    dimer_unpaired_probabilities=result.unpaired_probabilities[complex_obj],
                    dimer_id_map=complex_strand_map,
                    n_bases=n_bases,
                )
            )

    return ComprehensiveAnalysisResult(
        primary_monomer_fraction=monomer_fraction,
        dimer_fraction=dimer_fraction_dict,
        weighted_three_prime_unpaired_prob=weighted_unpaired_prob,
        weighted_three_prime_unpaired_probs=weighted_unpaired_probs,
        weighted_dimer_three_prime_paired_prob=weighted_dimer_three_prime_paired_prob,
        weighted_dimer_three_prime_paired_probs=weighted_dimer_three_prime_paired_probs,
    )


def print_bound_fraction_summary(
    anchor_bf: float,
    toehold_bf: float,
    result: BoundFractionResult,
    primer_name: str = "Primer",
) -> None:
    """Print a summary of bound fraction analysis."""
    print(f"\n{primer_name} Bound Fraction Analysis:")
    print(
        f"  Anchor:   {anchor_bf:.3f} (target: {BOUND_FRACTION_TARGETS['anchor_min']:.2f}-{BOUND_FRACTION_TARGETS['anchor_max']:.2f}) {'✓' if result.anchor_valid else '✗'}"
    )
    print(
        f"  Toehold:  {toehold_bf:.3f} (target: {BOUND_FRACTION_TARGETS['toehold_min']:.2f}-{BOUND_FRACTION_TARGETS['toehold_max']:.2f}) {'✓' if result.toehold_valid else '✗'}"
    )
    print(
        f"  Overall:  {'VALID' if result.is_valid else 'INVALID'} (score: {result.score:.3f})"
    )


def get_bound_fraction_targets() -> Dict[str, float]:
    """Get current bound fraction targets."""
    return BOUND_FRACTION_TARGETS.copy()


def update_bound_fraction_targets(
    toehold_min: float = None,
    toehold_max: float = None,
    anchor_min: float = None,
    anchor_max: float = None,
) -> None:
    """Update bound fraction target ranges."""
    if toehold_min is not None:
        BOUND_FRACTION_TARGETS['toehold_min'] = toehold_min
    if toehold_max is not None:
        BOUND_FRACTION_TARGETS['toehold_max'] = toehold_max
    if anchor_min is not None:
        BOUND_FRACTION_TARGETS['anchor_min'] = anchor_min
    if anchor_max is not None:
        BOUND_FRACTION_TARGETS['anchor_max'] = anchor_max
