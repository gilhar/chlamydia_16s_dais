#!/usr/bin/env python3
"""
NUPACK Complex Analysis Module

This module provides functions for analyzing sequence complexes using NUPACK API 4.0.1.9.
It calculates complex concentrations, unpaired probabilities, and base-pairing probabilities
for sets of DNA/RNA sequences under specified thermodynamic conditions.

Author: Claude (Anthropic)
Date: 2025
"""

import nupack
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import click
from chlamydia_16s_dais.nupack_subprocess import (
    analyze_sequence_complexes_subprocess,
    SequenceParam,
)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SequenceInput:
    """Represents a sequence with name and concentration for analysis."""

    name: str
    sequence: str
    concentration_M: float  # Concentration in Molar


@dataclass
class ComplexResult:
    """Results for a single complex from NUPACK analysis."""

    complex_id: str
    size: int
    concentration_molar: float
    sequence_id_map: Optional[Dict[int, str]] = (
        None  # Maps sequence_id (0-based) to sequence names
    )
    unpaired_probability: Optional[Dict[Tuple[int, int], float]] = (
        None  # (seq_id, base_offset) -> prob
    )
    pairing_probability: Optional[Dict[Tuple[int, int, int, int], float]] = (
        None  # (seq_id_1, base_1, seq_id_2, base_2) -> prob
    )


@dataclass
class ComplexAnalysisResult:
    """Complete results from complex analysis."""

    temperature_celsius: float
    ionic_conditions: Dict[str, float]
    max_complex_size: int
    total_sequences: int
    complexes: List[ComplexResult]

    def get_complex_by_size(self, size: int) -> List[ComplexResult]:
        """Get all complexes of a specific size."""
        return [c for c in self.complexes if c.size == size]

    def get_monomer_fraction(
        self, sequence_name: str, sequence_input_conc_molar: float
    ) -> float:
        """Return the fraction of a sequence that remains as monomer using the known input concentration.

        Specifically:
            fraction = [monomer(sequence_name)] / sequence_input_conc_M

        Args:
            sequence_name: Name of the sequence whose monomer fraction is requested
            sequence_input_conc_molar: Tube input concentration (M) of the sequence

        Raises:
            ValueError: On invalid name or non-positive/None input concentration.
        """
        if not isinstance(sequence_name, str) or not sequence_name:
            raise ValueError("sequence_name must be a non-empty string")
        if sequence_input_conc_molar is None:
            raise ValueError("sequence_input_conc_M must not be None")
        if sequence_input_conc_molar <= 0:
            raise ValueError("sequence_input_conc_M must be positive")

        # Sum concentrations of monomer complexes for this sequence
        monomer_conc = 0.0
        for c in self.complexes:
            if (
                c.size == 1
                and c.sequence_id_map
                and sequence_name in c.sequence_id_map.values()
            ):
                monomer_conc += c.concentration_molar

        return monomer_conc / sequence_input_conc_molar

    def get_hetero_dimer_fraction(
        self,
        seq1_name: str,
        seq2_name: str,
        seq1_input_conc_molar: float,
        seq2_input_conc_molar: float,
    ) -> float:
        """Return the hetero-dimer fraction relative to the limiting input concentration.

        Specifically:
            fraction = [hetero-dimer] / min(seq1_input_conc_M, seq2_input_conc_M)

        Args:
            seq1_name: Name of the first strand (must differ from seq2_name)
            seq2_name: Name of the second strand (must differ from seq1_name)
            seq1_input_conc_molar: Tube input concentration (M) of seq1
            seq2_input_conc_molar: Tube input concentration (M) of seq2

        Raises:
            ValueError: On invalid names, equal names, or non-positive/None input concentrations.
        """
        # Validate parameters
        if not isinstance(seq1_name, str) or not seq1_name:
            raise ValueError("seq1_name must be a non-empty string")
        if not isinstance(seq2_name, str) or not seq2_name:
            raise ValueError("seq2_name must be a non-empty string")
        if seq1_name == seq2_name:
            raise ValueError(
                "seq1_name and seq2_name must be different for hetero-dimer fraction"
            )
        if seq1_input_conc_molar is None or seq2_input_conc_molar is None:
            raise ValueError("Input concentrations must not be None")
        if seq1_input_conc_molar <= 0 or seq2_input_conc_molar <= 0:
            raise ValueError("Input concentrations must be positive")

        # Sum concentration of all hetero-dimer complexes containing exactly one copy of each
        hetero_dimer_conc = 0.0
        for c in self.complexes:
            if c.size == 2 and c.sequence_id_map:
                names = list(c.sequence_id_map.values())
                if names.count(seq1_name) == 1 and names.count(seq2_name) == 1:
                    hetero_dimer_conc += c.concentration_molar

        limiting = min(seq1_input_conc_molar, seq2_input_conc_molar)
        return hetero_dimer_conc / limiting

    def get_homo_dimer_fraction(
        self, sequence_name: str, sequence_input_conc_molar: float
    ) -> float:
        """Return the homodimer fraction of the given strand relative to its input concentration.

        For a homodimer A2, the fraction of A monomers that are in A2 is:
            fraction = (2 * [A2]) / [A]_input

        Args:
            sequence_name: Name of the strand forming the homodimer (A in A2)
            sequence_input_conc_molar: Tube input concentration (M) of the strand

        Raises:
            ValueError: On invalid name or non-positive/None input concentration.
        """
        if not isinstance(sequence_name, str) or not sequence_name:
            raise ValueError("sequence_name must be a non-empty string")
        if sequence_input_conc_molar is None:
            raise ValueError("sequence_input_conc_M must not be None")
        if sequence_input_conc_molar <= 0:
            raise ValueError("sequence_input_conc_M must be positive")

        homo_dimer_conc = 0.0
        for c in self.complexes:
            if c.size == 2 and c.sequence_id_map:
                names = list(c.sequence_id_map.values())
                # Homo-dimer: the same sequence appears twice
                if names.count(sequence_name) == 2:
                    homo_dimer_conc += c.concentration_molar

        # Each A2 contains 2 copies of A
        return (2.0 * homo_dimer_conc) / sequence_input_conc_molar

    def get_all_dimer_fractions(
        self, sequence_name: str, input_concentrations_molar: Dict[str, float]
    ) -> Dict[str, float]:
        """Get comprehensive dimer analysis for a sequence using provided input concentrations.

        Args:
            sequence_name: Name of the focal sequence
            input_concentrations_molar: Mapping from the sequence name to its tube input concentration (M)

        Returns:
            Dict with keys:
                - 'monomer': fraction of the sequence present as monomer (uses input concentration as denominator)
                - 'homo_dimer': fraction of monomers of sequence_name present in homodimer (2*[A2]/[A]_in)
                - 'hetero_dimers': dict mapping partner name -> fraction of limiting strand in AB
        """
        if not isinstance(sequence_name, str) or not sequence_name:
            raise ValueError("sequence_name must be a non-empty string")
        if (
            not isinstance(input_concentrations_molar, dict)
            or not input_concentrations_molar
        ):
            raise ValueError("input_concentrations_M must be a non-empty dict")
        if sequence_name not in input_concentrations_molar:
            raise ValueError(f"Missing input concentration for '{sequence_name}'")

        result = {
            'monomer': self.get_monomer_fraction(
                sequence_name, input_concentrations_molar[sequence_name]
            ),
            'homo_dimer': self.get_homo_dimer_fraction(
                sequence_name, input_concentrations_molar[sequence_name]
            ),
            'hetero_dimers': {},
        }

        # Find all hetero-dimers involving this sequence and compute limiting-strand fractions
        for c in self.complexes:
            if c.size == 2 and c.sequence_id_map:
                names = list(c.sequence_id_map.values())
                if sequence_name in names and names.count(sequence_name) == 1:
                    # Identify the partner
                    other_name = [name for name in names if name != sequence_name][0]
                    if other_name not in input_concentrations_molar:
                        raise ValueError(
                            f"Missing input concentration for hetero partner '{other_name}'"
                        )
                    frac = self.get_hetero_dimer_fraction(
                        sequence_name,
                        other_name,
                        seq1_input_conc_molar=input_concentrations_molar[sequence_name],
                        seq2_input_conc_molar=input_concentrations_molar[other_name],
                    )
                    # Store the maximum fraction observed across all heterodimers with the same partner
                    # (though there should be at most one dimer species per pair)
                    if other_name not in result['hetero_dimers']:
                        result['hetero_dimers'][other_name] = 0.0
                    result['hetero_dimers'][other_name] = max(
                        result['hetero_dimers'][other_name], frac
                    )

        return result


# ============================================================================
# NUPACK COMPLEX ANALYSIS FUNCTIONS
# ============================================================================


def analyze_sequence_complexes_inprocess(
    temperature_celsius: float,
    sequences: List[SequenceInput],
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
    max_complex_size: int = 4,
    base_pairing_analysis: bool = False,
) -> ComplexAnalysisResult:
    """
    Analyze sequence complexes using NUPACK API 4.0.1.9 (in-process).

    This function performs the computation in the current process.
    """

    # Create NUPACK model with ionic conditions
    model = nupack.Model(
        material='DNA',  # Assume DNA unless specified otherwise
        celsius=temperature_celsius,
        sodium=sodium_millimolar * 1e-3,  # Convert mM to M
        magnesium=magnesium_millimolar * 1e-3,  # Convert mM to M
    )

    # Create NUPACK strands
    nupack_strands = {}
    sequence_name_map = {}  # Maps strand index to sequence name

    for i, seq_input in enumerate(sequences):
        strand = nupack.Strand(seq_input.sequence, name=seq_input.name)
        nupack_strands[strand] = seq_input.concentration_M
        sequence_name_map[i] = seq_input.name

    # Create the tube with strands and concentrations
    tube = nupack.Tube(
        strands=nupack_strands,
        complexes=nupack.SetSpec(max_size=max_complex_size),
        name='analysis_tube',
    )

    # Run complex analysis
    compute_list = ['pfunc'] + (
        ['pairs'] if base_pairing_analysis else []
    )  # noqa: typo
    results = nupack.tube_analysis(tubes=[tube], model=model, compute=compute_list)

    # Extract complex results
    complexes = []
    tube_result = results[tube]

    for complex_spec, concentration in tube_result.complex_concentrations.items():
        # Build a sequence ID map for this complex
        complex_size = len(complex_spec.strands)
        seq_id_map = {}

        for seq_id, strand in enumerate(complex_spec.strands):
            # Find the original sequence name for this strand
            for orig_strand in nupack_strands.keys():
                if str(strand) == str(orig_strand):
                    seq_id_map[seq_id] = orig_strand.name
                    break

        complex_result = ComplexResult(
            complex_id=str(complex_spec),
            size=complex_size,
            concentration_molar=concentration,
            sequence_id_map=seq_id_map,
        )

        # Add base-pairing analysis if requested
        if base_pairing_analysis:
            # Extract pairing probabilities from NUPACK results
            if not hasattr(results, 'complexes'):
                raise RuntimeError("NUPACK results do not contain complexes")
            if not complex_spec in results.complexes:
                raise RuntimeError(
                    f"NUPACK results do not contain complex {complex_spec}"
                )
            complex_result_obj = results.complexes[complex_spec]

            if not hasattr(complex_result_obj, 'pairs'):
                raise RuntimeError(
                    "NUPACK results do not contain base-pairing probabilities"
                )
            # Get the pair's array - this is a numpy 2D array
            pairs_array = complex_result_obj.pairs.to_array()

            # Build unpaired and pairing probabilities
            unpaired_prob = {}
            pairing_prob = {}

            # Calculate cumulative base positions for each strand in the complex
            strand_lengths = [len(str(strand)) for strand in complex_spec.strands]
            cumulative_positions = [0]  # Start positions for each strand
            for length in strand_lengths:
                cumulative_positions.append(cumulative_positions[-1] + length)

            # Extract unpaired probabilities (diagonal elements)
            for seq_id in range(complex_size):
                strand_start = cumulative_positions[seq_id]
                strand_length = strand_lengths[seq_id]

                for base_idx in range(strand_length):
                    global_base_idx = strand_start + base_idx
                    base_offset = base_idx + 1  # Convert to 1-based indexing

                    # Unpaired probability is the diagonal element
                    if global_base_idx < pairs_array.shape[0]:
                        unpaired_prob[(seq_id, base_offset)] = pairs_array[
                            global_base_idx, global_base_idx
                        ]

            # Extract pairing probabilities (off-diagonal elements)
            for seq_id_1 in range(complex_size):
                strand_start_1 = cumulative_positions[seq_id_1]
                strand_length_1 = strand_lengths[seq_id_1]

                for base_idx_1 in range(strand_length_1):
                    global_base_idx_1 = strand_start_1 + base_idx_1
                    base_offset_1 = base_idx_1 + 1  # 1-based indexing

                    for seq_id_2 in range(complex_size):
                        strand_start_2 = cumulative_positions[seq_id_2]
                        strand_length_2 = strand_lengths[seq_id_2]

                        for base_idx_2 in range(strand_length_2):
                            global_base_idx_2 = strand_start_2 + base_idx_2
                            base_offset_2 = base_idx_2 + 1  # 1-based indexing

                            # Skip diagonal (unpaired) and self-pairs
                            if global_base_idx_1 != global_base_idx_2:
                                if (
                                    global_base_idx_1 >= pairs_array.shape[0]
                                    or global_base_idx_2 >= pairs_array.shape[1]
                                ):
                                    raise RuntimeError(
                                        f'indices ({global_base_idx_1}, {global_base_idx_2}) '
                                        f'out of range for pairs_array shape {pairs_array.shape}'
                                    )
                                pair_prob = pairs_array[
                                    global_base_idx_1, global_base_idx_2
                                ]

                                pairing_prob[
                                    (seq_id_1, base_offset_1, seq_id_2, base_offset_2)
                                ] = pair_prob

            complex_result.unpaired_probability = unpaired_prob
            complex_result.pairing_probability = pairing_prob

        complexes.append(complex_result)

    # Sort complexes by size then by concentration
    complexes.sort(key=lambda x: (x.size, -x.concentration_molar))

    return ComplexAnalysisResult(
        temperature_celsius=temperature_celsius,
        ionic_conditions={
            'sodium_mM': sodium_millimolar,
            'magnesium_mM': magnesium_millimolar,
        },
        max_complex_size=max_complex_size,
        total_sequences=len(sequences),
        complexes=complexes,
    )


def analyze_sequence_complexes(
    temperature_celsius: float,
    sequences: List[SequenceInput],
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
    max_complex_size: int = 4,
    base_pairing_analysis: bool = False,
) -> ComplexAnalysisResult:
    """
    Analyze sequence complexes using a short-lived worker subprocess to avoid leaks.

    This wrapper launches a one-shot worker process, passes the inputs as JSON,
    parses the returned JSON, and rehydrates the standard ComplexAnalysisResult
    data structures expected by callers.
    """
    # Build payload sequences for the worker
    seq_params = [
        SequenceParam(
            name=s.name, sequence=s.sequence, concentration_M=s.concentration_M
        )
        for s in sequences
    ]

    result_dict = analyze_sequence_complexes_subprocess(
        temperature_celsius=temperature_celsius,
        sequences=seq_params,
        sodium_millimolar=sodium_millimolar,
        magnesium_millimolar=magnesium_millimolar,
        max_complex_size=max_complex_size,
        base_pairing_analysis=base_pairing_analysis,
    )

    # Rehydrate ComplexAnalysisResult / ComplexResult from JSON-friendly dict
    complexes: List[ComplexResult] = []
    for c in result_dict.get("complexes", []):
        # sequence_id_map keys were stringified
        seq_id_map = {int(k): v for k, v in (c.get("sequence_id_map") or {}).items()}

        # Parse unpaired probabilities "seq:base" -> (seq, base)
        unpaired = None
        if c.get("unpaired_probability") is not None:
            unpaired = {}
            for k, v in c["unpaired_probability"].items():
                seq_str, base_str = k.split(":")
                unpaired[(int(seq_str), int(base_str))] = float(v)

        # Parse pairing probabilities "s1:b1|s2:b2" -> (s1,b1,s2,b2)
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
        temperature_celsius=float(result_dict["temperature_celsius"]),
        ionic_conditions={
            'sodium_mM': float(result_dict["ionic_conditions"]["sodium_mM"]),
            'magnesium_mM': float(result_dict["ionic_conditions"]["magnesium_mM"]),
        },
        max_complex_size=int(result_dict["max_complex_size"]),
        total_sequences=int(result_dict["total_sequences"]),
        complexes=complexes,
    )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR NASBA VALIDATION
# ============================================================================


def analyze_primer_dais_binding(
    primer_sequence: str,
    dais_sequence: str,
    primer_name: str = "primer",
    dais_name: str = "dais",
    temperature_celsius: float = 41.0,
    concentration_nanomolar: float = 250.0,
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
) -> Tuple[float, float]:
    """
    Analyze primer-dais binding and return hetero-dimer fraction and 3'-end unpaired probability.

    Returns:
        Tuple of (hetero_dimer_fraction, three_prime_unpaired_probability)
    """

    concentration_molar = concentration_nanomolar * 1e-9  # Convert nM to M

    sequences = [
        SequenceInput(primer_name, primer_sequence, concentration_molar),
        SequenceInput(dais_name, dais_sequence, concentration_molar),
    ]

    results = analyze_sequence_complexes(
        temperature_celsius=temperature_celsius,
        sequences=sequences,
        sodium_millimolar=sodium_millimolar,
        magnesium_millimolar=magnesium_millimolar,
        max_complex_size=2,
        base_pairing_analysis=True,
    )

    # Calculate the hetero-dimer fraction (normalized by limiting input concentration)
    hetero_dimer_fraction = results.get_hetero_dimer_fraction(
        primer_name,
        dais_name,
        seq1_input_conc_molar=concentration_molar,
        seq2_input_conc_molar=concentration_molar,
    )

    # Calculate 3'-end unpaired probability for primer
    three_prime_unpaired_prob = 0.0
    primer_length = len(primer_sequence)

    # Find hetero-dimer complex and get 3'-end unpaired probability
    for complex_result in results.complexes:
        if (
            complex_result.size == 2
            and complex_result.unpaired_probability
            and primer_name in complex_result.sequence_id_map.values()
        ):

            # Find sequence ID for primer in this complex
            primer_seq_id = None
            for seq_id, name in complex_result.sequence_id_map.items():
                if name == primer_name:
                    primer_seq_id = seq_id
                    break

            if primer_seq_id is not None:
                # Get unpaired probability for the last two bases (3'-end)
                base1_key = (primer_seq_id, primer_length - 1)  # Second to last base
                base2_key = (primer_seq_id, primer_length)  # Last base

                prob1 = complex_result.unpaired_probability.get(base1_key, 0.0)
                prob2 = complex_result.unpaired_probability.get(base2_key, 0.0)

                # Probability that both bases are unpaired
                three_prime_unpaired_prob = prob1 * prob2
                break

    return hetero_dimer_fraction, three_prime_unpaired_prob


def analyze_four_primer_cross_reactivity(
    primer_sequences: Dict[str, str],
    temperature_celsius: float = 41.0,
    concentration_nanomolar: float = 250.0,
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Analyze cross-reactivity between four primers.

    Args:
        primer_sequences: Dictionary mapping primer names to sequences

    Returns:
        Dictionary mapping primer names to (monomer_fraction, three_prime_unpaired_prob)
        :param primer_sequences:
        :param magnesium_millimolar:
        :param sodium_millimolar:
        :param concentration_nanomolar:
        :param temperature_celsius:
    """

    concentration_molar = concentration_nanomolar * 1e-9  # Convert nM to M

    sequences = [
        SequenceInput(name, seq, concentration_molar)
        for name, seq in primer_sequences.items()
    ]

    results = analyze_sequence_complexes(
        temperature_celsius=temperature_celsius,
        sequences=sequences,
        sodium_millimolar=sodium_millimolar,
        magnesium_millimolar=magnesium_millimolar,
        max_complex_size=4,
        base_pairing_analysis=True,
    )

    primer_results = {}

    for primer_name, primer_seq in primer_sequences.items():
        # Calculate monomer fraction (normalized by known input concentration)
        monomer_fraction = results.get_monomer_fraction(
            primer_name, concentration_molar
        )

        # Calculate 3'-end unpaired probability
        three_prime_unpaired_prob = 0.0
        primer_length = len(primer_seq)

        # Look through all complexes containing this primer
        for complex_result in results.complexes:
            if (
                complex_result.unpaired_probability
                and primer_name in complex_result.sequence_id_map.values()
            ):

                # Find sequence ID for primer in this complex
                primer_seq_id = None
                for seq_id, name in complex_result.sequence_id_map.items():
                    if name == primer_name:
                        primer_seq_id = seq_id
                        break

                if primer_seq_id is not None:
                    # Get unpaired probability for the last two bases (3'-end)
                    base1_key = (
                        primer_seq_id,
                        primer_length - 1,
                    )  # Second to last base
                    base2_key = (primer_seq_id, primer_length)  # Last base

                    prob1 = complex_result.unpaired_probability.get(base1_key, 0.0)
                    prob2 = complex_result.unpaired_probability.get(base2_key, 0.0)

                    # Weight by complex concentration and take maximum
                    complex_weight = complex_result.concentration_molar
                    weighted_prob = (prob1 * prob2) * complex_weight
                    three_prime_unpaired_prob = max(
                        three_prime_unpaired_prob, weighted_prob
                    )

        primer_results[primer_name] = (monomer_fraction, three_prime_unpaired_prob)

    return primer_results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================


@click.command()
@click.option('--temperature', '-T', default=41.0, help='Temperature in Celsius')
@click.option('--sodium', default=80.0, help='Sodium concentration in mM')
@click.option('--magnesium', default=12.0, help='Magnesium concentration in mM')
@click.option('--max-size', default=4, help='Maximum complex size to analyze')
@click.option('--concentration', default=250.0, help='Sequence concentration in nM')
@click.option('--base-pairing', is_flag=True, help='Enable base-pairing analysis')
@click.argument('sequences', nargs=-1, required=True)
def main(
    temperature, sodium, magnesium, max_size, concentration, base_pairing, sequences
):
    """
    Analyze sequence complexes using NUPACK.

    SEQUENCES should be provided as name:sequence pairs, e.g.:
    python nupack_complex_analysis.py primer1: ATCGATCG primer2: CGATCGAT # noqa: typo
    """

    # Parse sequence inputs
    sequence_inputs = []
    concentration_molar = concentration * 1e-9  # Convert nM to M

    for seq_str in sequences:
        if ':' not in seq_str:
            click.echo(f"Error: Invalid sequence format '{seq_str}'. Use name:sequence")
            return

        name, sequence = seq_str.split(':', 1)
        sequence_inputs.append(SequenceInput(name, sequence, concentration_molar))

    # Run analysis
    results = analyze_sequence_complexes(
        temperature_celsius=temperature,
        sequences=sequence_inputs,
        sodium_millimolar=sodium,
        magnesium_millimolar=magnesium,
        max_complex_size=max_size,
        base_pairing_analysis=base_pairing,
    )

    # Print results
    click.echo("=" * 60)
    click.echo("NUPACK COMPLEX ANALYSIS RESULTS")
    click.echo("=" * 60)
    click.echo(f"Temperature: {results.temperature_celsius}°C")
    click.echo(f"Ionic conditions: {results.ionic_conditions}")
    click.echo(f"Max complex size: {results.max_complex_size}")
    click.echo(f"Total sequences: {results.total_sequences}")
    click.echo()

    for complex_result in results.complexes:
        click.echo(f"Complex: {complex_result.complex_id}")
        click.echo(f"  Size: {complex_result.size}")
        click.echo(f"  Concentration: {complex_result.concentration_molar:.2e} M")
        if complex_result.sequence_id_map:
            click.echo(f"  Sequences: {complex_result.sequence_id_map}")

        if base_pairing and complex_result.unpaired_probability:
            click.echo("  Unpaired probabilities (seq_id, base_offset): prob")
            for key, prob in sorted(complex_result.unpaired_probability.items()):
                if prob > 0.01:  # Only show significant probabilities
                    click.echo(f"    {key}: {prob:.3f}")

        click.echo()


if __name__ == "__main__":
    main()
