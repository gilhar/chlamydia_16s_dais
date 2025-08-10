#!/usr/bin/env python3
"""
Debug script to compare direct NUPACK calls with our internal API
to identify where the difference occurs.
"""
import numpy as np
import nupack
from nupack import SetSpec
from nasba_primer_thermodynamics import _analyze_sequences_with_nupack


def direct_nupack_call():
    """Reproduce the working direct NUPACK call from test_nupack_pairs.py"""
    print("=" * 60)
    print("DIRECT NUPACK CALL")
    print("=" * 60)

    seq1 = "ATCG"
    seq2 = "CGAT"

    # Create NUPACK strands
    strand1 = nupack.Strand(seq1, name='strand1', material='dna')
    strand2 = nupack.Strand(seq2, name='strand2', material='dna')

    # Create thermodynamic model
    model = nupack.Model(material='dna', celsius=41.0, sodium=0.08, magnesium=0.012)

    # Create tube
    tube = nupack.Tube(
        name='test_tube',
        strands={strand1: 250e-9, strand2: 250e-9},
        complexes=SetSpec(max_size=2),
    )

    # Run analysis
    result = nupack.tube_analysis(tubes=[tube], model=model, compute=['pfunc', 'pairs'])

    # Get dimer complex pairs matrix
    dimer_complex = nupack.Complex([strand1, strand2])
    pairs_matrix = result.complexes[dimer_complex].pairs.to_array()

    print(f"Direct call - Matrix shape: {pairs_matrix.shape}")
    print(f"Direct call - Matrix dtype: {pairs_matrix.dtype}")
    print(f"Direct call - Sample diagonal values:")
    for i in range(min(4, pairs_matrix.shape[0])):
        print(f"  [{i}, {i}] = {pairs_matrix[i, i]}")

    return pairs_matrix


def internal_api_call():
    """Test our internal API with the same parameters"""
    print("\n" + "=" * 60)
    print("INTERNAL API CALL")
    print("=" * 60)

    try:
        result = _analyze_sequences_with_nupack(
            sequences=["ATCG", "CGAT"],
            sequence_names=['strand1', 'strand2'],
            concentrations_molar=[250e-9, 250e-9],
            temp_celsius=41.0,
            sodium_molar=0.08,
            magnesium_molar=0.012,
            max_complex_size=2,
            include_base_pairing=True,
        )

        print(f"Internal API - Analysis completed successfully")
        print(
            f"Internal API - Unpaired probabilities found: {len(result.unpaired_probabilities)}"
        )

        if result.unpaired_probabilities:
            print(f"Internal API - Sample unpaired probabilities:")
            for key, value in list(result.unpaired_probabilities.items())[:4]:
                print(f"  {key} = {value}")

        return result

    except Exception as exc:
        print(f"Internal API - Error: {exc}")
        print(f"Internal API - Error type: {type(exc)}")
        return None


def debug_internal_api():
    """Debug the internal API step by step"""
    print("\n" + "=" * 60)
    print("DEBUGGING INTERNAL API STEP BY STEP")
    print("=" * 60)

    sequences = ["ATCG", "CGAT"]
    sequence_names = ['strand1', 'strand2']
    concentrations_molar = [250e-9, 250e-9]

    # Create NUPACK strands (same as internal function)
    strands = []
    strand_concentrations = {}

    for i, (seq, name, conc) in enumerate(
        zip(sequences, sequence_names, concentrations_molar)
    ):
        strand = nupack.Strand(seq, name=name, material='dna')
        strands.append(strand)
        strand_concentrations[strand] = conc
        print(f"Created strand {i}: {strand}")

    # Create model (same as internal function)
    model = nupack.Model(material='dna', celsius=41.0, sodium=0.08, magnesium=0.012)
    print(f"Created model: {model}")

    # Create tube (same as internal function)
    tube = nupack.Tube(
        name='analysis_tube',
        strands=strand_concentrations,
        complexes=SetSpec(max_size=2),
    )
    print(f"Created tube: {tube}")

    # Run analysis (same as internal function)
    compute_params = ['pfunc', 'pairs']
    print(f"Running analysis with compute params: {compute_params}")

    result = nupack.tube_analysis(tubes=[tube], model=model, compute=compute_params)
    print(f"Analysis completed")

    # Check available complexes
    print(f"Available complexes: {list(result.complexes.keys())}")

    # Try to get dimer complex
    dimer_complex = nupack.Complex(strands)
    print(f"Looking for dimer complex: {dimer_complex}")

    if dimer_complex in result.complexes:
        complex_result = result.complexes[dimer_complex]
        print(f"Found dimer complex")

        if hasattr(complex_result, 'pairs'):
            pairs_matrix = complex_result.pairs.to_array()
            print(f"Pairs matrix shape: {pairs_matrix.shape}")
            print(f"Pairs matrix dtype: {pairs_matrix.dtype}")
            print(f"Sample matrix values:")
            print(f"  [0, 0] = {pairs_matrix[0, 0]}")
            print(f"  [0, 1] = {pairs_matrix[0, 1]}")
            print(f"  [1, 1] = {pairs_matrix[1, 1]}")
            return pairs_matrix
        else:
            print("No pairs attribute found")
    else:
        print("Dimer complex not found")

    return None


if __name__ == "__main__":
    try:
        direct_matrix = direct_nupack_call()
        internal_result = internal_api_call()
        debug_matrix = debug_internal_api()

        # Compare results
        if direct_matrix is not None and debug_matrix is not None:
            print("\n" + "=" * 60)
            print("COMPARISON")
            print("=" * 60)
            print(f"Direct matrix [0,0]: {direct_matrix[0, 0]}")
            print(f"Debug matrix [0,0]:  {debug_matrix[0, 0]}")

            # reassure linters that (direct_matrix == debug_matrix) is a numpy array
            comp_matrix = direct_matrix == debug_matrix
            assert isinstance(comp_matrix, np.ndarray)
            print(f"Matrices are equal: {comp_matrix.all()}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
