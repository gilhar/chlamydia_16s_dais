#!/usr/bin/env python3
"""
Test program to investigate NUPACK pairs matrix structure.

This program creates a simple two-strand system and examines the pairs matrix
to understand its structure and indexing scheme.
"""

import numpy as np
import nupack
from nupack import SetSpec
from chlamydia_16s_dais.nasba_primer_thermodynamics import NASBA_CONDITIONS


def test_nupack_pairs_matrix():
    """Test NUPACK pairs matrix with two simple complementary strands."""

    print("=" * 80)
    print("NUPACK PAIRS MATRIX INVESTIGATION")
    print("=" * 80)

    # Create two simple complementary strands
    seq1 = "ATCG"
    seq2 = "CGAT"  # Reverse complement of seq1

    print(f"\nSequence 1: {seq1}")
    print(f"Sequence 2: {seq2} (reverse complement)")
    print(f"Expected pairing: A-T, T-A, C-G, G-C")

    # Create NUPACK strands
    strand1 = nupack.Strand(seq1, name='strand1', material='dna')
    strand2 = nupack.Strand(seq2, name='strand2', material='dna')

    print(f"\nStrand 1: {strand1} (length: {len(seq1)})")
    print(f"Strand 2: {strand2} (length: {len(seq2)})")

    # Create thermodynamic model (NASBA conditions)
    model = nupack.Model(
        material='dna',
        celsius=NASBA_CONDITIONS['target_temp_C'],
        sodium=NASBA_CONDITIONS['Na_mM'] / 1e3,
        magnesium=NASBA_CONDITIONS['Mg_mM'] / 1e3
    )

    # Create tube with both strands
    tube = nupack.Tube(
        name='test_tube',
        strands={strand1: 250e-9, strand2: 250e-9},  # 250nM  # 250nM
        complexes=SetSpec(max_size=2),
    )

    print(f"\nRunning NUPACK analysis with pairs computation...")

    # Run analysis with pairs computation
    result = nupack.tube_analysis(tubes=[tube], model=model, compute=['pfunc', 'pairs'])

    # Get the dimer complex
    dimer_complex = nupack.Complex([strand1, strand2])

    print(f"\nDimer complex: {dimer_complex}")
    print(f"Available complexes: {list(result.complexes.keys())}")

    if dimer_complex in result.complexes:
        complex_result = result.complexes[dimer_complex]
        print(f"\nComplex result type: {type(complex_result)}")
        print(f"Complex result attributes: {dir(complex_result)}")

        if hasattr(complex_result, 'pairs'):
            print(f"\nPairs object type: {type(complex_result.pairs)}")

            # Get the pairs matrix
            pairs_matrix = complex_result.pairs.to_array()

            print(f"\nPairs matrix shape: {pairs_matrix.shape}")
            print(f"Pairs matrix dtype: {pairs_matrix.dtype}")
            print(f"\nFull pairs matrix:")
            print(pairs_matrix)

            # Analyze the matrix structure
            print(f"\nMatrix Analysis:")
            print(f"First row: {pairs_matrix[0, :]}")
            print(f"First column: {pairs_matrix[:, 0]}")

            # Check for specific patterns
            print(f"\nDiagonal elements:")
            for i in range(pairs_matrix.shape[0]):
                print(f"  [{i}, {i}] = {pairs_matrix[i, i]:.6f}")

            print(f"\nOff-diagonal elements (significant > 1e-6):")
            for i in range(pairs_matrix.shape[0]):
                for j in range(pairs_matrix.shape[1]):
                    if i != j and abs(pairs_matrix[i, j]) > 1e-6:
                        print(f"  [{i}, {j}] = {pairs_matrix[i, j]:.6f}")

            # Try to understand row/column meaning
            print(f"\nRow sums (should sum to 1 if probabilities):")
            for i in range(pairs_matrix.shape[0]):
                row_sum = np.sum(pairs_matrix[i, :])
                print(f"  Row {i}: {row_sum:.6f}")

            print(f"\nColumn sums:")
            for j in range(pairs_matrix.shape[1]):
                col_sum = np.sum(pairs_matrix[:, j])
                print(f"  Column {j}: {col_sum:.6f}")

        else:
            print("No pairs attribute found in complex result!")
    else:
        print(f"Dimer complex not found in results!")
        print(f"Available complexes:")
        for complex_key in result.complexes.keys():
            print(f"  {complex_key}")


def test_single_strand():
    """Test with a single strand to see if that changes the matrix structure."""

    print("\n" + "=" * 80)
    print("SINGLE STRAND TEST")
    print("=" * 80)

    seq = "ATCGATCG"
    print(f"\nSingle sequence: {seq}")

    strand = nupack.Strand(seq, name='single_strand', material='dna')

    model = nupack.Model(
        material='dna',
        celsius=NASBA_CONDITIONS['target_temp_C'],
        sodium=NASBA_CONDITIONS['Na_mM'] / 1e3,
        magnesium=NASBA_CONDITIONS['Mg_mM'] / 1e3
    )

    tube = nupack.Tube(
        name='single_tube',
        strands={strand: 250e-9},
        complexes=SetSpec(max_size=1),
    )

    result = nupack.tube_analysis(tubes=[tube], model=model, compute=['pfunc', 'pairs'])

    single_complex = nupack.Complex([strand])

    if single_complex in result.complexes:
        complex_result = result.complexes[single_complex]

        if hasattr(complex_result, 'pairs'):
            pairs_matrix = complex_result.pairs.to_array()

            print(f"\nSingle strand pairs matrix shape: {pairs_matrix.shape}")
            print(f"Single strand pairs matrix:")
            print(pairs_matrix)

            print(f"\nDiagonal elements (unpaired probabilities):")
            for i in range(pairs_matrix.shape[0]):
                if i < len(seq):
                    base = seq[i]
                    print(f"  Base {i} ({base}): {pairs_matrix[i, i]:.6f}")
                else:
                    print(f"  Position {i}: {pairs_matrix[i, i]:.6f}")


if __name__ == "__main__":
    try:
        test_nupack_pairs_matrix()
        test_single_strand()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
