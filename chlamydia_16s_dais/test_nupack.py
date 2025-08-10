#!/usr/bin/env python3
"""
Test script for NUPACK complex analysis integration.
"""

from nupack_complex_analysis import (
    SequenceInput,
    analyze_sequence_complexes,
    analyze_primer_dais_binding,
    analyze_four_primer_cross_reactivity
)


def test_basic_analysis():
    """Test basic complex analysis with simple sequences."""
    print("Testing basic complex analysis...")
    
    # Simple complementary sequences
    sequences = [
        SequenceInput("seq1", "ATCGATCG", 250e-9),  # 250 nM
        SequenceInput("seq2", "CGATCGAT", 250e-9)   # Reverse complement
    ]
    
    try:
        results = analyze_sequence_complexes(
            temperature_celsius=25.0,
            sequences=sequences,
            sodium_millimolar=100.0,
            magnesium_millimolar=5.0,
            max_complex_size=2,
            base_pairing_analysis=True
        )
        
        print(f"Found {len(results.complexes)} complexes")
        for complex_result in results.complexes:
            print(f"  Complex size {complex_result.size}: {complex_result.concentration_M:.2e} M")
            if complex_result.sequence_id_map:
                print(f"    Sequences: {complex_result.sequence_id_map}")
        
        print("✓ Basic analysis test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic analysis test failed: {e}")
        return False


def test_primer_dais_binding():
    """Test primer-dais binding analysis."""
    print("\nTesting primer-dais binding analysis...")
    
    # Example primer and its complementary dais
    primer_seq = "ATCGATCGATCGATCG"
    dais_seq = "CGATCGATCGATCGAT"  # Reverse complement
    
    try:
        hetero_dimer_frac, three_prime_unpaired = analyze_primer_dais_binding(
            primer_sequence=primer_seq,
            dais_sequence=dais_seq,
            primer_name="test_primer",
            dais_name="test_dais",
            temperature_celsius=41.0,
            concentration_nanomolar=250.0
        )
        
        print(f"  Hetero-dimer fraction: {hetero_dimer_frac:.3f}")
        print(f"  3'-end unpaired probability: {three_prime_unpaired:.3f}")
        print("✓ Primer-dais binding test passed")
        return True
        
    except Exception as e:
        print(f"✗ Primer-dais binding test failed: {e}")
        return False


def test_four_primer_analysis():
    """Test four-primer cross-reactivity analysis."""
    print("\nTesting four-primer cross-reactivity analysis...")
    
    # Four different primer sequences
    primers = {
        "primer1": "ATCGATCGATCG",
        "primer2": "GCTAGCTAGCTA",
        "primer3": "CGTACGTACGTA",
        "primer4": "TACGTACGTACG"
    }
    
    try:
        results = analyze_four_primer_cross_reactivity(
            primer_sequences=primers,
            temperature_celsius=41.0,
            concentration_nanomolar=250.0
        )
        
        print("  Primer analysis results:")
        for primer_name, (monomer_frac, unpaired_prob) in results.items():
            print(f"    {primer_name}: monomer={monomer_frac:.3f}, 3'-unpaired={unpaired_prob:.3f}")
        
        print("✓ Four-primer analysis test passed")
        return True
        
    except Exception as e:
        print(f"✗ Four-primer analysis test failed: {e}")
        return False


if __name__ == "__main__":
    print("NUPACK Integration Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    all_passed &= test_basic_analysis()
    all_passed &= test_primer_dais_binding()
    all_passed &= test_four_primer_analysis()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! NUPACK integration is working.")
    else:
        print("✗ Some tests failed. Check NUPACK installation.")