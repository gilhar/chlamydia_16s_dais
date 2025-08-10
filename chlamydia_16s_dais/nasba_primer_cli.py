#!/usr/bin/env python3
"""
NASBA Primer CLI

Command-line interface for NASBA primer generation and validation with bound-fraction criteria.
This orchestrates the full pipeline: generation → thermodynamic filtering → comprehensive validation.

Author: Claude (Anthropic)
Date: 2025
"""

import click

from chlamydia_nasba_primer import (
    get_base_primers,
    GenericPrimerSet,
    analyze_all_combinations,
)
from nasba_primer_thermodynamics import (
    update_bound_fraction_targets,
    get_bound_fraction_targets,
    NASBA_TEMPERATURE_CELSIUS,
    NASBA_SODIUM_MOLAR,
    NASBA_MAGNESIUM_MOLAR,
    NASBA_PRIMER_CONCENTRATION_MOLAR,
)
from nasba_primer_validation import (
    run_comprehensive_validation,
    build_validation_dataframe,
)


@click.group()
def cli():
    """NASBA Primer Generation and Validation CLI"""
    pass


@cli.command()
@click.option(
    '--anchor-min-bf',
    type=float,
    default=0.97,
    show_default=True,
    help=f'Minimum bound fraction for anchor segment at {NASBA_TEMPERATURE_CELSIUS}°C',
)
@click.option(
    '--anchor-max-bf',
    type=float,
    default=1.0,
    show_default=True,
    help=f'Maximum bound fraction for anchor segment at {NASBA_TEMPERATURE_CELSIUS}°C',
)
@click.option(
    '--toehold-min-bf',
    type=float,
    default=0.6,
    show_default=True,
    help=f'Minimum bound fraction for toehold segment at {NASBA_TEMPERATURE_CELSIUS}°C',
)
@click.option(
    '--toehold-max-bf',
    type=float,
    default=0.9,
    show_default=True,
    help=f'Maximum bound fraction for toehold segment at {NASBA_TEMPERATURE_CELSIUS}°C',
)
def generate(anchor_min_bf, anchor_max_bf, toehold_min_bf, toehold_max_bf):
    """
    Generate NASBA primer candidates.

    This generates all possible NASBA primer candidates with different anchor/toehold
    lengths but does not perform thermodynamic validation or comprehensive testing.
    """

    print("=" * 80)
    print("NASBA PRIMER CANDIDATE GENERATION")
    print("=" * 80)

    # Update bound fraction targets if provided
    update_bound_fraction_targets(
        anchor_min=anchor_min_bf,
        anchor_max=anchor_max_bf,
        toehold_min=toehold_min_bf,
        toehold_max=toehold_max_bf,
    )

    bf_targets = get_bound_fraction_targets()
    print(f"\nBound fraction targets at {NASBA_TEMPERATURE_CELSIUS}°C:")
    print(
        f"  Anchor:   {bf_targets['anchor_min']:.2f} - {bf_targets['anchor_max']:.2f}"
    )
    print(
        f"  Toehold:  {bf_targets['toehold_min']:.2f} - {bf_targets['toehold_max']:.2f}"
    )

    # Define generic primer sets
    generic_sets = [
        GenericPrimerSet.from_sequences(
            name="gen5",
            forward_seq="TTATGTTCGTGGTT",
            reverse_concat="AATTCTAATACGACTCACTATAGGGTAAATACGTGC",
        ),
        GenericPrimerSet.from_sequences(
            name="gen6",
            forward_seq="TTTTGGTGGGTGGAT",
            reverse_concat="AATTCTAATACGACTCACTATAGGGTAAATATCCGGC",
        ),
    ]

    # Get base primers and generate candidates
    base_primers = get_base_primers()
    results = analyze_all_combinations(base_primers, generic_sets)

    return results


@cli.command()
def validate():
    """
    Run comprehensive NASBA primer validation.

    This performs the full pipeline:
    1. Generate primer candidates
    2. Filter by bound-fraction criteria using NUPACK
    3. Run 9 comprehensive validation tests
    """

    # Run comprehensive validation
    validation_data = run_comprehensive_validation()
    # Export results
    export_results = "validation_results.csv"
    print(f"\nExporting results to {export_results}...")
    df = build_validation_dataframe(validation_data['all_pairs'])
    df.to_csv(export_results, index=False)
    print(f"Wrote {len(df)} rows to {export_results}")

    return validation_data


@cli.command()
def targets():
    """Show current bound fraction targets."""
    bf_targets = get_bound_fraction_targets()

    print(f"Current bound fraction targets at {NASBA_TEMPERATURE_CELSIUS}°C:")
    print(
        f"  Anchor segment:   {bf_targets['anchor_min']:.3f} - {bf_targets['anchor_max']:.3f}"
    )
    print(
        f"  Toehold segment:  {bf_targets['toehold_min']:.3f} - {bf_targets['toehold_max']:.3f}"
    )

    print("\nInterpretation:")
    print("  - Anchor should be almost fully bound (high specificity)")
    print("  - Toehold should be moderately bound (allows strand displacement)")
    print("  - Values are calculated using NUPACK thermodynamics at NASBA conditions")
    print(
        f"    ({NASBA_TEMPERATURE_CELSIUS}°C, {NASBA_SODIUM_MOLAR*1000:.0f}mM Na+, {NASBA_MAGNESIUM_MOLAR*1000:.0f}mM Mg++, {NASBA_PRIMER_CONCENTRATION_MOLAR*1e9:.0f}nM primer concentration)"
    )


@cli.command()
@click.argument('sequence')
@click.option(
    '--temp',
    type=float,
    default=NASBA_TEMPERATURE_CELSIUS,
    show_default=True,
    help='Temperature in Celsius',
)
def test_sequence(sequence, temp):
    """
    Test bound fraction calculation for a single sequence.

    SEQUENCE: DNA sequence to test (will calculate self-complement binding)
    """
    from nasba_primer_thermodynamics import calculate_bound_fraction_nupack
    from Bio.Seq import Seq

    print(f"Testing sequence: {sequence}")
    print(f"Temperature: {temp}°C")

    # Calculate bound fraction with its reverse complement
    rc_sequence = str(Seq(sequence).reverse_complement())
    print(f"Reverse complement: {rc_sequence}")

    bound_fraction = calculate_bound_fraction_nupack(
        primer_sequence=sequence,
        target_sequence=rc_sequence,
        temp_celsius=temp,
        sodium_molar=NASBA_SODIUM_MOLAR,
        magnesium_molar=NASBA_MAGNESIUM_MOLAR,
        primer_concentration_molar=NASBA_PRIMER_CONCENTRATION_MOLAR,
    )

    print(f"\nBound fraction: {bound_fraction:.4f}")

    # Compare to targets
    bf_targets = get_bound_fraction_targets()

    print(f"\nComparison to targets:")
    print(
        f"  Anchor range:   {bf_targets['anchor_min']:.3f} - {bf_targets['anchor_max']:.3f}"
    )
    print(
        f"  Toehold range:  {bf_targets['toehold_min']:.3f} - {bf_targets['toehold_max']:.3f}"
    )

    if bf_targets['anchor_min'] <= bound_fraction <= bf_targets['anchor_max']:
        print("  → Suitable for ANCHOR segment ✓")
    elif bf_targets['toehold_min'] <= bound_fraction <= bf_targets['toehold_max']:
        print("  → Suitable for TOEHOLD segment ✓")
    else:
        print("  → Outside target ranges for both anchor and toehold ✗")


if __name__ == "__main__":
    cli()
