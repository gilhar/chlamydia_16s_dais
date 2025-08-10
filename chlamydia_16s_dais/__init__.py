"""
Chlamydia 16S DAIS - NASBA Primer Validation Framework

A comprehensive framework for generating and validating NASBA primers for
Chlamydia trachomatis 16S rRNA detection using NUPACK thermodynamic calculations.

Modules:
- chlamydia_nasba_primer: NASBA primer construction from base primers
- nasba_primer_thermodynamics: NUPACK-based bound-fraction calculations
- nasba_primer_validation: Comprehensive validation framework with 9 tests
- nupack_complex_analysis: NUPACK complex formation analysis utilities
- nasba_primer_cli: Command-line interface

Author: Claude (Anthropic)
Date: 2025
"""

__version__ = "0.1.0"
__author__ = "Claude (Anthropic)"

# Import main functions for easy access
from .chlamydia_nasba_primer import (
    get_base_primers,
    GenericPrimerSet,
    generate_nasba_primer_candidates,
    TWIST_CT_16S,
)

from .nasba_primer_thermodynamics import (
    calculate_bound_fraction_nupack,
    calculate_primer_bound_fractions,
    validate_bound_fractions,
    BOUND_FRACTION_TARGETS,
)

from .nasba_primer_validation import (
    run_comprehensive_validation,
)

__all__ = [
    "get_base_primers",
    "GenericPrimerSet",
    "generate_nasba_primer_candidates",
    "TWIST_CT_16S",
    "calculate_bound_fraction_nupack",
    "calculate_primer_bound_fractions",
    "validate_bound_fractions",
    "BOUND_FRACTION_TARGETS",
    "run_comprehensive_validation",
]
