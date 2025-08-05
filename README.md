# Chlamydia 16S DAIS Project

A comprehensive NASBA primer validation framework for Chlamydia trachomatis 16S rRNA detection using thermodynamic analysis.

## Overview

This project provides tools for analyzing and validating NASBA (Nucleic Acid Sequence-Based Amplification) primers for Chlamydia trachomatis and Neisseria gonorrhoeae 16S rRNA detection. It includes comprehensive validation frameworks with thermodynamic calculations using NUPACK API 4.0.1.9.

## Features

- **NASBA Primer Generation**: Generate primer candidates with anchor/toehold analysis
- **Comprehensive Validation**: 9-test validation framework covering primer specificity, cross-reactivity, and binding efficiency
- **Thermodynamic Analysis**: Accurate complex analysis using NUPACK API 4.0.1.9
- **Multi-pathogen Support**: Analysis for both Chlamydia trachomatis and Neisseria gonorrhoeae

## Files

- `chlamydia_primer_analysis.py` - PCR primer analysis with modern Biopython alignment
- `chlamydia_nasba_primer.py` - NASBA primer generation framework
- `nasba_primer_validation.py` - Comprehensive 9-test validation system
- `nupack_complex_analysis.py` - NUPACK API integration for thermodynamic calculations
- `test_nupack.py` - Test suite for NUPACK integration
- `twist_ct_16s_canonical.fasta` - Canonical Chlamydia trachomatis 16S rRNA sequence
- `twist_ng_16s_canonical.fasta` - Canonical Neisseria gonorrhoeae 16S rRNA sequence

## Requirements

- Python 3.8+
- Biopython
- NUPACK 4.0.1.9
- Click
- NumPy

## Usage

### NASBA Primer Validation
```bash
python nasba_primer_validation.py
```

### NUPACK Complex Analysis
```bash
python nupack_complex_analysis.py --temperature 41 --base-pairing primer1:ATCGATCG primer2:CGATCGAT
```

### Test NUPACK Integration
```bash
python test_nupack.py
```

## License

Research use only.