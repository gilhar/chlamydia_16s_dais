#!/usr/bin/env python3
"""
Helpers to run NUPACK analysis in a short-lived subprocess.

Usage example:
    from chlamydia_16s_dais.nupack_subprocess import analyze_sequence_complexes_subprocess, SequenceParam

    sequences = [
        SequenceParam(name="primer", sequence="ACGT...", concentration_M=250e-9),
        SequenceParam(name="signal", sequence="TGCA...", concentration_M=10e-12),
    ]
    result = analyze_sequence_complexes_subprocess(
        temperature_celsius=NASBA_CONDITIONS['target_temp_C'],
        sequences=sequences,
        sodium_millimolar=NASBA_CONDITIONS['Na_mM'],
        magnesium_millimolar=NASBA_CONDITIONS['Mg_mM'],
        max_complex_size=2,
        base_pairing_analysis=True,
        timeout_seconds=120,
    )
    # 'result' is a JSON-serializable dict as emitted by the worker.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from chlamydia_16s_dais.nasba_primer_thermodynamics import ComprehensiveAnalysisResult


@dataclass
class SequenceParam:
    name: str
    sequence: str
    concentration_M: float


def _build_payload_analyze_sequence_complexes(
    temperature_celsius: float,
    sequences: List[SequenceParam],
    sodium_millimolar: float,
    magnesium_millimolar: float,
    max_complex_size: int,
    base_pairing_analysis: bool,
) -> Dict[str, Any]:
    return {
        "temperature_celsius": float(temperature_celsius),
        "sequences": [
            {
                "name": s.name,
                "sequence": s.sequence,
                "concentration_M": float(s.concentration_M),
            }
            for s in sequences
        ],
        "sodium_millimolar": float(sodium_millimolar),
        "magnesium_millimolar": float(magnesium_millimolar),
        "max_complex_size": int(max_complex_size),
        "base_pairing_analysis": bool(base_pairing_analysis),
    }


def _build_payload_analyze_sequence_comprehensive(
    primary_sequence: str,
    primary_sequence_name: str,
    primary_sequence_concentration: float,
    other_sequences: dict[str, str],
    other_sequence_concentrations: dict[str, float],
    temp_celsius: float,
    n_bases: int = 3,
) -> Dict[str, Any]:
    return {
        "primary_sequence": primary_sequence,
        "primary_sequence_name": primary_sequence_name,
        "primary_sequence_concentration": float(primary_sequence_concentration),
        "other_sequences": other_sequences,
        "other_sequence_concentrations": {
            k: float(v) for k, v in other_sequence_concentrations.items()
        },
        "temp_celsius": float(temp_celsius),
        "n_bases": int(n_bases),
    }

def analyze_sequence_complexes_subprocess(
    temperature_celsius: float,
    sequences: List[SequenceParam],
    sodium_millimolar: float = 80.0,
    magnesium_millimolar: float = 12.0,
    max_complex_size: int = 4,
    base_pairing_analysis: bool = False,
    timeout_seconds: int = 250,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Launch a fresh Python process to run NUPACK analysis and return its JSON result.

    The worker process exits after the call, which helps reclaim any leaked native memory.
    Raises RuntimeError on failure, including worker-side exceptions or timeouts.
    """
    payload = _build_payload_analyze_sequence_complexes(
        temperature_celsius=temperature_celsius,
        sequences=sequences,
        sodium_millimolar=sodium_millimolar,
        magnesium_millimolar=magnesium_millimolar,
        max_complex_size=max_complex_size,
        base_pairing_analysis=base_pairing_analysis,
    )
    payload["function"] = "analyze_sequence_complexes"

    cmd = [sys.executable, "-m", "chlamydia_16s_dais.nupack_worker"]

    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
            start_new_session=True,  # Make it its own process group/session
        )
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(
            f"NUPACK worker timed out after {timeout_seconds}s"
        ) from te

    if completed.returncode != 0:
        # Worker printed a JSON error object to stderr
        err_msg = completed.stderr
        try:
            err_obj = json.loads(err_msg) if err_msg else {}
        except Exception:
            err_obj = {
                "error": err_msg or "unknown worker error",
                "traceback": None,
            }
        raise RuntimeError(
            f"NUPACK worker failed: {err_obj.get('error')}\n{err_obj.get('traceback')}"
        )

    stdout = completed.stdout or ""
    if not stdout:
        raise RuntimeError("NUPACK worker produced no output")

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as jde:
        raise RuntimeError(
            f"Failed to parse worker JSON output: {stdout[:500]}"
        ) from jde

    return result


def analyze_sequence_comprehensive_subprocess(
    primary_sequence: str,
    primary_sequence_name: str,
    primary_sequence_concentration: float,
    other_sequences: dict[str, str],  # from sequence name to sequence
    other_sequence_concentrations: dict[str, float],
    temp_celsius: float,
    n_bases: int = 3,
    timeout_seconds: int = 250,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    payload = _build_payload_analyze_sequence_comprehensive(
        primary_sequence=primary_sequence,
        primary_sequence_name=primary_sequence_name,
        primary_sequence_concentration=primary_sequence_concentration,
        other_sequences=other_sequences,
        other_sequence_concentrations=other_sequence_concentrations,
        temp_celsius=temp_celsius,
        n_bases=n_bases,
    )
    payload["function"] = "analyze_sequence_comprehensive"

    cmd = [sys.executable, "-m", "chlamydia_16s_dais.nupack_worker"]

    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
            start_new_session=True,  # Make it its own process group/session
        )
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"NUPACK worker timed out after {timeout_seconds}s") from te

    if completed.returncode != 0:
        # Worker printed a JSON error object to stderr
        err_msg = completed.stderr
        try:
            err_obj = json.loads(err_msg) if err_msg else {}
        except Exception:
            err_obj = {"error": err_msg or "unknown worker error", "traceback": None}
        raise RuntimeError(
            f"NUPACK worker failed: {err_obj.get('error')}\n{err_obj.get('traceback')}"
        )

    stdout = completed.stdout or ""
    if not stdout:
        raise RuntimeError("NUPACK worker produced no output")

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as jde:
        raise RuntimeError(
            f"Failed to parse worker JSON output: {stdout[:500]}"
        ) from jde

    # Deserialize back to ComprehensiveAnalysisResult
    return result

