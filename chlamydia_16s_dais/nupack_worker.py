#!/usr/bin/env python3
"""
One-shot NUPACK worker process.

Reads a JSON payload from stdin describing a tube analysis job, runs it, and writes a JSON result to stdout.
Intended to be launched as a fresh process per call to avoid memory leaks in native libraries.

Input JSON schema:
{
  "temperature_celsius": float,
  "sequences": [{"name": str, "sequence": str, "concentration_M": float}, ...],
  "sodium_millimolar": float,
  "magnesium_millimolar": float,
  "max_complex_size": int,
  "base_pairing_analysis": bool
}

Output JSON schema (simplified and JSON-friendly):
{
  "temperature_celsius": float,
  "ionic_conditions": {"sodium_mM": float, "magnesium_mM": float},
  "max_complex_size": int,
  "total_sequences": int,
  "complexes": [
    {
      "complex_id": str,
      "size": int,
      "concentration_molar": float,
      "sequence_id_map": {"0": "name0", ...},
      "unpaired_probability": {"seqId:baseOffset": float} | null,
      "pairing_probability": {"s1:b1|s2:b2": float} | null
    },
    ...
  ]
}
"""

import json
import sys
import traceback
from typing import Any, Dict

from chlamydia_16s_dais.nupack_complex_analysis import (
    analyze_sequence_complexes_inprocess,
    SequenceInput,
    ComplexAnalysisResult,
)


def _serialize_result(res: ComplexAnalysisResult) -> Dict[str, Any]:
    """Convert ComplexAnalysisResult into a JSON-serializable dict."""
    out: Dict[str, Any] = {
        "temperature_celsius": res.temperature_celsius,
        "ionic_conditions": res.ionic_conditions,
        "max_complex_size": res.max_complex_size,
        "total_sequences": res.total_sequences,
        "complexes": [],
    }

    for c in res.complexes:
        entry = {
            "complex_id": c.complex_id,
            "size": c.size,
            "concentration_molar": c.concentration_molar,
            "sequence_id_map": {
                str(k): v for k, v in (c.sequence_id_map or {}).items()
            },
            "unpaired_probability": None,
            "pairing_probability": None,
        }
        if c.unpaired_probability is not None:
            # keys are tuples (seq_id, base_offset) -> stringify as "seq:base"
            entry["unpaired_probability"] = {
                f"{k[0]}:{k[1]}": v for k, v in c.unpaired_probability.items()
            }
        if c.pairing_probability is not None:
            # keys are 4-tuples -> stringify as "s1:b1|s2:b2"
            entry["pairing_probability"] = {
                f"{k[0]}:{k[1]}|{k[2]}:{k[3]}": v
                for k, v in c.pairing_probability.items()
            }
        out["complexes"].append(entry)

    return out


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)

        # Basic validation
        required_fields = [
            "temperature_celsius",
            "sequences",
            "sodium_millimolar",
            "magnesium_millimolar",
            "max_complex_size",
            "base_pairing_analysis",
        ]
        for f in required_fields:
            if f not in payload:
                raise ValueError(f"Missing required field '{f}'")

        seq_inputs = [
            SequenceInput(
                name=s["name"],
                sequence=s["sequence"],
                concentration_M=s["concentration_M"],
            )
            for s in payload["sequences"]
        ]

        res = analyze_sequence_complexes_inprocess(
            temperature_celsius=payload["temperature_celsius"],
            sequences=seq_inputs,
            sodium_millimolar=payload["sodium_millimolar"],
            magnesium_millimolar=payload["magnesium_millimolar"],
            max_complex_size=payload["max_complex_size"],
            base_pairing_analysis=payload["base_pairing_analysis"],
        )

        output = _serialize_result(res)
        sys.stdout.write(json.dumps(output))
        sys.stdout.flush()
        return 0

    except Exception as e:
        # Print a structured error to stderr for the caller to capture
        tb = traceback.format_exc()
        sys.stderr.write(json.dumps({"error": str(e), "traceback": tb}))
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
