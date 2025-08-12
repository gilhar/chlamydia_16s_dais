#!/usr/bin/env python3
"""
Lightweight JSON file cache utilities for storing per-(test_id, primer_id) results.

- Files are written under: cache/<test_id>/<primer_id>[--<hash>].json
- Atomic writes to reduce risk of partial files after crashes
- Deterministic hashing of "inputs that matter" to avoid stale reuse when parameters change
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# Default cache root directory (relative to project root)
DEFAULT_CACHE_ROOT = Path("cache")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _stable_json_dumps(data: Any) -> str:
    """Serialize data to a deterministic JSON string suitable for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), cls=NumpyEncoder)


def _sanitize(s: str) -> str:
    """Make a string safe for filesystem paths."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(s))


def make_cache_path(
    test_id: str,
    primer_id: str,
    inputs_for_hash: Optional[Dict[str, Any]] = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    """
    Build a cache file path for a given (test_id, primer_id).

    If inputs_for_hash is provided, append a short 8-character SHA-256 hash to the filename
    to disambiguate results under different parameterization.

    Returns:
        Path to the JSON cache file (parent directories are ensured to exist).
    """
    safe_test = _sanitize(test_id)
    safe_primer = _sanitize(primer_id)

    subdir = cache_root / safe_test
    subdir.mkdir(parents=True, exist_ok=True)

    if inputs_for_hash:
        digest = hashlib.sha256(_stable_json_dumps(inputs_for_hash).encode("utf-8")).hexdigest()[:8]
        filename = f"{safe_primer}--{digest}.json"
    else:
        filename = f"{safe_primer}.json"

    return subdir / filename


def read_cached_result(path: Path) -> Optional[Dict[str, Any]]:
    """
    Read and return a JSON object from the given cache path, or None if not present or invalid.
    """
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_cached_result(path: Path, result: Dict[str, Any]) -> None:
    """
    Atomically write a JSON-serializable dict to the cache path.

    Writes to a temporary file in the same directory and then replaces the destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"), sort_keys=False, cls=NumpyEncoder)
    # replacement is atomic on POSIX; on Windows this will replace if destination doesn't exist
    tmp.replace(path)
