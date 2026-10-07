"""Access layer for the packaged reference points and the data tables of the Letter.

Public names: ``reference_points``, ``letter_dir``, ``load``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

__all__ = ["reference_points", "letter_dir", "load"]

_HERE = Path(__file__).resolve().parent          # .../dscorr/data
_SENTINEL = "triangle_conformal_33config.csv"
_SUBDIR = Path("examples") / "Letter" / "data"


def reference_points():
    """Content of the packaged ``reference_points.json`` (needs no repository checkout)."""
    with open(_HERE / "reference_points.json", encoding="utf-8") as f:
        return json.load(f)


def _candidates():
    env = os.environ.get("DSCORR_DATA")
    if env:
        yield Path(env)
    # examples/Letter/data of a source checkout (src/dscorr/data -> repository root)
    yield _HERE.parents[2] / _SUBDIR
    cwd = Path.cwd()
    yield cwd / _SUBDIR
    yield cwd / "data"                     # running inside examples/Letter
    yield cwd.parent / _SUBDIR
    yield cwd.parent.parent / _SUBDIR


def letter_dir(required=True):
    """Absolute path of the directory with the data tables of the Letter.

    These are the values of record of the one-loop production runs, shipped in the repository
    under ``examples/Letter/data`` (not installed with the package).  Candidates in order:
    ``$DSCORR_DATA``; ``examples/Letter/data`` of the source checkout; ``./examples/Letter/data``;
    ``./data``; ``../examples/Letter/data``; ``../../examples/Letter/data``.  The first one
    that contains ``triangle_conformal_33config.csv`` is returned.
    """
    for c in _candidates():
        if (c / _SENTINEL).is_file():
            return str(c.resolve())
    if required:
        raise FileNotFoundError(
            "Letter data not found; set DSCORR_DATA to the repository's examples/Letter/data directory")
    return None


def _convert(field):
    if field is None or field == "":
        return None
    try:
        return float(field)
    except ValueError:
        return field


def load(name):
    """Load a data table of the Letter: JSON parsed, CSV as a list of row dictionaries
    (``#`` lines skipped, empty fields -> None, numeric fields -> float)."""
    path = Path(letter_dir()) / name
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    rows = []
    for r in reader:
        if not r:
            continue
        rows.append({h: _convert(v) for h, v in zip(header, r)})
    return rows
