"""Values of record: the numbers quoted in the paper, and a small comparison helper.

The table lives in values_of_record.json next to this package. Each entry is
    "script/quantity": {"value": x or [re, im], "rtol": r, "where": "Sec. ...", "note": "..."}
or, for statements of the form "agrees to 1e-4" or "below 1e-9",
    "script/quantity": {"max": x, "where": ...}
`compare(key, computed)` prints one line and returns True if |computed - value| <= rtol |value|
(or |computed| <= max).
"""
import json
import os

import mpmath as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(os.path.dirname(_HERE), "values_of_record.json")


def load():
    with open(_PATH) as f:
        return json.load(f)


def _as_number(v):
    if isinstance(v, (list, tuple)):
        return mp.mpc(v[0], v[1])
    return mp.mpf(v)


def compare(key, computed, table=None, quiet=False):
    """Compare a computed value with the value of record `key`. Returns (ok, relative error)."""
    table = table or load()
    ent = table[key]
    comp = mp.mpmathify(computed)
    if "max" in ent:
        bound = mp.mpf(ent["max"])
        ok = abs(comp) <= bound
        if not quiet:
            flag = "ok " if ok else "FAIL"
            print(f"  [{flag}] {key:<46s} computed {mp.nstr(comp, 10):>28s}   record < {mp.nstr(bound, 3):>26s}")
        return ok, abs(comp)
    ref = _as_number(ent["value"])
    err = abs(comp - ref) / abs(ref) if ref != 0 else abs(comp)
    ok = err <= ent["rtol"]
    if not quiet:
        flag = "ok " if ok else "FAIL"
        print(f"  [{flag}] {key:<46s} computed {mp.nstr(comp, 10):>28s}   record {mp.nstr(ref, 10):>28s}   rel {mp.nstr(err, 2)}")
    return ok, err


def report(results):
    """Summarize a list of (ok, err) pairs; return the process exit code."""
    bad = sum(1 for ok, _ in results if not ok)
    print(f"  {len(results) - bad}/{len(results)} values of record reproduced")
    return 1 if bad else 0
