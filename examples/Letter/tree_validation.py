"""Mellin-Barnes tree values against the direct Schwinger-Keldysh route.

    python tree_validation.py --points FILE.json [--workers 4] [--regenerate]

``direct_refined`` is the refined direct reference of the Letter's tree-validation statement
(setting A: mpanels=36, window 70, soft-tail width 0.25/k, extent 22/k, soft rule below 0.1;
setting B: 28, 60, 0.35, 18, 1.5).  The input file is a JSON list of points
{"diagram": "double"|"triple", "masses": [...], "k": [k1, k2, k3]} (max k = 1).  With
``--regenerate`` the refined direct values are recomputed (minutes per point); otherwise only
the four Mellin-Barnes settings are evaluated.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from dscorr import DoubleExchange, TripleExchange
from dscorr._engines.direct import SETTING_A, SETTING_B, direct_refined  # noqa: F401

SETTINGS = {
    "half_step": {"hL": 0.002},
    "default": {},
    "fast": {"L0": -54.0, "L1": 10.6, "hL": 0.005, "T": 8.0, "h": 0.05, "T_s": 8.0, "h_s": 0.025},
    "fast_contours_default_step": {"L0": -54.0, "L1": 10.6, "hL": 0.004, "T": 8.0, "h": 0.05,
                                   "T_s": 8.0, "h_s": 0.025},
}


def mb_value(diagram, masses, k, setting):
    kw = dict(SETTINGS[setting])
    if diagram == "double":
        if "T" in kw:
            kw["T_sp"] = kw.pop("T")
        return DoubleExchange(*masses, **kw)(*k)
    return TripleExchange(*masses, **kw)(*k)


def _one(p, regenerate):
    out = dict(p)
    for s in SETTINGS:
        out["S_MB_" + s] = mb_value(p["diagram"], p["masses"], p["k"], s)
    if regenerate:
        out["S_direct_refined"] = direct_refined(p["diagram"], p["masses"], p["k"], **SETTING_A)
        out["S_direct_B"] = direct_refined(p["diagram"], p["masses"], p["k"], **SETTING_B)
    if "S_direct_refined" in out:
        ref = out["S_direct_refined"]
        for s in SETTINGS:
            out["rel_" + s] = abs(out["S_MB_" + s] / ref - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    pts = json.load(open(a.points))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_one, pts, [a.regenerate] * len(pts)))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
