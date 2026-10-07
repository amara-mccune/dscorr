"""run_triangle_correlator.py <tasks.json> [--out DIR] [--workers N] [--grid Q0[,Q1]] [--tags T,..]

Brute-force driver of the triangle correlator.  Points are grouped by
mass; one pool of worker processes per mass; nodes sorted by yA + yB + yC and split into
12 x workers chunks.  A point whose raw file exists is re-assembled, not recomputed; summary.json
is rewritten after every point.  Cost: about 6 minutes per Q0 point on one workstation core
(0.22 core-hours each at 16 processes in the production run), Q1 about 4 times that.
"""
import argparse
import json
import multiprocessing as mpc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corr  # noqa: E402

_W = {}


def _init(mu, k, grid):
    lv = corr.LEVELS.get(grid, corr.G2)
    _W["st"] = corr.Structs(mu, corr.time_grid(k, lv["tpan_base"], lv["tngl"]))
    _W["k"] = k


def _chunk(L):
    k = _W["k"]
    return corr._eval(_W["st"], k, L)


def run_pou_parallel(mus, k, grid, workers):
    if workers <= 1:
        return corr.run_point_pou(mus, k, grid)
    t0 = time.time()
    lv = corr.LEVELS[grid]
    cents, redges = corr.pou_nodes(k, lv)
    allL = np.concatenate([c[1] for c in cents])
    allW = np.concatenate([c[2] for c in cents])
    cid = np.concatenate([np.full(len(c[2]), i) for i, c in enumerate(cents)])
    pid = np.concatenate([c[3] for c in cents])
    yA, yB, yC = corr.line_energies(allL, k)
    order = np.argsort(yA + yB + yC, kind="stable")
    chunks = np.array_split(order, 12 * workers)
    ctx = mpc.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(float(mus[0]), tuple(k), grid)) as pool:
        res = pool.map(_chunk, [allL[c] for c in chunks])
    vals = {n: np.concatenate([r[n] for r in res]) for n in corr.NAMES}
    tg = corr.time_grid(k, lv["tpan_base"], lv["tngl"])
    panels = {}
    for i, (s, L, W, panel) in enumerate(cents):
        npan = len(redges[s]) - 1
        sel = cid[order] == i
        re = np.zeros((npan, 4))
        im = np.zeros((npan, 4))
        for j, n in enumerate(corr.NAMES):
            v = allW[order][sel] * vals[n][sel]
            re[:, j] = np.bincount(pid[order][sel], weights=v.real, minlength=npan)
            im[:, j] = np.bincount(pid[order][sel], weights=v.imag, minlength=npan)
        panels[s] = {"re": re.tolist(), "im": im.tolist()}
    return {"mus": list(mus), "k": list(k), "grid": grid,
            "gspec": [tg.tmin, tg.tmax, tg.npan, tg.ngl], "n_nodes": int(allW.size),
            "wall_s": time.time() - t0, "names": list(corr.NAMES), "centres": ["C", "A", "B"],
            "r_edges": {s: list(map(float, redges[s])) for s in redges}, "panels": panels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--tags", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.tasks))
    workers = a.workers or cfg.get("workers", 16)
    grids = a.grid.split(",") if a.grid else None
    tags = a.tags.split(",") if a.tags else None
    os.makedirs(a.out, exist_ok=True)
    sfn = os.path.join(a.out, "summary.json")
    summary = json.load(open(sfn)) if os.path.exists(sfn) else {"points": []}
    done = {p["tag"]: p for p in summary["points"]}
    for p in cfg["points"]:
        if (grids and p["grid"] not in grids) or (tags and p["tag"] not in tags):
            continue
        rfn = os.path.join(a.out, f"raw_{p['tag']}.json")
        recomputed = False
        if os.path.exists(rfn):
            raw = json.load(open(rfn))
        else:
            if p["grid"] == "G2":
                raw = corr.run_point(p["mus"], tuple(p["k"]))
            else:
                raw = run_pou_parallel(p["mus"], tuple(p["k"]), p["grid"], workers)
            json.dump(raw, open(rfn, "w"))
            recomputed = True
        asm = corr.assemble(raw) if p["grid"] == "G2" else corr.assemble_pou(raw)
        asm.update({"tag": p["tag"], "mus": p["mus"], "k": p["k"], "grid": p["grid"],
                    "wall_s": raw["wall_s"], "workers": workers, "recomputed": recomputed})
        done[p["tag"]] = asm
        json.dump({"points": list(done.values())}, open(sfn, "w"), indent=1)


if __name__ == "__main__":
    main()
