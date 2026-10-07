"""Sections, caches, results and analysis of the bubble pipeline."""
from __future__ import annotations

import json
import math
import time

import mpmath as mp

from terms import (Families, Problem, assemble, grading, n_terms, nsum_family, scheme_chain,
                   signal_coeff)

STOP_TOL = mp.mpf("1e-16")


def S(z, d=25):
    return mp.nstr(z, d)


def enc(Sig, vals):
    sig = [S(mp.re(Sig)), S(mp.im(Sig))] if Sig is not None else ["0", "0"]
    return {"Sig": sig, "vals": [[S(mp.re(v)), S(mp.im(v))] for v in vals]}


def dec(e):
    Sig = mp.mpc(e["Sig"][0], e["Sig"][1])
    return Sig, [mp.mpc(a, b) for a, b in e["vals"]]


class Progress:
    def __init__(self, fn=None):
        self.fn = fn
        self.t0 = time.time()

    def key(self, fam, key, v0):
        if self.fn:
            with open(self.fn, "a") as f:
                f.write(json.dumps({"t": time.time() - self.t0, "fam": fam, "key": key, "v0": S(v0, 15)}) + "\n")

    def mp_done(self, fam, m, big):
        if self.fn:
            with open(self.fn, "a") as f:
                f.write(json.dumps({"t": time.time() - self.t0, "fam": fam, "mp": m, "big": S(big, 5)}) + "\n")


# ============================================================================ D_II


def old_rule_sum(F, fam, m, rs, N0=10, K=6, Nn=0):
    """n-sum without the tail rule: fixed Nn terms, or N0+K terms with Neville extrapolation."""
    nr = len(rs)
    N = Nn if Nn > 0 else N0 + K
    parts = [[] for _ in range(nr)]
    acc = [mp.mpc(0)] * nr
    last = None
    for n in range(N):
        t = F.U_term(m, n, rs) if fam == "U" else F.V_term(m, n, rs)
        for i in range(nr):
            acc[i] += t[i]
            parts[i].append(acc[i])
        last = t
    if Nn > 0:
        return acc, [abs(x) for x in last]
    vals, errs = [], []
    for i in range(nr):
        xs = [mp.mpf(1) / (n + 1) for n in range(N0 - 1, N0 + K)]
        ys = parts[i][N0 - 1:N0 + K]
        full = Families.neville0(xs, ys)
        short = Families.neville0(xs[:-1], ys[:-1])
        vals.append(full)
        errs.append(abs(full - short))
    return vals, errs


def dii_mprime(F, m, rs, tailrule=True, rmax=None, tailscale=1.3, Ktail=3, N0=10, K=6, Nn=0,
               termdump=None, do_V=True, do_U=True):
    """All D_II keys of one m' (U task also carries the rung families)."""
    out = {}
    if do_U:
        if tailrule:
            N = n_terms("U", m, rmax, tailscale=tailscale)
            v, e = nsum_family(F, "U", m, rs, N, Ktail, termdump)
        else:
            v, e = old_rule_sum(F, "U", m, rs, N0, K, Nn)
        out[f"U{m}"] = (None, v)
        out[f"EU{m}"] = (None, e)
        for kind in ("U", "V"):
            fin, pole = F.rung(m, kind)
            out[f"R0{kind}{m}"] = (None, [fin * mp.mpf(r) ** m for r in rs])
            out[f"PR0{kind}{m}"] = (None, [pole * mp.mpf(r) ** m for r in rs])
    if do_V:
        if m % 2 == 0:
            out[f"V{m}"] = (None, [mp.mpc(0)] * len(rs))
            out[f"EV{m}"] = (None, [mp.mpc(0)] * len(rs))
        else:
            if tailrule:
                N = n_terms("V", m, rmax, tailscale=tailscale)
                v, e = nsum_family(F, "V", m, rs, N, Ktail, termdump)
            else:
                v, e = old_rule_sum(F, "V", m, rs, N0, K, Nn)
            out[f"V{m}"] = (None, v)
            out[f"EV{m}"] = (None, e)
    for key, (_, vals) in out.items():
        for z in vals:
            if not mp.isfinite(z):
                raise FloatingPointError(f"non-finite value {key}")
    return out


def big_of(entries, keys=None):
    b = mp.mpf(0)
    for k, (_, vals) in entries.items():
        if keys and k not in keys:
            continue
        if k.startswith(("E", "P")):
            continue
        b = max(b, abs(vals[-1]))
    return b


def dii_section(F, rs, mpmax, prog=None, stop=True, **kw):
    cache = {}
    tw = F.towers(rs)
    for k, (Sig, vals, _) in tw.items():
        cache[k] = (Sig, vals)
        if prog:
            prog.key("DII", k, vals[0])
    below = 0
    last_m = None
    for m in range(mpmax + 1):
        ent = dii_mprime(F, m, rs, **kw)
        cache.update(ent)
        b = big_of(ent)
        last_m = m
        if prog:
            for k, (_, vals) in ent.items():
                prog.key("DII", k, vals[0])
            prog.mp_done("DII", m, b)
        below = below + 1 if b < STOP_TOL else 0
        if stop and below >= 2:
            break
    return cache, last_m


# ============================================================================ D_IK and XX


def dik_section(F, rs, mpmax, prog=None):
    cache = {}
    nmax = max(F.Nmax, F.Mmax)
    for n1 in range(F.Nmax + 1):
        for m in range(F.Mmax + 1):
            if n1 + m > nmax:
                continue
            for sig, sg in ((1, "+"), (-1, "-")):
                Sig, vals, _ = F.TIK(n1, sig, m, rs)
                cache[f"T{n1}_{sg}{m}"] = (Sig, vals)
                if prog:
                    prog.key("DIK", f"T{n1}_{sg}{m}", vals[0])
    below = 0
    last_m = None
    for m in range(mpmax + 1):
        ent = {f"J{m}": (None, F.JIK(m, rs))}
        for n1 in range(F.Nmax + 1):
            if F.MIK_included(n1, m):
                ent[f"M{n1}_{m}"] = (None, F.MIK(n1, m, rs))
        cache.update(ent)
        b = big_of(ent)
        last_m = m
        if prog:
            prog.mp_done("DIK", m, b)
        below = below + 1 if b < STOP_TOL else 0
        if below >= 2:
            break
    return cache, last_m


def xx_section(F, rs, mpmax, prog=None):
    cache = {}
    for s1, a in ((1, "+"), (-1, "-")):
        for s2, b in ((1, "+"), (-1, "-")):
            for m1 in range(F.Mmax + 1):
                for m2 in range(F.Mmax + 1 - m1):
                    Sig, vals, _ = F.TXX(s1, m1, s2, m2, rs)
                    cache[f"T{a}{m1}_{b}{m2}"] = (Sig, vals)
    below = 0
    last_m = None
    for m in range(mpmax + 1):
        ent = {f"J{m}": (None, F.JXX(m, rs))}
        for s1, a in ((1, "+"), (-1, "-")):
            for m1 in range(F.Mmax + 1):
                if F.MXX_included(m1, m):
                    ent[f"M{a}{m1}_{m}"] = (None, F.MXX(s1, m1, m, rs))
        cache.update(ent)
        b = big_of(ent)
        last_m = m
        if prog:
            prog.mp_done("XX", m, b)
        below = below + 1 if b < STOP_TOL else 0
        if below >= 2:
            break
    return cache, last_m


# ============================================================================ caches


def save_cache(fn, prob, rs, fam, extra=None):
    d = {"prob": prob, "rs": [str(r) for r in rs],
         "fam": {s: {k: enc(Sig, v) for k, (Sig, v) in c.items()} for s, c in fam.items()}}
    if extra:
        d.update(extra)
    json.dump(d, open(fn, "w"))


def load_cache(fn):
    d = json.load(open(fn))
    fam = {s: {k: dec(e) for k, e in c.items()} for s, c in d["fam"].items()}
    for s in fam:
        for k, (Sig, v) in list(fam[s].items()):
            if Sig == 0 and not k.startswith(("T", "R0T")):
                fam[s][k] = (None, v)
    return d, fam


# ============================================================================ error terms


def truncation_proxy(cache, i, r):
    import re
    groups = {}
    for s, c in cache.items():
        for k, (_, vals) in c.items():
            m = re.fullmatch(r"(U|V|R0U|R0V|J)(\d+)", k)
            pre = None
            if m:
                pre, mm = m.group(1), int(m.group(2))
            else:
                m = re.fullmatch(r"(M[+-]?\d+_)(\d+)", k)
                if m:
                    pre, mm = m.group(1), int(m.group(2))
            if pre is None:
                continue
            g = (s, pre)
            if g not in groups or mm > groups[g][0]:
                groups[g] = (mm, vals[i])
    rr = mp.mpf(r)
    return 16 * rr * (rr ** 2 / 2) * 2 * mp.fsum(abs(v) for _, v in groups.values())


def errors(cache, row, i, calib=None):
    tp = truncation_proxy(cache, i, row["r"])
    bound = row["err_richardson"] + tp
    if calib:
        emp = 10 * max(abs(calib["halving"]), abs(calib["order"]), tp)
    else:
        emp = mp.mpf(0)
    err = max(emp, bound)
    basis = "empirical_x10" if emp > bound else "richardson_bound"
    return {"bound": bound, "err_empirical": emp, "err": err, "err_basis": basis, "trunc_proxy": tp}


def truncation_scan(cache, F, rs, S_full, tols=("1e-6", "1e-8")):
    res = {}
    for t in tols:
        tol = mp.mpf(t)
        res[t] = []
    rows_g = {}
    for g in range(0, 52, 2):
        rows_g[g] = assemble(cache, F, rs, gmax=g)
    for t in tols:
        tol = mp.mpf(t)
        out = []
        for i in range(len(rs)):
            ok = None
            for g in sorted(rows_g, reverse=True):
                if abs(rows_g[g][i]["S_house"] - S_full[i]) <= tol * abs(S_full[i]):
                    ok = g
                else:
                    break
            nterms = sum(1 for s in cache for k, (_, v) in cache[s].items()
                         if grading(k) is not None and ok is not None and grading(k) <= ok and v[i] != 0)
            out.append((ok, nterms))
        res[t] = out
    return res


def n_graded(cache):
    return sum(1 for s in cache for k, (_, v) in cache[s].items() if grading(k) is not None
               and any(z != 0 for z in v))
