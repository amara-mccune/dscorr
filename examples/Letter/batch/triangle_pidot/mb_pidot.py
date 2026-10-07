"""Mellin-Barnes route of the loop triangle with one pi-dot leg per vertex.

psi_3^loop = 1/8 int prod du/(2 pi i) prod[pi/sin(pi b_l) 2^(2 delta_l)] T^(rv)(u) C_3(delta),
with the banded trace over the inner s-lattice, symmetric windows in the slow direction, and
C_3 = core of the loop integral on contours that separate its poles.
float64/complex128; Gamma products as exponentials of sums of log-Gamma.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import loggamma as lg


def frange(a, b, h):
    """Floating-point range from a with stop value b + h/2."""
    return np.arange(a, b + h / 2.0, h)


def line_log(s, u, nu):
    return lg(-s) + lg(s - u) - lg(nu + s + 1.0) - lg(nu + u - s + 1.0)


class C3Core:
    """C_3(delta_A, delta_B, delta_C) with only the double Barnes integral (no residue terms)."""

    def __init__(self, k, delta, rho, sig_u=-0.05, hst=0.0125, tmax_u=14.0):
        assert sig_u < 0, "C3: pole sequence Gamma(-s): sig_u must be < 0"
        assert sig_u < 1.5 + 2 * rho, "C3: pole sequence Gamma(delta_B - lambda - s) crossed"
        assert 2 * sig_u > 3 * rho + 1.5, "C3: pole sequence Gamma(lambda + s + t) crossed"
        assert 2 * sig_u > rho, "C3: pole sequence Gamma(delta_A + s + t) crossed"
        self.margin = min(-sig_u, 1.5 + 2 * rho - sig_u, 2 * sig_u - 3 * rho - 1.5,
                          2 * sig_u - rho)
        k1, k2, k3 = k
        self.lz, self.lw = math.log(k1 ** 2 / k3 ** 2), math.log(k2 ** 2 / k3 ** 2)
        self.lk3 = math.log(k3 ** 2)
        self.delta = delta
        nv = len(delta)
        self.nv = nv
        hv = (delta[0] - delta[1]).imag
        rd = delta[0].real
        jj = np.arange(2 * nv - 1)
        self.dp = 2 * rd + 1j * (2 * delta[0].imag - jj * hv)
        jt = np.arange(3 * nv - 2)
        self.dt = 3 * rd + 1j * (3 * delta[0].imag - jt * hv)
        ts = frange(-tmax_u, tmax_u, hst)
        wt = np.full(ts.size, hst)
        wt[0] = wt[-1] = hst / 2.0
        s = sig_u + 1j * ts
        base = lg(-s)
        F1 = np.exp(base[None, :] + s[None, :] * self.lz + lg(1.5 - self.dp[:, None] - s[None, :])) * wt
        F2 = np.exp(base[None, :] + s[None, :] * self.lw + lg(1.5 - self.dp[:, None] - s[None, :])) * wt
        n = s.size
        L = 1
        while L < 2 * n:
            L *= 2
        self.FF1 = np.fft.fft(F1, L, axis=1)
        self.FF2 = np.fft.fft(F2, L, axis=1)
        self.n = n
        sig2 = 2 * sig_u + 1j * (2 * ts[0] + hst * np.arange(2 * n - 1))
        lam = self.dt - 1.5
        self.T3 = np.exp(lg(lam[:, None] + sig2[None, :]))
        self.TA = np.exp(lg(delta[:, None] + sig2[None, :]))

    def for_a(self, a):
        nv = self.nv
        b = np.arange(nv)
        X = self.FF1[a + b][None, :, :] * self.FF2[a + b][:, None, :]
        conv = np.fft.ifft(X, axis=2)[:, :, :2 * self.n - 1]
        tri = a + b[:, None] + b[None, :]
        C = np.einsum("bcs,bcs->bc", conv, self.T3[tri] * self.TA[a][None, None, :]) / (2 * np.pi) ** 2
        dA = self.delta[a]
        dB = self.delta[:, None]
        dC = self.delta[None, :]
        S = dA + dB + dC
        pre = np.exp(1.5 * math.log(np.pi) + (1.5 - S) * self.lk3 - lg(3.0 - S) - lg(dA) - lg(dB)
                     - lg(dC))
        return pre * C

    def full(self):
        return np.stack([self.for_a(a) for a in range(self.nv)])


def windows(Ns, nblk, hs):
    ic = (Ns - 1) // 2
    radii = [int(ic * m / nblk) for m in range(1, nblk + 1)]
    pieces = []
    prev = -1
    for m, r in enumerate(radii):
        if prev < 0:
            pieces.append([(ic - r, ic + r + 1)])
        else:
            pieces.append([(ic - r, ic - prev), (ic + prev + 1, ic + r + 1)])
        prev = r
    # uncovered indices (even Ns) go to the last shell
    covered = np.zeros(Ns, bool)
    for p in pieces:
        for a, b in p:
            covered[a:b] = True
    extra = np.where(~covered)[0]
    for i in extra:
        pieces[-1].append((int(i), int(i) + 1))
    return [r * hs for r in radii], pieces


def mb_loop_point(mus, k, rho=-0.55, rv=2.0, sig=None, sig_u=-0.05, hst=0.0125, tmax_u=14.0,
                  hs=0.1, hv=0.1, taumax=100.0, Vhalf=4.0, nblk=24, Dband=None, bchunk=9,
                  verbose=False):
    t0 = time.time()
    mus = tuple(float(m) for m in mus)
    nus = [1j * m for m in mus]
    x0, x1, x2 = (float(v) for v in k)
    sig = rho / 2.0 if sig is None else sig
    taus = frange(-taumax, taumax, hs)
    Ns = len(taus)
    v = frange(-Vhalf, Vhalf, hv)
    Nv = len(v)
    us = [rho + 1j * (-m + v) for m in mus]
    delta = -rho - 1j * v
    D = int(math.ceil((10.0 + Vhalf) / hs)) if Dband is None else int(Dband)
    s = sig + 1j * taus
    fA = np.exp(line_log(s[None, :], us[0][:, None], nus[0]))
    fB = np.exp(line_log(s[None, :], us[1][:, None], nus[1]))
    fC = np.exp(line_log(s[None, :], us[2][:, None], nus[2]))
    d = np.arange(-D, D + 1)

    def vrow(n1, n2, u, x):
        z = rv + n1 + n2 + 2.0 * u[:, None] + 2j * d[None, :] * hs
        return np.exp(lg(z) - z * math.log(x))

    g1 = vrow(nus[0], nus[1], us[0], x1)
    g2 = vrow(nus[1], nus[2], us[1], x2)
    g0 = vrow(nus[0], nus[2], us[2], x0)
    band_edge = max(float(np.max(np.maximum(np.abs(g[:, 0]), np.abs(g[:, -1])) / np.abs(g).max(axis=1)))
                    for g in (g1, g2, g0))
    W = 2 * D + 1
    pad = lambda f: np.concatenate([np.zeros((f.shape[0], D), complex), f,
                                    np.zeros((f.shape[0], D), complex)], axis=1)
    fBp, fCp = pad(fB), pad(fC)
    idx = np.arange(Ns)[:, None] + np.arange(W)[None, :]               # i + d + D
    # H[c, i, e] = f_C[c, i+e] g0[c, -e]
    H = fCp[:, idx] * g0[:, ::-1][:, None, :]
    H = H.reshape(Nv, Ns * W)
    Nl, pieces = windows(Ns, nblk, hs)
    nsh = len(pieces)
    Tsh = np.zeros((nsh, Nv, Nv, Nv), dtype=complex)
    dd = d[None, :] - d[:, None]                                        # e - d1
    for b in range(Nv):
        Fb = fBp[b][idx]                                                # (Ns, W): f_B[b, i+d1]
        G2 = np.where(np.abs(dd) <= D, g2[b][np.clip(dd + D, 0, W - 1)], 0.0)   # [d1, e]
        for a0 in range(0, Nv, bchunk):
            al = slice(a0, min(Nv, a0 + bchunk))
            X = Fb[None, :, :] * g1[al][:, None, :]
            Y = (X @ G2) * fA[al][:, :, None]                           # (na, Ns, W)
            Y = Y.reshape(Y.shape[0], Ns * W)
            for m, pc in enumerate(pieces):
                for i0, i1 in pc:
                    Tsh[m, al, b, :] += Y[:, i0 * W:i1 * W] @ H[:, i0 * W:i1 * W].T
        if verbose and b % 10 == 0:
            print(f"b {b}/{Nv}  {time.time() - t0:.0f} s", flush=True)
    Tsh *= (hs / (2 * np.pi)) ** 3
    c3 = C3Core(k, delta, rho, sig_u=sig_u, hst=hst, tmax_u=tmax_u)
    C3 = c3.full()
    w = np.pi / np.sin(np.pi * (1.0 - delta)) * np.exp(2.0 * delta * math.log(2.0))
    Wt = w[:, None, None] * w[None, :, None] * w[None, None, :]
    pref = 0.125 * (hv / (2 * np.pi)) ** 3
    contrib = np.array([pref * np.sum(Wt * C3 * Tsh[m]) for m in range(nsh)])
    S_ladder = np.cumsum(contrib)
    full = Tsh.sum(axis=0) * Wt * C3
    V_ladder = {}
    for Vp in (2, 2.5, 3, 3.5, 4, 5, 6):
        if Vp <= Vhalf + 1e-12:
            msk = np.abs(v) <= Vp + 1e-9
            val = pref * np.sum(full[np.ix_(msk, msk, msk)])
            V_ladder[f"{Vp:g}"] = [val.real, val.imag]
    S2hv = None
    if Nv % 2 == 1:
        val = 8 * pref * np.sum(full[::2, ::2, ::2])
        S2hv = [val.real, val.imag]
    return {"mus": list(mus), "k": [x0, x1, x2], "rv": rv, "rho": rho, "sig": sig, "sig_u": sig_u,
            "hst": hst, "tmax_u": tmax_u, "hs": hs, "taumax": taumax, "Vhalf": Vhalf, "hv": hv,
            "Dband": D, "Ns": Ns, "Nv": Nv, "band_edge_ratio": band_edge,
            "w3_pole_margin": c3.margin, "N_ladder": Nl,
            "S_ladder": [[z.real, z.imag] for z in S_ladder], "V_ladder": V_ladder,
            "S_2hv": S2hv, "wall_s": time.time() - t0}


def extrapolate_window(N, S, p, nterms=3, nfit=10, weights=None):
    """Least-squares fit S(N) = S_inf + N^-p (c0 + c1/N + ...) on the last nfit points."""
    N = np.asarray(N, float)[-nfit:]
    S = np.asarray(S, complex)[-nfit:]
    cols = [np.ones_like(N)] + [N ** (-p - j) for j in range(nterms)]
    X = np.column_stack(cols).astype(complex)
    c, *_ = np.linalg.lstsq(X, S, rcond=None)
    resid = float(np.max(np.abs(X @ c - S)))
    return complex(c[0]), resid


def extrapolate_record(rec, Nmin=20.0):
    """Assembly step 9: S_inf, error, N_last, S_last, tail_last, fit residual."""
    N = np.array(rec["N_ladder"])
    S = np.array([complex(*z) for z in rec["S_ladder"]])
    Nmax = rec["taumax"] - rec["Dband"] * rec["hs"]
    sel = (N >= Nmin - 1e-12) & (N <= Nmax + 1e-12)
    N, S = N[sel], S[sel]
    p = 5.0 + 6.0 * rec["rho"]
    S3, resid = extrapolate_window(N, S, p, 3, 10)
    S2, _ = extrapolate_window(N, S, p, 2, 10)
    S15, _ = extrapolate_window(N, S, p, 3, 15)
    err = max(abs(S3 - S2), abs(S3 - S15))
    return {"S_inf": S3, "extrap_err": err, "N_last": float(N[-1]), "S_last": S[-1],
            "tail_last": abs(S3 - S[-1]), "fit_resid": resid, "p": p}
