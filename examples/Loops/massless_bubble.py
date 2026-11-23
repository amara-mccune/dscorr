"""The bubble of two massless lines (nu = 3/2) at r_v = 2 in closed form (Sec. 4.5, App. A, Eq. masslessbubble).

psi_ren is the finite part at coincidence of the integral over the invariant distance, Eq. (psiren):
    psi_ren = (1/2 pi^2) FP int_1^oo dZ g_{3/2}(Z)^2 K(Z),   g_{3/2} = Z/(Z^2-1) - L/2,   L = ln((Z+1)/(Z-1)).
The closed form holds for k < x_1, x_2 (the physical region). The script compares it with the integral over Z
at seven points and writes massless_bubble_record.json.

Usage:  python massless_bubble.py [--json out.json]
"""
import json, sys
import mpmath as mp

mp.mp.dps = 60


def S12(x):
    """Nielsen polylogarithm S_{1,2}(x) = (1/2) int_0^x ln^2(1-t) dt/t, 0 <= x <= 1."""
    if x == 0:
        return mp.mpf(0)
    if x == 1:
        return mp.zeta(3)
    return mp.zeta(3) - mp.polylog(3, 1 - x) + mp.log(1 - x) * mp.polylog(2, 1 - x) + mp.log(x) * mp.log(1 - x) ** 2 / 2


def B(a, b):
    """B(a, b) = int_0^1 ln(1 - b w)/(1 - a w) dw for 0 < a < 1, 0 < b <= 1 (Eq. masslessB)."""
    rho = 1 - a / b
    return (mp.log(a / b) * mp.log(1 - a) - mp.log(1 - a) ** 2 / 2 + mp.polylog(2, rho) - mp.polylog(2, rho / (1 - a))) / a


def Lam(a, b):
    """Lambda(a, b) = int_0^1 ln(1 - a w) ln(1 - b w) dw/w = sum_{m,n>=1} a^m b^n/(m n (m+n)), 0 <= a, b <= 1."""
    if a == 1 and b == 1:
        return 2 * mp.zeta(3)
    if a == 1 or b == 1:
        c = b if a == 1 else a
        return mp.polylog(3, c) + S12(c)
    if a == b:
        return 2 * S12(a)
    # double sum: sum_n b^n/(n (m+n)) = [-ln(1-b) - T_m]/m, T_m = sum_n b^n/(m+n), from T_{m-1} = b/m + b T_m
    eps = mp.mpf(10) ** (-mp.mp.dps - 5)
    Ma = int(mp.ceil(mp.log(eps) / mp.log(a))) + 10
    Mb = Ma + int(mp.ceil(mp.log(eps) / mp.log(b))) + 10
    T, Ts = b / (Mb + 1) / (1 - b), {}
    for m in range(Mb, 0, -1):
        if m <= Ma:
            Ts[m] = T
        T = b / m + b * T
    return mp.fsum(a ** m / m ** 2 * (-mp.log(1 - b) - Ts[m]) for m in range(1, Ma + 1))


def psi_closed(x1, x2, k):
    """4 pi^2 psi_ren from Eq. (masslessbubble)."""
    x1, x2, k = map(mp.mpf, (x1, x2, k))
    E, Dx = x1 + x2, x1 - x2
    u1, u2 = (x1 - k) / (x1 + k), (x2 - k) / (x2 + k)
    beta, s = [mp.mpf(1), u1 * u2, u1, u2], [-1, -1, 1, 1]
    I1 = mp.zeta(2) + mp.polylog(2, u1 * u2) - mp.polylog(2, u1) - mp.polylog(2, u2)
    I2 = mp.fsum(s[i] * s[j] * Lam(beta[i], beta[j]) for i in range(4) for j in range(4))
    M = lambda a: mp.fsum(s[i] * B(a, beta[i]) for i in range(4))
    dlog = mp.log((x1 + k) / (x2 + k)) / (4 * Dx) if Dx != 0 else 1 / (4 * (x1 + k))
    return ((1 + mp.log(2 * E ** 2 / ((x1 + k) * (x2 + k)))) / (4 * E) + dlog - 3 * I1 / (4 * k)
            + x1 * x2 * I2 / (4 * k ** 3)
            + (E * u1 * u2 * M(u1 * u2) - x2 * u1 * M(u1) - x1 * u2 * M(u2)) / k ** 2)


def psi_Z(x1, x2, k, eps='1e-24'):
    """4 pi^2 psi_ren from the integral over Z, Eq. (psiren), with Z = 1 + e."""
    x1, x2, k = map(mp.mpf, (x1, x2, k))
    E = x1 + x2
    g = lambda e: (1 + e) / (e * (2 + e)) - mp.log((2 + e) / e) / 2
    K = lambda e: mp.log((k * mp.sqrt(E ** 2 + 2 * x1 * x2 * e + k ** 2 * e * (2 + e)) + k ** 2 * (1 + e) + x1 * x2)
                         / ((x1 + k) * (x2 + k))) / (2 * k)
    C = 1 / (8 * E)                                       # g^2 K -> C/e at coincidence
    f = lambda e: g(e) ** 2 * K(e) - C / e
    ep = mp.mpf(eps)
    with mp.workdps(mp.mp.dps + 60):                       # f ~ A + B ln e below ep
        f1, f2 = f(ep), f(ep / 10)
    Bc = (f1 - f2) / mp.log(10)
    Ac = f1 - Bc * mp.log(ep)
    near = mp.quad(f, [ep, ep * 1e3, mp.mpf('1e-9'), mp.mpf('1e-6'), mp.mpf('1e-3'), mp.mpf('0.01'), mp.mpf('0.1'), 1])
    far = mp.quad(lambda e: g(e) ** 2 * K(e), [1, 3, 10, 100, mp.inf])
    return 2 * (near + ep * (Ac + Bc * (mp.log(ep) - 1)) + far)


POINTS = [(1, 1, 0.2), (1.3, 0.7, 0.4), (2, 1, 0.9), (1, 0.5, 0.1), (1, 0.9, 0.85), (3, 1, 0.05), (1, 1, 0.99)]

if __name__ == '__main__':
    rows, worst = [], 0
    # the identities behind the closed form
    Zt = mp.mpf('1.7'); Lt = mp.log((Zt + 1) / (Zt - 1))
    H = Zt / (2 * (Zt ** 2 - 1)) + Lt / 4 - Zt * Lt ** 2 / 4
    Hq = mp.quad(lambda Z: (Z / (Z ** 2 - 1) - mp.log((Z + 1) / (Z - 1)) / 2) ** 2, [Zt, 3, 10, mp.inf])
    print('H(Z) = int_Z^oo g^2 at Z = 1.7: |closed - quad| = %s' % mp.nstr(abs(H - Hq), 3))
    for a, b in [(mp.mpf('0.3'), mp.mpf('0.7')), (mp.mpf('0.7'), mp.mpf('0.3')), (mp.mpf('0.4'), mp.mpf(1))]:
        qB = mp.quad(lambda w: mp.log(1 - b * w) / (1 - a * w), [0, 1])
        qL = mp.quad(lambda w: mp.log(1 - a * w) * mp.log(1 - b * w) / w, [0, 1])
        print('B, Lambda at (%s, %s): %s, %s' % (a, b, mp.nstr(abs(qB - B(a, b)), 3), mp.nstr(abs(qL - Lam(a, b)), 3)))
    for p in POINTS:
        c, z = psi_closed(*p), psi_Z(*p)
        d = abs(c / z - 1)
        worst = max(worst, d)
        rows.append(dict(x1=p[0], x2=p[1], k=p[2], four_pi2_psi_ren=mp.nstr(c, 25), rel_diff=float(d)))
        print('(x1, x2, k) = %-16s 4 pi^2 psi_ren = %s   closed/Z - 1 = %s' % (p, mp.nstr(c, 20), mp.nstr(d, 3)))
    print('worst relative difference: %.1e' % worst)
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else 'massless_bubble_record.json'
    json.dump(dict(description='4 pi^2 psi_ren of the bubble of two massless lines at r_v = 2, Eq. (masslessbubble), '
                               'against the integral over Z, Eq. (psiren)', points=rows, worst=float(worst)),
              open(out, 'w'), indent=1)
