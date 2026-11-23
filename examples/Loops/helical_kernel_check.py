"""Numerical check of Eq. (8.3) (Sec. 8.1): the ordered helical kernel as a double Barnes integral.
We close the s' contour to the right. The poles of Gamma(-s') give Kummer's series at t'; the poles b = 1 + k give
terms whose s integrand is rational times (t/t')^s, computed here numerically and added. For t > t' these vanish
(the text's statement); for t < t' they supply the other ordering. Compared with Eq. (8.2) via mpmath's Whittaker functions, in both time orderings, on the principal and
complementary series and at the four Dirac indices K = +-(1/2 +- i kappa). Writes helical_kernel_check_record.json.

Usage:  python helical_kernel_check.py [case indices]"""
import mpmath as mp
mp.mp.dps = 25

def lhs(y, t, tp, K, nu):
    ap = mp.mpf(1)/2 + nu - K
    tg, tl = max(t, tp), min(t, tp)
    return mp.gamma(ap)/mp.gamma(1+2*nu) * mp.whitw(K, nu, 2*y*tg)*mp.whitm(K, nu, 2*y*tl)/(2*y*mp.sqrt(t*tp))

def rhs(y, t, tp, K, nu, c):
    ap = mp.mpf(1)/2 + nu - K; am = mp.mpf(1)/2 - nu - K
    f = lambda x: (lambda s: mp.gamma(-s)*mp.gamma(ap+s)/mp.gamma(1+2*nu+s)*mp.pi/mp.sin(mp.pi*(1+2*nu+s))
                   * (2*y)**s * t**(nu+s))(c + 1j*x)
    Js = mp.quad(f, [-mp.inf, -20, -5, 0, 5, 20, mp.inf])/(2*mp.pi)
    # poles of Gamma(a_+ + s) that lie right of the line Re s = c belong to the left of the separating contour
    m = 0
    while mp.re(-ap - m) > c:
        s0 = -ap - m
        Js += (-1)**m/mp.factorial(m) * mp.gamma(-s0)/mp.gamma(1+2*nu+s0)*mp.pi/mp.sin(mp.pi*(1+2*nu+s0))*(2*y)**s0*t**(nu+s0)
        m += 1
    tot = mp.nsum(lambda n: 1/(mp.factorial(n)*mp.gamma(1+2*nu+n)*mp.gamma(am-n)) * (2*y)**(2*nu+n) * tp**(nu+n), [0, mp.inf])
    main = tot*Js
    # b-pole terms: s integrand R_k(s) (t/t')^s, rational; closed to the left (no poles, zero) for t > t',
    # to the right for t < t', where I_k = -(sum of residues at s = j and s = m - 2 nu, j, m = 0..k).
    bp = 0
    if t < tp:
        for k in range(0, 200):
            res = 0
            for j in range(k+1):
                den = mp.fprod([i-j for i in range(k+1) if i != j]) * mp.rf(2*nu+j-k, k+1)
                res += -mp.rf(ap-k+j, k)/den * (t/tp)**j
            for m in range(k+1):
                s0 = m - 2*nu
                den = mp.rf(-s0, k+1) * mp.fprod([m-k+i for i in range(k+1) if i != k-m])
                res += mp.rf(ap-k+s0, k)/den * (t/tp)**s0
            term = (-1)**k * (2*y)**k * t**nu * tp**(k-nu) * (-res)
            bp += term
            if k > 5 and abs(term) < mp.mpf(10)**(-mp.mp.dps-3) * abs(main): break
    return mp.e**(-y*(t-tp)) * (main + bp), mp.e**(-y*(t-tp)) * bp

cases = [('principal h=+1, mu=1, kappa=0.5', -0.5j, 1j, mp.mpf(-0.25)),
         ('principal h=-1, mu=2, kappa=0.7', 0.7j, 2j, mp.mpf(-0.25)),
         ('complementary nu=0.3, kappa=0.5', -0.5j, mp.mpf('0.3'), mp.mpf(-0.7)),
         ('Dirac K=1/2-i kappa, mu=1.2', mp.mpf(0.5)-0.5j, 1.2j, mp.mpf(-0.25)),
         ('Dirac K=-1/2+i kappa, mu=1.2', mp.mpf(-0.5)+0.5j, 1.2j, mp.mpf(-0.25)),
         ('Dirac K=-1/2-i kappa, mu=1.2', mp.mpf(-0.5)-0.5j, 1.2j, mp.mpf(-0.25)),
         ('Dirac K=1/2+i kappa, mu=1.2', mp.mpf(0.5)+0.5j, 1.2j, mp.mpf(-0.25))]
import sys
sel = cases if len(sys.argv) < 2 else [cases[int(i)] for i in sys.argv[1:]]
import json
rows = []
for name, K, nu, c in sel:
    for (t, tp) in [(1.3, 0.6), (0.6, 1.3), (2.0, 1.9), (1.9, 2.0)]:
        a = lhs(1, mp.mpf(t), mp.mpf(tp), K, nu); b, bp = rhs(1, mp.mpf(t), mp.mpf(tp), K, nu, c)
        d = abs(b/a-1)
        rows.append(dict(case=name, t=t, tp=tp, rel_diff=float(d), bpole_part=float(abs(bp/a))))
        print(f"{name:34s} t={t} t'={tp}:  |rhs/lhs - 1| = {mp.nstr(d, 3)}   |b-pole part/lhs| = {mp.nstr(abs(bp/a), 3)}")
if len(sys.argv) < 2:
    json.dump(rows, open('helical_kernel_check_record.json', 'w'), indent=1)
