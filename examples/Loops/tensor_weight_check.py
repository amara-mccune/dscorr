"""Check of Eqs. (3.23)-(3.24) (Sec. 3.5): the polarization weight of the tensor bubble acts on the loop block as
    int d^3l T2(c) y1^(p1-1) y2^(p2-1) = 2 pi k^(p1+p2+1) varrho(alpha, beta) C2hat(p1, p2),
with alpha = (1-p1)/2, beta = (1-p2)/2, T2(c) = (c^4 + 6c^2 + 1)/4, and C2hat of Sec. 2.2. The left side is integrated
directly over the loop momentum at three pairs of powers where it converges. Also checks varrho = 2 on alpha+beta = 3/2.

Usage:  python tensor_weight_check.py
"""
import mpmath as mp
import sympy as sp

mp.mp.dps = 20


def C2(d1, d2):
    S = d1 + d2
    return mp.pi ** 1.5 * mp.gamma(1.5 - d1) * mp.gamma(1.5 - d2) / (mp.gamma(d1) * mp.gamma(d2)) * mp.gamma(S - 1.5) / mp.gamma(3 - S)


def C2hat(p1, p2):
    return C2((1 - p1) / 2, (1 - p2) / 2) / (2 * mp.pi)


def varrho(a, b):
    return (16 * a ** 2 * b ** 2 + 16 * a * b - 12 * a ** 2 - 12 * b ** 2 + 27) / (8 * a * (a + 1) * b * (b + 1))


def T2(c):
    return (c ** 4 + 6 * c ** 2 + 1) / 4


def lhs(p1, p2, k=1):
    def f(r, u):                      # l = (r, u = cos theta) with k along z; y1 = r, y2 = |k - l|
        y2 = mp.sqrt(r ** 2 + k ** 2 - 2 * r * k * u)
        c = (r * k * u - r ** 2) / (r * y2)
        return 2 * mp.pi * r ** 2 * T2(c) * r ** (p1 - 1) * y2 ** (p2 - 1)
    return mp.quad(f, [0, 0.5, 1, 2, mp.inf], [-1, 0, 1])


if __name__ == '__main__':
    worst = 0
    for p1, p2 in [('-0.8', '-0.8'), ('-0.6', '-1.1'), ('-1.3', '-0.4')]:
        p1, p2 = mp.mpf(p1), mp.mpf(p2)
        L, R = lhs(p1, p2), 2 * mp.pi * varrho((1 - p1) / 2, (1 - p2) / 2) * C2hat(p1, p2)
        worst = max(worst, abs(L / R - 1))
        print('(p1, p2) = (%s, %s):  direct %s   Eq. (3.23) %s   rel. diff. %s' % (p1, p2, mp.nstr(L, 12), mp.nstr(R, 12), mp.nstr(abs(L / R - 1), 3)))
    al = sp.symbols('alpha')
    print('varrho(alpha, 3/2 - alpha) - 2 =', sp.simplify(varrho(al, sp.Rational(3, 2) - al) - 2))
    print('worst: %.1e' % worst)
