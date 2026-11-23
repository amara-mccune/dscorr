"""Ranges of the leading squeezed term (Sec. 10.1 and the models-at-a-glance table).

For each loop, the largest R = 2k3/(k1 + k2) on k1 = k2 up to which the leading squeezed term, with its exact
coefficient, reproduces the pair signal to 10% as a complex amplitude, |T_lead/S - 1| <= 0.1 for every R' <= R
(T_lead the leading term, or the leading term times the factor). Same criterion as the series ranges of Sec. 10.1.
'factor' repeats the test for S_lead times the phase factor of Sec. 10.2:
    scalar triangle:                 Eq. (scalarrapidity), [(1 + sqrt(1 - R^2/4))/2]^(-nu_BC) (1 - R^2/4)^(-2);
    lines with a chemical potential: (1 + R/2)^(-2 nu).
Series rows use data/fig23_series.json (trusted while the last two orders change the sum by less than 0.2%);
two-vertex rows use the sampled curves of data/fig23_two_vertex.json and data/fig_smshape.json.
A range that reaches the end of the reliable data is printed as '>R'.

Usage:  python shape_ranges.py [--json out.json]      (default record: data/shape_ranges_record.json)
"""
import json, sys
import numpy as np

SER = json.load(open('data/fig23_series.json'))['series']
TV = json.load(open('data/fig23_two_vertex.json'))
SM = json.load(open('data/fig_smshape.json'))


def binom(a, k):
    out = 1.0 + 0j
    for j in range(k):
        out *= (a - j) / (j + 1)
    return out


def series_T(e, R):
    """S/S_lead = sum_n a_n R^n (a_n = A_N/A_0 of the data file), trusted while the last two orders change the sum
    by less than 0.2% (the trust criterion of the series ranges)."""
    a = [complex(*z) for z in e['ratio']]
    a = [x / a[0] for x in a]
    N = len(a) - 1
    T = sum(a[n] * R ** n for n in range(N + 1))
    T2 = sum(a[n] * R ** n for n in range(N - 1))
    bad = np.nonzero(np.abs(T - T2) > 0.002 * np.abs(T))[0]
    stop = bad[0] if bad.size else R.size
    return R[:stop], T[:stop], complex(*e['two_nu'])


def first_exit(R, ratio, tol=0.1):
    """ratio = template / signal"""
    bad = np.nonzero(np.abs(ratio - 1) > tol)[0]
    if bad.size == 0:
        return R[-1], True
    i = bad[0]
    if i == 0:
        return 0.0, False
    # linear interpolation of |ratio - 1| between the last good and the first bad point
    d0, d1 = abs(ratio[i - 1] - 1) - tol, abs(ratio[i] - 1) - tol
    return R[i - 1] + (R[i] - R[i - 1]) * (-d0) / (d1 - d0), False


def fmt(x):
    r, open_ = x
    return ('>%.2f' % r) if open_ else '%.3f' % r


rows = []
R = np.arange(1, 3201) / 2000.0
for e in SER:
    Rmax = 1.6 if e['group'] == 'scalar' else 1.0
    r, T, two_nu = series_T(e, R[R <= Rmax + 1e-12])
    lead = first_exit(r, 1 / T)
    if e['group'] == 'scalar':
        F = ((1 + np.sqrt(1 - r ** 2 / 4)) / 2) ** (-two_nu) / (1 - r ** 2 / 4) ** 2
    else:
        F = (1 + r / 2) ** (-two_nu)
    fac = first_exit(r, F / T)
    rows.append((e['group'], e['style'], e['label'], fmt(lead), fmt(fac), r[-1]))


def sampled(u, a, p):
    u = np.asarray(u); return 2 * u, np.asarray(a) * np.exp(1j * np.unwrap(np.asarray(p)))


for key in ('mu=0.5', 'mu=1', 'mu=2', 'mu=4', 'nu=1.4', 'nu=1.45'):
    Rr, rat = sampled(TV['higgs_portal']['u'], TV['higgs_portal'][key]['abs'], TV['higgs_portal'][key]['arg'])
    rows.append(('higgs', 'bub', key, fmt(first_exit(Rr, 1 / rat)), '', Rr[-1]))
for key in ('lam=40,mu=4', 'lam=20,mu=2', 'lam=10,mu=1'):
    Rr, rat = sampled(TV['charged_scalar']['u'], TV['charged_scalar'][key]['abs'], TV['charged_scalar'][key]['arg'])
    rows.append(('charged', 'bub', key, fmt(first_exit(Rr, 1 / rat)), '', Rr[-1]))
for sec, keys in (('wz', ('mu=1', 'mu=2', 'mu=4')), ('top_quark_full', ('m=1', 'm=2'))):
    for key in keys:
        Rr, rat = sampled(SM[sec]['u'], SM[sec][key]['abs'], SM[sec][key]['arg'])
        rows.append((sec, 'bub', key, fmt(first_exit(Rr, 1 / rat)), '', Rr[-1]))

for r in rows:
    print('%-15s %-4s %-26s lead %-9s factor %-9s (data to R=%.2f)' % r)
out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else 'data/shape_ranges_record.json'
json.dump([dict(group=a, style=b, label=c, lead=d, factor=f, Rmax=float(g)) for a, b, c, d, f, g in rows],
          open(out, 'w'), indent=1)
print('record:', out)
