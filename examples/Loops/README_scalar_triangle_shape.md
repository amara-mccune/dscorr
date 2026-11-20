# The scalar triangle at general shape

Follow-up to Sec. 5.2 of "Massive de Sitter correlators at one loop". New files only; nothing in the paper's scripts changes.

| File | Content |
|---|---|
| `scalar_triangle_shape.py` | Script in the style of the other scripts here: Eq. (5.6) on k1 = k2, the recurrence for the hard moments, the exchange of the hard legs, the in-in assembly (M_tri and D_tri of Ref. [26]), the size of the series at the shapes quoted in the write-up, and the comparison with the triangle integral of Ref. [26] at delta = 0.5 and 1. Compares with `scalar_triangle_shape_record.json` and exits nonzero if a value is not reproduced. |
| `ext/scalartri.py` | The series: `ScalarTriangle(nuA, nuB, nuC, r1, r2, Nmax)` gives the coefficients of R^N delta^m of the pair family of psi_3 at r3 = 2, `correlator_family` the in-in assembly, `to_symmetric` and `exchanged_angle` the changes of variables used in the checks. |
| `scalar_triangle_shape_record.json` | Values of record (Nov 19, 2026). |
| `scalar_triangle_shape_direct.json` | psi_3 from the Mellin-Barnes integral of Ref. [26] at 17 values of R between 0.035 and 0.5, for delta = 0.5 and delta = 1 (runs of Nov 19, 2026; about five minutes and 2.3 GB per point with `examples/Letter/batch/triangle_pidot.py` and the block C_3 from the expansion of Appendix A). |
| `../../tests/test_scalar_triangle_shape.py` | The same checks at lower orders (about 30 s), and the parameter integral by quadrature at real orders (needs scipy). |

    cd examples/Loops
    python scalar_triangle_shape.py                              # mu = (1, 1.3, 0.7), N <= 24, 30 digits (about 1 min)
    python scalar_triangle_shape.py --mu 2 2 2 --nmax 30 --dps 40

Needs Python >= 3.10 and mpmath >= 1.3.

Equation numbers are those of v5.12 to v5.14 of the paper. In v5.16 the hard moment is Eq. (B.3), the scaling relation Eq. (B.5), and the two-line block with a numerator Eq. (B.6). From v5.19 on, the coefficients c_N on k1 = k2 (Eq. (5.6) of the earlier drafts) and the expansion of C_3 are cited from the Letter, the series is Sec. 5.1, Eqs. (5.1)–(5.5), and the appendix equations are (A.3), (A.5), (A.6), and (A.11).

What is computed. For r3 = 2 the soft loop momentum of the pair family sits on the segment from 0 to k3 (the term of Eq. (B.7) with j powers of J^2 carries 1/Gamma(r3 - 1 - j)), so the family is one parameter integral over three-site chains,

    psi_3  contains  pi^2 Gamma(2 + nuBC) Gamma(-1 - nuBC) k3 int_0^1 dlam [lam (1 - lam)]^(1/2)
                     int dt1 dt2 t1^(r1-1) t2^(r2-1) e^(-k1 t1 - k2 t2)
                     I_nuC(lam k3 t1) K_nuA(y t>) I_nuA(y t<) I_nuB((1 - lam) k3 t2),
    y^2 = (1 - lam) k1^2 + lam k2^2 - lam (1 - lam) k3^2.

With K = (k1 + k2)/2, R = k3/K, and delta = (k1 - k2)/k3, it expands as

    psi_3  contains  K^(1 - r1 - r2) R^(1 + nuB + nuC) sum_N D_N(delta) R^N.

Conventions worth knowing:

- **Labels.** Vertex v has energy k_v and power r_v. Line A joins vertices 1 and 2, line B joins 2 and 3, line C joins 3 and 1, and k3 is the soft leg. `coefficients()` returns `{(N, m): coefficient of R^N delta^m}`.
- **Parity.** D_N has degree at most N and parity (-1)^N in delta. Exchanging the hard legs maps delta to -delta together with (nuB, r2) <-> (nuC, r1). The D_N are even in delta only for equal soft lines and equal hard vertices; at mu = (1, 1.3, 0.7) the odd orders are present away from k1 = k2 (|D_1/D_0| = 0.044 at delta = 1).
- **One hypergeometric value.** All hard moments follow from H_nuA(r1 - 1 + nuC, r2 - 1 + nuB) of Eq. (B.4) by
  (2 a0 + 3) H(a0 + 1, a1) = [(a0 + 1)^2 - nu^2] H(a0, a1) + Gamma(a0 + a1 + 2)/2^(a0 + a1 + 2), and the same in a1. `recurrence=False` evaluates each from Eq. (B.4) instead.
- **Convergence.** At fixed delta the radius is R = 2 (the collinear configuration k3 = k1 + k2). Thirty orders take about 40 s at 40 digits.
- **Correlator.** `correlator_family(mus, sB, sC)` returns A_N(delta) with S = (k1 k2 k3)^2 <pi pi pi>' containing (1 - R^2 delta^2/4) 2 Re[R^(2 + i (sB muB + sC muC)) sum A_N R^N] (weights of Eq. (2.10); A_0 at (+, +) is M_tri and at (+, -) is D_tri).
- **Other soft-vertex powers.** For r3 different from 2 the terms with j >= 1 contribute as well, and this module does not include them. They were checked through the first correction at r3 = 2 + 0.5 i (see the notes of the run).
- **A closed factor.** On k1 = k2 the sum of the series is reproduced to about 1% up to R = 1 by (1 - R^2/4)^(-2) [(1 + sqrt(1 - R^2/4))/2]^(-nuB - nuC), its leading form at large masses.
