# The fermion triangle beyond the leading squeezed term, at general shape

Behind Sec. 9.3 of "Massive de Sitter correlators at one loop" (arXiv v1). New files only; nothing in the paper's scripts changes.

| File | Content |
|---|---|
| `fermion_triangle_shape.py` | Script in the style of the other scripts here: the leading coefficient against the closed form of Song et al., the exchange symmetry k1 <-> k2, and the series in the symmetric variables. Compares with `fermion_triangle_shape_record.json` and exits nonzero if a value is not reproduced. |
| `ext/fermitri.py` | The expansion: spinor components on the rotated contour, the hard fermion line in closed form (Gamma functions and 4F3(-1)), its energy and the energy of the second hard leg by scaling, spin traces as polynomials, and the assembly over branches, helicities, and chiralities. |
| `ext/shapevars.py` | The exchange identity for a series in (r, c), and the re-expansion in R = 2 k3/(k1 + k2), delta = (k1 - k2)/k3. Also used by `helical_triangle.py --general`. |
| `fermion_triangle_shape_record.json` | Values of record (Nov 18, 2026, with the orders added on Nov 19: n = 9 and 10 at kappa = 1.3 and 5, n = 7 and 8 at kappa = 10). |
| `direct/` | Direct time-domain integration of the triangle at fixed shape (nothing expanded) and the fits that compare it with the series. See `direct/README.md`. The runs of Nov 19, 2026 are stored in `direct/data/`. |
| `../../tests/test_fermion_triangle_shape.py` | Fast checks of the hard-line formula and of the shape variables; one slow check of the leading and first orders (`RUN_SLOW=1`). |
| `../../tests/test_fermion_triangle_direct.py` | The stored direct runs against the series (needs numpy). |

    cd examples/Loops
    python fermion_triangle_shape.py                        # kappa = 1.3, m = 1 through n = 2 (about 3 min)
    python fermion_triangle_shape.py --point 5 1 --nmax 6   # one point (about 15 min)
    python fermion_triangle_shape.py --all                  # kappa = 1.3 and 5 (n <= 10), 10 (n <= 8); about three hours
    python fermion_triangle_shape.py --point 1.3 1 --nmax 10 --save series.json    # coefficients for direct/fermi_shape_fit.py

Needs Python >= 3.10 and mpmath >= 1.3. Soft moments are cached in `cache/`.

Conventions worth knowing:

- **Normalization.** Model and conventions of `ext/fermion.py` and `fermion_triangle.py` (Aoki and Strumia): Dirac fermion, axial chemical potential kappa, couplings kappa at the vertices with 1/taudot_0 stripped. `C[(0, 0)]` is the coefficient C of Sec. 9.3, both orientations of the loop included. For the Majorana fermion of Song et al., divide by 2 kappa^3 Lambda^3.
- **Two forms of the series.** `C[(n, mc)]`: k1^6 <dtau^3>' = sum C r^(2 nu + n) c^mc + c.c., with r = k3/k1 and c = k1hat.k3hat (converges for k3 < k1 only). `D[(N, md)]`: K^6 <dtau^3>' = sum D R^(2 nu + N) delta^md + c.c. (even md only; at kappa = 1.3 the last two of ten orders change the equilateral value by 0.4%).
- **Unit steps.** The first index K of a fermion never vanishes, so odd and even orders are both present.
- **The hard line** is in closed form, so there is no contour angle. At large kappa the soft and hard moments lose digits (about six at kappa = 10); use 40 digits there. The exchange symmetry measures the digits that survive.
- **Large kappa.** For kappa much larger than m, |D_N/D_0| grows like |nu|^N/N!. Most of it is the phase of (2 k3/k_T)^(2 nu). This is a numerical finding, not a derived result, and it fails at m = kappa. The direct integration confirms the first two corrections at kappa = 5 (`direct/README.md`).
- **Structures.** `FermiTriangle(structures=[('1', '1', '1')])` keeps the time-derivative part of every vertex only ('k' is the gradient part). Each choice is a complete series by itself and can be compared with `direct_fermi.py` under `STRUCT=111`.
