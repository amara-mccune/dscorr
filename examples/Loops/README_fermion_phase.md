# The total-energy phase of the fermion loops at large chemical potential

Follow-up to Secs. 6.4 and 8.4 of "Massive de Sitter correlators at one loop" (v5.19). For kappa >> m the pair signal of the Dirac loops oscillates in k3/k_T, with k_T = k1 + k2 + k3 [Eq. (8.6)]. These files hold the forms derived for that regime and their comparison with the exact series. New files only; nothing in the paper's scripts changes. The derivation is in `kT_phase_derivation.pdf` (session of Nov 20, 2026), and "notes" below refers to it.

| File | Content |
|---|---|
| `fermion_phase.py` | Script in the style of the other scripts here. Closed forms against the moments of the model loop; the tree-level saddle phase of Qin, Renaux-Petel, Tong, Werth, and Zhu (2506.01555) in the limit kappa -> mubar; the average of (y1 + y2)/k3 over the soft moment; the Dirac bubble; and the triangle as a contact term plus a remainder. Compares with `fermion_phase_record.json` and exits nonzero if a value is not reproduced. |
| `ext/fermicontact.py` | `Contact(FermiTriangle)`: the two diagrams of the identity below, built on `ext/fermitri.py`. Closed forms: `pair_closed`, `bubble_closed`, `eps_factor`, `contact_leading`, `contact_first_order`, `segment_average`, `tree_saddle_phase`. Series tools: `log_series`, `strip_total_energy`. |
| `fermion_phase_record.json` | Values of record (Nov 20, 2026), kappa = 5, m = 1. Entries with `max` are bounds on the difference between an exact and a derived quantity. |
| `../../tests/test_fermion_phase.py` | Fast checks of the closed forms and of the shape dependence of the contact term; one slow check of the identity (`RUN_SLOW=1`). |

    cd examples/Loops
    python fermion_phase.py                        # kappa = 5, m = 1 through N = 2 (about 2 min)
    python fermion_phase.py --quick                # closed forms only (seconds)
    python fermion_phase.py --point 10 1 --dps 40  # another point; nothing is compared with the record
    python fermion_phase.py --point 20 2 --dps 60 --nmax 1

Needs Python >= 3.10 and mpmath >= 1.3. Soft moments are cached in `cache/` (shared with `fermion_triangle_shape.py`).

What is derived, and what is numerical:

- **The identity (exact).** Rotate the fermion by the fluctuation on the two hard legs, Psi -> exp(-i alpha gamma^5) Psi with alpha = -+ kappa dtau_h. The axial vertices of the hard legs disappear and the mass term becomes m Psibar exp(-2 i alpha gamma^5) Psi. At third order the triangle is the sum of a two-vertex loop with a local quartic vertex (the contact term B') and a triangle with two pseudoscalar mass vertices (T3'). The rotation is standard (Hook, Huang, and Racco, 1907.10624; Aoki and Strumia, 2604.05548). The script checks the identity coefficient by coefficient on k1 = k2 and at general shape, to 1e-16.
- **The contact term (derived).** Its hard part is elementary and exact, `contact_coefficients`. Its soft vertex is of total-energy type, so for kappa >> m its loop integral is an end-point integral at y1 + y2 = k3, and B'(R, delta)/B'(0) -> (1 + R/2)^(-2 i kappa) (1 - R/2)^-2 (1 - R^2 delta^2/4)^-2 times R^(2 nu). The exponent of the total energy is 2 i kappa, and the squeezed frequency stays 2 mubar.
- **The Dirac bubble (derived).** The same steps give (1 + R/2)^(-2 i kappa) (1 - R^2/4)^-2 [1 + 4 rho/c + 6 rho^2/(c (c + 1))], rho = 2R/(2 - R), c = 2 + 2 nu, with a remainder of order 1/kappa^2 in the coefficients of the logarithm.
- **The remainder T3' (numerical).** At R -> 0 it is about -2 m^2/kappa of the contact term, and it varies on the scale R ~ kappa^(-1/2). No closed form. It is the largest correction to the triangle at the benchmarks of the paper.
- **Single branch assignments of T3'** have a pole when both mass vertices go to late times. It cancels in the sum over the branches of the hard vertices. `Contact.t3_accumulate(delta)` shifts the power t^-1 by delta, and the script averages over delta = +-1e-8 (the two values differ by 2e-8, and the mean satisfies the identity to 1e-16).
- **Normalization.** As in `fermion_triangle_shape.py`: `tri[N]` is D_N on k1 = k2 with both orientations, and `contact_coefficients` returns D_N(B') in the same normalization.
