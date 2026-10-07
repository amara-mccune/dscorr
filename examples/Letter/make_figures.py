"""Regenerate the figures of the Letter and its Supplemental Material.

    python make_figures.py all                 # every figure (about 45 CPU-minutes, cached afterwards)
    python make_figures.py fig2 figS1          # selected figures
    python make_figures.py fig2 --no-direct    # skip the direct Schwinger-Keldysh squares
    python make_figures.py all --quick         # coarse grids, no direct points (a few minutes)

Targets and outputs (default directory ``figures/`` next to this script):

    fig1   fig1.pdf   diagrams: double exchange and triangle (drawing only)
    fig2   fig2.pdf   triple exchange (a) and its squeezed collider signal (b)      [dscorr, live]
    fig3   fig3.pdf   one-loop triangle along (k1, 1, 1.1)                         [data/ values of record]
    figS1  figS1.pdf  double exchange                                    [dscorr, live; data/]
    figS2  figS2.pdf  (a),(b) multi-exchange sums [dscorr, live]; (c) triangle correlator [data/]
    figS3  figS3.pdf  renormalized bubble                                           [data/]

Tree curves are computed with dscorr; the squares are direct Schwinger-Keldysh integrations
(``dscorr.direct_sk``, 28 outer panels, 10-30 s each).  Results are cached in ``cache/`` (delete
it to recompute).  The one-loop values are read from ``data/`` (see data/README.md).

Comparison points from other groups' codes and papers are drawn only if the corresponding file
is present in the directory given by ``--external`` (default ``data/external``):

    cosmoflow.csv                  columns diagram,mu1,mu2,mu3,kappa,S   Fig. S2(a),(b) circles
                                   (diagram = double|triple, mu3 empty for double; S summed over the
                                   six leg assignments at (1,1,kappa))
    herderschee_lu.csv             columns mu,k3_over_k1,S_over_S_equilateral   Fig. S2(c) circles

Lines starting with '#' are ignored.  Two comparison sets are computed by the author from formulas
printed in other papers and ship in data/ with that credit: the diamonds of Fig. S1 (series solution
of Aoki, Pinol, Sano, Yamaguchi and Zhu, JHEP 09 (2024) 176, aoki_et_al_double_series.csv) and the
spectral-decomposition circles of Fig. S3 (Xianyu and Zhang, bubble_spectral_xianyu_zhang.csv).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

BLUE, YELLOW, RED = "#4053d3", "#ddb310", "#b51d14"
GRAY, ZERO, FRAME = "#595959", "#bfbfbf", "#4c4d4c"
DIAGRAM = "#1f4e79"
PT = 1 / 72.0

TRIPLES = {"neq": (1.0, 1.5, 0.7), "eq": (1.0, 1.0, 1.0), "neq2": (0.5, 1.0, 1.5)}
PAIRS = {"eq": (1.0, 1.0), "neq": (1.0, 1.5), "neq2": (0.7, 1.5)}
COLOR3 = {"neq": BLUE, "eq": YELLOW, "neq2": RED}
COLOR2 = {"eq": BLUE, "neq": YELLOW, "neq2": RED}


# ------------------------------------------------------------------------------------- style
def style():
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.38, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": FRAME,
        "legend.framealpha": 1.0, "legend.borderpad": 0.3, "legend.handlelength": 1.3,
        "legend.handletextpad": 0.4, "legend.labelspacing": 0.3, "legend.columnspacing": 0.8,
        "patch.linewidth": 0.8, "savefig.dpi": 300, "pdf.fonttype": 42, "path.simplify": False,
    })
    return plt


def figure(plt, w_pt, h_pt):
    return plt.figure(figsize=(w_pt * PT, h_pt * PT))


def axes_pt(fig, w_pt, h_pt, box):
    """Axes at the box (x0, x1, y0, y1) in points, measured from the top-left corner."""
    x0, x1, y0, y1 = box
    return fig.add_axes([x0 / w_pt, 1 - y1 / h_pt, (x1 - x0) / w_pt, (y1 - y0) / h_pt])


def legend_at(ax, box, frame, **kw):
    """Legend centred where the frame (x0, x1, y0, y1; points from the top-left) is drawn."""
    x0, x1, y0, y1 = box
    cx = ((frame[0] + frame[1]) / 2 - x0) / (x1 - x0)
    cy = (y1 - (frame[2] + frame[3]) / 2) / (y1 - y0)
    leg = ax.legend(loc="center", bbox_to_anchor=(cx, cy), **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


def label_at(ax, box, text_box, s):
    x0, x1, y0, y1 = box
    cx = ((text_box[0] + text_box[1]) / 2 - x0) / (x1 - x0)
    cy = (y1 - (text_box[2] + text_box[3]) / 2) / (y1 - y0)
    ax.text(cx, cy, s, transform=ax.transAxes, ha="center", va="center", fontsize=9)


def zero_line(ax):
    ax.axhline(0.0, color=ZERO, lw=0.6, zorder=0)


# ------------------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, name, fn):
        path = self.dir / (name + ".json")
        if path.is_file():
            with open(path) as f:
                return json.load(f)
        t = time.time()
        val = fn()
        print(f"  computed {name} in {time.time() - t:.0f} s", flush=True)
        with open(path, "w") as f:
            json.dump(val, f)
        return val


def read_csv(path):
    with open(path, newline="") as f:
        rows = [ln for ln in f if not ln.startswith("#")]
    out = []
    for r in csv.DictReader(rows):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v) if v not in ("", None) else None
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def external(args, name):
    path = Path(args.external) / name
    if path.is_file():
        print(f"  using {path}")
        return read_csv(path)
    return None


# ------------------------------------------------------------------------------------- grids
def scan_grid(quick=False):
    """321 abscissae: 281 log-spaced in [2e-3, 1] and 0.02 * 50^(i/40), rounded to 12 decimals."""
    if quick:
        return np.geomspace(2e-3, 1.0, 81)
    a = np.geomspace(2e-3, 1.0, 281)
    b = 0.02 * 50.0 ** (np.arange(40) / 40)
    return np.unique(np.round(np.concatenate([a, b]), 12))


def sk_grid(quick=False):
    """The 15 abscissae of the direct Schwinger-Keldysh points, log-spaced from 1 to 0.002."""
    return np.geomspace(1.0, 0.002, 5 if quick else 15)


KAPPA40 = [float(np.exp(np.log(0.05) * (1 - j / 39))) for j in range(40)]


# ------------------------------------------------------------------------------------- dscorr
def triple_curve(mus, quick):
    from dscorr import TripleExchange
    k = scan_grid(quick)
    return {"kappa": k.tolist(), "S": TripleExchange(*mus).scan(k).tolist()}


def double_curve(mus, quick):
    from dscorr import DoubleExchange
    k = scan_grid(quick)
    return {"kappa": k.tolist(), "S": DoubleExchange(*mus).scan(k).tolist()}


def direct_points(diagram, mus, quick):
    from dscorr import double_exchange_direct, triple_exchange_direct
    f = double_exchange_direct if diagram == "double" else triple_exchange_direct
    k = sk_grid(quick)
    return {"kappa": k.tolist(), "S": [float(f(1.0, 1.0, float(q), *mus, mpanels=28)) for q in k]}


def signal_panel(mus, quick):
    """Squeezed signal of Fig. 2(b): S on the merged deep + scan abscissae (kappa <= 0.25), an
    unweighted fit of the degree-six background plus the pinned oscillations (model 3 of
    dscorr.squeezed), and (S - S_an)/sqrt(kappa) on the points with kappa <= 0.2."""
    from dscorr import M3, TripleExchange, ValidityWarning
    from dscorr.squeezed import fit_panel_c
    ks = scan_grid(quick)
    kd = np.geomspace(1e-4, 1e-2, 60 if quick else 320)
    S_scan = TripleExchange(*mus).scan(ks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        S_deep = TripleExchange(*mus, L1=16.0).scan(kd)
    k = np.concatenate([kd[kd < ks.min()], ks[ks <= 0.25]])
    S = np.concatenate([S_deep[kd < ks.min()], S_scan[ks <= 0.25]])
    M_fit, _, S_an = fit_panel_c(k, S, mus[2])
    sig = (S - S_an(k)) / np.sqrt(k)
    M = M3(*mus)
    return {"kappa": k.tolist(), "signal": sig.tolist(), "M3": [M.real, M.imag],
            "M_fit": [M_fit.real, M_fit.imag]}


def sum6_lines(diagram, mus):
    from dscorr import DoubleExchange, TripleExchange
    if diagram == "double":
        D = DoubleExchange(*mus)
        return [float(D(1.0, 1.0, k, sum_perms=True)) for k in KAPPA40]
    T = TripleExchange(*mus)
    return [float(sum(T(*p) for p in itertools.permutations((1.0, 1.0, k)))) for k in KAPPA40]


# ------------------------------------------------------------------------------------- figures
def fig1(plt, args, cache):
    """Diagrams (a) double exchange, (b) triangle; drawn in points on a 200.35 x 74.29 pt canvas."""
    W, H = 200.35, 74.29
    fig = figure(plt, W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    leg = dict(color="#99a6b0", lw=0.9, solid_capstyle="butt", zorder=1)
    line = dict(color=DIAGRAM, lw=1.9, solid_capstyle="butt", zorder=2)
    bnd = dict(color="#333333", lw=1.1, zorder=1)

    def dot(x, y, r, filled):
        ax.add_patch(plt.Circle((x, y), r, facecolor=DIAGRAM if filled else "white",
                                edgecolor=DIAGRAM, lw=1.0, zorder=3))

    def txt(x, y, s, size=7.0, color="black", **kw):
        ax.text(x, y, s, fontsize=size, color=color, ha="center", va="center", **kw)

    # (a) double exchange
    ax.plot([4.1, 109.6], [20.4, 20.4], **bnd)
    V1, V3, B2 = (19.9, 41.0), (82.9, 42.1), (44.0, 20.4)   # cubic vertex, k3 mixing, k2 boundary point
    V2 = (B2[0] + 0.45 * (V1[0] - B2[0]), B2[1] + 0.45 * (V1[1] - B2[1]))   # k2 mixing, collinear with B2 and V1
    ax.plot([9.0, V1[0]], [20.4, V1[1]], **leg)
    ax.plot([B2[0], V2[0]], [B2[1], V2[1]], **leg)
    ax.plot([V3[0], V3[0]], [20.4, V3[1]], **leg)
    ax.plot([V1[0], V2[0]], [V1[1], V2[1]], **line)
    t = np.linspace(0, 1, 120)[:, None]
    P0, P1, P2 = np.array(V1), np.array([52.40, 82.52]), np.array(V3)
    arc = (1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t ** 2 * P2
    ax.plot(arc[:, 0], arc[:, 1], **line)
    dot(*V1, 2.2, True)
    dot(*V2, 1.7, False)
    dot(*V3, 1.7, False)
    txt(8.3, 4.2, "(a)", size=7.5)
    for x, s in ((8.75, r"$\mathbf{k}_1$"), (B2[0], r"$\mathbf{k}_2$"), (V3[0], r"$\mathbf{k}_3$")):
        txt(x, 14.7, s)
    txt(45.4, 29.6, r"$r_v=\frac{1}{2}$", size=6.2, color=DIAGRAM)
    txt(8.6, 47.0, r"$r_v=2$", size=6.2, color=DIAGRAM)
    txt(93.25, 46.3, r"$r_v=\frac{1}{2}$", size=6.2, color=DIAGRAM)
    txt(52.0, 40.2, r"$(s_1,\,s_1'),\;\nu_1,\;y_1$")
    txt(53.55, 69.2, r"$(s_2,\,s_2'),\;\nu_2,\;y_2$")
    # (b) triangle
    ax.plot([125.2, 190.0], [20.5, 20.5], **bnd)
    ax.plot([157.6, 157.6], [20.5, 34.2], **leg)
    ax.plot([131.1, 144.9], [20.5, 56.3], **leg)
    ax.plot([184.1, 170.4], [20.5, 56.3], **leg)
    for (xa, ya), (xb, yb) in (((157.6, 34.2), (144.9, 56.3)), ((157.6, 34.2), (170.4, 56.3)),
                               ((144.9, 56.3), (170.4, 56.3))):
        ax.plot([xa, xb], [ya, yb], **line)
    for x, y in ((157.6, 34.2), (144.9, 56.3), (170.4, 56.3)):
        dot(x, y, 2.0, True)
    txt(129.6, 4.35, "(b)", size=7.5)
    for x, s in ((130.8, r"$\mathbf{k}_1$"), (157.3, r"$\mathbf{k}_3$"), (183.85, r"$\mathbf{k}_2$")):
        txt(x, 14.8, s)
    txt(145.25, 40.4, r"$y_C$")
    txt(169.95, 40.4, r"$y_B$")
    txt(157.15, 61.0, r"$y_A$")
    txt(157.6, 49.6, r"$\boldsymbol{\ell}$", size=7.0, color=DIAGRAM)
    txt(169.2, 30.3, r"$r_v=1$", size=6.2, color=DIAGRAM)
    return fig


def fig2(plt, args, cache):
    W, H = 244.8, 205.0
    fig = figure(plt, W, H)
    q = args.quick
    tag = "_quick" if q else ""
    # (a) triple exchange
    box = (46.71, 240.1, 4.13, 66.39)
    ax = axes_pt(fig, W, H, box)
    zero_line(ax)
    for n in ("neq", "eq", "neq2"):
        c = cache.get(f"triple_curve_{n}{tag}", lambda n=n: triple_curve(TRIPLES[n], q))
        lab = "$(" + ",\\,".join("%g" % m for m in TRIPLES[n]) + ")$"
        ax.plot(c["kappa"], c["S"], color=COLOR3[n], lw=1.1, label=lab)
    if args.direct:
        for i, n in enumerate(("eq", "neq", "neq2")):
            d = cache.get(f"direct_triple_{n}{tag}", lambda n=n: direct_points("triple", TRIPLES[n], q))
            ax.plot(d["kappa"], d["S"], "s", ms=2.9, mfc="none", mec="black", mew=0.7,
                    label="direct SK" if i == 0 else None, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 1.0)
    ax.set_ylim(-0.027557, 0.0032122)
    ax.set_yticks([-0.02, -0.01, 0.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("$%.2f$"))
    ax.set_xlabel("$k_3/k_1$", labelpad=1.5)
    ax.set_ylabel("$S$, triple")
    h, lab = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3] if len(h) == 4 else list(range(len(h)))
    legend_at(ax, box, (48.56, 159.98, 48.5, 65.0), handles=[h[i] for i in order],
              labels=[lab[i] for i in order], ncol=2)
    label_at(ax, box, (224.2, 235.0, 7.65, 13.67), "(a)")
    # (b) squeezed signal
    box = (46.71, 240.1, 102.49, 174.67)
    ax = axes_pt(fig, W, H, box)
    zero_line(ax)
    from dscorr import signal
    kk = np.geomspace(1e-4, 0.2, 600)
    lead = None
    for n in ("neq", "eq", "neq2"):
        mus = TRIPLES[n]
        p = cache.get(f"signal_panel_{n}{tag}", lambda mus=mus: signal_panel(mus, q))
        k = np.array(p["kappa"])
        sel = k <= 0.2
        ax.plot(k[sel], np.array(p["signal"])[sel], color=COLOR3[n], lw=1.1,
                label=r"$\mu=%g$" % mus[2])
        M = complex(*p["M3"])
        lead, = ax.plot(kk, signal(kk, M, mus[2]), color="black", lw=0.8, ls=(0, (3, 2)),
                        zorder=3)
    lead.set_label("leading term")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.2)
    ax.set_ylim(-0.034535, 0.049654)
    ax.set_yticks([-0.02, 0.0, 0.02, 0.04])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("$%.2f$"))
    ax.set_xlabel("$k_3/k_1$", labelpad=1.5)
    ax.set_ylabel(r"$(S-S_{\rm an})\,(k_1/k_3)^{1/2}$, triple")
    h, lab = ax.get_legend_handles_labels()
    legend_at(ax, box, (55.48, 231.33, 103.93, 113.0), handles=h, labels=lab, ncol=4,
              handlelength=1.6)
    label_at(ax, box, (50.6, 61.9, 165.18, 171.34), "(b)")
    return fig


def figS1(plt, args, cache):
    W, H = 244.8, 111.5
    fig = figure(plt, W, H)
    q = args.quick
    tag = "_quick" if q else ""
    box = (46.71, 240.1, 4.82, 87.48)
    ax = axes_pt(fig, W, H, box)
    zero_line(ax)
    for n in ("eq", "neq", "neq2"):
        c = cache.get(f"double_curve_{n}{tag}", lambda n=n: double_curve(PAIRS[n], q))
        ax.plot(c["kappa"], c["S"], color=COLOR2[n], lw=1.1,
                label="$(" + ",\\,".join("%g" % m for m in PAIRS[n]) + ")$")
    handles = ax.get_legend_handles_labels()
    if args.direct:
        for i, n in enumerate(("eq", "neq", "neq2")):
            d = cache.get(f"direct_double_{n}{tag}", lambda n=n: direct_points("double", PAIRS[n], q))
            ax.plot(d["kappa"], d["S"], "s", ms=2.9, mfc="none", mec="black", mew=0.7,
                    label="direct SK" if i == 0 else None, zorder=4)
    ap = Path(args.data) / "aoki_et_al_double_series.csv"
    ext = read_csv(ap) if ap.is_file() else None
    if ext:
        for i, n in enumerate(("eq", "neq", "neq2")):
            pts = sorted((r["kappa"], r["S"]) for r in ext if (r["mu1"], r["mu2"]) == PAIRS[n])
            if pts:
                k, s = zip(*pts)
                ax.plot(k, s, "D", ms=3.6, mfc="none", mec=GRAY, mew=0.7, zorder=5,
                        label=r"Aoki $et\ al.$" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 1.0)
    ax.set_ylim(-0.018084, 0.0049108)
    ax.set_yticks([-0.015, -0.010, -0.005, 0.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("$%.3f$"))
    ax.set_xlabel("$k_3/k_1$", labelpad=1.5)
    ax.set_ylabel("$S$, double")
    h, lab = ax.get_legend_handles_labels()
    # legend: three lines in the first column, markers in the second
    nl = len(handles[0])
    if len(h) > nl:
        h = h[:nl] + h[nl:] + [plt.Line2D([], [], alpha=0)] * (2 * nl - len(h))
        lab = lab[:nl] + lab[nl:] + [""] * (2 * nl - len(lab))
    legend_at(ax, box, (48.56, 150.68, 54.25, 85.63), handles=h, labels=lab,
              ncol=2 if len(h) > nl else 1)
    return fig


def fig3(plt, args, cache):
    from matplotlib.lines import Line2D
    W, H = 244.8, 122.0
    fig = figure(plt, W, H)
    box = (49.16, 240.1, 12.88, 94.8)
    ax = axes_pt(fig, W, H, box)
    tri = read_csv(Path(args.data) / "triangle_conformal_33config.csv")
    pid = read_csv(Path(args.data) / "triangle_pidot_14config.csv")
    for mt, colr in (("M111", BLUE), ("Mmix", YELLOW)):
        r = sorted([x for x in tri if x["family"] == "momentum" and x["tag"].endswith(mt)],
                   key=lambda x: x["shape_x"])
        x = [k["k1"] / (k["k2"] + k["k3"]) for k in r]
        ax.plot(x, [abs(complex(k["MB_re"], k["MB_im"])) for k in r], color=colr, lw=1.1)
        ax.plot(x, [abs(complex(k["Breg_re"], k["Breg_im"])) for k in r], "o", ms=3.4, mfc="none",
                mec=colr, mew=0.9)
        r = sorted([x for x in pid if x["family"] == "momentum" and x["tag"].endswith(mt)],
                   key=lambda x: x["shape_x"])
        x = [k["k1"] / (k["k2"] + k["k3"]) for k in r]
        ax.plot(x, [abs(complex(k["MB_re"], k["MB_im"])) for k in r], color=colr, lw=1.1, ls="--")
        ax.plot(x, [abs(complex(k["B_re"], k["B_im"])) for k in r], "s", ms=3.2, mfc="none",
                mec=colr, mew=0.9)
    ax.set_yscale("log")
    ax.set_xlim(0.027386, 0.99641)
    ax.set_ylim(0.025, 4.4991)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("$%.1f$"))
    ax.set_xlabel("$k_1/(k_2+k_3)$", labelpad=1.5)
    ax.set_ylabel(r"$|\psi_3|$, triangle")
    g = "#595959"
    h = [Line2D([], [], color=BLUE, lw=1.1), Line2D([], [], color=YELLOW, lw=1.1),
         Line2D([], [], color=g, lw=1.1, marker="o", ms=3.4, mfc="none", mew=0.9),
         Line2D([], [], color=g, lw=1.1, ls="--", marker="s", ms=3.2, mfc="none", mew=0.9)]
    lab = [r"$\mu=(1,\,1,\,1)$", r"$\mu=(1,\,1.3,\,0.7)$",
           r"$r_v=1,\ \psi_3^{\rm loop,\,reg}$", r"$r_v=2,\ \psi_3^{\rm loop}$"]
    legend_at(ax, box, (86.97, 221.39, 32.0, 53.4), handles=h, labels=lab, ncol=2)
    return fig


def figS2(plt, args, cache):
    """Multi-exchange sums (a), (b) and the triangle correlator (c) (printed as Fig. S2)."""
    W, H = 244.8, 352.8
    fig = figure(plt, W, H)
    cf = external(args, "cosmoflow.csv")
    panels = [
        ("double", [("D11", (1.0, 1.0), BLUE), ("D115", (1.0, 1.5), YELLOW)],
         (48.96, 239.9, 3.53, 96.02), (-0.10635, 0.014029), [-0.1, -0.075, -0.05, -0.025, 0.0],
         "$%.3f$", (51.17, 105.77, 63.71, 93.81), (223.0, 234.0, 10.0, 19.0), "(a)"),
        ("triple", [("T111", (1.0, 1.0, 1.0), BLUE), ("T1157", (1.0, 1.5, 0.7), YELLOW)],
         (48.96, 239.9, 96.02, 188.52), (-0.163216, 0.0077801), [-0.15, -0.10, -0.05, 0.0],
         "$%.2f$", (51.17, 106.15, 164.4, 186.31), (222.0, 234.0, 102.0, 111.0), "(b)")]
    axes = []
    for diagram, sets, box, ylim, yt, fmt, frame, lbox, plab in panels:
        ax = axes_pt(fig, W, H, box)
        axes.append(ax)
        zero_line(ax)
        for t, mus, colr in sets:
            S = cache.get(f"sum6_{t}", lambda d=diagram, m=mus: sum6_lines(d, m))
            ax.plot(KAPPA40, S, color=colr, lw=1.2,
                    label="$(" + ",\\,".join("%g" % m for m in mus) + ")$")
            if cf:
                pts = sorted((r["kappa"], r["S"]) for r in cf if r["diagram"] == diagram and
                             tuple(r[c] for c in ("mu1", "mu2", "mu3") if r[c] is not None) == mus)
                if pts:
                    k, s = zip(*pts)
                    ax.plot(k, s, "o", ms=3.2, mfc="none", mec=colr, mew=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.045, 1.05)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
        ax.set_ylabel("$S$, " + diagram)
        h, lab = ax.get_legend_handles_labels()
        if cf and diagram == "double":
            from matplotlib.lines import Line2D
            h.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=GRAY, mew=0.9))
            lab.append("CosmoFlow")
        legend_at(ax, box, frame, handles=h, labels=lab)
        label_at(ax, box, lbox, plab)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("$k_3/k_1$", labelpad=1.5)
    # (c) triangle correlator, equilateral-normalized
    box = (48.96, 239.9, 219.97, 326.34)
    ax = axes_pt(fig, W, H, box)
    rows = read_csv(Path(args.data) / "triangle_correlator_Q1.csv")
    hl = external(args, "herderschee_lu.csv")
    from matplotlib.lines import Line2D
    h, lab = [], []
    for mu, colr in ((0.5, BLUE), (0.7, YELLOW)):
        r = sorted((x["k3"] / x["k1"], x["S_over_S_equilateral"]) for x in rows if x["mu"] == mu)
        k, s = zip(*r)
        h.append(ax.plot(k, s, color=colr, lw=1.2)[0])
        lab.append(r"$\mu=%g$" % mu)
        if hl:
            pts = sorted((x["k3_over_k1"], x["S_over_S_equilateral"]) for x in hl if x["mu"] == mu)
            if pts:
                k, s = zip(*pts)
                ax.plot(k, s, "o", ms=3.2, mfc="none", mec=colr, mew=0.9, zorder=4)
    if hl:
        h.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=GRAY, mew=0.9))
        lab.append("Herderschee, Lu")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7e-4, 1.4)
    ax.set_ylim(7e-4, 2.0)
    ax.set_xlabel("$k_3/k_1$", labelpad=1.5)
    ax.set_ylabel(r"$S/S_{\rm eq}$, triangle")
    leg = ax.legend(handles=h, labels=lab, loc="lower right", borderaxespad=0.35)
    leg.get_frame().set_linewidth(0.8)
    label_at(ax, box, (57.0, 68.0, 227.0, 236.0), "(c)")
    return fig


def figS3(plt, args, cache):
    """Renormalized bubble (printed as Fig. S3)."""
    W, H = 244.8, 123.0
    fig = figure(plt, W, H)
    box = (49.16, 240.1, 3.85, 96.83)
    ax = axes_pt(fig, W, H, box)
    res = read_csv(Path(args.data) / "bubble_S_of_r.csv")
    ax.plot([x["r"] for x in res], [x["S_residue_families"] for x in res], color=BLUE, lw=1.2,
            label="this work")
    sp = Path(args.data) / "bubble_spectral_xianyu_zhang.csv"
    if sp.is_file():
        c = [x for x in read_csv(sp) if x["plotted_circle"] == 1]
        ax.plot([x["r"] for x in c], [x["S_spectral"] for x in c], "o", ms=2.6, mfc="none",
                mec=RED, mew=1.0, label="spectral decomposition")
    ax.set_xlim(0.077503, 0.57249)
    ax.set_ylim(-1.0009e-05, 1.9e-04)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0.0, 5e-5, 1e-4, 1.5e-4])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("$%.1f$"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("$%.5f$"))
    ax.set_xlabel("$r=k_3/2k_1$", labelpad=1.5)
    ax.set_ylabel("$S$, bubble")
    legend_at(ax, box, (51.0, 141.32, 5.69, 25.05))
    return fig


TARGETS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "figS1": figS1, "figS2": figS2,
           "figS3": figS3}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("targets", nargs="*", default=["all"], help="all | " + " ".join(TARGETS))
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--external", default=str(HERE / "data" / "external"))
    ap.add_argument("--quick", action="store_true", help="coarse grids and no direct points")
    ap.add_argument("--no-direct", dest="direct", action="store_false",
                    help="skip the direct Schwinger-Keldysh squares of Figs. 2(a) and S1")
    ap.add_argument("--png", action="store_true", help="also write PNG files")
    args = ap.parse_args()
    if args.quick:
        args.direct = False
    names = list(TARGETS) if "all" in args.targets else args.targets
    for n in names:
        if n not in TARGETS:
            ap.error(f"unknown target {n}")
    plt = style()
    cache = Cache(args.cache)
    os.makedirs(args.outdir, exist_ok=True)
    for n in names:
        print(n, flush=True)
        fig = TARGETS[n](plt, args, cache)
        out = Path(args.outdir) / f"{n}.pdf"
        fig.savefig(out)
        if args.png:
            fig.savefig(out.with_suffix(".png"), dpi=200)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
