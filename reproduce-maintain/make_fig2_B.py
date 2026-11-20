"""Figure 2 from production round B: resource flux × exclusivity."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": ["Montserrat", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 12, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
DIV = LinearSegmentedColormap.from_list("rm", [BLUE, "#f0efec", ORANGE])
s = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/round_B/summary.csv")
s["flux"] = (s.resource_capacity * 0.2).round(2)
fl = sorted(s.flux.unique()); ex = sorted(s.exclusivity.unique())
n = s.groupby(["flux", "exclusivity"]).size().min()
cell = lambda f, e: s[(s.flux == f) & (s.exclusivity == e)]
P = np.array([[(cell(f, e).first_extinct == "Reproducer").mean() * 100 for f in fl] for e in ex])
A = np.array([[cell(f, e).first_extinct.isna().mean() for f in fl] for e in ex])   # share of runs undecided (cap)
S = np.array([[cell(f, e).share_31_first.mean() for f in fl] for e in ex])
fig, axs = plt.subplots(1, 1, figsize=(6.6, 5.0)); axs = [axs]
for ax, M, norm, fmt, t, cl in (
        (axs[0], P, TwoSlopeNorm(50, 0, 100), lambda v: f"{v:.0f}", "% runs where Reproducers die out first", "% of runs"),):
    im = ax.imshow(M, cmap=DIV, norm=norm, aspect="auto", origin="lower")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if ax is axs[0] and A[i, j] > 0.5:   # runs stopped at the population cap
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#d3d1c7", edgecolor="white", lw=1.5))
                ax.text(j, i, "cap", ha="center", va="center", fontsize=11, color=INK2); continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=12.5,
                    color="white" if abs(norm(v) - 0.5) > 0.32 else INK)
    ax.set_xticks(range(len(fl)), [f"{f:g}" for f in fl]); ax.set_yticks(range(len(ex)), [f"{e:g}" for e in ex])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("Resource per cell per step  (abundance →)"); ax.set_ylabel("Exclusivity  (size advantage in sharing →)")
    ax.set_title(t, loc="left", fontsize=13, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(cl, color=INK2); cb.outline.set_visible(False)
fig.tight_layout()
fig.canvas.draw()
_tb = axs[0]._left_title.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _tb.y1 + 0.012, "Fig 2 · Resource: abundance × size advantage", ha="left", va="bottom", fontsize=15, color=INK)
import textwrap
_note = ("Round B · Hazard 4×10⁻⁴, aging slope 1×10⁻⁴ (on the Fig 1 boundary); environment unchanging. Exclusivity: how much bigger agents win when competing for the same resource (0 = equal shares; higher = bigger agents take more). 20 runs per cell. cap: Reproducers reached 30,000 agents, run stopped.").replace("\n", " ")
_xb = axs[0].xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _xb.y0 - 0.03, textwrap.fill(_note.replace(" = ", "\u00a0=\u00a0"), 92), ha="left", va="top", fontsize=9, color=GRAY, linespacing=1.4)
import os; os.makedirs("figs", exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"figs/fig2_resource_B.{ext}", dpi=200, bbox_inches="tight")
