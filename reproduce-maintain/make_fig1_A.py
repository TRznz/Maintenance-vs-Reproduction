"""Figure 1 from production round A (new rules)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": ["Montserrat", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 12, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
DIV = LinearSegmentedColormap.from_list("rm", [BLUE, "#f0efec", ORANGE])
s = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/round_A/summary.csv")
hz = sorted(s.hazard_level.unique()); bs = sorted(s.age_damage_slope.unique())
n = s.groupby(["hazard_level", "age_damage_slope"]).size().min()
P = np.array([[(s[(s.hazard_level == h) & (s.age_damage_slope == b)].first_extinct == "Reproducer").mean() * 100 for b in bs] for h in hz])
S = np.array([[s[(s.hazard_level == h) & (s.age_damage_slope == b)].share_51_400.mean() for b in bs] for h in hz])
fig, axs = plt.subplots(1, 1, figsize=(6.6, 5.2)); axs = [axs]
xl = [f"{b*1e4:g}" for b in bs]
yl = ["0" if h == 0 else f"{h*1e4:g}" for h in hz]
for ax, M, norm, fmt, t, cl in (
        (axs[0], P, TwoSlopeNorm(50, 0, 100), lambda v: f"{v:.0f}", "% runs where Reproducers die out first", "% of runs"),):
    im = ax.imshow(M, cmap=DIV, norm=norm, aspect="auto", origin="lower")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ax.text(j, i, fmt(v), ha="center", va="center", fontsize=12.5,
                                 color="white" if abs(norm(v) - 0.5) > 0.32 else INK)
    ax.set_xticks(range(len(bs)), xl); ax.set_yticks(range(len(hz)), yl)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("Age-damage slope b  (×10⁻⁴)  (aging →)"); ax.set_ylabel("Hazard per cell  (×10⁻⁴)  (size-scaled damage →)")
    ax.set_title(t, loc="left", fontsize=13, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(cl, color=INK2); cb.outline.set_visible(False)
fig.tight_layout()
fig.canvas.draw()
_tb = axs[0]._left_title.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _tb.y1 + 0.012, "Fig 1 · Damage: aging × environmental hazard", ha="left", va="bottom", fontsize=15, color=INK)
import textwrap
_note = ("Round A · Hazard is the same in every cell and the environment does not change; resource 2, exclusivity 0.5. Offspring start at age 0. 20 runs per cell; each run stops when the first type dies out. Orange: Maintainers usually outlast Reproducers; blue: Reproducers usually outlast Maintainers.").replace("\n", " ")
_xb = axs[0].xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _xb.y0 - 0.03, textwrap.fill(_note.replace(" = ", "\u00a0=\u00a0"), 92), ha="left", va="top", fontsize=9, color=GRAY, linespacing=1.4)
import os; os.makedirs("figs", exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"figs/fig1_damage_A.{ext}", dpi=200, bbox_inches="tight")
