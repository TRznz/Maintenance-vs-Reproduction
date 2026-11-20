"""Fig 3 (poster): a = one field fluctuates, b = both fluctuate; x = patch size, period 100 (Round C = D2)."""
import textwrap, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": ["Montserrat", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 12,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c9c7c1"})
D2 = pd.read_csv("results/round_D2/summary.csv")
st = D2[D2._cond == "static"]; p0 = (st.first_extinct == "Reproducer").mean() * 100
Ls = [1, 2, 5, 10, 25, 50]; xs = np.arange(len(Ls))
def pe(x):
    p = (x.first_extinct == "Reproducer").mean(); return p * 100, np.sqrt(p * (1 - p) / len(x)) * 100
def curve(cond, field):
    return np.array([pe(D2[(D2._cond == cond) & (D2[f"{field}_period"] == 100) & (D2[f"{field}_patch_size"] == L)]) for L in Ls])
panels = [("a · Resource or hazard alone", True, [("Hazard", curve("hazard", "hazard"), "#a3304f", "s"),
                                              ("Resource", curve("resource", "resource"), "#3c8f55", "o")]),
          ("b · Resource and hazard together", False, [("Rich = safe", curve("both-negative", "resource"), "#1f7a8c", "o"),
                                                  ("Independent", curve("both-independent", "resource"), "#8a8780", "D"),
                                                  ("Rich = risky", curve("both-positive", "resource"), "#9a5b2e", "s")])]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
for ax, (title, direct, lines) in zip(axs, panels):
    right = xs[-1] + (1.45 if direct else 0.35)
    ax.axhspan(p0, 103, color=ORANGE, alpha=0.07, lw=0); ax.axhspan(-3, p0, color=BLUE, alpha=0.07, lw=0)
    ax.axhline(p0, color=GRAY, ls=(0, (4, 3)), lw=1.3)
    for lab, c, col, mk in lines:
        ax.errorbar(xs, c[:, 0], yerr=c[:, 1], color=col, lw=2.2, marker=mk, ms=8, mec="white", mew=1.2, elinewidth=1.2, zorder=3, label=lab)
        if direct:
            ax.text(xs[-1] + 0.18, c[-1, 0], lab, color=col, fontsize=11.5, va="center", ha="left", fontweight="bold")
    if not direct:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, fontsize=11, handlelength=2.2, labelcolor=INK2)
    ax.set_xticks(xs, [str(L) for L in Ls[:-1]] + ["50\n(whole map)"])
    ax.set_xlim(-0.3, right); ax.set_ylim(-3, 103)
    ax.set_xlabel("Patch size of fluctuation (cells)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.text(-0.2, 99, "Maintainers favored", color=ORANGE, fontsize=10.5, va="top")
    ax.text(right - 0.08, 1, "Reproducers favored", color=BLUE, fontsize=10.5, va="bottom", ha="right")
    ax.text(right - 0.08, p0 + 1.5, f"static {p0:.0f}%", color=GRAY, fontsize=10, ha="right", va="bottom")
    ax.text(xs[-1] / 2, -0.27, "more varied  →  more uniform", transform=ax.get_xaxis_transform(), fontsize=10.5, color=INK2, ha="center", va="top")
axs[0].set_ylabel("% runs where Reproducers die out first")
fig.tight_layout(w_pad=2.5)
fig.canvas.draw()
_tb = axs[0]._left_title.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _tb.y1 + 0.015, "Fig 3 · Fluctuating environments", ha="left", va="bottom", fontsize=15, color=INK)
note = ("Round C · Fluctuation period 100 steps; amplitude 0.9: values swing between 10% and 190% of their mean. "
        "a: only the named one fluctuates; the other stays constant. b: both fluctuate over the same patches; rich = safe: resource peaks when hazard is lowest. "
        "All other settings as the Fig 2 default (hazard 4×10⁻⁴, aging slope 1×10⁻⁴, resource 2, exclusivity 0.5). 60 runs per point; bars ±1 SE.")
_xb = axs[0].texts[-1].get_window_extent().transformed(fig.transFigure.inverted())
fig.text(0.01, _xb.y0 - 0.03, textwrap.fill(note.replace(" = ", " = "), 150), ha="left", va="top", fontsize=9, color=GRAY, linespacing=1.4)
import os; os.makedirs("figs", exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"figs/fig3_final.{ext}", dpi=200, bbox_inches="tight")
