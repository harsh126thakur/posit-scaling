"""Generate all figures into paper/figures/ from the CSVs in results/."""
import _common
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "figure.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
ORDER = ["none", "ruiz", "logmean", "lp_inf"]
R, F = _common.RESULTS, _common.FIGURES


def need(name):
    p = R / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run the experiments first (make all)")
    return pd.read_csv(p)


# ---- fig 1: LP optimum vs Karp max mean cycle -------------------------
th = need("theory.csv")
fig, ax = plt.subplots(figsize=(4.0, 3.4))
for fam, c in zip(["centered", "tree", "mixed", "cyclic"],
                  [GRAY, AQUA, ORANGE, BLUE]):
    s = th[th.family == fam]
    ax.scatter(s.mmc, s.lp, s=14, color=c, label=fam, alpha=.8)
lim = [0, th.mmc.max() * 1.05]
ax.plot(lim, lim, "--", color=GRAY, lw=1, zorder=0)
dev = np.max(np.abs(th.lp - th.mmc) / np.maximum(np.abs(th.mmc), 1e-12))
ax.text(.05, .92, f"max rel. deviation {dev:.1e}", transform=ax.transAxes, fontsize=8)
ax.set_xlabel("max mean cycle weight (Karp)")
ax.set_ylabel(r"LP optimum: min max $|\log_2$ scaled entry$|$")
ax.set_title("Graph theory predicts the LP optimum exactly", fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig1_theory.png"); plt.close(fig)

# ---- fig 2: representation fidelity -----------------------------------
rep = need("representation.csv")
g = rep.groupby("scaling")[["posit32", "float32", "bits"]].median().reindex(ORDER)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
x, w = np.arange(4), .36
axes[0].bar(x - w/2, g.posit32, w, color=BLUE, label="posit32")
axes[0].bar(x + w/2, g.float32, w, color=ORANGE, label="float32")
axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(ORDER)
axes[0].set_ylabel("worst-entry representation error")
axes[0].set_title("Scaling improves posits, does nothing for floats", fontsize=9)
axes[0].legend(frameon=False, fontsize=8)
axes[1].bar(x, g.bits, .55, color=AQUA)
axes[1].axhline(23, ls="--", color=ORANGE, lw=1)
axes[1].text(3.4, 23.15, "float32", color=ORANGE, fontsize=8, ha="right")
axes[1].set_ylim(22, 28); axes[1].set_xticks(x); axes[1].set_xticklabels(ORDER)
axes[1].set_ylabel("mean fraction bits held (posit32)")
axes[1].set_title("Bits recovered by scaling", fontsize=9)
fig.tight_layout(); fig.savefig(F / "fig2_representation.png"); plt.close(fig)

# ---- fig 3: solve error vs spread -------------------------------------
sw = need("sweep.csv")
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
for ax, fam in zip(axes, ["tree", "mixed", "cyclic"]):
    s = sw[(sw.family == fam) & (sw.fmt == "posit32")]
    for sc, c in zip(ORDER, [GRAY, ORANGE, AQUA, BLUE]):
        t = s[s.scaling == sc].groupby("spread").bwd.median()
        ax.plot(t.index, t.values, "o-", color=c, ms=3.5, lw=1.6, label=sc)
    ax.set_yscale("log"); ax.set_xlabel("magnitude spread (bits)")
    ax.set_title(fam, fontsize=9)
axes[0].set_ylabel("backward error, posit32 LU solve")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Which scaling wins depends on whether the spread is removable",
             fontsize=9.5, y=1.02)
fig.tight_layout(); fig.savefig(F / "fig3_solve.png", bbox_inches="tight")
plt.close(fig)

# ---- fig 4: decision rule ---------------------------------------------
dec = need("decision.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.4))
for fam, c in zip(["tree", "mixed", "cyclic"], [AQUA, ORANGE, BLUE]):
    s = dec[dec.family == fam]
    ax.scatter(s.removable, s.ratio, s=16, color=c, alpha=.75, label=fam)
ax.axhline(1, ls="--", color=GRAY, lw=1)
ax.set_yscale("log")
ax.set_xlabel("removable fraction of spread")
ax.set_ylabel("error(Ruiz) / error(LP)")
ax.set_title("Above 1: the posit-optimal LP wins", fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig4_decision.png"); plt.close(fig)

# ---- fig 5: the precision profile itself -------------------------------
from positscale import Posit, Float
s = np.arange(-120, 121)
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(s, Posit(32, 2).frac_bits(s), color=BLUE, lw=2, label="posit32")
ax.axhline(Float(32).fb, color=ORANGE, lw=2, ls="--", label="float32")
ax.set_xlabel(r"$\log_2 |x|$"); ax.set_ylabel("fraction bits")
ax.set_title("Tapered vs flat precision", fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig5_profile.png"); plt.close(fig)

print(f"figures written to {F}")
for p in sorted(F.glob("*.png")):
    print("  ", p.name)
