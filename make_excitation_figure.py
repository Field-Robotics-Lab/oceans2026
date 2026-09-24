#!/usr/bin/env python3
"""Plot Diagnostic 1's improvement against how separable the two regressors are.

Reads data/processed/diagnostic1_manual.csv, written by the manual-mode sweep.
Used only by the extended paper.

  PYTHONNOUSERSITE=1 /usr/bin/python3 make_excitation_figure.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
# IEEE submission systems reject Type 3 fonts; 42 emits TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

VERM, BLACK, MUTED, GRID = "#D55E00", "#1A1A1A", "#5A5A5A", "#DDDDDD"

rows = [r for r in csv.DictReader(open("data/processed/diagnostic1_manual.csv"))
        if float(r["v_hi"]) > 0.5]
cc = np.array([float(r["corr"]) for r in rows])
dR = np.array([float(r["dR2"]) for r in rows])
n = np.array([float(r["n"]) for r in rows])
vh = np.array([float(r["v_hi"]) for r in rows])

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": BLACK, "text.color": BLACK,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
sc = ax.scatter(cc, dR, s=np.clip(n / 40, 16, 80), c=vh, cmap="viridis",
                edgecolor="white", linewidth=0.8, zorder=3)
ax.scatter([0.992], [-0.032], marker="X", s=65, color=VERM,
           edgecolor="white", lw=0.9, zorder=4)
ax.set_xlim(0.835, 1.012)
ax.set_ylim(-0.075, 0.185)
ax.annotate("trial used in the paper", xy=(0.992, -0.032), xytext=(0.968, -0.062),
            fontsize=6.5, ha="right", color=BLACK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("0.5--4.5 m/s", xy=(0.928, 0.130), xytext=(0.955, 0.155),
            fontsize=6.5, ha="left", color=BLACK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("max speed under rudder (m/s)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_xlabel(r"collinearity of $\delta$ and $\delta v$")
ax.set_ylabel(r"$\Delta R^2$ for $k_\delta v$")
fig.tight_layout(pad=0.3)
fig.savefig("paper/figures/diagnostic1_excitation.pdf", bbox_inches="tight")
fig.savefig("paper/figures/diagnostic1_excitation.png", dpi=200, bbox_inches="tight")
print("wrote paper/figures/diagnostic1_excitation.{pdf,png}")
