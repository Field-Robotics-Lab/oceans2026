#!/usr/bin/env python3
"""Plot the steady-state response of each level against the measured speeds.

Reads the curves produced by steady_state_test.py. Run that first if
data/processed/steady_state_curves.json is missing or stale.

  PYTHONNOUSERSITE=1 python3 make_steady_figure.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
# IEEE submission systems reject Type 3 fonts; 42 emits TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

BLUE, GREEN, VERM = "#0072B2", "#009E73", "#D55E00"
BLACK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#DDDDDD"

d = json.load(open("data/processed/steady_state_curves.json"))
u = np.asarray(d["throttle"])
meas = np.asarray([np.nan if x is None else x for x in d["measured"]])

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": BLACK, "text.color": BLACK,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
})
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.plot(u, d["Level 1"], color=BLUE, lw=1.2, ls=(0, (4, 2)), label="Level 1", zorder=3)
ax.plot(u, d["Level 2"], color=GREEN, lw=1.2, ls=(0, (5, 1.5, 1, 1.5)), label="Level 2", zorder=3)
ax.plot(u, d["Level 3 (curated)"], color=VERM, lw=1.4, label="Level 3", zorder=4)
m = np.isfinite(meas)
ax.plot(u[m], meas[m], "o", ms=4.2, color=BLACK, mec="white", mew=0.9,
        label="measured", zorder=6)
ax.set_xlabel("throttle command (normalized)")
ax.set_ylabel("settled speed (m/s)")
ax.set_ylim(0, 2.75)
ax.set_xlim(u.min(), u.max())
ax.legend(loc="upper left", frameon=False, handlelength=2.2,
          borderaxespad=0.3, labelspacing=0.3)
fig.tight_layout(pad=0.3)
fig.savefig("paper/figures/steady_state_gain.pdf", bbox_inches="tight")
fig.savefig("paper/figures/steady_state_gain.png", dpi=200, bbox_inches="tight")
print("wrote paper/figures/steady_state_gain.{pdf,png}")
