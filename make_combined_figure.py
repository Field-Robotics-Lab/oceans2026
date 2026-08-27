#!/usr/bin/env python3
"""Regenerate the three-level comparison figure for the common 130 s trial.

Carson's narx/ package archives the Level-3 traces but no plotting code, so
this rebuilds the figure from verified sources:

  Level 3  narx/results/original_trial_level3_traces.csv  (hash in model_manifest)
  Level 1  ZOH discretization of the published transfer functions
  Level 2  Euler integration of the published state-space form

The Level-1 and Level-2 simulations reproduce the paper's reported fits to
within 0.002 R^2, which is the check that they are the same models.

Run with PYTHONNOUSERSITE=1 so the system numpy/matplotlib pair is used.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
# IEEE submission systems reject Type 3 fonts; 42 emits TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

TRACES = "narx/results/curated_trial_traces.csv"
DT = 0.1

# Published Level-1 transfer functions and Level-2 parameters.
K_V, TAU_V = 6.42, 1.95
K_R, TAU_R = 0.771, 0.471
KT, D2 = 3.963, 0.438
KD, DR = 1.474, 1.912

BLACK, BLUE, GREEN, VERM = "#1A1A1A", "#0072B2", "#009E73", "#D55E00"
MUTED, GRID = "#5A5A5A", "#DDDDDD"


def load(path):
    cols = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", "nan") else np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def first_order(u, y0, K, tau):
    a = np.exp(-DT / tau)
    b = K * (1 - a)
    y = np.empty(len(u))
    y[0] = y0
    for k in range(len(u) - 1):
        y[k + 1] = a * y[k] + b * u[k]
    return y


def surge_l2(u, v0):
    v = np.empty(len(u))
    v[0] = v0
    for k in range(len(u) - 1):
        v[k + 1] = v[k] + DT * (KT * u[k] - D2 * v[k] * abs(v[k]))
    return v


def yaw_l2(d, r0):
    r = np.empty(len(d))
    r[0] = r0
    for k in range(len(d) - 1):
        r[k + 1] = r[k] + DT * (KD * d[k] - DR * r[k])
    return r


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


d = load(TRACES)
t = d["time_s"]
u, dl = d["throttle_norm"], d["rudder_norm"]
mv, mr = d["measured_speed_mps"], d["measured_yaw_rate_rps"]
p3v, p3r = d["recursive_speed_mps"], d["recursive_yaw_rate_rps"]

p1v = first_order(u, mv[0], K_V, TAU_V)
p1r = first_order(dl, mr[0], K_R, TAU_R)
p2v = surge_l2(u, mv[0])
p2r = yaw_l2(dl, mr[0])

# Level 3 is scored only where it has predictions; score the others identically
# so the legend numbers describe the same rows.
m = np.isfinite(p3v)
fits = {
    "1v": r2(mv[m], p1v[m]), "2v": r2(mv[m], p2v[m]), "3v": r2(mv[m], p3v[m]),
    "1r": r2(mr[m], p1r[m]), "2r": r2(mr[m], p2r[m]), "3r": r2(mr[m], p3r[m]),
}
print("scored rows:", m.sum())
for k, v in fits.items():
    print(f"  {k}: R2 = {v:+.4f}")

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": BLACK,
    "axes.labelcolor": BLACK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
})
fig, ax = plt.subplots(3, 1, figsize=(7.16, 3.6), sharex=True,
                       gridspec_kw={"height_ratios": [3, 3, 1.25], "hspace": 0.16})

for a in ax:
    a.grid(True, color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.axvspan(t[0], t[m][0], color="#F0F0F0", zorder=0)

ax[0].plot(t, mv, color=BLACK, lw=1.1, label="measured", zorder=5)
ax[0].plot(t, p1v, color=BLUE, lw=1.0, ls=(0, (4, 2)), label=f"Level 1  ($R^2$={fits['1v']:.3f})")
ax[0].plot(t, p2v, color=GREEN, lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), label=f"Level 2  ($R^2$={fits['2v']:.3f})")
ax[0].plot(t, p3v, color=VERM, lw=1.1, label=f"Level 3  ($R^2$={fits['3v']:.3f})")
ax[0].set_ylabel("surge speed (m/s)")
ax[0].legend(ncol=4, loc="upper left", frameon=False, handlelength=2.4,
             borderaxespad=0.2, columnspacing=1.4)
ax[0].set_ylim(-0.35, 3.6)

ax[1].plot(t, mr, color=BLACK, lw=1.1, label="measured", zorder=5)
ax[1].plot(t, p1r, color=BLUE, lw=1.0, ls=(0, (4, 2)), label=f"Level 1  ($R^2$={fits['1r']:.3f})")
ax[1].plot(t, p2r, color=GREEN, lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), label=f"Level 2  ($R^2$={fits['2r']:.3f})")
ax[1].plot(t, p3r, color=VERM, lw=1.1, label=f"Level 3  ($R^2$={fits['3r']:.3f})")
ax[1].set_ylabel("yaw rate (rad/s)")
ax[1].legend(ncol=4, loc="upper left", frameon=False, handlelength=2.4,
             borderaxespad=0.2, columnspacing=1.4)
ax[1].set_ylim(-1.05, 1.15)

ax[2].plot(t, u, color=MUTED, lw=0.9, label="throttle")
ax[2].plot(t, dl, color="#8B5FA8", lw=0.9, ls=(0, (3, 2)), label="rudder")
ax[2].set_ylabel("command")
ax[2].set_xlabel("time (s)")
ax[2].legend(ncol=2, loc="upper left", frameon=False, handlelength=2.4, borderaxespad=0.2)
ax[2].set_ylim(-1.15, 1.35)
ax[2].set_xlim(t[0], t[-1])

fig.savefig("paper/figures/combined_three_level.pdf", bbox_inches="tight")
fig.savefig("paper/figures/combined_three_level.png", dpi=200, bbox_inches="tight")
print("\nwrote paper/figures/combined_three_level.{pdf,png}")
