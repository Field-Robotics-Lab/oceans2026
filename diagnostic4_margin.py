"""Diagnostic 4: does the Level-1 to Level-2 discrepancy change the control design?

Diagnostics 1-3 ask whether a term is PRESENT in the data. This one asks whether it
CHANGES THE DESIGN, by treating the robustness margin of a Level-1-designed
compensator as a budget for model error.

Identified parameters (see first_order_models.ipynb, nonlinear_ss_model.ipynb):
  Level 1  surge  G1(s)  = 6.42/(1.95 s + 1)          R2 = 0.930
           yaw    G1r(s) = 0.771/(0.471 s + 1)        R2 = 0.826
  Level 2  surge  dv/dt = kT u - d2 v|v|,  kT = 3.963, d2 = 0.438
           yaw    dr/dt = kd delta - dr r, kd = 1.474, dr = 1.912

Linearizing Level-2 surge about forward speed v0 > 0 (d/dv of v|v| is 2 v0):
  G2(s,v0) = kT/(s + 2 d2 v0) = K2(v0)/(tau2(v0) s + 1)
  tau2(v0) = 1/(2 d2 v0),   K2(v0) = kT/(2 d2 v0)
Both the gain and the time constant fall as 1/v0.

Outputs the two forms of the test:
  Step 3  robust stability   ||l T||_inf < 1
  Step 4  performance        achieved closed-loop bandwidth vs design intent
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Identified parameters ────────────────────────────────────────────────────
K1, TAU1 = 6.42, 1.95          # Level 1, surge
KT, D2 = 3.963, 0.438          # Level 2, surge
K1R, TAU1R = 0.771, 0.471      # Level 1, yaw
KD, DR = 1.474, 1.912          # Level 2, yaw

LAM = 1.0                      # IMC closed-loop time constant, surge speed loop, s
VMIN, VMAX = 0.2, 2.9          # tested speed envelope

w = np.logspace(-3, 3, 8000)
jw = 1j * w


def lvl2_surge(v0):
    """Gain and time constant of the Level-2 surge model linearized at speed v0."""
    a = 2.0 * D2 * v0
    return KT / a, 1.0 / a


def surge_l(v0):
    """Multiplicative model error l(w, v0) = |G2/G1 - 1| for the surge channel."""
    K2, t2 = lvl2_surge(v0)
    return np.abs((K2 / (t2 * jw + 1.0)) / (K1 / (TAU1 * jw + 1.0)) - 1.0)


# ── Step 1: nominal compensator from LEVEL 1 ALONE ───────────────────────────
# IMC PI, C(s) = (tau1 s + 1)/(K1 lam s): cancels the Level-1 pole so the nominal
# loop is L = 1/(lam s) and T = 1/(lam s + 1). PM = 90 deg, GM infinite.
def C(s):
    return (TAU1 * s + 1.0) / (K1 * LAM * s)


kp, ki = TAU1 / (K1 * LAM), 1.0 / (K1 * LAM)
T_nom = 1.0 / (LAM * jw + 1.0)

# ── Sweep ────────────────────────────────────────────────────────────────────
speeds = np.linspace(VMIN, VMAX, 600)
lT = np.array([np.max(surge_l(v) * np.abs(T_nom)) for v in speeds])

bw_ratio = []
for v0 in speeds:
    K2, t2 = lvl2_surge(v0)
    L = C(jw) * (K2 / (t2 * jw + 1.0))
    m = np.abs(L / (1.0 + L))
    b = np.where(m < m[0] / np.sqrt(2))[0]
    bw_ratio.append((w[b[0]] if len(b) else w[-1]) * LAM)
bw_ratio = np.array(bw_ratio)

# Lower bound: robust stability. l(0) = |K2/K1 - 1| = 1 gives a closed form
# independent of the design knob lambda.
V_LO = KT / (4.0 * D2 * K1)
# Upper bound: achieved bandwidth falls to half the design intent.
_hi = np.where(bw_ratio < 0.5)[0]
V_HI = speeds[_hi[0]] if len(_hi) else np.nan

# Yaw channel
K2R, TAU2R = KD / DR, 1.0 / DR
l_yaw = np.abs((K2R / (TAU2R * jw + 1.0)) / (K1R / (TAU1R * jw + 1.0)) - 1.0)
lT_yaw = np.max(l_yaw * np.abs(T_nom))

# ── Report ───────────────────────────────────────────────────────────────────
print(f"Nominal surge PI from Level 1 alone (IMC, lambda = {LAM} s):")
print(f"  kp = {kp:.4f}, ki = {ki:.4f};  nominal PM = 90 deg, GM = inf\n")
print("Level 1 coincides with the Level-2 linearization at:")
print(f"  K2(v0) = K1     at v0 = {KT/(2*D2*K1):.2f} m/s")
print(f"  tau2(v0) = tau1 at v0 = {1/(2*D2*TAU1):.2f} m/s")
print("  -> the Level-1 fit is anchored near 0.6-0.7 m/s\n")
print(f"HF limit |G2/G1| = kT tau1/K1 = {KT*TAU1/K1:.4f}, independent of v0\n")
print("SUFFICIENCY WINDOW for the Level-1 surge model:")
print(f"  lower bound  v_lo = kT/(4 d2 K1) = {V_LO:.2f} m/s  (robust stability)")
print(f"  upper bound  v_hi = {V_HI:.2f} m/s  (50% of design bandwidth)")
print(f"  -> Level 1 supports the design only over [{V_LO:.2f}, {V_HI:.2f}] m/s,")
print(f"     under half of the {VMAX} m/s envelope tested.\n")
print(f"  below v_lo: quadratic-drag linearization is singular, K2,tau2 ~ 1/v0")
print(f"  above v_hi: loop goes sluggish, not unstable; drag is stabilizing")
print(f"  at {VMAX} m/s the achieved bandwidth is {bw_ratio[-1]*100:.0f}% of intent,")
print(f"     while ||l T||_inf = {lT[-1]:.2f} still satisfies robust stability\n")
print(f"Yaw channel: DC gain differs {abs(K2R/K1R-1)*100:.1f}%, "
      f"tau differs {abs(TAU2R/TAU1R-1)*100:.1f}%")
print(f"  max |l_yaw| = {l_yaw.max():.3f}, ||l T||_inf = {lT_yaw:.3f}"
      f"  -> Level 2 changes nothing for yaw control")

# ── Figure ───────────────────────────────────────────────────────────────────
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
GRID, INK, MUTED, BAND = "#D9D9D9", "#1A1A1A", "#5A5A5A", "#EFEFEF"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True)

for ax in (ax1, ax2):
    ax.axvspan(V_LO, V_HI, color=BAND, zorder=0)
    ax.grid(True, color=GRID, lw=0.5, zorder=1)
    ax.set_axisbelow(True)

# (a) robust stability -- binds only at the low-speed end
ax1.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax1.plot(speeds, lT, color=BLUE, lw=1.6, label="surge", zorder=4)
ax1.plot(speeds, np.full_like(speeds, lT_yaw), color=GREEN, lw=1.6,
         ls=(0, (1.5, 1.5)), label="yaw", zorder=4)
ax1.set_ylim(0, 1.6)
ax1.set_ylabel(r"$\|\ell\,T\|_\infty$")
ax1.legend(loc="upper right", frameon=False, handlelength=2.2)
ax1.plot([V_LO], [1.0], "o", ms=4.5, color=BLUE, mec="white", mew=1.1, zorder=5)
ax1.annotate(rf"$v_{{\mathrm{{lo}}}} = {V_LO:.2f}$ m/s", xy=(V_LO, 1.0),
             xytext=(V_LO + 0.35, 1.30), fontsize=7, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax1.set_title("(a)  robust stability binds at low speed",
              fontsize=7.5, loc="left", color=INK, pad=4)

# (b) performance -- binds at the high-speed end
ax2.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax2.plot(speeds, bw_ratio, color=VERM, lw=1.6, zorder=4)
ax2.set_ylim(0, 1.45)
ax2.set_xlim(VMIN, VMAX)
ax2.set_xlabel("forward speed $v_0$ (m/s)")
ax2.set_ylabel("achieved BW / design BW")
ax2.plot([V_HI], [0.5], "o", ms=4.5, color=VERM, mec="white", mew=1.1, zorder=5)
ax2.annotate(rf"$v_{{\mathrm{{hi}}}} = {V_HI:.2f}$ m/s", xy=(V_HI, 0.5),
             xytext=(V_HI + 0.28, 0.82), fontsize=7, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax2.set_title("(b)  performance binds at high speed",
              fontsize=7.5, loc="left", color=INK, pad=4)

ax2.annotate("", xy=(V_LO, 0.13), xytext=(V_HI, 0.13),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
ax2.text((V_LO + V_HI) / 2, 0.20, "Level 1\nsufficient",
         ha="center", va="bottom", fontsize=6.5, color=INK, linespacing=1.15)

fig.tight_layout(pad=0.4)
fig.savefig("paper/figures/diagnostic4_margin.pdf", bbox_inches="tight")
fig.savefig("paper/figures/diagnostic4_margin.png", dpi=200, bbox_inches="tight")
print("\nwrote paper/figures/diagnostic4_margin.{pdf,png}")
