#!/usr/bin/env python3
"""Can a neural model learn the equilibrium from this data, given a better-posed task?

The paper's NARX maps 19 lagged signals to a state increment. That lets it lean
on persistence: the recent past predicts the near future well enough to score
0.998 one-step without ever learning what steady speed belongs to a throttle.

This fits the same data with the same tool, changing only what the network is
asked to represent: the acceleration as a function of the current state and
command, dv/dt = f(v, u). The equilibrium is then wherever f crosses zero, so
transient data alone determines it.
"""
import numpy as np, pandas as pd, torch
from torch import nn

torch.manual_seed(0); np.random.seed(0)
tr = pd.read_csv('narx/data/partitions/training.csv')

rows = []
for _, g in tr.groupby('segment_id'):
    g = g.sort_values('time_s')
    v = g.speed_mps.to_numpy(); u = g.throttle_norm.to_numpy(); t = g.time_s.to_numpy()
    ok = np.isclose(np.diff(t), 0.1, atol=0.02)
    dv = (v[1:] - v[:-1]) / 0.1
    rows.append(np.column_stack([v[:-1][ok], u[:-1][ok], dv[ok]]))
D = np.concatenate(rows)
print(f"training pairs: {len(D)}   speed {D[:,0].min():.2f}-{D[:,0].max():.2f}   "
      f"throttle {D[:,1].min():.2f}-{D[:,1].max():.2f}")

X = torch.tensor(D[:, :2], dtype=torch.float32)
Y = torch.tensor(D[:, 2:3], dtype=torch.float32)
xm, xs = X.mean(0), X.std(0); ym, ys = Y.mean(0), Y.std(0)
Xn, Yn = (X - xm) / xs, (Y - ym) / ys

net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
for ep in range(400):
    p = torch.randperm(len(Xn))
    for i in range(0, len(p), 512):
        j = p[i:i+512]
        opt.zero_grad(); loss = ((net(Xn[j]) - Yn[j])**2).mean(); loss.backward(); opt.step()
print(f"final one-step accel MSE (normalized): {float(loss):.4f}")

def accel(v, u):
    x = (torch.tensor([[v, u]], dtype=torch.float32) - xm) / xs
    return float(net(x) * ys + ym)

# equilibrium = where predicted acceleration crosses zero
def equil(u, lo=0.0, hi=4.0):
    if accel(lo, u) <= 0: return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if accel(mid, u) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2

KT, D2 = 3.963, 0.438
tl = pd.read_csv('usv_step_response_150_280s.csv')
print(f"\n{'throttle':>9} {'ML f(v,u)':>10} {'Level 2':>9} {'measured':>9}")
for u in [0.10, 0.15, 0.20, 0.25, 0.30, 0.39]:
    m = tl[(tl.throttle_norm > u-0.02) & (tl.throttle_norm < u+0.02) & (tl.speed_mps > 0.2)]
    meas = f"{m.speed_mps.median():9.3f}" if len(m) > 20 else f"{'-':>9}"
    print(f"{u:9.2f} {equil(u):10.3f} {np.sqrt(KT*u/D2):9.3f} {meas}")

# recursive simulation of the trial using the learned acceleration
v = tl.speed_mps.to_numpy(); u = tl.throttle_norm.to_numpy()
s = np.empty_like(v); s[0] = v[0]
for k in range(len(v)-1):
    s[k+1] = s[k] + 0.1*accel(float(s[k]), float(u[k]))
r2 = 1 - np.sum((v-s)**2)/np.sum((v-v.mean())**2)
print(f"\nrecursive surge R2 on the common trial: {r2:+.4f}   (paper NARX: -1.988)")
