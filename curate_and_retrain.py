#!/usr/bin/env python3
"""Curate the training logs on a physical-response criterion, then refit.

A source log earns its place only if the vessel demonstrably responds to
throttle. Logs that are stationary throughout, or that show no speed response
under sustained throttle, teach the model that throttle does nothing, and one
such log supplies 27% of the frozen training partition.

Rule, applied per source log:
  drop if the 95th-percentile speed is below 0.5 m/s   (never got under way)
  drop if, given >20 samples above throttle 0.2, the mean speed there is
     below 0.6 m/s                                     (no response to thrust)
"""
import numpy as np, pandas as pd, torch
from torch import nn

torch.manual_seed(0); np.random.seed(0)
tr = pd.read_csv('narx/data/partitions/training.csv')

keep, drop = [], []
for lg, g in tr.groupby('source_log'):
    hi = g.throttle_norm > 0.2
    stationary = np.percentile(g.speed_mps, 95) < 0.5
    unresponsive = hi.sum() > 20 and g.speed_mps[hi].mean() < 0.6
    (drop if (stationary or unresponsive) else keep).append(lg)
cur = tr[tr.source_log.isin(keep)]
print(f"kept {len(keep)} logs / {len(cur)} rows;  dropped {len(drop)} logs / {len(tr)-len(cur)} rows "
      f"({1-len(cur)/len(tr):.1%})")
print("dropped:", ", ".join(drop))
print(f"active fraction: {(tr.throttle_norm.abs()>0.02).mean():.1%} -> {(cur.throttle_norm.abs()>0.02).mean():.1%}")


def fit_accel(df, epochs=400):
    rows = []
    for _, g in df.groupby('segment_id'):
        g = g.sort_values('time_s')
        v = g.speed_mps.to_numpy(); u = g.throttle_norm.to_numpy(); t = g.time_s.to_numpy()
        ok = np.isclose(np.diff(t), 0.1, atol=0.02)
        rows.append(np.column_stack([v[:-1][ok], u[:-1][ok], ((v[1:]-v[:-1])/0.1)[ok]]))
    D = np.concatenate(rows)
    X = torch.tensor(D[:, :2], dtype=torch.float32); Y = torch.tensor(D[:, 2:3], dtype=torch.float32)
    xm, xs, ym, ys = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    Xn, Yn = (X-xm)/xs, (Y-ym)/ys
    net = nn.Sequential(nn.Linear(2,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        p = torch.randperm(len(Xn))
        for i in range(0, len(p), 512):
            j = p[i:i+512]; opt.zero_grad()
            ((net(Xn[j])-Yn[j])**2).mean().backward(); opt.step()
    def accel(v, u):
        with torch.no_grad():
            return float(net((torch.tensor([[v,u]],dtype=torch.float32)-xm)/xs)*ys+ym)
    return accel, len(D)


def equil(accel, u, lo=0.0, hi=4.0):
    if accel(lo, u) <= 0: return 0.0
    for _ in range(60):
        mid = (lo+hi)/2
        if accel(mid, u) > 0: lo = mid
        else: hi = mid
    return (lo+hi)/2


tl = pd.read_csv('usv_step_response_150_280s.csv')
KT, D2 = 3.963, 0.438
results = {}
for tag, df in [("all data", tr), ("curated", cur)]:
    accel, n = fit_accel(df)
    v = tl.speed_mps.to_numpy(); u = tl.throttle_norm.to_numpy()
    s = np.empty_like(v); s[0] = v[0]
    for k in range(len(v)-1):
        s[k+1] = s[k] + 0.1*accel(float(s[k]), float(u[k]))
    r2 = 1-np.sum((v-s)**2)/np.sum((v-v.mean())**2)
    results[tag] = (accel, r2, n)
    print(f"\n{tag}: {n} pairs, recursive surge R2 on the trial = {r2:+.4f}")

print(f"\n{'throttle':>9} {'all data':>10} {'curated':>9} {'Level 2':>9} {'measured':>9}")
for u in [0.10, 0.15, 0.20, 0.25, 0.30, 0.39]:
    m = tl[(tl.throttle_norm>u-0.02)&(tl.throttle_norm<u+0.02)&(tl.speed_mps>0.2)]
    meas = f"{m.speed_mps.median():9.3f}" if len(m)>20 else f"{'-':>9}"
    print(f"{u:9.2f} {equil(results['all data'][0],u):10.3f} {equil(results['curated'][0],u):9.3f} "
          f"{np.sqrt(KT*u/D2):9.3f} {meas}")
cur.to_csv('data/processed/training_curated.csv', index=False)
print("\nwrote data/processed/training_curated.csv")
