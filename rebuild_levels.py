#!/usr/bin/env python3
"""Rebuild Levels 1 and 2 from data and compare all three levels on equal terms.

The comparison in the paper is confounded three ways: Levels 1 and 2 are fitted
and scored on the same short trial, Level 3 is fitted on 21x more data, and
Level 3 is rolled out 25x beyond the 5 s horizon it was trained for. This script
removes the first two by refitting Levels 1 and 2 on Carson's training partition
(the same 28,023 rows Level 3 saw) and scoring every level on the held-out
validation segments, and addresses the third by reporting error against rollout
horizon rather than a single full-segment number.

Level 3 is not retrained here; its predictions come from the archived traces.

  python3 rebuild_levels.py
"""
import json

import numpy as np
import pandas as pd

DT = 0.1
CORPUS = "narx/data/slow_manual_mode_10hz.csv"
TRAIN = "narx/data/partitions/training.csv"
TRACES = "narx/results/paper_validation_traces.csv"
HORIZONS = [1, 5, 10, 20, 50, 100]          # samples; matches Carson's grid


def contiguous(g):
    """Split a segment into runs with an unbroken 10 Hz grid."""
    t = g.time_s.to_numpy()
    brk = np.where(~np.isclose(np.diff(t), DT, atol=0.02))[0]
    edges = np.concatenate(([0], brk + 1, [len(g)]))
    return [g.iloc[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]


# ── Fit Levels 1 and 2 on the training partition ────────────────────────────
def fit_levels(train):
    Av, bv, Ar, br = [], [], [], []
    coast_v, coast_dv, pow_u, pow_dv, pow_v = [], [], [], [], []
    for _, g in train.groupby("segment_id"):
        for run in contiguous(g.sort_values("time_s")):
            if len(run) < 5:
                continue
            v = run.speed_mps.to_numpy()
            r = run.yaw_rate_rps.to_numpy()
            u = run.throttle_norm.to_numpy()
            d = run.rudder_norm.to_numpy()
            Av.append(np.column_stack([v[:-1], u[:-1]])); bv.append(v[1:])
            Ar.append(np.column_stack([r[:-1], d[:-1]])); br.append(r[1:])
            dv = (v[1:] - v[:-1]) / DT
            coast = (np.abs(u[:-1]) < 0.02) & (v[:-1] > 0.4)
            coast_v.append(v[:-1][coast]); coast_dv.append(dv[coast])
            pw = np.abs(u[:-1]) >= 0.02
            pow_u.append(u[:-1][pw]); pow_dv.append(dv[pw]); pow_v.append(v[:-1][pw])

    a_v, b_v = np.linalg.lstsq(np.vstack(Av), np.concatenate(bv), rcond=None)[0]
    a_r, b_r = np.linalg.lstsq(np.vstack(Ar), np.concatenate(br), rcond=None)[0]
    L1 = dict(a_v=a_v, b_v=b_v, a_r=a_r, b_r=b_r,
              tau_v=-DT / np.log(a_v), K_v=b_v / (1 - a_v),
              tau_r=-DT / np.log(a_r), K_r=b_r / (1 - a_r))

    # Phase 1: drag from coasting alone, where dv/dt = -d2 v|v|
    cv, cdv = np.concatenate(coast_v), np.concatenate(coast_dv)
    d2 = float(-np.linalg.lstsq((cv * np.abs(cv))[:, None], cdv, rcond=None)[0][0])
    # Phase 2: thrust gain from powered samples with d2 held fixed
    pu, pdv, pv = np.concatenate(pow_u), np.concatenate(pow_dv), np.concatenate(pow_v)
    kT = float(np.linalg.lstsq(pu[:, None], pdv + d2 * pv * np.abs(pv), rcond=None)[0][0])
    # Yaw: dr/dt = kd*delta - dr*r
    Ay, by = [], []
    for _, g in train.groupby("segment_id"):
        for run in contiguous(g.sort_values("time_s")):
            if len(run) < 5:
                continue
            r = run.yaw_rate_rps.to_numpy(); d = run.rudder_norm.to_numpy()
            Ay.append(np.column_stack([d[:-1], -r[:-1]])); by.append((r[1:] - r[:-1]) / DT)
    kd, dr = np.linalg.lstsq(np.vstack(Ay), np.concatenate(by), rcond=None)[0]
    L2 = dict(kT=kT, d2=d2, kd=float(kd), dr=float(dr),
              n_coast=len(cv), n_pow=len(pu))
    return L1, L2


# ── Rollout predictors ──────────────────────────────────────────────────────
def roll_l1(L1, v0, u, n):
    v = v0
    for k in range(n):
        v = L1["a_v"] * v + L1["b_v"] * u[k]
    return v


def roll_l2(L2, v0, u, n):
    v = v0
    for k in range(n):
        v = v + DT * (L2["kT"] * u[k] - L2["d2"] * v * abs(v))
    return v


def horizon_rmse(segs, L1, L2, l3map):
    out = {h: {"L1": [], "L2": [], "L3": []} for h in HORIZONS}
    for seg, run, l3 in segs:
        v = run.speed_mps.to_numpy(); u = run.throttle_norm.to_numpy()
        for h in HORIZONS:
            if len(v) <= h:
                continue
            for i in range(0, len(v) - h, max(1, h // 2)):
                truth = v[i + h]
                out[h]["L1"].append(roll_l1(L1, v[i], u[i:i + h], h) - truth)
                out[h]["L2"].append(roll_l2(L2, v[i], u[i:i + h], h) - truth)
    return out


if __name__ == "__main__":
    corp = pd.read_csv(CORPUS)
    train = pd.read_csv(TRAIN)
    tr_segs = set(train.segment_id.unique())
    L1, L2 = fit_levels(corp[corp.segment_id.isin(tr_segs)])

    print("Refit on Carson's 28,023-row training partition")
    print(f"  Level 1 surge: K={L1['K_v']:.3f}  tau={L1['tau_v']:.3f} s"
          f"   (paper, old trial: K=6.42  tau=1.95)")
    print(f"  Level 1 yaw:   K={L1['K_r']:.3f}  tau={L1['tau_r']:.3f} s"
          f"   (paper: K=0.771  tau=0.471)")
    print(f"  Level 2 surge: kT={L2['kT']:.3f}  d2={L2['d2']:.3f}"
          f"   (paper: kT=3.963  d2=0.438)   coast n={L2['n_coast']}, powered n={L2['n_pow']}")
    print(f"  Level 2 yaw:   kd={L2['kd']:.3f}  dr={L2['dr']:.3f}"
          f"   (paper: kd=1.474  dr=1.912)")

    # Held-out evaluation: every validation segment, full recursive rollout
    tr = pd.read_csv(TRACES)
    rows, pool = [], {"m": [], "L1": [], "L2": [], "L3": []}
    for seg, g in tr.groupby("segment_id"):
        g = g.sort_values("validation_index").reset_index(drop=True)
        cs = corp[corp.segment_id == seg].sort_values("time_s").reset_index(drop=True)
        if len(cs) != len(g):
            continue
        st = np.where(np.isfinite(g.recursive_speed_mps.to_numpy()))[0]
        if len(st) < 50:
            continue
        i0 = st[0]
        u = cs.throttle_norm.to_numpy()[i0:]
        meas = g.measured_speed_mps.to_numpy()[i0:]
        p3 = g.recursive_speed_mps.to_numpy()[i0:]
        p1 = np.empty(len(u)); p1[0] = meas[0]
        p2 = np.empty(len(u)); p2[0] = meas[0]
        for k in range(len(u) - 1):
            p1[k + 1] = L1["a_v"] * p1[k] + L1["b_v"] * u[k]
            p2[k + 1] = p2[k] + DT * (L2["kT"] * u[k] - L2["d2"] * p2[k] * abs(p2[k]))
        for key, p in (("L1", p1), ("L2", p2), ("L3", p3)):
            pool[key].append(p)
        pool["m"].append(meas)
        rows.append(dict(seg=seg[:32], n=len(u)))

    M = np.concatenate(pool["m"])
    print(f"\nHeld-out validation, {len(rows)} segments, {len(M)} samples")
    print("full-segment recursive surge R^2, all levels trained on the same data:")
    for key in ("L1", "L2", "L3"):
        P = np.concatenate(pool[key])
        m = np.isfinite(M) & np.isfinite(P)
        r2 = 1 - np.sum((M[m] - P[m]) ** 2) / np.sum((M[m] - M[m].mean()) ** 2)
        rmse = np.sqrt(np.mean((M[m] - P[m]) ** 2))
        print(f"  {key}: R^2 = {r2:+.4f}   RMSE = {rmse:.4f} m/s")

    # Horizon sweep for the structured levels, against Carson's archived Level 3
    segs = []
    for seg, g in tr.groupby("segment_id"):
        cs = corp[corp.segment_id == seg].sort_values("time_s").reset_index(drop=True)
        if len(cs) == len(g):
            segs.append((seg, cs, None))
    hs = horizon_rmse(segs, L1, L2, None)
    l3h = json.load(open("narx/results/paper_validation_results.json"))["horizons"]
    print(f"\n{'horizon':>8} {'L1 RMSE':>9} {'L2 RMSE':>9} {'L3 RMSE':>9}")
    for h in HORIZONS:
        e1 = np.array(hs[h]["L1"]); e2 = np.array(hs[h]["L2"])
        l3 = l3h.get(str(h), {}).get("speed_rmse", float("nan"))
        print(f"{h*DT:7.1f}s {np.sqrt((e1**2).mean()):9.4f} "
              f"{np.sqrt((e2**2).mean()):9.4f} {l3:9.4f}")

    json.dump({"L1": {k: float(v) for k, v in L1.items()},
               "L2": {k: float(v) for k, v in L2.items()}},
              open("data/processed/refit_levels.json", "w"), indent=2)
    print("\nwrote data/processed/refit_levels.json")
