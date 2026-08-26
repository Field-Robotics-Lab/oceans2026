#!/usr/bin/env python3
"""Does each model level know the steady-state throttle-to-speed relationship?

A long recursive rollout is dominated by the model's equilibrium: hold throttle
constant and whatever speed the model settles to is what the rollout reports.
Levels 1 and 2 have an equilibrium by construction. The NARX has one only if it
learned it, and the training corpus holds throttle steady for a median of 0.1 s
and a 90th percentile of 1.6 s against a vehicle time constant of two to four
seconds, so the corpus almost never reaches equilibrium.

This drives each model from rest at constant throttle and records where it
settles, against the speeds actually measured on the trial.

  .venv-narx/bin/python steady_state_test.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from narx import model as nm
from narx.model import file_sha256, MAX_LAG

KT, D2 = 3.963, 0.438
K_V, TAU_V = 6.42, 1.95
THROTTLES = np.round(np.arange(0.04, 0.42, 0.02), 3)
SETTLE_S = 60.0

CHECKPOINTS = {
    "uncurated": "narx/checkpoints/final_8202026.pt",
    "curated": "narx/checkpoints/curated.pt",
}


def narx_equilibrium(ckpt, throttles, settle_s=SETTLE_S):
    p = Path(ckpt)
    nm.PAPER_CHECKPOINT_SHA256 = file_sha256(p)
    mdl, pre, _ = nm.load_paper_checkpoint(p)
    pred = nm.ResidualNarxPredictor(mdl, pre)
    out = []
    n = int(settle_s / 0.1) + MAX_LAG + 1
    for u in throttles:
        vals = np.zeros((n, 4), np.float32)
        vals[:, 2] = u
        for k in range(MAX_LAG, n - 1):
            vals[k + 1, :2] = pred.predict_next(vals, k)
        out.append(float(vals[-1, 0]))
    return np.asarray(out)


def measured_steady(trial, throttles, tol=0.02):
    v = []
    for u in throttles:
        m = trial[(trial.throttle_norm > u - tol) & (trial.throttle_norm < u + tol)
                  & (trial.speed_mps > 0.2)]
        v.append(m.speed_mps.median() if len(m) > 20 else np.nan)
    return np.asarray(v, float)


if __name__ == "__main__":
    trial = pd.read_csv("usv_step_response_150_280s.csv")
    l2 = np.sqrt(KT * THROTTLES / D2)          # d(v)/dt = 0  ->  v = sqrt(kT u / d2)
    l1 = K_V * THROTTLES                        # DC gain of the first-order model
    meas = measured_steady(trial, THROTTLES)

    curves = {"Level 1": l1, "Level 2": l2}
    for name, ck in CHECKPOINTS.items():
        if Path(ck).exists():
            curves[f"Level 3 ({name})"] = narx_equilibrium(ck, THROTTLES)

    hdr = f"{'throttle':>9}" + "".join(f"{k:>24}" for k in curves) + f"{'measured':>10}"
    print(hdr)
    for i, u in enumerate(THROTTLES):
        row = f"{u:9.2f}" + "".join(f"{c[i]:24.3f}" for c in curves.values())
        row += f"{meas[i]:10.3f}" if np.isfinite(meas[i]) else f"{'-':>10}"
        print(row)

    ok = np.isfinite(meas)
    print("\nRMS error against measured steady speed, where measured exists:")
    for k, c in curves.items():
        print(f"  {k:32s} {np.sqrt(np.mean((c[ok]-meas[ok])**2)):.3f} m/s")
    print("\nmonotone in throttle (a physical requirement):")
    for k, c in curves.items():
        print(f"  {k:32s} {'yes' if np.all(np.diff(c) > -1e-6) else 'NO'}")

    json.dump({k: [float(x) for x in v] for k, v in curves.items()}
              | {"throttle": [float(x) for x in THROTTLES],
                 "measured": [None if not np.isfinite(x) else float(x) for x in meas]},
              open("data/processed/steady_state_curves.json", "w"), indent=2)
    print("\nwrote data/processed/steady_state_curves.json")
