#!/usr/bin/env python3
"""Write measured / recursive traces for a checkpoint on the common trial."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from narx import model as nm
from narx.model import file_sha256, MAX_LAG

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--trial", type=Path, default=Path("usv_step_response_150_280s.csv"))
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

nm.PAPER_CHECKPOINT_SHA256 = file_sha256(a.checkpoint)
mdl, pre, _ = nm.load_paper_checkpoint(a.checkpoint)
pred = nm.ResidualNarxPredictor(mdl, pre)

t = pd.read_csv(a.trial)
vals = t[["speed_mps", "yaw_rate_rps", "throttle_norm", "rudder_norm"]].to_numpy(np.float32)
rec = np.full((len(vals), 2), np.nan, np.float32)
hist = vals.copy()
rec[MAX_LAG] = vals[MAX_LAG, :2]
for k in range(MAX_LAG, len(vals) - 1):
    nxt = pred.predict_next(hist, k)
    rec[k + 1] = nxt
    hist[k + 1, :2] = nxt

out = pd.DataFrame({
    "time_s": t.time_s, "throttle_norm": t.throttle_norm, "rudder_norm": t.rudder_norm,
    "measured_speed_mps": t.speed_mps, "measured_yaw_rate_rps": t.yaw_rate_rps,
    "recursive_speed_mps": rec[:, 0], "recursive_yaw_rate_rps": rec[:, 1],
})
out.to_csv(a.out, index=False)
m = np.isfinite(rec[:, 0])
for j, n in enumerate(("speed", "yaw_rate")):
    y = vals[m, j]; p = rec[m, j]
    print(f"  {n}: recursive R2 = {1-np.sum((y-p)**2)/np.sum((y-y.mean())**2):+.4f}")
print(f"wrote {a.out}")
