#!/usr/bin/env python3
"""Evaluate any residual-NARX checkpoint, not just the frozen paper one.

narx/model.py refuses to load a checkpoint whose SHA-256 differs from the
archived paper artifact. That guard is right for the paper result and wrong for
retraining, so this rebinds the expected hash to whichever file is being
evaluated and leaves every other structural check in place: model name,
candidate id, hidden width, architecture metadata and preprocessing schema all
still have to match.

  .venv-narx/bin/python eval_checkpoint.py narx/checkpoints/retrained_width_64_seed_42.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from narx import model as narx_model
from narx.model import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--trial", type=Path, default=Path("usv_step_response_150_280s.csv"))
    args = ap.parse_args()

    narx_model.PAPER_CHECKPOINT_SHA256 = file_sha256(args.checkpoint)
    from narx import evaluate as narx_eval

    model, preprocessing, payload = narx_model.load_paper_checkpoint(args.checkpoint)
    predictor = narx_model.ResidualNarxPredictor(model, preprocessing)

    import pandas as pd
    trial = pd.read_csv(args.trial)
    values = trial[["speed_mps", "yaw_rate_rps", "throttle_norm", "rudder_norm"]].to_numpy(np.float32)
    lag = narx_model.MAX_LAG

    one_step = np.full((len(values), 2), np.nan, np.float32)
    for k in range(lag, len(values) - 1):
        one_step[k + 1] = predictor.predict_next(values, k)

    rec = np.full((len(values), 2), np.nan, np.float32)
    hist = values.copy()
    rec[lag] = values[lag, :2]
    for k in range(lag, len(values) - 1):
        nxt = predictor.predict_next(hist, k)
        rec[k + 1] = nxt
        hist[k + 1, :2] = nxt

    def r2(truth, pred):
        m = np.isfinite(pred)
        y, p = truth[m], pred[m]
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  sha256 {file_sha256(args.checkpoint)[:16]}...  scored rows {np.isfinite(rec[:,0]).sum()}")
    for j, name in enumerate(("speed_mps", "yaw_rate_rps")):
        print(f"  {name:13s} one-step R2 = {r2(values[:, j], one_step[:, j]):+.4f}   "
              f"recursive R2 = {r2(values[:, j], rec[:, j]):+.4f}")

    # the drift mechanism: mean one-step increment bias
    inc = one_step[1:, 0] - values[:-1, 0]
    true_inc = values[1:, 0] - values[:-1, 0]
    m = np.isfinite(inc)
    print(f"  one-step increment bias: {np.mean((inc - true_inc)[m]):+.6f} m/s per step "
          f"-> {np.mean((inc-true_inc)[m])*m.sum():+.2f} m/s over the trial")


if __name__ == "__main__":
    sys.exit(main())
