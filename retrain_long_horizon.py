#!/usr/bin/env python3
"""Retrain the Level-3 NARX with rollout horizons extended past 5 s.

The paper rolls the model out for 128 s but trains it with rollout losses to
5 s only. This asks whether the resulting drift is a property of black-box
models or of that training choice, by rerunning the identical recipe with the
horizon ladder extended and nothing else changed.

  .venv-narx/bin/python retrain_long_horizon.py --horizons 5 10 20 50 100 200 --out narx/checkpoints/long200.pt
"""
import argparse, sys
from pathlib import Path
from narx import train as narx_train

ap = argparse.ArgumentParser()
ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200])
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

narx_train.ROLLOUT_HORIZONS = tuple(a.horizons)
print(f"rollout horizons (samples): {narx_train.ROLLOUT_HORIZONS}  "
      f"= {[h/10 for h in narx_train.ROLLOUT_HORIZONS]} s")
sys.argv = ["narx.train", "--device", "cpu", "--output", str(a.out)]
raise SystemExit(narx_train.main())
