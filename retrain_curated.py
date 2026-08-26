#!/usr/bin/env python3
"""Retrain the paper NARX with the unresponsive source logs removed.

Drops a source log if the vessel never got under way (95th-percentile speed
below 0.5 m/s) or showed no speed response under sustained throttle (mean speed
below 0.6 m/s given more than 20 samples above throttle 0.2). Everything else
about the recipe is unchanged.
"""
import argparse, sys
from pathlib import Path
import numpy as np
from narx import data as narx_data, train as narx_train

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 50])
a = ap.parse_args()

_orig = narx_data.FrozenData.load_partition

def responsive(seg):
    v, u = seg.values[:, 0], seg.values[:, 2]
    return not (np.percentile(v, 95) < 0.5)

def patched(self, name):
    segs = _orig(self, name)
    if name not in ("training", "tuning_validation"):
        return segs
    bylog = {}
    for s in segs:
        bylog.setdefault(s.source_log, []).append(s)
    keep = []
    for lg, ss in bylog.items():
        v = np.concatenate([s.values[:, 0] for s in ss])
        u = np.concatenate([s.values[:, 2] for s in ss])
        hi = u > 0.2
        if np.percentile(v, 95) < 0.5:            continue   # never under way
        if hi.sum() > 20 and v[hi].mean() < 0.6:  continue   # no thrust response
        keep.extend(ss)
    b = sum(len(s.values) for s in segs); k = sum(len(s.values) for s in keep)
    print(f"  {name}: {len(bylog)} logs / {b} rows -> "
          f"{len({s.source_log for s in keep})} logs / {k} rows ({1-k/b:.1%} dropped)")
    if not keep:
        raise SystemExit("everything was dropped")
    return tuple(keep)

narx_data.FrozenData.load_partition = patched
narx_train.FrozenData = narx_data.FrozenData
narx_train.ROLLOUT_HORIZONS = tuple(a.horizons)
sys.argv = ["narx.train", "--device", "cpu", "--output", str(a.out)]
raise SystemExit(narx_train.main())
