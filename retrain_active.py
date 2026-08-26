#!/usr/bin/env python3
"""Retrain the Level-3 NARX on the powered portion of the training data.

Only 16.3% of the frozen training partition is active (|throttle| > 0.02),
while the common evaluation trial is active 70% of the time and holds throttle
steady for stretches of ten seconds. A model fitted mostly to idle learns that
speed decays toward zero, which is exactly the failure the paper reports: a
small negative increment bias that integrates into recursive divergence.

This keeps Carson's training recipe untouched and changes only which samples it
sees, by splitting each training segment into powered sub-segments. If the
divergence is a data-coverage artifact rather than a property of black-box
models, it should shrink.

  .venv-narx/bin/python retrain_active.py --out narx/checkpoints/active.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from narx import data as narx_data
from narx import train as narx_train
from narx.data import Segment
from narx.model import MAX_LAG


def powered_subsegments(segments, thr=0.02, pad=MAX_LAG, min_len=None):
    """Split each segment into runs where the throttle is doing something."""
    min_len = min_len or (MAX_LAG + 60)
    out = []
    for seg in segments:
        u = seg.values[:, 2]
        active = np.abs(u) > thr
        if not active.any():
            continue
        # close short gaps so a brief neutral blip does not chop a run in two
        idx = np.flatnonzero(active)
        breaks = np.flatnonzero(np.diff(idx) > 30)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        for k, (a, b) in enumerate(zip(starts, ends)):
            a = max(0, a - pad)
            b = min(len(u) - 1, b + pad)
            if b - a + 1 < min_len:
                continue
            out.append(Segment(
                segment_id=f"{seg.segment_id}__pw{k:02d}",
                source_log=seg.source_log, source_group=seg.source_group,
                source_path=seg.source_path,
                time_s=seg.time_s[a:b + 1], values=seg.values[a:b + 1],
            ))
    return tuple(out)


ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 50, 100])
a = ap.parse_args()

_orig = narx_data.FrozenData.load_partition


def patched(self, name):
    segs = _orig(self, name)
    if name in ("training", "tuning_validation"):
        filt = powered_subsegments(segs)
        before = sum(len(s.values) for s in segs)
        after = sum(len(s.values) for s in filt)
        act_b = sum(int((np.abs(s.values[:, 2]) > 0.02).sum()) for s in segs)
        act_a = sum(int((np.abs(s.values[:, 2]) > 0.02).sum()) for s in filt)
        print(f"  {name}: {len(segs)} segs / {before} rows (active {act_b/before:.1%})"
              f"  ->  {len(filt)} segs / {after} rows (active {act_a/max(after,1):.1%})")
        if not filt:
            raise SystemExit(f"no powered sub-segments survived in {name}")
        return filt
    return segs


narx_data.FrozenData.load_partition = patched
narx_train.FrozenData = narx_data.FrozenData
narx_train.ROLLOUT_HORIZONS = tuple(a.horizons)
print(f"rollout horizons: {[h/10 for h in a.horizons]} s")
sys.argv = ["narx.train", "--device", "cpu", "--output", str(a.out)]
raise SystemExit(narx_train.main())
