#!/usr/bin/env python3
"""Extract the control-relevant signals from ArduPilot .BIN logs into small CSVs.

The raw logs carry ~60 message types at up to 150 Hz; this paper needs five
signals. Pulling just those and resampling to 10 Hz shrinks the corpus by
roughly two orders of magnitude and makes it tractable to version.

Conventions follow data_cleaning.ipynb exactly, so output is directly
comparable with usv_step_response.csv:
  - PWM normalized as (pwm - 1500) / 400
  - common 10 Hz grid, linear interpolation, per segment
  - segments split at RCOU gaps wider than 5 s, never interpolated across
  - duplicate IMU timestamps dropped, first occurrence kept

Columns beyond that schema (mode, gps_status, nsats, hdop) support the
mode and quality gating the Level-3 pipeline needs; the first five columns
are unchanged so existing notebooks read these files without modification.

Usage:
  python3 bin2csv.py data/42926_boat_tests_2801/LOGS/*.BIN -o data/processed
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from pymavlink import mavutil

PWM_NEUTRAL = 1500
PWM_HALF = 400.0
OUT_HZ = 10.0
OUT_DT = 1.0 / OUT_HZ
GAP_THRESHOLD_S = 5.0

# Only these are parsed. Everything else in the log is discarded.
WANTED = ["IMU", "RCOU", "GPS", "MODE", "THR", "STER"]


def parse(path):
    """Pull the wanted message types out of one log, keyed by type."""
    mlog = mavutil.mavlink_connection(path)
    out = {k: [] for k in WANTED}
    while True:
        msg = mlog.recv_match(type=WANTED)
        if msg is None:
            break
        d = msg.to_dict()
        t = msg.get_type()
        if t == "IMU" and d.get("I", 0) != 0:
            continue  # first IMU only; the others are redundant here
        out[t].append(d)
    return {k: pd.DataFrame(v) for k, v in out.items() if v}


def tsec(df):
    return df["TimeUS"].to_numpy(dtype=np.float64) / 1e6


def segments_from(t, gap=GAP_THRESHOLD_S):
    breaks = np.where(np.diff(t) > gap)[0]
    edges = np.concatenate(([0], breaks + 1, [len(t)]))
    return [(t[edges[i]], t[edges[i + 1] - 1]) for i in range(len(edges) - 1)]


def interp(t_query, t_src, y_src):
    if len(t_src) < 2:
        return np.full_like(t_query, np.nan, dtype=np.float64)
    return np.interp(t_query, t_src, y_src, left=np.nan, right=np.nan)


def hold(t_query, t_src, y_src):
    """Zero-order hold, for event streams such as MODE."""
    if len(t_src) == 0:
        return np.full(len(t_query), np.nan)
    idx = np.searchsorted(t_src, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return np.asarray(y_src)[idx]


def convert(path, outdir):
    f = parse(path)
    for need in ("IMU", "RCOU", "GPS"):
        if need not in f:
            return None, f"missing {need}"

    imu, rcou, gps = f["IMU"], f["RCOU"], f["GPS"]
    t0 = min(tsec(imu)[0], tsec(rcou)[0], tsec(gps)[0])

    imu_t, gyr = tsec(imu) - t0, imu["GyrZ"].to_numpy(float)
    keep = np.concatenate(([True], np.diff(imu_t) > 0))  # drop duplicate stamps
    imu_t, gyr = imu_t[keep], gyr[keep]

    rc_t = tsec(rcou) - t0
    throttle = (rcou["C3"].to_numpy(float) - PWM_NEUTRAL) / PWM_HALF
    rudder = (rcou["C1"].to_numpy(float) - PWM_NEUTRAL) / PWM_HALF

    gps_t = tsec(gps) - t0
    spd = gps["Spd"].to_numpy(float)

    mode_t = tsec(f["MODE"]) - t0 if "MODE" in f else np.array([])
    mode_n = f["MODE"]["ModeNum"].to_numpy() if "MODE" in f else np.array([])

    frames = []
    for i, (a, b) in enumerate(segments_from(rc_t)):
        if b - a < 1.0:
            continue
        grid = np.arange(a, b + 1e-9, OUT_DT)
        df = pd.DataFrame({
            "time_s": grid,
            "throttle_norm": interp(grid, rc_t, throttle),
            "rudder_norm": interp(grid, rc_t, rudder),
            "speed_mps": interp(grid, gps_t, spd),
            "yaw_rate_rps": interp(grid, imu_t, gyr),
            "segment": i,
            "mode_num": hold(grid, mode_t, mode_n) if len(mode_t) else np.nan,
            "gps_status": hold(grid, gps_t, gps["Status"].to_numpy()),
            "nsats": hold(grid, gps_t, gps["NSats"].to_numpy()),
            "hdop": hold(grid, gps_t, gps["HDop"].to_numpy(float)),
        })
        frames.append(df)

    if not frames:
        return None, "no usable segment"

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["throttle_norm", "rudder_norm", "speed_mps", "yaw_rate_rps"])
    if out.empty:
        return None, "all rows NaN after alignment"

    name = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(outdir, f"{name}_10hz.csv")
    out.to_csv(dest, index=False, float_format="%.6g")
    return out, dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--outdir", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for path in sorted(args.logs):
        raw_mb = os.path.getsize(path) / 1e6
        try:
            df, dest = convert(path, args.outdir)
        except Exception as exc:  # a truncated log should not stop the batch
            print(f"  {os.path.basename(path):16s} FAILED: {exc}")
            continue
        if df is None:
            print(f"  {os.path.basename(path):16s} skipped: {dest}")
            continue
        csv_mb = os.path.getsize(dest) / 1e6
        rows.append({
            "log": os.path.basename(path),
            "raw_MB": round(raw_mb, 1),
            "csv_MB": round(csv_mb, 3),
            "rows": len(df),
            "dur_s": round(df.time_s.max() - df.time_s.min(), 1),
            "segments": df.segment.nunique(),
            "modes": "/".join(str(int(m)) for m in sorted(df.mode_num.dropna().unique())),
            "spd_max": round(df.speed_mps.max(), 2),
            "thr_min": round(df.throttle_norm.min(), 2),
            "thr_max": round(df.throttle_norm.max(), 2),
            "rud_absmax": round(df.rudder_norm.abs().max(), 2),
            "nsats_min": int(df.nsats.min()),
        })
        print(f"  {rows[-1]['log']:16s} {raw_mb:7.1f} MB -> {csv_mb:6.3f} MB"
              f"  {len(df):6d} rows  {rows[-1]['dur_s']:7.1f} s")

    if not rows:
        sys.exit("no logs converted")
    man = pd.DataFrame(rows)
    mpath = os.path.join(args.outdir, "manifest.csv")
    man.to_csv(mpath, index=False)
    print(f"\n{len(man)} logs, {man.raw_MB.sum():.0f} MB raw -> {man.csv_MB.sum():.2f} MB csv "
          f"({man.raw_MB.sum()/max(man.csv_MB.sum(),1e-9):.0f}x smaller)")
    print(f"total {man.rows.sum()} rows, {man.dur_s.sum()/60:.1f} min")
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()
