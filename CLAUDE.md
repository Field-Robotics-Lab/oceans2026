# oceans2026 — Claude Code Project Context

## What this project is

Research prototype for an **IEEE OCEANS 2026** paper.
**Hypothesis:** Machine learning can quickly generate a hierarchy of dynamic models
of an uncrewed surface vessel (USV) that support design, testing, and implementation
of closed-loop motion control.

The three-level model hierarchy, following Box's "all models are wrong, but some are
useful," balances interpretability against fidelity:

| Level | Notebook | Model | Status |
|-------|----------|-------|--------|
| 1 | `first_order_models.ipynb` | First-order SISO (linear ARX, decoupled) | ✅ Done |
| 2 | `nonlinear_ss_model.ipynb` | Nonlinear state-space (quadratic drag) | ✅ Done |
| 3 | `blackbox_model.ipynb` | NARX MLP (black-box) | ✅ Done |

**Current status:** Full paper drafted in `paper/paper.tex` (8 pages, builds clean).
The `traditional-models` and `vogt_dev` branches were merged into `main` on 2026-08-25;
`traditional-models` was identical to `main` and `vogt_dev` fast-forwarded, so no
content was reconciled at merge time. Sec. IV is complete: the platform is described from the ME2801 course
site and the ArduRover parameter set. No `\needsdata{}` placeholders remain.

---

## Platform

| Item | Value | Source |
|------|-------|--------|
| Hull | Pro Boat Blackjack 42, RC monohull, 1.07 m (42 in) | BSB, direct observation |
| Mass | ~5 kg as instrumented | ME2801 autopilot page |
| Propulsion | Single four-pole water-cooled brushless motor, one propeller, 160 A ESC, 8S LiPo | product listing |
| Steering | Single rudder, one digital servo | product listing |
| Autopilot | Cube Orange, ArduRover 4.6.3 (commit `3fc7011a`) | ME2801 autopilot page |
| GNSS | HERE3 over DroneCAN | ME2801 autopilot page |
| Cruise speed | `CRUISE_SPEED` = 5 m/s (trial covers only 0–2.9 m/s) | `2026_05_29_proto.param` |
| Actuator config | `SERVO1_FUNCTION=26` (steering), `SERVO3_FUNCTION=70` (throttle), all others 0 | `2026_05_29_proto.param` |

The single-motor / single-rudder configuration is confirmed by the parameter file.

⚠️ The Horizon Hobby listing calls this product a "Brushless Catamaran." BSB confirms
the actual hull is a **monohull** (2026-08-25); the spec sheet is wrong. The paper says
monohull. Beam is not published and is not stated in the paper.

Reference material lives at
`~/WorkingCopies/me2801/introduction-to-feedback-control/site/` (the `autopilot/`
and `weeks/` directories).

## Data

| File | Description |
|------|-------------|
| `step_resp_example.BIN.proc.mat` | Raw source — MATLAB v5, ArduPilot BIN log, 341 s recording |
| `usv_step_response.csv` | Full cleaned dataset, 10 Hz, two segments (gap at t≈37–66 s) |
| `usv_step_response_150_280s.csv` | Working subset, 130 s, time re-zeroed — all analysis uses this |

**Signals:**

| Column | Units | Source |
|--------|-------|--------|
| `time_s` | s | Relative time from recording start |
| `throttle_norm` | — | RCOU_C3 PWM normalized: (pwm−1500)/400 |
| `rudder_norm` | — | RCOU_C1 PWM normalized: (pwm−1500)/400 |
| `speed_mps` | m/s | GPS forward speed |
| `yaw_rate_rps` | rad/s | IMU yaw rate (+ = port turn) |

Data coverage is intentionally narrow (throttle 0–39%, speed 0–3 m/s, one trial).
This is flagged throughout the notebooks. More data will be collected before the full paper.

---

## Notebooks — run order and dependencies

All notebooks read from `usv_step_response_150_280s.csv`.
Execute with `jupyter nbconvert --to notebook --execute <nb>.ipynb --output <nb>.ipynb`.

1. **`data_cleaning.ipynb`** — loads `.mat`, cleans multi-rate streams, exports CSVs
2. **`first_order_models.ipynb`** — levels-1 SISO fits + diagnostics that motivate level 2
3. **`nonlinear_ss_model.ipynb`** — level-2 nonlinear SS, two-phase identification
4. **`blackbox_model.ipynb`** — level-3 NARX MLP, train/test split, simulation comparison

---

## Key results

**Level 1 — First-order SISO**
- Surge: `G(s) = 6.42 / (1.95s + 1)` — R² = 0.930
- Yaw:   `G_r(s) = 0.771 / (0.471s + 1)` — R² = 0.826

**Level 2 — Nonlinear state-space** (two-phase identification)
- `dv/dt = 3.963·u − 0.438·v|v|`  — R² = 0.909
- `dr/dt = 1.474·δ − 1.912·r`     — R² = 0.820
- `d2` (drag) identified from coasting-only segments; `k_T` from powered segments

**Level 3 — NARX MLP** — the numbers below are the ones the paper reports.
Architecture: fixed-tap residual NARX, 19 lagged inputs, one 64-neuron tanh hidden
layer, two state-increment outputs. Trained on 65 source logs / 44,133 samples,
grouped by source log (37 train, 28 validation).

- Common 130 s trial (the cross-level comparison): one-step speed R² = 0.998,
  yaw R² = 0.989; recursive speed R² = **−1.988**, yaw R² = 0.259
- Slow-region validation set (30 segments, 28 held-out logs): one-step speed
  R² = 0.998, yaw R² = 0.991; recursive speed R² = 0.836, yaw R² = 0.768
- Key finding: recursive simulation diverges for surge while Levels 1 and 2 track,
  so physical structure is essential regularization at this data scale

⚠️ **`blackbox_model.ipynb` is STALE and does not reproduce these numbers.** It still
contains the earlier 3-lag / 20-neuron ReLU model trained on the single 130 s trial
(one-step 0.927/0.876, recursive 0.050/0.822). The code and logs behind the published
Level-3 results are not in this repository. Reconcile before the full paper.

**Diagnostic 4 — margin-based sufficiency** (`diagnostic4_margin.py`)
- The Level-1 surge model supports the control design only over **[0.35, 1.28] m/s**,
  under half the 0–2.9 m/s tested envelope
- Lower bound is robust stability, closed form `v_lo = kT/(4·d2·K1)`, independent of
  the design knob λ. Below it the quadratic-drag linearization is singular
- Upper bound is loss of closed-loop bandwidth. The 50% criterion is a **convention,
  not a result**: 70% gives 1.03 m/s, 30% gives 1.91 m/s. Only the lower bound is analytic.
- Upper bound detail (50% of design intent): Robust
  stability never fails at high speed: drag is stabilizing, so the Level-1 design
  goes sluggish rather than unstable. This contradicts the expectation recorded in
  `classicalmodelling_notes.tex`
- Yaw: ‖ℓT‖∞ = 0.034 across the envelope, so Level 2 changes nothing for yaw control

**Diagnostic findings** (in `first_order_models.ipynb`, final sections):
- Drag is purely quadratic (D1 = −0.015, **D2 = 0.448**) — confirms level-2 structure.
  Note 0.448 is the *diagnostic's* coasting fit; the *Level-2 two-phase* value is 0.4377
  (→ 0.438). These are two different fits and were conflated in an earlier draft.
- Speed-dependent rudder gain (K_r · v) not supported by this dataset
- Rudder-induced surge drag negligible (r = −0.08)

---

## Dependencies

```
python3-scipy      1.11.4
python3-pandas     2.1.4
python3-matplotlib 3.6.3
python3-sklearn    1.4.1
jupyter-notebook
jupyter-nbconvert
```

Install on a fresh Ubuntu machine:
```bash
sudo apt-get install -y python3-scipy python3-pandas python3-matplotlib \
     python3-sklearn jupyter-notebook jupyter-nbconvert
```

**Note:** a user-local numpy 2.2.6 in `~/.local/lib/python3.10/site-packages`
shadows the system numpy 1.21.5 that the system matplotlib was built against, so
plotting fails with `_ARRAY_API not found`. Run with `PYTHONNOUSERSITE=1` to use the
system stack:

```bash
PYTHONNOUSERSITE=1 python3 diagnostic4_margin.py
```

---

## Where this stopped (2026-08-25)

Paper is submittable at 8 pages. Work halted on the **Level-3 reproducibility blocker**,
not on the writing.

A reviewer pass found the cross-level comparison was confounded: neither
`first_order_models.ipynb` nor `nonlinear_ss_model.ipynb` contains a train/test split,
so Levels 1 and 2 are fitted and scored on the same 1,300 samples while Level 3 is
scored on a trial it was not fitted to. The paper now states this and **withdraws** the
claim that physical structure beats data volume. It rests instead on the within-model
claim, which the protocol difference cannot touch: one-step accuracy is a poor guide to
simulation accuracy (0.998 → −1.988 on the common trial, 0.998 → 0.836 on Level 3's
held-out set).

Blocking an extension:
1. **Level-3 code, weights and the 65-log corpus are not in this repo.** Every Level-3
   number is unverifiable, including the one now carrying the paper's central claim.
2. **Unknown whether the common 130 s trial was inside Level 3's training partition.**
   If it was, the recursive failure is on training data, which is a stronger result and
   worth one sentence in the paper. Carson can answer this.
3. **No uncertainty quantification anywhere.** Single trial, no intervals.
   `nonlinear_ss_model.ipynb` warns `kT` and `d2` extrapolate beyond the observed
   envelope; the paper does not repeat that warning.
4. **Fig. 4 is CV's preview render** — web-styled, overlapping subtitle, clipped R²
   legend label. It carries the headline claim. Regenerating needs the Level-3
   predictions, i.e. blocker 1.

## Known inconsistency

`paper/abstract.tex` is the already-submitted early-decision abstract. It quotes the
slow-region validation numbers (recursive 0.836/0.768) while claiming surge
divergence, which those numbers do not show. The paper now headlines the common-trial
numbers instead. Left as-is because the abstract is already submitted, but a reviewer
holding both documents may notice.

## What to do next

1. **Regenerate the Level-3 figures at IEEE size.** `figures/preview/*` are
   web-styled: baked-in titles duplicate the captions, and in
   `combined_three_level_preview.pdf` the "Yaw-rate recursive simulation" heading
   overlaps the panel above it and the Level-3 R² legend label is clipped.

2. **Collect more data** — wider throttle range, multiple speed levels, dedicated
   rudder sweeps at fixed speeds (needed to resolve speed-dependent rudder gain).

3. **Iterate models** with richer data:
   - Level 2: add speed-dependent rudder term once data supports it
   - Level 3: explore deeper MLP or LSTM; larger lag window; cross-validation

3. **Write the full paper.** The current draft is the conference version.
