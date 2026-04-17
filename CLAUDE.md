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

**Current status:** Prototype complete. Next task is writing the conference abstract
for early-decision submission. Full paper follows after several months of additional
data collection.

---

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

**Level 3 — NARX MLP** (3 lags, 20-neuron hidden layer, trained on first 100 s)
- One-step prediction: speed R² = 0.927, yaw R² = 0.876
- Closed-loop simulation: speed R² = 0.050, yaw R² = 0.822
- Key finding: simulation diverges for surge — structured models outperform
  the black-box with limited data because physical constraints prevent drift

**Diagnostic findings** (in `first_order_models.ipynb`, final sections):
- Drag is purely quadratic (D1 ≈ 0, D2 = 0.438) — confirms level-2 structure
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

---

## What to do next

1. **Write the abstract** — early-decision submission for IEEE OCEANS 2026.
   The narrative is the three-level model hierarchy, the Box quote framing,
   and the prototype results demonstrating the workflow.

2. **Collect more data** — wider throttle range, multiple speed levels, dedicated
   rudder sweeps at fixed speeds (needed to resolve speed-dependent rudder gain).

3. **Iterate models** with richer data:
   - Level 2: add speed-dependent rudder term once data supports it
   - Level 3: explore deeper MLP or LSTM; larger lag window; cross-validation

4. **Write the full paper.**
