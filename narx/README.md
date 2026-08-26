# Paper Residual NARX

This directory freezes the single Level 3 neural network used for the paper
results on the `vogt_dev` branch. It intentionally excludes the absolute NARX,
LSTM, model-family selection harness, full-regime experiments, and every other
candidate checkpoint.

## Paper Checkpoint

`checkpoints/final_8202026.pt` is the trusted `width_64_seed_42` checkpoint:

```text
SHA-256: d46dc04d12c62309669950cd47f24446c9bfe4350dbfe177e1b36b9b645ca774
Inputs:  19 lagged state and control values
Network: Linear(19, 64), Tanh, Linear(64, 2)
Outputs: standardized speed and yaw-rate increments
Params:  1,410
```

The state taps are 0, 1, 2, and 5 samples; throttle taps are 0, 1, 2, 5,
10, and 20 samples; rudder taps are 0, 1, 2, 5, and 10 samples. At 10 Hz,
the longest history is two seconds. The scaler state is embedded in the
checkpoint.

PyTorch checkpoints use pickle. Load this file only as a trusted repository
artifact. `model.py` checks its SHA-256 before loading it.

## Data And Results

`data/` contains the frozen 44,133-row slow-region dataset and its source-log-
grouped partitions. The model was trained on 28,023 rows in 40 segments from
37 logs. Raw ArduPilot logs are intentionally excluded.

`results/` contains archived outputs used to prepare the paper:

- `paper_validation_results.json` and `paper_validation_traces.csv`
- `original_trial_level3_results.json` and `original_trial_level3_traces.csv`
- `paper_validation_figure_manifest.json`

The paper-validation result pools `tuning_validation`, `family_selection`, and
`final_test` into one 28-log nontraining evaluation. The checkpoint was not
retrained for that evaluation, but the nominal final-test partition was opened
and must not be described as untouched.

The checkpoint was manually frozen for the paper after broader evaluation. It
was not the winner selected by the earlier within-family tuning rule. No other
checkpoint is included here because this package documents the network that
actually produced the paper results.

## Code

Run modules from the repository root so package-relative imports resolve:

```bash
python3 -m narx.evaluate --mode paper-validation
python3 -m narx.evaluate --mode original-trial
python3 -m narx.train --device cpu
```

`narx.train` is restricted to the paper architecture, width 64 and seed 42. Its
default output is `checkpoints/retrained_width_64_seed_42.pt`, so it does not
overwrite the archived paper checkpoint.

The archived metrics and figures were not regenerated during this migration.
The original untracked research tree was under
`~/oceans2026/data/narx_models/model_selection/`; absolute paths in copied JSON
metadata were converted to paths relative to this repository.
