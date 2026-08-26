#!/usr/bin/env python3
"""Retrain only the paper's width-64, seed-42 residual NARX architecture."""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FrozenData, Segment
from .model import (
    HIDDEN_SIZE,
    MAX_LAG,
    ResidualNarxMLP,
    architecture_metadata,
    fit_preprocessing,
    residual_feature,
)


SEED = 42
ROLLOUT_HORIZONS = (5, 10, 20, 50)
ROLLOUT_WINDOW_SEED = 20260819


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def teacher_samples(
    segments: tuple[Segment, ...], preprocessing: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    increments = []
    for segment in segments:
        for index in range(MAX_LAG, len(segment.values) - 1):
            raw = residual_feature(segment.values, index)
            features.append((raw - preprocessing["feature_mean"]) / preprocessing["feature_scale"])
            increment = segment.values[index + 1, :2] - segment.values[index, :2]
            increments.append(
                (increment - preprocessing["increment_mean"])
                / preprocessing["increment_scale"]
            )
    return np.asarray(features, dtype=np.float32), np.asarray(increments, dtype=np.float32)


def warm_start(
    model: nn.Module,
    training: tuple[Segment, ...],
    tuning: tuple[Segment, ...],
    preprocessing: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict]:
    train_x, train_y = teacher_samples(training, preprocessing)
    tune_x, tune_y = teacher_samples(tuning, preprocessing)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    tune_x_tensor = torch.from_numpy(tune_x).to(device)
    tune_y_tensor = torch.from_numpy(tune_y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 301):
        model.train()
        total = 0.0
        examples = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
            examples += len(inputs)
        model.eval()
        with torch.inference_mode():
            tuning_loss = float(criterion(model(tune_x_tensor), tune_y_tensor))
        history.append(
            {
                "phase": "one_step_warm_start",
                "horizon_samples": 1,
                "epoch": epoch,
                "training_loss": total / examples,
                "tuning_loss": tuning_loss,
            }
        )
        if tuning_loss < best_loss - 1e-6:
            best_loss = tuning_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    if best_state is None:
        raise RuntimeError("warm start produced no checkpoint")
    model.load_state_dict(best_state)
    return history


def rollout_arrays(
    segments: tuple[Segment, ...], horizon: int, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stride = max(1, horizon // 4)
    candidates = [
        (segment, start)
        for segment in segments
        for start in range(MAX_LAG, len(segment.values) - horizon, stride)
    ]
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:maximum]
    return (
        np.asarray([segment.values[start - 5 : start + 1, :2] for segment, start in candidates], dtype=np.float32),
        np.asarray([segment.values[start - MAX_LAG : start + horizon, 2:4] for segment, start in candidates], dtype=np.float32),
        np.asarray([segment.values[start + 1 : start + horizon + 1, :2] for segment, start in candidates], dtype=np.float32),
    )


def rollout_batch(
    model: nn.Module,
    state_context: torch.Tensor,
    controls: torch.Tensor,
    horizon: int,
    preprocessing: dict[str, torch.Tensor],
) -> torch.Tensor:
    states = list(state_context.unbind(dim=1))
    predictions = []
    for step in range(horizon):
        parts = [
            (states[-1 - lag] - preprocessing["state_mean"]) / preprocessing["state_scale"]
            for lag in (0, 1, 2, 5)
        ]
        control_index = MAX_LAG + step
        parts.extend(
            ((controls[:, control_index - lag, 0] - preprocessing["control_mean"][0]) / preprocessing["control_scale"][0]).unsqueeze(1)
            for lag in (0, 1, 2, 5, 10, 20)
        )
        parts.extend(
            ((controls[:, control_index - lag, 1] - preprocessing["control_mean"][1]) / preprocessing["control_scale"][1]).unsqueeze(1)
            for lag in (0, 1, 2, 5, 10)
        )
        increment_scaled = model(torch.cat(parts, dim=1))
        increment = increment_scaled * preprocessing["increment_scale"] + preprocessing["increment_mean"]
        states.append(states[-1] + increment)
        predictions.append(states[-1])
    return torch.stack(predictions, dim=1)


def rollout_loss(predictions: torch.Tensor, targets: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    normalized = (predictions - targets) / state_scale
    return 0.2 * torch.mean(normalized[:, 0] ** 2) + 0.8 * torch.mean(normalized**2)


def tuning_rollout_loss(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    preprocessing: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    total = 0.0
    examples = 0
    context, controls, targets = arrays
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(context), 256):
            context_batch = torch.from_numpy(context[start : start + 256]).to(device)
            controls_batch = torch.from_numpy(controls[start : start + 256]).to(device)
            target_batch = torch.from_numpy(targets[start : start + 256]).to(device)
            predictions = rollout_batch(model, context_batch, controls_batch, targets.shape[1], preprocessing)
            loss = rollout_loss(predictions, target_batch, preprocessing["state_scale"])
            total += float(loss) * len(context_batch)
            examples += len(context_batch)
    return total / examples


def rollout_training(
    model: nn.Module,
    training: tuple[Segment, ...],
    tuning: tuple[Segment, ...],
    preprocessing_arrays: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict]:
    preprocessing = {
        name: torch.from_numpy(value).to(device)
        for name, value in preprocessing_arrays.items()
    }
    history = []
    for stage, horizon in enumerate(ROLLOUT_HORIZONS):
        train_arrays = rollout_arrays(training, horizon, 5000, ROLLOUT_WINDOW_SEED + 100 * stage)
        tune_arrays = rollout_arrays(tuning, horizon, 2000, ROLLOUT_WINDOW_SEED + 1000 + 100 * stage)
        train_context = torch.from_numpy(train_arrays[0])
        train_controls = torch.from_numpy(train_arrays[1])
        train_targets = torch.from_numpy(train_arrays[2])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        stale = 0
        generator = torch.Generator().manual_seed(SEED + stage)
        for epoch in range(1, 41):
            model.train()
            permutation = torch.randperm(len(train_context), generator=generator)
            total = 0.0
            examples = 0
            for start in range(0, len(permutation), 256):
                indices = permutation[start : start + 256]
                context = train_context[indices].to(device)
                controls = train_controls[indices].to(device)
                targets = train_targets[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = rollout_batch(model, context, controls, horizon, preprocessing)
                loss = rollout_loss(predictions, targets, preprocessing["state_scale"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                examples += len(indices)
            tune_loss = tuning_rollout_loss(model, tune_arrays, preprocessing, device)
            history.append(
                {
                    "phase": "multistep_rollout",
                    "horizon_samples": horizon,
                    "epoch": epoch,
                    "training_loss": total / examples,
                    "tuning_loss": tune_loss,
                }
            )
            if tune_loss < best_loss - 1e-6:
                best_loss = tune_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= 10:
                    break
        if best_state is None:
            raise RuntimeError(f"rollout stage {horizon} produced no checkpoint")
        model.load_state_dict(best_state)
    return history


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=package_dir / "checkpoints" / "retrained_width_64_seed_42.pt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    frozen = FrozenData(args.data_dir)
    training = frozen.load_partition("training")
    tuning = frozen.load_partition("tuning_validation")
    preprocessing = fit_preprocessing(training)
    seed_everything()
    model = ResidualNarxMLP().to(device)
    history = warm_start(model, training, tuning, preprocessing, device)
    history.extend(rollout_training(model, training, tuning, preprocessing, device))
    payload = {
        "schema_version": 1,
        "artifact_type": "paper_residual_narx_retraining",
        "model_name": "residual_narx",
        "candidate_id": "width_64_seed_42",
        "architecture": architecture_metadata(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "preprocessing": {name: torch.as_tensor(value).cpu() for name, value in preprocessing.items()},
        "dataset_sha256": frozen.dataset_sha256,
        "split_sha256": frozen.split_sha256,
        "seed": SEED,
        "hyperparameters": {"hidden_size": HIDDEN_SIZE},
        "training_source_logs": frozen.split["partitions"]["training"]["source_logs"],
        "tuning_validation_source_logs": frozen.split["partitions"]["tuning_validation"]["source_logs"],
        "training_history": history,
        "training_configuration": {
            "warm_start_epochs": 300,
            "warm_start_patience": 25,
            "rollout_horizons_samples": list(ROLLOUT_HORIZONS),
            "rollout_epochs_per_horizon": 40,
            "rollout_patience": 10,
            "maximum_training_windows": 5000,
            "maximum_tuning_windows": 2000,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
