"""Definition, preprocessing, and inference for the paper residual NARX."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


STATE_TAPS = (0, 1, 2, 5)
THROTTLE_TAPS = (0, 1, 2, 5, 10, 20)
RUDDER_TAPS = (0, 1, 2, 5, 10)
MAX_LAG = 20
HIDDEN_SIZE = 64
CANDIDATE_ID = "width_64_seed_42"
PAPER_CHECKPOINT_SHA256 = "d46dc04d12c62309669950cd47f24446c9bfe4350dbfe177e1b36b9b645ca774"

PREPROCESSING_SHAPES = {
    "state_mean": (2,),
    "state_scale": (2,),
    "control_mean": (2,),
    "control_scale": (2,),
    "increment_mean": (2,),
    "increment_scale": (2,),
    "feature_mean": (19,),
    "feature_scale": (19,),
}


class ResidualNarxMLP(nn.Module):
    """Fixed-tap MLP that predicts standardized state increments."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(19, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_metadata() -> dict:
    return {
        "name": "residual_narx",
        "parameter_count": 1410,
        "inputs": 19,
        "state_taps": list(STATE_TAPS),
        "throttle_taps": list(THROTTLE_TAPS),
        "rudder_taps": list(RUDDER_TAPS),
        "hidden_layers": [HIDDEN_SIZE],
        "activation": "Tanh",
        "target": "standardized state increment",
    }


def residual_feature(values: np.ndarray, index: int) -> np.ndarray:
    parts = [values[index - lag, :2] for lag in STATE_TAPS]
    parts.extend(np.asarray([values[index - lag, 2]], dtype=np.float32) for lag in THROTTLE_TAPS)
    parts.extend(np.asarray([values[index - lag, 3]], dtype=np.float32) for lag in RUDDER_TAPS)
    return np.concatenate(parts).astype(np.float32)


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale < 1e-7] = 1.0
    return scale.astype(np.float32)


def fit_preprocessing(segments: Iterable) -> dict[str, np.ndarray]:
    segments = tuple(segments)
    values = np.concatenate([segment.values for segment in segments])
    states = values[:, :2]
    controls = values[:, 2:]
    increments = np.concatenate(
        [segment.values[1:, :2] - segment.values[:-1, :2] for segment in segments]
    )
    state_mean = states.mean(axis=0).astype(np.float32)
    state_scale = _safe_scale(states)
    control_mean = controls.mean(axis=0).astype(np.float32)
    control_scale = _safe_scale(controls)
    increment_mean = increments.mean(axis=0).astype(np.float32)
    increment_scale = _safe_scale(increments)
    return {
        "state_mean": state_mean,
        "state_scale": state_scale,
        "control_mean": control_mean,
        "control_scale": control_scale,
        "increment_mean": increment_mean,
        "increment_scale": increment_scale,
        "feature_mean": np.concatenate(
            [
                *[state_mean for _ in STATE_TAPS],
                np.repeat(control_mean[0], len(THROTTLE_TAPS)),
                np.repeat(control_mean[1], len(RUDDER_TAPS)),
            ]
        ).astype(np.float32),
        "feature_scale": np.concatenate(
            [
                *[state_scale for _ in STATE_TAPS],
                np.repeat(control_scale[0], len(THROTTLE_TAPS)),
                np.repeat(control_scale[1], len(RUDDER_TAPS)),
            ]
        ).astype(np.float32),
    }


def load_paper_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[ResidualNarxMLP, dict[str, np.ndarray], dict]:
    """Load the trusted paper checkpoint and validate its identifying metadata."""
    path = Path(path)
    if file_sha256(path) != PAPER_CHECKPOINT_SHA256:
        raise ValueError("paper checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("model_name") != "residual_narx":
        raise ValueError("checkpoint is not a residual NARX")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise ValueError(f"checkpoint is not {CANDIDATE_ID}")
    if int(payload.get("hyperparameters", {}).get("hidden_size", 0)) != HIDDEN_SIZE:
        raise ValueError("checkpoint does not use the paper hidden width")
    if payload.get("architecture") != architecture_metadata():
        raise ValueError("checkpoint architecture metadata mismatch")

    preprocessing_tensors = payload.get("preprocessing")
    if not isinstance(preprocessing_tensors, dict) or set(preprocessing_tensors) != set(PREPROCESSING_SHAPES):
        raise ValueError("checkpoint preprocessing schema mismatch")
    preprocessing = {}
    for name, expected_shape in PREPROCESSING_SHAPES.items():
        value = preprocessing_tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            raise ValueError(f"invalid preprocessing tensor: {name}")
        preprocessing[name] = value.detach().cpu().numpy().astype(np.float32)

    model = ResidualNarxMLP().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, preprocessing, payload


class ResidualNarxPredictor:
    def __init__(
        self,
        model: ResidualNarxMLP,
        preprocessing: dict[str, np.ndarray],
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessing = preprocessing

    @torch.inference_mode()
    def predict_next(self, history: np.ndarray, index: int) -> np.ndarray:
        feature = residual_feature(history, index)
        standardized = (
            feature - self.preprocessing["feature_mean"]
        ) / self.preprocessing["feature_scale"]
        increment_scaled = self.model(torch.from_numpy(standardized).to(self.device))
        increment = (
            increment_scaled.cpu().numpy() * self.preprocessing["increment_scale"]
            + self.preprocessing["increment_mean"]
        )
        return (history[index, :2] + increment).astype(np.float32)
