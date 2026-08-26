#!/usr/bin/env python3
"""Evaluate the frozen paper NARX without modifying archived results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import COMMON_WARMUP_SAMPLES, FrozenData, Segment
from .model import ResidualNarxPredictor, load_paper_checkpoint


DIVERGENCE_BOUNDS = np.asarray([100.0, 20.0], dtype=np.float32)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(("speed_mps", "yaw_rate_rps")):
        error = actual[:, index] - predicted[:, index]
        residual = float(np.sum(error**2))
        total = float(np.sum((actual[:, index] - actual[:, index].mean()) ** 2))
        result[name] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": 1.0 - residual / total if total > 0.0 else None,
        }
    return result


def evaluate_segments(
    predictor: ResidualNarxPredictor,
    segments: tuple[Segment, ...],
    state_scale: np.ndarray,
) -> dict:
    one_step_actual = []
    one_step_predicted = []
    replay_actual = []
    replay_predicted = []
    one_step_scores = []
    replay_scores = []
    for segment in segments:
        actual = segment.values[COMMON_WARMUP_SAMPLES + 1 :, :2]
        one_step = np.asarray(
            [
                predictor.predict_next(segment.values, index)
                for index in range(COMMON_WARMUP_SAMPLES, len(segment.values) - 1)
            ],
            dtype=np.float32,
        )
        working = segment.values.copy()
        replay = []
        for index in range(COMMON_WARMUP_SAMPLES, len(working) - 1):
            prediction = predictor.predict_next(working, index)
            if not np.all(np.isfinite(prediction)) or np.any(np.abs(prediction) >= DIVERGENCE_BOUNDS):
                raise ValueError(f"recursive simulation diverged: {segment.segment_id}")
            working[index + 1, :2] = prediction
            replay.append(prediction)
        replay = np.asarray(replay, dtype=np.float32)
        one_step_actual.append(actual)
        one_step_predicted.append(one_step)
        replay_actual.append(actual)
        replay_predicted.append(replay)
        one_step_scores.append(float(np.mean(np.sqrt(np.mean((actual - one_step) ** 2, axis=0)) / state_scale)))
        replay_scores.append(float(np.mean(np.sqrt(np.mean((actual - replay) ** 2, axis=0)) / state_scale)))
    return {
        "segment_count": len(segments),
        "scored_rows": int(sum(len(values) for values in one_step_actual)),
        "one_step": {
            "macro_nrmse": float(np.mean(one_step_scores)),
            "metrics": regression_metrics(np.concatenate(one_step_actual), np.concatenate(one_step_predicted)),
        },
        "recursive_simulation": {
            "macro_nrmse": float(np.mean(replay_scores)),
            "metrics": regression_metrics(np.concatenate(replay_actual), np.concatenate(replay_predicted)),
        },
    }


def load_original_trial(path: Path) -> Segment:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    values = np.asarray(
        [
            [
                float(row["speed_mps"]),
                float(row["yaw_rate_rps"]),
                float(row["throttle_norm"]),
                float(row["rudder_norm"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    return Segment(
        segment_id="legacy_step_response_150_280s",
        source_log="step_resp_example",
        source_group="legacy_frl_trial",
        source_path="step_resp_example.BIN.proc.mat",
        time_s=np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32),
        values=values,
    )


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parent
    repository = package_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("paper-validation", "original-trial"),
        default="paper-validation",
    )
    parser.add_argument("--checkpoint", type=Path, default=package_dir / "checkpoints" / "final_8202026.pt")
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data")
    parser.add_argument("--trial", type=Path, default=repository / "usv_step_response_150_280s.csv")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen = FrozenData(args.data_dir)
    model, preprocessing, payload = load_paper_checkpoint(args.checkpoint)
    if payload.get("dataset_sha256") != frozen.dataset_sha256:
        raise ValueError("checkpoint dataset SHA-256 does not match frozen data")
    if payload.get("split_sha256") != frozen.split_sha256:
        raise ValueError("checkpoint split SHA-256 does not match frozen data")
    predictor = ResidualNarxPredictor(model, preprocessing, torch.device("cpu"))
    if args.mode == "paper-validation":
        segments = tuple(
            segment
            for name in ("tuning_validation", "family_selection", "final_test")
            for segment in frozen.load_partition(name)
        )
    else:
        segments = (load_original_trial(args.trial),)
    result = {
        "model": payload["candidate_id"],
        "mode": args.mode,
        **evaluate_segments(predictor, segments, frozen.training_state_scale()),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
