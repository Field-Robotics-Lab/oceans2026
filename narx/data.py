"""Frozen slow-region data and partition loading for the paper NARX."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import file_sha256


DT = 0.1
COMMON_WARMUP_SAMPLES = 20
SIGNAL_NAMES = ("speed_mps", "yaw_rate_rps", "throttle_norm", "rudder_norm")
PARTITION_NAMES = ("training", "tuning_validation", "family_selection", "final_test")


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source_log: str
    source_group: str
    source_path: str
    time_s: np.ndarray
    values: np.ndarray


def load_segments(path: Path, expected_sources: set[str] | None = None) -> tuple[Segment, ...]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            grouped[row["segment_id"]].append(row)
    segments = []
    for segment_id, rows in grouped.items():
        values = np.asarray(
            [[float(row[name]) for name in SIGNAL_NAMES] for row in rows],
            dtype=np.float32,
        )
        time_s = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32)
        if len(values) <= COMMON_WARMUP_SAMPLES + 1:
            raise ValueError(f"segment is too short: {segment_id}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"segment contains nonfinite signals: {segment_id}")
        if len(time_s) > 1 and not np.allclose(np.diff(time_s), DT, atol=1e-3, rtol=0.0):
            raise ValueError(f"segment is not uniformly sampled at 10 Hz: {segment_id}")
        segments.append(
            Segment(
                segment_id=segment_id,
                source_log=rows[0]["source_log"],
                source_group=rows[0]["source_group"],
                source_path=rows[0]["source_path"],
                time_s=time_s,
                values=values,
            )
        )
    if not segments:
        raise ValueError(f"no segments found in {path}")
    present_sources = {segment.source_log for segment in segments}
    if expected_sources is not None and present_sources != expected_sources:
        raise ValueError(f"partition sources do not match the frozen split: {path}")
    return tuple(segments)


class FrozenData:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.split_path = self.data_dir / "split.json"
        self.split_sha256 = file_sha256(self.split_path)
        self.split = json.loads(self.split_path.read_text(encoding="utf-8"))
        self.dataset_path = self.data_dir / self.split["dataset"]
        self.dataset_sha256 = self.split["dataset_sha256"]
        if self.split.get("schema_version") != 2:
            raise ValueError("unsupported split schema")
        if file_sha256(self.dataset_path) != self.dataset_sha256:
            raise ValueError("aggregate dataset SHA-256 mismatch")
        if set(self.split["partitions"]) != set(PARTITION_NAMES):
            raise ValueError("split does not contain the expected partitions")

    def load_partition(self, name: str) -> tuple[Segment, ...]:
        if name not in PARTITION_NAMES:
            raise ValueError(f"unknown partition: {name}")
        record = self.split["partitions"][name]
        dataset = record["dataset"]
        path = (self.data_dir / dataset["path"]).resolve()
        if self.data_dir not in path.parents:
            raise ValueError(f"unsafe partition path: {name}")
        if file_sha256(path) != dataset["sha256"]:
            raise ValueError(f"partition SHA-256 mismatch: {name}")
        return load_segments(path, set(record["source_logs"]))

    def training_state_scale(self) -> np.ndarray:
        values = np.concatenate(
            [segment.values[:, :2] for segment in self.load_partition("training")]
        )
        return values.std(axis=0).astype(np.float32)
