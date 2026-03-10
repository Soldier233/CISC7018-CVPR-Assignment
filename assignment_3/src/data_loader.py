"""Data loading utilities for a tiny LINEMOD-style subset.

This module supports a lightweight JSON-based dataset format so the assignment
can run without downloading the full LINEMOD package. The expected format is:

root/
  images/
    000001.png
    000002.png
  annotations.json

annotations.json schema:
[
  {
    "image": "images/000001.png",
    "keypoints": [[x1, y1], [x2, y2], ...],
    "pose": {
      "R": [[...],[...],[...]],
      "t": [tx, ty, tz]
    }
  },
  ...
]

If the dataset is missing, run the download script referenced in README.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class PoseSample:
    image_path: Path
    keypoints: np.ndarray  # (K, 2)
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)


@dataclass
class PoseDataset:
    samples: List[PoseSample]

    def split(self, train_ratio: float = 0.8, seed: int = 0) -> Tuple["PoseDataset", "PoseDataset"]:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        cut = int(len(indices) * train_ratio)
        train_idx, val_idx = indices[:cut], indices[cut:]
        return PoseDataset([self.samples[i] for i in train_idx]), PoseDataset(
            [self.samples[i] for i in val_idx]
        )


def load_dataset(root: str | Path) -> PoseDataset:
    root = Path(root)
    annotation_path = root / "annotations.json"
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"annotations.json not found at {annotation_path}. "
            "Download or create a tiny subset first."
        )

    with annotation_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    samples: List[PoseSample] = []
    for entry in entries:
        image_path = root / entry["image"]
        keypoints = np.asarray(entry["keypoints"], dtype=np.float32)
        rotation = np.asarray(entry["pose"]["R"], dtype=np.float32)
        translation = np.asarray(entry["pose"]["t"], dtype=np.float32)
        samples.append(PoseSample(image_path, keypoints, rotation, translation))

    return PoseDataset(samples)
