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
    "keypoints_2d": [[x1, y1], [x2, y2], ...],
    "keypoints_3d": [[X1, Y1, Z1], [X2, Y2, Z2], ...],
    "pose": {
      "R": [[...],[...],[...]],
      "t": [tx, ty, tz]
    },
    "camera": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}  # optional
  },
  ...
]

If the dataset is missing, run the download script referenced in README.
Optionally, a camera.json file at dataset root can provide default intrinsics:
{"fx": ..., "fy": ..., "cx": ..., "cy": ...}
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
    keypoints_2d: np.ndarray  # (K, 2)
    keypoints_3d: np.ndarray  # (K, 3)
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    camera_matrix: np.ndarray | None  # (3, 3)


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

    default_camera = None
    camera_path = root / "camera.json"
    if camera_path.exists():
        with camera_path.open("r", encoding="utf-8") as handle:
            cam = json.load(handle)
        default_camera = np.array(
            [[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    samples: List[PoseSample] = []
    for entry in entries:
        image_path = root / entry["image"]
        keypoints_2d = entry.get("keypoints_2d", entry.get("keypoints"))
        if keypoints_2d is None:
            raise KeyError("annotations.json missing keypoints_2d/keypoints field.")
        keypoints_3d = entry.get("keypoints_3d")
        if keypoints_3d is None:
            raise KeyError("annotations.json missing keypoints_3d field.")
        keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
        rotation = np.asarray(entry["pose"]["R"], dtype=np.float32)
        translation = np.asarray(entry["pose"]["t"], dtype=np.float32)
        camera_entry = entry.get("camera")
        if camera_entry is None:
            camera_matrix = default_camera
        elif isinstance(camera_entry, dict):
            camera_matrix = np.array(
                [
                    [camera_entry["fx"], 0.0, camera_entry["cx"]],
                    [0.0, camera_entry["fy"], camera_entry["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            camera_matrix = np.asarray(camera_entry, dtype=np.float32)
        samples.append(
            PoseSample(image_path, keypoints_2d, keypoints_3d, rotation, translation, camera_matrix)
        )

    return PoseDataset(samples)
