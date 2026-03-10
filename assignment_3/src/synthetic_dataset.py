from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("OpenCV (cv2) is required for synthetic dataset generation.") from exc


def _random_rotation(rng: np.random.Generator, max_angle_deg: float = 30.0) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    return (axis * angle).astype(np.float64)


def _camera_matrix(image_size: Tuple[int, int], focal_scale: float = 1.2) -> np.ndarray:
    height, width = image_size
    focal = focal_scale * max(height, width)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def generate_toy_dataset(
    root: str | Path,
    num_samples: int = 40,
    image_size: Tuple[int, int] = (480, 640),
    seed: int = 0,
    draw: bool = True,
) -> None:
    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    camera_matrix = _camera_matrix(image_size)
    keypoints_3d = np.array(
        [
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
            [-0.05, -0.05, 0.1],
            [0.05, -0.05, 0.1],
            [0.05, 0.05, 0.1],
            [-0.05, 0.05, 0.1],
            [0.0, 0.0, 0.05],
        ],
        dtype=np.float64,
    )

    entries = []
    height, width = image_size
    for idx in range(num_samples):
        for _ in range(50):
            rvec = _random_rotation(rng)
            tvec = np.array(
                [
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(0.4, 0.8),
                ],
                dtype=np.float64,
            )
            projected, _ = cv2.projectPoints(
                keypoints_3d, rvec.reshape(3, 1), tvec.reshape(3, 1), camera_matrix, None
            )
            points_2d = projected.reshape(-1, 2)
            if np.all(points_2d[:, 0] >= 5) and np.all(points_2d[:, 0] <= width - 5):
                if np.all(points_2d[:, 1] >= 5) and np.all(points_2d[:, 1] <= height - 5):
                    break

        image = np.full((height, width, 3), 245, dtype=np.uint8)
        if draw:
            for point in points_2d.astype(int):
                cv2.circle(image, tuple(point), 4, (30, 80, 200), -1)

        filename = f"{idx + 1:06d}.png"
        cv2.imwrite(str(images_dir / filename), image)

        entries.append(
            {
                "image": f"images/{filename}",
                "keypoints_2d": points_2d.tolist(),
                "keypoints_3d": keypoints_3d.tolist(),
                "pose": {
                    "R": cv2.Rodrigues(rvec.reshape(3, 1))[0].tolist(),
                    "t": tvec.tolist(),
                },
            }
        )

    with (root / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)

    with (root / "camera.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
            handle,
            indent=2,
        )

