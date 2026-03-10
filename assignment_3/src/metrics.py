from __future__ import annotations

import numpy as np

from pose_estimation import project_points


def reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> float:
    projected = project_points(points_3d, rvec, tvec, camera_matrix)
    residual = projected - points_2d
    return float(np.mean(np.linalg.norm(residual, axis=1)))


def rotation_error_deg(rotation_est: np.ndarray, rotation_gt: np.ndarray) -> float:
    delta = rotation_est @ rotation_gt.T
    trace = np.trace(delta)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est.reshape(3) - t_gt.reshape(3)))

