from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("OpenCV (cv2) is required for pose estimation.") from exc


@dataclass
class PoseEstimate:
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,)
    inliers: np.ndarray | None = None


def _as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def ensure_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    if dist_coeffs is None:
        return np.zeros((4, 1), dtype=np.float64)
    return _as_float64(dist_coeffs)


def project_points(
    points_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    points_3d = _as_float64(points_3d)
    rvec = _as_float64(rvec).reshape(3, 1)
    tvec = _as_float64(tvec).reshape(3, 1)
    camera_matrix = _as_float64(camera_matrix)
    dist_coeffs = ensure_dist_coeffs(dist_coeffs)
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return projected.reshape(-1, 2)


def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    reprojection_error: float = 5.0,
    iterations: int = 100,
    confidence: float = 0.99,
) -> PoseEstimate | None:
    points_3d = _as_float64(points_3d)
    points_2d = _as_float64(points_2d)
    camera_matrix = _as_float64(camera_matrix)
    dist_coeffs = ensure_dist_coeffs(dist_coeffs)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs,
        reprojectionError=reprojection_error,
        iterationsCount=iterations,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None
    return PoseEstimate(rvec.reshape(3), tvec.reshape(3), inliers)


def refine_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> PoseEstimate | None:
    points_3d = _as_float64(points_3d)
    points_2d = _as_float64(points_2d)
    camera_matrix = _as_float64(camera_matrix)
    dist_coeffs = ensure_dist_coeffs(dist_coeffs)
    rvec_init = _as_float64(rvec_init).reshape(3, 1)
    tvec_init = _as_float64(tvec_init).reshape(3, 1)

    success, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return PoseEstimate(rvec.reshape(3), tvec.reshape(3))


def map_refine(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    sigma_obs: float = 2.0,
    max_iters: int = 15,
    tol: float = 1e-6,
    damping: float = 1e-3,
    dist_coeffs: np.ndarray | None = None,
) -> PoseEstimate:
    points_3d = _as_float64(points_3d)
    points_2d = _as_float64(points_2d)
    camera_matrix = _as_float64(camera_matrix)
    dist_coeffs = ensure_dist_coeffs(dist_coeffs)
    prior_mean = _as_float64(prior_mean).reshape(6)
    prior_std = np.maximum(_as_float64(prior_std).reshape(6), 1e-6)
    sigma_obs = max(float(sigma_obs), 1e-6)

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = params[:3]
        tvec = params[3:]
        proj = project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        return (proj - points_2d).reshape(-1)

    def numeric_jacobian(params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        base = residuals(params)
        jac = np.zeros((base.size, params.size), dtype=np.float64)
        for i in range(params.size):
            step = np.zeros_like(params)
            step[i] = eps
            f_plus = residuals(params + step)
            f_minus = residuals(params - step)
            jac[:, i] = (f_plus - f_minus) / (2.0 * eps)
        return jac

    params = np.concatenate([_as_float64(rvec_init).reshape(3), _as_float64(tvec_init).reshape(3)])
    inv_sigma_obs2 = 1.0 / (sigma_obs**2)
    prior_weight = 1.0 / (prior_std**2)

    for _ in range(max_iters):
        res = residuals(params)
        jac = numeric_jacobian(params)
        hessian = inv_sigma_obs2 * (jac.T @ jac)
        hessian += np.diag(prior_weight)
        hessian += damping * np.eye(6)
        gradient = inv_sigma_obs2 * (jac.T @ res) + prior_weight * (params - prior_mean)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        params_new = params - step
        if np.linalg.norm(step) < tol:
            params = params_new
            break
        params = params_new

    return PoseEstimate(params[:3], params[3:])

