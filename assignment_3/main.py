import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("OpenCV (cv2) is required. Install opencv-python.") from exc

import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from metrics import reprojection_error, rotation_error_deg, translation_error
from pose_estimation import map_refine, refine_pnp, solve_pnp_ransac
from synthetic_dataset import generate_toy_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 3: Pose Estimation with ML + MAP")
    parser.add_argument("--data-root", type=str, default="data_linemod", help="Dataset root.")
    parser.add_argument("--generate-toy", action="store_true", help="Generate a toy dataset first.")
    parser.add_argument("--num-samples", type=int, default=40, help="Toy dataset size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0],
        help="Pixel noise levels for evaluation.",
    )
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Scale factor for prior std.")
    parser.add_argument("--max-iters", type=int, default=15, help="MAP refinement iterations.")
    return parser.parse_args()


def infer_camera_from_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    height, width = image.shape[:2]
    focal = 1.2 * max(height, width)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def compute_pose_prior(samples) -> tuple[np.ndarray, np.ndarray]:
    rvecs = []
    tvecs = []
    for sample in samples:
        rvec, _ = cv2.Rodrigues(sample.rotation)
        rvecs.append(rvec.reshape(3))
        tvecs.append(sample.translation.reshape(3))
    if not rvecs:
        return np.zeros(6), np.ones(6)
    params = np.hstack([np.stack(rvecs), np.stack(tvecs)])
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    std = np.maximum(std, 1e-3)
    return mean, std


def evaluate(
    samples,
    noise_sigma: float,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    rng: np.random.Generator,
    max_iters: int,
):
    metrics = {
        "ML-RANSAC": {"reproj": [], "rot": [], "trans": []},
        "ML-Refined": {"reproj": [], "rot": [], "trans": []},
        "MAP": {"reproj": [], "rot": [], "trans": []},
    }

    for sample in samples:
        camera_matrix = sample.camera_matrix
        if camera_matrix is None:
            camera_matrix = infer_camera_from_image(sample.image_path)

        noisy_keypoints = sample.keypoints_2d + rng.normal(
            0.0, noise_sigma, size=sample.keypoints_2d.shape
        )

        ml_base = solve_pnp_ransac(sample.keypoints_3d, noisy_keypoints, camera_matrix)
        if ml_base is None:
            continue

        ml_refined = refine_pnp(
            sample.keypoints_3d, noisy_keypoints, camera_matrix, ml_base.rvec, ml_base.tvec
        )
        if ml_refined is None:
            continue

        sigma_obs = max(noise_sigma, 1.0)
        map_est = map_refine(
            sample.keypoints_3d,
            noisy_keypoints,
            camera_matrix,
            ml_refined.rvec,
            ml_refined.tvec,
            prior_mean,
            prior_std,
            sigma_obs=sigma_obs,
            max_iters=max_iters,
        )

        for label, estimate in [
            ("ML-RANSAC", ml_base),
            ("ML-Refined", ml_refined),
            ("MAP", map_est),
        ]:
            rotation_est, _ = cv2.Rodrigues(estimate.rvec.reshape(3, 1))
            metrics[label]["reproj"].append(
                reprojection_error(
                    sample.keypoints_3d,
                    sample.keypoints_2d,
                    estimate.rvec,
                    estimate.tvec,
                    camera_matrix,
                )
            )
            metrics[label]["rot"].append(rotation_error_deg(rotation_est, sample.rotation))
            metrics[label]["trans"].append(translation_error(estimate.tvec, sample.translation))

    summary = {}
    for label, values in metrics.items():
        summary[label] = {
            "reproj": float(np.mean(values["reproj"])) if values["reproj"] else float("nan"),
            "rot": float(np.mean(values["rot"])) if values["rot"] else float("nan"),
            "trans": float(np.mean(values["trans"])) if values["trans"] else float("nan"),
        }
    return summary


def plot_metric_curves(noise_levels, curves, metric, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for method, stats in curves.items():
        plt.plot(noise_levels, stats[metric], marker="o", label=method)
    plt.xlabel("Noise Sigma (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{metric.title()} Error vs Noise")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_overlay(
    image_path: Path,
    points_gt: np.ndarray,
    points_noisy: np.ndarray,
    points_ml: np.ndarray,
    points_map: np.ndarray,
    save_path: Path,
):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    plt.scatter(points_gt[:, 0], points_gt[:, 1], c="#2ca02c", s=25, label="GT 2D")
    plt.scatter(points_noisy[:, 0], points_noisy[:, 1], c="#ff7f0e", s=25, label="Noisy 2D")
    plt.scatter(points_ml[:, 0], points_ml[:, 1], c="#1f77b4", s=25, label="ML-Refined")
    plt.scatter(points_map[:, 0], points_map[:, 1], c="#d62728", s=25, label="MAP")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / args.data_root
    img_dir = base_dir / "img"
    img_dir.mkdir(exist_ok=True)

    if args.generate_toy:
        generate_toy_dataset(data_root, num_samples=args.num_samples, seed=args.seed)

    dataset = load_dataset(data_root)
    train_set, val_set = dataset.split(train_ratio=0.7, seed=args.seed)

    prior_mean, prior_std = compute_pose_prior(train_set.samples)
    prior_std = prior_std * max(args.prior_scale, 1e-3)

    curves = {
        "ML-RANSAC": {"reproj": [], "rot": [], "trans": []},
        "ML-Refined": {"reproj": [], "rot": [], "trans": []},
        "MAP": {"reproj": [], "rot": [], "trans": []},
    }

    for noise in args.noise_levels:
        summary = evaluate(
            val_set.samples, noise, prior_mean, prior_std, rng, max_iters=args.max_iters
        )
        for method in curves:
            curves[method]["reproj"].append(summary[method]["reproj"])
            curves[method]["rot"].append(summary[method]["rot"])
            curves[method]["trans"].append(summary[method]["trans"])
        print(f"[Noise {noise:.1f}] {summary}")

    plot_metric_curves(
        args.noise_levels,
        curves,
        "reproj",
        "Mean Reprojection Error (pixels)",
        img_dir / "reprojection_error.png",
    )
    plot_metric_curves(
        args.noise_levels,
        curves,
        "rot",
        "Rotation Error (deg)",
        img_dir / "rotation_error.png",
    )
    plot_metric_curves(
        args.noise_levels,
        curves,
        "trans",
        "Translation Error (units)",
        img_dir / "translation_error.png",
    )

    if val_set.samples:
        sample = val_set.samples[0]
        camera_matrix = sample.camera_matrix
        if camera_matrix is None:
            camera_matrix = infer_camera_from_image(sample.image_path)
        noise = args.noise_levels[-1]
        noisy_keypoints = sample.keypoints_2d + rng.normal(
            0.0, noise, size=sample.keypoints_2d.shape
        )
        ml_base = solve_pnp_ransac(sample.keypoints_3d, noisy_keypoints, camera_matrix)
        if ml_base is not None:
            ml_refined = refine_pnp(
                sample.keypoints_3d, noisy_keypoints, camera_matrix, ml_base.rvec, ml_base.tvec
            )
            if ml_refined is not None:
                map_est = map_refine(
                    sample.keypoints_3d,
                    noisy_keypoints,
                    camera_matrix,
                    ml_refined.rvec,
                    ml_refined.tvec,
                    prior_mean,
                    prior_std,
                    sigma_obs=max(noise, 1.0),
                    max_iters=args.max_iters,
                )
                from pose_estimation import project_points

                points_ml = project_points(
                    sample.keypoints_3d, ml_refined.rvec, ml_refined.tvec, camera_matrix
                )
                points_map = project_points(
                    sample.keypoints_3d, map_est.rvec, map_est.tvec, camera_matrix
                )
                plot_overlay(
                    sample.image_path,
                    sample.keypoints_2d,
                    noisy_keypoints,
                    points_ml,
                    points_map,
                    img_dir / "overlay.png",
                )

    print(f"[Info] Outputs saved to: {img_dir}")


if __name__ == "__main__":
    main()
