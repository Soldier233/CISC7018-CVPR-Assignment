import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAFromScratch:
    n_components: int
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCAFromScratch":
        self.mean_ = x.mean(axis=0)
        x_centered = x - self.mean_

        # SVD is numerically stable and avoids forming a huge covariance matrix.
        u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        eigenvalues = (s**2) / (x_centered.shape[0] - 1)

        self.components_ = vt[: self.n_components]
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / max(total_var, 1e-12)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA model is not fitted.")
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA model is not fitted.")
        return z @ self.components_ + self.mean_


@dataclass
class LDAFromScratch:
    n_components: int
    reg: float = 1e-4
    means_: dict[int, np.ndarray] | None = None
    overall_mean_: np.ndarray | None = None
    scalings_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LDAFromScratch":
        classes = np.unique(y)
        n_features = x.shape[1]

        self.means_ = {}
        self.overall_mean_ = x.mean(axis=0)

        sw = np.zeros((n_features, n_features), dtype=np.float64)
        sb = np.zeros((n_features, n_features), dtype=np.float64)

        for cls in classes:
            x_c = x[y == cls]
            mean_c = x_c.mean(axis=0)
            self.means_[int(cls)] = mean_c

            x_centered = x_c - mean_c
            sw += x_centered.T @ x_centered

            mean_diff = (mean_c - self.overall_mean_).reshape(-1, 1)
            sb += x_c.shape[0] * (mean_diff @ mean_diff.T)

        sw += self.reg * np.eye(n_features)

        mat = np.linalg.pinv(sw) @ sb
        eigvals, eigvecs = np.linalg.eig(mat)

        order = np.argsort(eigvals.real)[::-1]
        eigvecs = eigvecs[:, order]

        max_dims = min(len(classes) - 1, self.n_components)
        self.scalings_ = eigvecs[:, :max_dims].real
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scalings_ is None:
            raise RuntimeError("LDA model is not fitted.")
        return x @ self.scalings_


def plot_explained_variance(cumsum_var: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    xs = np.arange(1, len(cumsum_var) + 1)
    plt.plot(xs, cumsum_var, marker="o", markersize=3)
    plt.axhline(0.95, color="#c0392b", linestyle="--", linewidth=1.2, label="95% threshold")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance on Digits")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_embedding(z: np.ndarray, y: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=12, alpha=0.8)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.colorbar(scatter, ticks=range(10), label="Digit class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_reconstruction(
    x_original: np.ndarray,
    x_recon: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    n_samples: int = 8,
) -> None:
    fig, axes = plt.subplots(2, n_samples, figsize=(1.7 * n_samples, 4.2))
    idx = np.arange(n_samples)

    for i, j in enumerate(idx):
        axes[0, i].imshow(x_original[j].reshape(8, 8), cmap="gray")
        axes[0, i].set_title(f"GT:{y[j]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_recon[j].reshape(8, 8), cmap="gray")
        axes[1, i].set_title("PCA rec")
        axes[1, i].axis("off")

    fig.suptitle("Digits Reconstruction (Top: Original, Bottom: PCA)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_accuracy_curve(
    pca_dims: list[int],
    pca_acc: list[float],
    lda_dims: list[int],
    lda_acc: list[float],
    baseline_acc: float,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(pca_dims, pca_acc, marker="o", label="PCA + kNN")
    plt.plot(lda_dims, lda_acc, marker="s", label="LDA + kNN")
    plt.axhline(baseline_acc, linestyle="--", color="#444", label=f"Raw pixels + kNN ({baseline_acc:.3f})")
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Test Accuracy")
    plt.title("Classification Accuracy vs Embedding Dimensions")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def evaluate_knn(train_z: np.ndarray, test_z: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, np.ndarray]:
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_z, y_train)
    pred = clf.predict(test_z)
    return accuracy_score(y_test, pred), confusion_matrix(y_test, pred)


def run_experiment(seed: int, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    x = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    base_acc, _ = evaluate_knn(x_train_std, x_test_std, y_train, y_test)

    pca_full = PCAFromScratch(n_components=x_train_std.shape[1]).fit(x_train_std)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plot_explained_variance(cumsum_var, output_dir / "pca_explained_variance.png")

    pca_dims = [2, 5, 10, 20, 30, 40, 50]
    pca_acc = []
    pca_best = {"dim": -1, "acc": -1.0, "cm": None}
    for dim in pca_dims:
        pca = PCAFromScratch(n_components=dim).fit(x_train_std)
        z_train = pca.transform(x_train_std)
        z_test = pca.transform(x_test_std)
        acc, cm = evaluate_knn(z_train, z_test, y_train, y_test)
        pca_acc.append(acc)
        if acc > pca_best["acc"]:
            pca_best = {"dim": dim, "acc": acc, "cm": cm}

    lda_max = len(np.unique(y_train)) - 1
    lda_dims = list(range(2, lda_max + 1))
    lda_acc = []
    lda_best = {"dim": -1, "acc": -1.0, "cm": None}
    for dim in lda_dims:
        lda = LDAFromScratch(n_components=dim, reg=1e-3).fit(x_train_std, y_train)
        z_train = lda.transform(x_train_std)
        z_test = lda.transform(x_test_std)
        acc, cm = evaluate_knn(z_train, z_test, y_train, y_test)
        lda_acc.append(acc)
        if acc > lda_best["acc"]:
            lda_best = {"dim": dim, "acc": acc, "cm": cm}

    plot_accuracy_curve(
        pca_dims, pca_acc, lda_dims, lda_acc, base_acc, output_dir / "accuracy_vs_dims.png"
    )

    pca_2d = PCAFromScratch(n_components=2).fit(x_train_std)
    z_pca_2d = pca_2d.transform(x_test_std)
    plot_embedding(z_pca_2d, y_test, "PCA 2D Embedding (Digits)", output_dir / "pca_2d_embedding.png")

    lda_2d = LDAFromScratch(n_components=2, reg=1e-3).fit(x_train_std, y_train)
    z_lda_2d = lda_2d.transform(x_test_std)
    plot_embedding(z_lda_2d, y_test, "LDA 2D Embedding (Digits)", output_dir / "lda_2d_embedding.png")

    pca_for_recon = PCAFromScratch(n_components=20).fit(x_train_std)
    z_recon = pca_for_recon.transform(x_test_std[:8])
    x_recon_std = pca_for_recon.inverse_transform(z_recon)
    x_recon = scaler.inverse_transform(x_recon_std)
    plot_reconstruction(x_test[:8], x_recon, y_test[:8], output_dir / "pca_reconstruction.png")

    if pca_best["cm"] is not None:
        save_confusion_matrix(
            pca_best["cm"],
            output_dir / "pca_best_confusion_matrix.png",
            f"PCA Best ({pca_best['dim']}D) Confusion Matrix",
        )

    if lda_best["cm"] is not None:
        save_confusion_matrix(
            lda_best["cm"],
            output_dir / "lda_best_confusion_matrix.png",
            f"LDA Best ({lda_best['dim']}D) Confusion Matrix",
        )

    n95 = int(np.argmax(cumsum_var >= 0.95) + 1)

    results = {
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "baseline_knn_acc": float(base_acc),
        "pca_dims": pca_dims,
        "pca_acc": [float(v) for v in pca_acc],
        "lda_dims": lda_dims,
        "lda_acc": [float(v) for v in lda_acc],
        "pca_best_dim": int(pca_best["dim"]),
        "pca_best_acc": float(pca_best["acc"]),
        "lda_best_dim": int(lda_best["dim"]),
        "lda_best_acc": float(lda_best["acc"]),
        "pca_n_components_95_var": n95,
    }

    summary_path = output_dir / "results_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Assignment 4: PCA and LDA on Digits\n")
        f.write(f"Samples: {results['num_samples']} | Features: {results['num_features']}\n")
        f.write(f"Baseline kNN acc (raw pixels): {results['baseline_knn_acc']:.4f}\n")
        f.write(f"PCA best: dim={results['pca_best_dim']}, acc={results['pca_best_acc']:.4f}\n")
        f.write(f"LDA best: dim={results['lda_best_dim']}, acc={results['lda_best_acc']:.4f}\n")
        f.write(f"PCA components for >=95% variance: {results['pca_n_components_95_var']}\n")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 4: PCA and LDA Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="img", help="Output image directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir
    results = run_experiment(seed=args.seed, output_dir=output_dir)

    print("=== Assignment 4 Experiment Finished ===")
    print(f"Baseline kNN acc (raw pixels): {results['baseline_knn_acc']:.4f}")
    print(f"PCA best: dim={results['pca_best_dim']}, acc={results['pca_best_acc']:.4f}")
    print(f"LDA best: dim={results['lda_best_dim']}, acc={results['lda_best_acc']:.4f}")
    print(f"PCA components for >=95% variance: {results['pca_n_components_95_var']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
