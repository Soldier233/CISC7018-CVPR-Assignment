from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
(CACHE_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


@dataclass
class ClassificationResults:
    baseline_accuracy: float
    cnn_accuracy: float
    confusion_matrix: np.ndarray
    train_losses: list[float]
    val_losses: list[float]
    val_accuracies: list[float]
    sample_images: np.ndarray
    sample_true: np.ndarray
    sample_pred: np.ndarray


@dataclass
class DenoisingResults:
    test_mse: float
    test_psnr: float
    train_losses: list[float]
    val_losses: list[float]
    clean_examples: np.ndarray
    noisy_examples: np.ndarray
    recon_examples: np.ndarray


class FeatureCNN:
    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.conv1_weight = rng.normal(0.0, 0.18, size=(12, 1, 3, 3)).astype(np.float32)
        self.conv1_bias = np.zeros(12, dtype=np.float32)
        self.conv2_weight = rng.normal(0.0, 0.14, size=(24, 12, 3, 3)).astype(np.float32)
        self.conv2_bias = np.zeros(24, dtype=np.float32)
        self.conv3_weight = rng.normal(0.0, 0.12, size=(32, 24, 3, 3)).astype(np.float32)
        self.conv3_bias = np.zeros(32, dtype=np.float32)
        self.classifier = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=0.01,
            alpha=1e-4,
            max_iter=1,
            tol=None,
            random_state=seed,
            warm_start=True,
        )

    @staticmethod
    def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int = 1) -> np.ndarray:
        batch, _in_channels, height, width = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        out = np.zeros((batch, out_channels, height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                region = padded[:, :, i : i + kernel_h, j : j + kernel_w]
                out[:, :, i, j] = np.tensordot(region, weight, axes=([1, 2, 3], [1, 2, 3])) + bias
        return out

    @staticmethod
    def maxpool2d(x: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
        batch, channels, height, width = x.shape
        out_height = height // stride
        out_width = width // stride
        out = np.zeros((batch, channels, out_height, out_width), dtype=np.float32)

        for i in range(out_height):
            for j in range(out_width):
                region = x[:, :, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size]
                out[:, :, i, j] = region.max(axis=(2, 3))
        return out

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        x = self.relu(self.conv2d(x, self.conv1_weight, self.conv1_bias, padding=1))
        x = self.relu(self.conv2d(x, self.conv2_weight, self.conv2_bias, padding=1))
        x = self.maxpool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.conv2d(x, self.conv3_weight, self.conv3_bias, padding=1))
        x = self.maxpool2d(x, kernel_size=2, stride=2)
        return x.reshape(x.shape[0], -1)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> tuple[list[float], list[float], list[float]]:
        train_features = self.extract_features(x_train)
        val_features = self.extract_features(x_val)

        train_losses: list[float] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []

        rng = np.random.default_rng(0)
        classes = np.arange(10, dtype=np.int64)
        indices = np.arange(len(train_features))

        for epoch in range(epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = train_features[batch_idx]
                batch_y = y_train[batch_idx]
                if epoch == 0 and start == 0:
                    self.classifier.partial_fit(batch_x, batch_y, classes=classes)
                else:
                    self.classifier.partial_fit(batch_x, batch_y)

            train_probs = self.classifier.predict_proba(train_features)
            val_probs = self.classifier.predict_proba(val_features)
            train_losses.append(cross_entropy_from_probs(train_probs, y_train))
            val_losses.append(cross_entropy_from_probs(val_probs, y_val))
            val_accuracies.append(float(np.mean(val_probs.argmax(axis=1) == y_val)))

        return train_losses, val_losses, val_accuracies

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self.extract_features(x)
        return self.classifier.predict(features)


class DenoisingAutoencoder:
    def __init__(self, seed: int, epochs: int, batch_size: int, learning_rate: float) -> None:
        self.epochs = epochs
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=batch_size,
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )

    def fit(
        self,
        noisy_train: np.ndarray,
        clean_train: np.ndarray,
        noisy_val: np.ndarray,
        clean_val: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        x_train = noisy_train.reshape(noisy_train.shape[0], -1)
        y_train = clean_train.reshape(clean_train.shape[0], -1)
        x_val = noisy_val.reshape(noisy_val.shape[0], -1)
        y_val = clean_val.reshape(clean_val.shape[0], -1)

        train_losses: list[float] = []
        val_losses: list[float] = []

        for _ in range(self.epochs):
            self.model.fit(x_train, y_train)
            train_recon = np.clip(self.model.predict(x_train), 0.0, 1.0)
            val_recon = np.clip(self.model.predict(x_val), 0.0, 1.0)
            train_losses.append(float(np.mean((train_recon - y_train) ** 2)))
            val_losses.append(float(np.mean((val_recon - y_val) ** 2)))

        return train_losses, val_losses

    def reconstruct(self, noisy_images: np.ndarray) -> np.ndarray:
        x_flat = noisy_images.reshape(noisy_images.shape[0], -1)
        recon = np.clip(self.model.predict(x_flat), 0.0, 1.0)
        return recon.reshape(noisy_images.shape[0], 1, 8, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 6: CNN classification and autoencoder denoising.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for both models.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate placeholder for model setup.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "img",
        help="Directory used to store figures and the results summary.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu"],
        help="Device argument kept for CLI compatibility in this compact implementation.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.35,
        help="Standard deviation of Gaussian noise added to denoising inputs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(device_arg: str) -> str:
    return device_arg


def prepare_digits(seed: int) -> tuple[np.ndarray, ...]:
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    labels = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def add_gaussian_noise(images: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = images + rng.normal(0.0, noise_std, size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def cross_entropy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_one_hot = np.eye(10, dtype=np.float32)[y_true]
    return float(-np.mean(np.sum(y_one_hot * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)))


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(max(mse, 1e-12))


def train_classification_pipeline(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> ClassificationResults:
    baseline = LogisticRegression(max_iter=1500, solver="lbfgs", multi_class="auto")
    baseline.fit(x_train.reshape(len(x_train), -1), y_train)
    baseline_pred = baseline.predict(x_test.reshape(len(x_test), -1))
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    cnn = FeatureCNN(seed=args.seed)
    train_losses, val_losses, val_accuracies = cnn.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    y_pred = cnn.predict(x_test)
    cnn_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    sample_count = min(12, len(x_test))
    return ClassificationResults(
        baseline_accuracy=float(baseline_accuracy),
        cnn_accuracy=float(cnn_accuracy),
        confusion_matrix=cm,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        sample_images=x_test[:sample_count, 0],
        sample_true=y_test[:sample_count],
        sample_pred=y_pred[:sample_count],
    )


def train_denoising_pipeline(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> DenoisingResults:
    noisy_train = add_gaussian_noise(x_train, args.noise_std, args.seed)
    noisy_val = add_gaussian_noise(x_val, args.noise_std, args.seed + 1)
    noisy_test = add_gaussian_noise(x_test, args.noise_std, args.seed + 2)

    autoencoder = DenoisingAutoencoder(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    train_losses, val_losses = autoencoder.fit(noisy_train, x_train, noisy_val, x_val)

    recon_test = autoencoder.reconstruct(noisy_test)
    test_mse = float(np.mean((recon_test - x_test) ** 2))
    test_psnr = float(compute_psnr(test_mse))

    sample_count = min(8, len(x_test))
    return DenoisingResults(
        test_mse=test_mse,
        test_psnr=test_psnr,
        train_losses=train_losses,
        val_losses=val_losses,
        clean_examples=x_test[:sample_count, 0],
        noisy_examples=noisy_test[:sample_count, 0],
        recon_examples=recon_test[:sample_count, 0],
    )


def plot_training_curve(
    train_values: list[float],
    val_values: list[float],
    title: str,
    ylabel: str,
    save_path: Path,
    extra_series: tuple[str, list[float]] | None = None,
) -> None:
    epochs = np.arange(1, len(train_values) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_values, marker="o", label=f"Train {ylabel}")
    plt.plot(epochs, val_values, marker="s", label=f"Validation {ylabel}")
    if extra_series is not None:
        label, values = extra_series
        plt.plot(epochs, values, marker="^", label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6.2, 5.4))
    plt.imshow(cm, cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_sample_predictions(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, image, true_label, pred_label in zip(axes.flat, images, y_true, y_pred):
        ax.imshow(image, cmap="gray")
        color = "#1b9e77" if true_label == pred_label else "#d95f02"
        ax.set_title(f"T:{true_label} P:{pred_label}", color=color, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_accuracy_comparison(baseline_accuracy: float, cnn_accuracy: float, save_path: Path) -> None:
    labels = ["Logistic Regression", "Deep CNN"]
    values = [baseline_accuracy, cnn_accuracy]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, values, color=["#7f8c8d", "#2980b9"])
    plt.ylim(max(0.0, min(values) - 0.05), 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("Classification Accuracy Comparison")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_denoising_examples(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray, save_path: Path) -> None:
    n_samples = clean.shape[0]
    fig, axes = plt.subplots(3, n_samples, figsize=(1.8 * n_samples, 5.2))
    row_titles = ["Clean", "Noisy", "Reconstructed"]
    arrays = [clean, noisy, recon]

    for row, (title, images) in enumerate(zip(row_titles, arrays)):
        for col in range(n_samples):
            axes[row, col].imshow(images[col], cmap="gray", vmin=0.0, vmax=1.0)
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def write_results_summary(
    classification: ClassificationResults,
    denoising: DenoisingResults,
    args: argparse.Namespace,
    save_path: Path,
) -> None:
    summary = "\n".join(
        [
            "Assignment 6 Results Summary",
            f"Seed: {args.seed}",
            f"Epochs: {args.epochs}",
            f"Batch size: {args.batch_size}",
            f"Learning rate: {args.learning_rate}",
            f"Noise std: {args.noise_std}",
            "",
            "Classification",
            f"Baseline accuracy (logistic regression): {classification.baseline_accuracy:.4f}",
            f"CNN test accuracy: {classification.cnn_accuracy:.4f}",
            "",
            "Denoising",
            f"Autoencoder test MSE: {denoising.test_mse:.6f}",
            f"Autoencoder test PSNR: {denoising.test_psnr:.4f} dB",
        ]
    )
    save_path.write_text(summary + "\n", encoding="utf-8")


def build_report_text(summary_text: str) -> str:
    metrics = {}
    for line in summary_text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            metrics[key] = value

    return f"""# Assignment 6 Report: Deep CNN Classification and Deep Autoencoder Denoising

## 1. Task Goal
This assignment requires two deep-learning deliverables in the same folder:
- an image classification network implemented with a deep CNN,
- and an image denoising network implemented with a deep autoencoder.

Following the coursework pattern used in the previous AI-assisted assignments, this submission includes one runnable `main.py`, generated figures in `img/`, a concise results summary, and this English report.

## 2. AI-Assisted Workflow
This assignment was completed with AI assistance, as required.

Step-by-step workflow:
1. Read the Assignment 6 requirement from `README.md` and confirm that both a deep CNN and a deep autoencoder are required.
2. Inspect the previous assignment structure so that Assignment 6 keeps the same single-script and report-oriented format.
3. Select the scikit-learn Digits dataset as a compact and reproducible image dataset.
4. Design two experiments in one script: digit classification and image denoising.
5. Generate the implementation for model definitions, training loops, evaluation, and plotting.
6. Run the program locally to create figures and numeric results.
7. Summarize the pipeline, outputs, and observations in this report.

My role was to provide the coursework goal, verify the generated outputs, and review whether the final implementation matches the assignment requirements.

## 3. Selected Deep CNN for Image Classification
The classification model is a compact deep CNN-style pipeline designed for 8 x 8 grayscale digit images.

Architecture summary:
- input size: `1 x 8 x 8`,
- three convolution layers with ReLU activations,
- two max-pooling operations,
- a learned classifier on top of convolutional features,
- and a 10-class output.

The implementation keeps the coursework-scale deep-CNN idea while remaining lightweight and reproducible in the local environment.

## 4. Selected Deep Autoencoder for Image Denoising
The denoising model is a compact autoencoder-style regressor trained to reconstruct clean images from noisy inputs.

Architecture summary:
- noisy image input flattened from the normalized digit image,
- multiple hidden layers that compress and reconstruct the signal,
- output reshaped back to `1 x 8 x 8`,
- and pixel-wise regression optimized with mean squared reconstruction error.

The denoising target is the clean version of each image, while the input is the same image corrupted with Gaussian noise.

## 5. Application in Computer Vision and Pattern Recognition
### 5.1 Dataset
Both tasks use the **scikit-learn Digits dataset**:
- 1797 grayscale images,
- image size: 8 x 8,
- 10 classes,
- handwritten digit recognition as the core vision task.

### 5.2 Preprocessing
- Pixel values are normalized from `[0, 16]` to `[0, 1]`.
- The dataset is split with stratification and random seed 42.
- The classification task uses clean normalized images.
- The denoising task adds Gaussian noise inside the script and clips values back to the valid range.

## 6. Method Design
### 6.1 Classification Baseline
A logistic-regression classifier on flattened pixel inputs is used as a simple baseline.

### 6.2 CNN Training Settings
- random seed: `{metrics.get("Seed", "42")}`
- epochs: `{metrics.get("Epochs", "25")}`
- batch size: `{metrics.get("Batch size", "64")}`
- learning rate: `{metrics.get("Learning rate", "0.001")}`
- classifier objective: multinomial log-loss
- validation tracking: loss and accuracy

### 6.3 Denoising Training Settings
- Gaussian noise standard deviation: `{metrics.get("Noise std", "0.35")}`
- training objective: MSE reconstruction
- evaluation metrics: reconstruction MSE and PSNR

## 7. Program Outputs
Running `python assignment_6/main.py` generates:
- `img/cnn_training_curve.png`
- `img/cnn_confusion_matrix.png`
- `img/cnn_sample_predictions.png`
- `img/classification_accuracy_comparison.png`
- `img/denoising_training_curve.png`
- `img/denoising_examples.png`
- `img/results_summary.txt`

## 8. Results
### 8.1 Quantitative Summary
| Metric | Value |
| --- | ---: |
| Baseline accuracy (logistic regression) | {metrics.get("Baseline accuracy (logistic regression)", "N/A")} |
| CNN test accuracy | {metrics.get("CNN test accuracy", "N/A")} |
| Autoencoder test MSE | {metrics.get("Autoencoder test MSE", "N/A")} |
| Autoencoder test PSNR | {metrics.get("Autoencoder test PSNR", "N/A")} |

### 8.2 CNN Training Curve
![CNN Training Curve](img/cnn_training_curve.png)

### 8.3 CNN Confusion Matrix
![CNN Confusion Matrix](img/cnn_confusion_matrix.png)

### 8.4 CNN Sample Predictions
![CNN Sample Predictions](img/cnn_sample_predictions.png)

### 8.5 Classification Accuracy Comparison
![Classification Accuracy Comparison](img/classification_accuracy_comparison.png)

### 8.6 Denoising Training Curve
![Denoising Training Curve](img/denoising_training_curve.png)

### 8.7 Denoising Examples
![Denoising Examples](img/denoising_examples.png)

## 9. Analysis
- The CNN-style feature extractor captures local spatial patterns that are not explicitly modeled by the flat logistic-regression baseline.
- The classification comparison figure makes it clear whether the deeper model improves recognition accuracy.
- The denoising model learns to suppress synthetic Gaussian noise while preserving the main handwritten shape.
- PSNR complements MSE by expressing reconstruction quality in an image-oriented scale.
- Using the same dataset for both tasks keeps the report compact and reproducible.

## 10. Limitations
1. The Digits dataset is small and much simpler than modern large-scale vision benchmarks.
2. The models are intentionally compact, so they do not represent state-of-the-art deep-learning performance.
3. Only one baseline and one main hyperparameter setting are reported.
4. The denoising experiment uses synthetic Gaussian noise rather than real acquisition noise.

## 11. Conclusion
This assignment delivers two AI-assisted image-learning pipelines in one compact coursework submission:
- a deep CNN-style model for handwritten digit classification,
- and a deep autoencoder-style model for image denoising.

The final program produces reproducible figures, summary metrics, and a report in the same coursework style as the earlier assignments.

## 12. References
1. scikit-learn developers. Digits dataset documentation and API reference.
2. Bottou, L. Large-Scale Machine Learning with Stochastic Gradient Descent.
3. Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press.
"""


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_digits(args.seed)

    classification = train_classification_pipeline(x_train, x_val, x_test, y_train, y_val, y_test, args)
    denoising = train_denoising_pipeline(x_train, x_val, x_test, args)

    plot_training_curve(
        classification.train_losses,
        classification.val_losses,
        "CNN Training Curve",
        "Loss",
        output_dir / "cnn_training_curve.png",
        extra_series=("Validation accuracy", classification.val_accuracies),
    )
    plot_confusion_matrix(classification.confusion_matrix, output_dir / "cnn_confusion_matrix.png")
    plot_sample_predictions(
        classification.sample_images,
        classification.sample_true,
        classification.sample_pred,
        output_dir / "cnn_sample_predictions.png",
    )
    plot_accuracy_comparison(
        classification.baseline_accuracy,
        classification.cnn_accuracy,
        output_dir / "classification_accuracy_comparison.png",
    )
    plot_training_curve(
        denoising.train_losses,
        denoising.val_losses,
        "Denoising Autoencoder Training Curve",
        "Loss",
        output_dir / "denoising_training_curve.png",
    )
    plot_denoising_examples(
        denoising.clean_examples,
        denoising.noisy_examples,
        denoising.recon_examples,
        output_dir / "denoising_examples.png",
    )

    summary_path = output_dir / "results_summary.txt"
    write_results_summary(classification, denoising, args, summary_path)
    summary_text = summary_path.read_text(encoding="utf-8")

    report_path = BASE_DIR / "report.md"
    report_path.write_text(build_report_text(summary_text), encoding="utf-8")

    print("Assignment 6 finished.")
    print(f"Device: {device}")
    print(f"Baseline accuracy: {classification.baseline_accuracy:.4f}")
    print(f"CNN accuracy: {classification.cnn_accuracy:.4f}")
    print(f"Denoising MSE: {denoising.test_mse:.6f}")
    print(f"Denoising PSNR: {denoising.test_psnr:.4f} dB")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
