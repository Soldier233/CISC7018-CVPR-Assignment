from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class LearnerBundle:
    model: SVC
    scaler: StandardScaler
    feature_idx: np.ndarray
    feature_count: int


class SupportVectorMachineChains:
    def __init__(
        self,
        chain_size: int = 3,
        min_feature_ratio: float = 0.5,
        tournament_size: int = 3,
        kernel: str = "linear",
        c_value: float = 4.0,
        gamma: str | float = "scale",
        random_state: int = 42,
    ) -> None:
        self.chain_size = chain_size
        self.min_feature_ratio = min_feature_ratio
        self.tournament_size = max(2, tournament_size)
        self.kernel = kernel
        self.c_value = c_value
        self.gamma = gamma
        self.random_state = random_state

        self.mi_scores_: np.ndarray | None = None
        self.feature_order_: np.ndarray | None = None
        self.feature_counts_: list[int] = []
        self.learners_: list[LearnerBundle] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SupportVectorMachineChains":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape

        self.mi_scores_ = mutual_info_classif(x, y, random_state=self.random_state)
        self.feature_order_ = np.argsort(self.mi_scores_)[::-1]

        min_features = max(2, int(np.ceil(n_features * self.min_feature_ratio)))
        self.feature_counts_ = list(range(n_features, min_features - 1, -1))
        self.learners_.clear()

        for feature_count in self.feature_counts_:
            feature_idx = self.feature_order_[:feature_count]
            x_stage = x[:, feature_idx]

            for _ in range(self.chain_size):
                sample_idx = rng.integers(0, n_samples, size=n_samples)
                x_boot = x_stage[sample_idx]
                y_boot = y[sample_idx]

                scaler = StandardScaler()
                x_boot_std = scaler.fit_transform(x_boot)

                model = SVC(
                    kernel=self.kernel,
                    C=self.c_value,
                    gamma=self.gamma,
                    decision_function_shape="ovr",
                )
                model.fit(x_boot_std, y_boot)

                self.learners_.append(
                    LearnerBundle(
                        model=model,
                        scaler=scaler,
                        feature_idx=feature_idx.copy(),
                        feature_count=feature_count,
                    )
                )

        return self

    def _predict_all(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.learners_:
            raise RuntimeError("SVMC model is not fitted.")

        labels = []
        confidences = []
        stage_ids = []

        for learner in self.learners_:
            x_stage = learner.scaler.transform(x[:, learner.feature_idx])
            pred = learner.model.predict(x_stage)
            scores = learner.model.decision_function(x_stage)
            if scores.ndim == 1:
                conf = np.abs(scores)
            else:
                conf = np.max(scores, axis=1)

            labels.append(pred.astype(np.int64))
            confidences.append(conf.astype(np.float64))
            stage_ids.append(np.full(x.shape[0], learner.feature_count, dtype=np.int64))

        return (
            np.column_stack(labels),
            np.column_stack(confidences),
            np.column_stack(stage_ids),
        )

    @staticmethod
    def _pick_winner(labels: np.ndarray, confidences: np.ndarray) -> int:
        counts = Counter(labels.tolist())
        conf_by_label: dict[int, float] = defaultdict(float)
        for label, conf in zip(labels.tolist(), confidences.tolist()):
            conf_by_label[int(label)] += float(conf)

        winner = max(
            counts.keys(),
            key=lambda label: (counts[label], conf_by_label[label], -label),
        )
        return int(winner)

    def _majority_vote(self, labels: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        preds = np.empty(labels.shape[0], dtype=np.int64)
        for i in range(labels.shape[0]):
            preds[i] = self._pick_winner(labels[i], confidences[i])
        return preds

    def _tournament_vote(self, labels: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        preds = np.empty(labels.shape[0], dtype=np.int64)

        for i in range(labels.shape[0]):
            round_labels = labels[i].astype(np.int64)
            round_conf = confidences[i].astype(np.float64)

            while round_labels.shape[0] > 1:
                next_labels = []
                next_conf = []
                for start in range(0, round_labels.shape[0], self.tournament_size):
                    group_labels = round_labels[start : start + self.tournament_size]
                    group_conf = round_conf[start : start + self.tournament_size]

                    winner = self._pick_winner(group_labels, group_conf)
                    winner_conf = float(group_conf[group_labels == winner].sum())
                    next_labels.append(winner)
                    next_conf.append(winner_conf)

                round_labels = np.asarray(next_labels, dtype=np.int64)
                round_conf = np.asarray(next_conf, dtype=np.float64)

            preds[i] = int(round_labels[0])

        return preds

    def predict(self, x: np.ndarray, voting: str = "tournament") -> np.ndarray:
        labels, confidences, _ = self._predict_all(x)
        if voting == "majority":
            return self._majority_vote(labels, confidences)
        if voting == "tournament":
            return self._tournament_vote(labels, confidences)
        raise ValueError(f"Unsupported voting strategy: {voting}")

    def stage_statistics(self, x: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        labels, confidences, stage_ids = self._predict_all(x)
        stats: list[dict[str, float]] = []

        for feature_count in self.feature_counts_:
            mask = stage_ids[0] == feature_count
            stage_labels = labels[:, mask]
            stage_conf = confidences[:, mask]

            learner_acc = [
                accuracy_score(y, stage_labels[:, idx]) for idx in range(stage_labels.shape[1])
            ]
            stage_majority = self._majority_vote(stage_labels, stage_conf)
            stats.append(
                {
                    "feature_count": float(feature_count),
                    "mean_learner_acc": float(np.mean(learner_acc)),
                    "stage_ensemble_acc": float(accuracy_score(y, stage_majority)),
                }
            )

        return stats


def plot_accuracy_bars(scores: dict[str, float], save_path: Path) -> None:
    labels = list(scores.keys())
    values = [scores[key] for key in labels]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#34495e", "#16a085", "#d35400"])
    plt.ylim(max(0.0, min(values) - 0.03), min(1.0, max(values) + 0.02))
    plt.ylabel("Accuracy")
    plt.title("Digits Classification Accuracy")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_stage_curves(stage_stats: list[dict[str, float]], save_path: Path) -> None:
    feature_counts = [int(item["feature_count"]) for item in stage_stats]
    mean_learner_acc = [item["mean_learner_acc"] for item in stage_stats]
    stage_ensemble_acc = [item["stage_ensemble_acc"] for item in stage_stats]

    plt.figure(figsize=(7, 4.5))
    plt.plot(feature_counts, mean_learner_acc, marker="o", label="Mean single learner")
    plt.plot(feature_counts, stage_ensemble_acc, marker="s", label="Stage majority vote")
    plt.gca().invert_xaxis()
    plt.xlabel("Number of retained features")
    plt.ylabel("Accuracy")
    plt.title("SVMC Stage Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_mi_heatmap(mi_scores: np.ndarray, save_path: Path) -> None:
    heatmap = mi_scores.reshape(8, 8)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(heatmap, cmap="magma")
    plt.colorbar(label="Mutual information")
    plt.title("Mutual Information per Pixel")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6.2, 5.4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    n_samples: int = 12,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    idx = np.arange(min(n_samples, len(images)))

    for ax, i in zip(axes.flat, idx):
        ax.imshow(images[i], cmap="gray")
        color = "#1b9e77" if y_true[i] == y_pred[i] else "#d95f02"
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict[str, float | int]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    x = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    images = digits.images

    x_train, x_test, y_train, y_test, img_train, img_test = train_test_split(
        x,
        y,
        images,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    baseline_scaler = StandardScaler()
    x_train_std = baseline_scaler.fit_transform(x_train)
    x_test_std = baseline_scaler.transform(x_test)

    baseline = SVC(
        kernel=args.kernel,
        C=args.c_value,
        gamma=args.gamma,
        decision_function_shape="ovr",
    )
    baseline.fit(x_train_std, y_train)
    baseline_pred = baseline.predict(x_test_std)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    svmc = SupportVectorMachineChains(
        chain_size=args.chain_size,
        min_feature_ratio=args.min_feature_ratio,
        tournament_size=args.tournament_size,
        kernel=args.kernel,
        c_value=args.c_value,
        gamma=args.gamma,
        random_state=args.seed,
    ).fit(x_train, y_train)

    majority_pred = svmc.predict(x_test, voting="majority")
    tournament_pred = svmc.predict(x_test, voting="tournament")
    majority_acc = accuracy_score(y_test, majority_pred)
    tournament_acc = accuracy_score(y_test, tournament_pred)

    stage_stats = svmc.stage_statistics(x_test, y_test)
    tournament_cm = confusion_matrix(y_test, tournament_pred)

    plot_accuracy_bars(
        {
            "Single SVM": baseline_acc,
            "SVMC Majority": majority_acc,
            "SVMC Tournament": tournament_acc,
        },
        output_dir / "accuracy_comparison.png",
    )
    plot_stage_curves(stage_stats, output_dir / "stage_accuracy.png")
    plot_mi_heatmap(svmc.mi_scores_, output_dir / "mi_heatmap.png")
    plot_confusion_matrix(
        tournament_cm,
        output_dir / "confusion_matrix.png",
        "SVMC Tournament Voting Confusion Matrix",
    )
    plot_sample_predictions(
        img_test,
        y_test,
        tournament_pred,
        output_dir / "sample_predictions.png",
    )

    best_stage = max(stage_stats, key=lambda item: item["stage_ensemble_acc"])
    top_features = svmc.feature_order_[:10].tolist() if svmc.feature_order_ is not None else []

    summary = {
        "seed": args.seed,
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "baseline_acc": float(baseline_acc),
        "majority_acc": float(majority_acc),
        "tournament_acc": float(tournament_acc),
        "best_stage_features": int(best_stage["feature_count"]),
        "best_stage_acc": float(best_stage["stage_ensemble_acc"]),
    }

    summary_lines = [
        "Assignment 5: Support Vector Machine Chains (SVMC)",
        f"Random seed: {summary['seed']}",
        f"Train size: {summary['train_size']}",
        f"Test size: {summary['test_size']}",
        f"Single SVM accuracy: {summary['baseline_acc']:.4f}",
        f"SVMC majority-vote accuracy: {summary['majority_acc']:.4f}",
        f"SVMC tournament-vote accuracy: {summary['tournament_acc']:.4f}",
        f"Best stage feature count: {summary['best_stage_features']}",
        f"Best stage ensemble accuracy: {summary['best_stage_acc']:.4f}",
        f"Top-10 MI-ranked feature indices: {top_features}",
    ]
    (output_dir / "results_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Support Vector Machine Chains for handwritten digit recognition."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--chain-size", type=int, default=3)
    parser.add_argument("--min-feature-ratio", type=float, default=0.5)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "img"))
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    summary = run_experiment(args)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
