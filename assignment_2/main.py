import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tarfile
import requests
from skimage.feature import hog
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


# ============================
# 1. Dataset Module (CIFAR-10)
# ============================
class CIFAR10Manager:
    """Download and manage the CIFAR-10 dataset."""

    CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, data_dir="data_cifar10"):
        self.data_dir = data_dir
        self.cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")

    def download_and_extract(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if os.path.exists(self.cifar_dir):
            print("[Info] CIFAR-10 dataset found. Ready to run.")
            return

        archive_path = os.path.join(self.data_dir, "cifar-10-python.tar.gz")
        print("[Info] Downloading CIFAR-10 dataset (~170 MB)...")
        response = requests.get(self.CIFAR10_URL, stream=True)
        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("[Info] Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir)
        print("[Info] Done.")

    def _load_batch(self, filepath):
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b'labels'])
        return images, labels

    def load_data(self):
        """Load all training and test data."""
        train_images, train_labels = [], []
        for i in range(1, 6):
            imgs, lbls = self._load_batch(os.path.join(self.cifar_dir, f"data_batch_{i}"))
            train_images.append(imgs)
            train_labels.append(lbls)

        train_images = np.concatenate(train_images)
        train_labels = np.concatenate(train_labels)

        test_images, test_labels = self._load_batch(os.path.join(self.cifar_dir, "test_batch"))

        return train_images, train_labels, test_images, test_labels


# ============================
# 2. Feature Extraction Module
# ============================
class FeatureExtractor:
    """Extract features from images for classification."""

    @staticmethod
    def extract_hog(images):
        """Extract HOG features from a batch of images."""
        features = []
        for img in images:
            gray = np.mean(img, axis=2).astype(np.uint8)
            feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
            features.append(feat)
        return np.array(features)

    @staticmethod
    def extract_color_histogram(images, bins=32):
        """Extract color histogram features from a batch of images."""
        features = []
        for img in images:
            hist_r = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))[0]
            hist_g = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))[0]
            hist_b = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))[0]
            hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
            hist = hist / (hist.sum() + 1e-7)  # normalize
            features.append(hist)
        return np.array(features)

    @staticmethod
    def extract_combined(images):
        """Combine HOG and color histogram features."""
        hog_feats = FeatureExtractor.extract_hog(images)
        color_feats = FeatureExtractor.extract_color_histogram(images)
        return np.hstack([hog_feats, color_feats])


# ============================
# 3. Bayes Classifier Module
# ============================
class BayesClassifier:
    """Wrapper for different Naive Bayes classifiers."""

    def __init__(self, method='gaussian'):
        if method == 'gaussian':
            self.model = GaussianNB()
        elif method == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=CIFAR10Manager.CLASS_NAMES,
                                       output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return acc, report, cm, y_pred


# ============================
# 4. Visualization Module
# ============================
class Visualizer:

    @staticmethod
    def plot_confusion_matrix(cm, class_names, title, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=14)
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7)

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    @staticmethod
    def plot_accuracy_comparison(results, save_path=None):
        """Bar chart comparing accuracy of different configurations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(results.keys())
        accs = [results[k]['accuracy'] * 100 for k in names]

        bars = ax.bar(names, accs, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Naive Bayes Classification Accuracy on CIFAR-10', fontsize=14)
        ax.set_ylim(0, max(accs) + 10)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    @staticmethod
    def plot_per_class_accuracy(report, class_names, title, save_path=None):
        """Bar chart showing per-class precision/recall/f1."""
        precisions = [report[c]['precision'] for c in class_names]
        recalls = [report[c]['recall'] for c in class_names]
        f1s = [report[c]['f1-score'] for c in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision', color='#4C72B0')
        ax.bar(x, recalls, width, label='Recall', color='#55A868')
        ax.bar(x + width, f1s, width, label='F1-Score', color='#C44E52')

        ax.set_ylabel('Score')
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    @staticmethod
    def plot_sample_predictions(images, true_labels, pred_labels, class_names, save_path=None):
        """Show sample predictions with correct/incorrect labels."""
        fig, axes = plt.subplots(3, 6, figsize=(14, 7))
        fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14)

        indices = np.random.choice(len(images), 18, replace=False)
        for i, idx in enumerate(indices):
            row, col = i // 6, i % 6
            axes[row, col].imshow(images[idx])
            axes[row, col].axis('off')
            correct = true_labels[idx] == pred_labels[idx]
            color = 'green' if correct else 'red'
            axes[row, col].set_title(
                f"T:{class_names[true_labels[idx]]}\nP:{class_names[pred_labels[idx]]}",
                fontsize=8, color=color
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')


# ============================
# 5. Main Program
# ============================
def main():
    np.random.seed(42)

    # Create output directory
    img_dir = "img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # --- Step 1: Download & load dataset ---
    dm = CIFAR10Manager()
    dm.download_and_extract()
    train_images, train_labels, test_images, test_labels = dm.load_data()
    print(f"[Info] Training set: {train_images.shape}, Test set: {test_images.shape}")

    # Use a subset for faster experimentation
    n_train = 10000
    n_test = 2000
    train_idx = np.random.choice(len(train_images), n_train, replace=False)
    test_idx = np.random.choice(len(test_images), n_test, replace=False)

    train_imgs = train_images[train_idx]
    train_lbls = train_labels[train_idx]
    test_imgs = test_images[test_idx]
    test_lbls = test_labels[test_idx]

    # --- Step 2: Extract features ---
    print("[Info] Extracting HOG features...")
    train_hog = FeatureExtractor.extract_hog(train_imgs)
    test_hog = FeatureExtractor.extract_hog(test_imgs)

    print("[Info] Extracting color histogram features...")
    train_color = FeatureExtractor.extract_color_histogram(train_imgs)
    test_color = FeatureExtractor.extract_color_histogram(test_imgs)

    print("[Info] Extracting combined features...")
    train_combined = np.hstack([train_hog, train_color])
    test_combined = np.hstack([test_hog, test_color])

    # --- Step 3: Train & evaluate different configurations ---
    experiments = {
        'GaussianNB + HOG': ('gaussian', train_hog, test_hog),
        'GaussianNB + Color': ('gaussian', train_color, test_color),
        'GaussianNB + Combined': ('gaussian', train_combined, test_combined),
        'BernoulliNB + HOG': ('bernoulli', train_hog, test_hog),
        'BernoulliNB + Color': ('bernoulli', train_color, test_color),
        'BernoulliNB + Combined': ('bernoulli', train_combined, test_combined),
    }

    results = {}
    best_acc = 0
    best_name = ""

    for name, (method, X_train, X_test) in experiments.items():
        print(f"[Info] Running: {name}...")
        clf = BayesClassifier(method=method)
        clf.train(X_train, train_lbls)
        acc, report, cm, y_pred = clf.evaluate(X_test, test_lbls)
        results[name] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
        }
        print(f"       Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name

    print(f"\n[Result] Best configuration: {best_name} with accuracy {best_acc:.4f}")

    # --- Step 4: Visualization ---
    print("[Info] Generating plots...")

    # Accuracy comparison bar chart
    Visualizer.plot_accuracy_comparison(results, save_path=os.path.join(img_dir, "accuracy_comparison.png"))

    # Confusion matrix for best model
    best_cm = results[best_name]['confusion_matrix']
    Visualizer.plot_confusion_matrix(
        best_cm, CIFAR10Manager.CLASS_NAMES,
        f"Confusion Matrix - {best_name}",
        save_path=os.path.join(img_dir, "confusion_matrix.png")
    )

    # Per-class metrics for best model
    best_report = results[best_name]['report']
    Visualizer.plot_per_class_accuracy(
        best_report, CIFAR10Manager.CLASS_NAMES,
        f"Per-Class Metrics - {best_name}",
        save_path=os.path.join(img_dir, "per_class_metrics.png")
    )

    # Sample predictions
    best_preds = results[best_name]['predictions']
    Visualizer.plot_sample_predictions(
        test_imgs, test_lbls, best_preds, CIFAR10Manager.CLASS_NAMES,
        save_path=os.path.join(img_dir, "sample_predictions.png")
    )

    print(f"[Info] All plots saved to '{img_dir}/' directory.")
    plt.show()


if __name__ == "__main__":
    main()
