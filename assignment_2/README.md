# Assignment 2 Report: Bayes Classification on CIFAR-10

## 1. Task Description
The assignment requires:
- finding and implementing a **Bayes classification algorithm** for a classification application,
- using an AI tool for most of the workflow (algorithm search, coding, and report writing),
- and submitting a report describing the process.

In this project, I selected **CIFAR-10 image classification** and used **Naive Bayes** as the core method.

---

## 2. AI-Assisted Workflow
Following the assignment requirement, I completed the work mainly with AI support:
1. Clarified the task goal and selected a feasible direction (image classification with Bayes methods).
2. Generated the overall code structure (dataset module, feature extraction module, classifier module, and visualization module).
3. Iteratively refined the experiment pipeline and output figures.
4. Generated this report based on the implemented code and produced outputs.

My own role was to:
- provide requirements and constraints,
- verify that the code structure meets course expectations,
- run the program and confirm outputs are generated correctly.

---

## 3. Method Design

### 3.1 Dataset
- Dataset: **CIFAR-10** (10 classes, 32×32 color images)
- Full training set: 50,000 images
- Full test set: 10,000 images
- For faster experimentation in this project:
  - Training subset: 10,000
  - Test subset: 2,000

### 3.2 Feature Engineering
The program uses three feature representations:
1. **HOG features** (shape/edge information)
2. **Color histogram features** (color distribution)
3. **Combined features** (HOG + color histogram)

### 3.3 Classifiers
Two Naive Bayes variants are compared:
- **GaussianNB**
- **BernoulliNB**

Each classifier is tested with three feature sets (6 experiments total):
- GaussianNB + HOG
- GaussianNB + Color
- GaussianNB + Combined
- BernoulliNB + HOG
- BernoulliNB + Color
- BernoulliNB + Combined

---

## 4. Program Pipeline
The main workflow in `main.py` is:
1. Download/load CIFAR-10.
2. Sample training/testing subsets.
3. Extract three types of features.
4. Train and evaluate 6 Bayes configurations.
5. Select the best-performing configuration.
6. Generate visualization outputs.

---

## 5. Experimental Results and Visualizations
The program generates the following files in `assignment_2/img/`:
- `accuracy_comparison.png`: accuracy comparison across model/feature configurations
- `confusion_matrix.png`: confusion matrix of the best model
- `per_class_metrics.png`: per-class Precision / Recall / F1
- `sample_predictions.png`: sample predictions (green = correct, red = wrong)

### Accuracy Comparison
![Accuracy Comparison](img/accuracy_comparison.png)

### Confusion Matrix (Best Model)
![Confusion Matrix](img/confusion_matrix.png)

### Per-Class Metrics
![Per-Class Metrics](img/per_class_metrics.png)

### Sample Predictions
![Sample Predictions](img/sample_predictions.png)

Main observations:
1. **Feature representation strongly affects Bayes classification performance**.
2. **Combined features (HOG + Color) are generally more stable than single-feature inputs**.
3. On CIFAR-10, Naive Bayes works as a useful baseline, but it is still limited compared with stronger models (e.g., CNNs).

---

## 6. Analysis
- **HOG** captures structural/edge cues and helps with classes that differ by shape.
- **Color histograms** provide complementary color-distribution information but ignore spatial layout.
- **Combined features** merge shape and color cues, which improves overall representation.
- **BernoulliNB** is more suitable for binary/discrete inputs, so it may be less compatible with continuous image descriptors.
- **GaussianNB** is naturally aligned with continuous features and is typically more suitable in this setup.

---

## 7. Limitations and Future Improvements
Current limitations:
1. Only Naive Bayes is used, so model capacity is limited.
2. A subset is used for speed, which restricts final performance.
3. Features are handcrafted rather than learned end-to-end.

Possible improvements:
- train on the full dataset and perform parameter tuning,
- add more descriptors (e.g., LBP, SIFT Bag-of-Words),
- compare systematically with Logistic Regression, SVM, and Random Forest,
- further test CNN-based methods and compare them with traditional features + Bayes.

---

## 8. Conclusion
This assignment delivers a complete **Bayes-based image classification pipeline**:
- CIFAR-10 as the benchmark dataset,
- HOG / color histogram / combined feature extraction,
- GaussianNB and BernoulliNB comparison,
- visualization of accuracy, confusion matrix, per-class metrics, and sample predictions.

The results show that **Naive Bayes is a clear and low-cost baseline for image classification experiments**, making it suitable as a starting point for method comparison in coursework.