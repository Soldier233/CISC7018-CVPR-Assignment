# Assignment 3 Report: Pose Estimation with ML + MAP

## 1. Task Description
The assignment requires:
- implementing **maximum likelihood (ML)** and **Bayesian (MAP)** estimation algorithms,
- selecting a **CV/PR application** and dataset,
- completing most steps using AI assistance,
- and submitting a report that documents the workflow.

This project focuses on **6D pose estimation from 2D keypoints** on a LINEMOD-style dataset.

---

## 2. AI-Assisted Workflow
1. Selected pose estimation as the target application and identified ML/MAP formulations.
2. Designed a compact dataset schema based on LINEMOD.
3. Implemented PnP-based ML estimation and a MAP refinement with Gaussian priors.
4. Added evaluation metrics and visualization outputs.
5. Wrote this report based on the implemented pipeline.

My own role was to:
- choose the dataset scope and constraints,
- review the generated code structure and results,
- and verify output paths and reproducibility settings.

---

## 3. Method Design

### 3.1 Dataset
**Target dataset:** LINEMOD single-object subset (e.g., `ape`), converted into:
```
root/
  images/
    000001.png
    000002.png
  annotations.json
  camera.json  (optional)
```

Each annotation stores per-image 2D/3D keypoints and the ground-truth pose:
```json
{
  "image": "images/000001.png",
  "keypoints_2d": [[x1, y1], [x2, y2], ...],
  "keypoints_3d": [[X1, Y1, Z1], [X2, Y2, Z2], ...],
  "pose": {"R": [[...],[...],[...]], "t": [tx, ty, tz]},
  "camera": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
}
```

For portability, the project also provides a **toy dataset generator** that produces a small LINEMOD-style dataset with synthetic poses and keypoints.

### 3.2 ML Estimator
ML estimation minimizes the reprojection error of 3D points onto 2D observations under Gaussian noise:

**Step 1:** `cv2.solvePnPRansac` (robust baseline).  
**Step 2:** `cv2.solvePnP` refinement (nonlinear least squares).

### 3.3 MAP Estimator
MAP adds a Gaussian prior on pose parameters (Rodrigues + translation):

```
E_MAP = (1/sigma^2) || reprojection_error ||^2
        + || (pose - mu) / sigma_prior ||^2
```

The prior mean and variance are estimated from the training split. A Gauss-Newton solver with damping refines the pose.

---

## 4. Program Pipeline
Main workflow in `assignment_3/main.py`:
1. Load (or generate) dataset in `annotations.json` format.
2. Split into train/validation.
3. Estimate ML and MAP poses under different noise levels.
4. Compute reprojection, rotation, and translation errors.
5. Save plots and overlay visualization to `assignment_3/img/`.

---

## 5. Results & Visualizations
Running the program generates:
- `img/reprojection_error.png`
- `img/rotation_error.png`
- `img/translation_error.png`
- `img/overlay.png`

These plots compare ML-RANSAC, ML-Refined, and MAP across noise levels, and visualize keypoint projections on a sample image.

---

## 6. Analysis
- **ML-RANSAC** is robust to outliers but can be less accurate than refined estimates.
- **ML-Refined** improves accuracy when the initial solution is good.
- **MAP** is more stable under higher noise by leveraging the pose prior.
- The prior helps especially in under-constrained or noisy cases, reducing jitter in translation/rotation.

---

## 7. Limitations and Future Work
Current limitations:
1. The MAP optimization uses a numerical Jacobian, which is slower than an analytic one.
2. The toy dataset is simplified compared with real LINEMOD scenes.
3. The prior is a simple diagonal Gaussian in Rodrigues space.

Future improvements:
- use analytic Jacobians for faster MAP refinement,
- incorporate real LINEMOD keypoints and object meshes,
- compare with learning-based pose estimators.

---

## 8. Conclusion
This project delivers a complete **ML + MAP pose estimation pipeline**:
- LINEMOD-style dataset schema,
- PnP + RANSAC ML estimation,
- Gaussian-prior MAP refinement,
- evaluation metrics and visualization outputs.

The framework is lightweight and can be extended to real LINEMOD data or other 6D pose estimation benchmarks.

---

## Usage

### Option A: Generate a Toy Dataset
```bash
cd assignment_3
python main.py --generate-toy --data-root data_toy --num-samples 40
```

### Option B: Use a Custom LINEMOD Subset
1. Download the LINEMOD dataset from its official source.
2. Select a single object (e.g., `ape`) and gather the RGB images.
3. Choose a small set of 3D keypoints on the object model.
4. For each image, pair 2D keypoints with their 3D correspondences and the ground-truth pose.
5. Save to `annotations.json` using the schema above (optionally add `camera.json`).
6. Run:
```bash
cd assignment_3
python main.py --data-root path/to/linemod_subset
```
