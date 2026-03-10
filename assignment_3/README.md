# Assignment 3 Notes: Pose Estimation with ML + MAP

## Goals (from README)
- Implement **recent ML** and **Bayesian (MAP)** estimation algorithms for a **CV/PR application**.
- Use AI tools for algorithm search, programming, and report writing.
- Submit a report describing how the assignment was completed.

## Chosen application and dataset
- **Application:** 6D pose estimation from 2D keypoints.
- **Dataset:** **LINEMOD**, **single object** subset (e.g., ape).
- **Schema:** Use the existing `assignment_3/src/data_loader.py` format.

## Expected dataset layout (data_loader.py)
```
root/
  images/
    000001.png
    000002.png
  annotations.json
```

`annotations.json` schema:
```json
[
  {
    "image": "images/000001.png",
    "keypoints": [[x1, y1], [x2, y2], ...],
    "pose": {
      "R": [[...],[...],[...]],
      "t": [tx, ty, tz]
    }
  }
]
```

**Decision:** store 3D keypoints **in `annotations.json`** (per-sample), along with 2D keypoints and pose.

## Implementation plan (high level)
1. **Dataset preparation (LINEMOD → schema conversion)**
   - Provide download steps (no code execution).
   - Convert to the `images/` + `annotations.json` format.
   - Include 3D keypoints in each annotation.

2. **ML estimator (Maximum Likelihood)**
   - Objective: minimize reprojection error (Gaussian noise).
   - Use **OpenCV PnP + RANSAC** for pose estimation.
   - Compare baseline vs refined solution.

3. **Bayesian estimator (MAP)**
   - Add Gaussian prior on pose parameters.
   - Solve MAP by minimizing reprojection error + prior penalty.
   - Compare MAP vs ML under added noise.

4. **Evaluation + visualization**
   - Metrics: reprojection error, rotation error, translation error.
   - Visuals: error curves, projected keypoint overlays.
   - Save outputs to `assignment_3/img/` for report inclusion.

5. **Report structure (follow assignment_2 format)**
   1) Task Description
   2) AI‑Assisted Workflow
   3) Method Design (dataset, ML objective, MAP objective)
   4) Program Pipeline
   5) Results & Visualizations
   6) Analysis
   7) Limitations & Future Work
   8) Conclusion

## Repo context
- `assignment_3/src/data_loader.py` provides the dataset schema and loader.
- `assignment_2/README.md` shows the preferred report format and structure.
- `assignment_2/main.py` demonstrates modular pipeline design and saving figures to `img/`.

## Open items to resolve when implementing
- Confirm LINEMOD download path and local folder names.
- Decide whether to add a conversion script or adapt the loader.
- Confirm availability of OpenCV (cv2) in the environment.
