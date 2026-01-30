# M‑SVDD TurtleBot Experiment

Date: 2026-01-20

## Started with
- Paper: https://arxiv.org/abs/2505.05811
- Reference implementation: https://github.com/jamesyang7/M-SVDD

## Summary of what I implemented
- Implemented a ROS2-based anomaly detection pipeline using two approaches:
  - SKLearn baseline: `OneClassSVM` (classical SVDD-like behavior).
  - PyTorch Mahalanobis‑SVDD (learned precision matrix, hinge-style loss).
- Implemented a ROS2 inference node that consumes `cmd_vel` and `imu` feature windows, computes anomaly scores, and publishes results.

## Simulation & data
- Data source: TurtleBot simulation (teleop driving). Topics recorded: `/cmd_vel` and `/imu`.
- Feature extraction: sliding-window statistics computed from aligned `/cmd_vel` and `/imu` windows (mean, std, correlations, derivatives). Note the feature vector dimensionality — ensure recorder/trainer use the same extractor implementation.

Example: record features (produces `features.npz` containing `X`):
```bash
python3 record_features.py --duration 120 --rate 10 --window-size 10 --out features.npz
```

## Training
- SKLearn OneClassSVM (example):
```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib, numpy as np
X = np.load('features.npz')['X']
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
oc = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
oc.fit(Xs)
joblib.dump({'model': oc, 'scaler': scaler}, 'ocsvm_joblib.pkl')
```

- PyTorch M‑SVDD (CPU-safe run):
```bash
CUDA_VISIBLE_DEVICES="" python3 train_ms_svdd.py --features features.npz --out msvdd_model.pt
```

Notes: I observed a feature-dimension mismatch during runtime (training features had 13 dims, runtime extractor produced 20). To fix: re-run `record_features.py` with the exact extractor used by the node, then retrain so saved scaler matches runtime extractor dimensions.

## Inference / ROS2 node
- Run the node (loads scaler + model, subscribes to `/cmd_vel` and `/imu`):
```bash
python3 ms_svdd_node.py msvdd_config.yaml
```
- Published topics:
  - `/ms_svdd/anomaly` (std_msgs/Bool) — True when a window is flagged anomalous.
  - `/ms_svdd/anomaly_score` (std_msgs/Float32) — continuous anomaly score (higher = more anomalous).

## Evaluation & testing
- Offline: load `features.npz` and compute scores with saved models; plot time series and overlay collision/impact events.
- Metrics: choose threshold on validation normal data, compute precision/recall/F1 and ROC/AUC against labeled collision intervals.

## Repro commands (summary)
```bash
# 1) record features during normal teleop
python3 record_features.py --duration 120 --rate 10 --window-size 10 --out features.npz

# 2) train M-SVDD (CPU)
CUDA_VISIBLE_DEVICES="" python3 train_ms_svdd.py --features features.npz --out msvdd_model.pt

# 3) run inference node
python3 ms_svdd_node.py msvdd_config.yaml
```

## Notes & next steps
- Ensure the feature extractor used when recording exactly matches the node's extractor; mismatch causes scaler errors at runtime.
- If using GPU, verify PyTorch/CUDA compatibility; otherwise force CPU via `CUDA_VISIBLE_DEVICES=""` or `device='cpu'` in code.
- Next: collect a balanced set of normal runs, retrain M‑SVDD, run inference during teleop and validate detections against collision events.

---