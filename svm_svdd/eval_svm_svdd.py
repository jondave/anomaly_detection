#!/usr/bin/env python3
"""Evaluate a trained OneClassSVM model and save anomaly scores.

This script computes a per-sample anomaly score as ``-decision_function`` so
that larger values indicate more anomalous samples (consistent with other
baselines that produce higher=more anomalous).
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

try:
    from svm_svdd.svm_svdd_model import SVM_SVDDModel
except Exception:
    from svm_svdd_model import SVM_SVDDModel
from ros2_svdd_monitor.features import extract_features_from_dataframe


def load_features(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        data = np.load(path)
        if 'X' in data:
            return data['X']
        return data[data.files[0]]
    elif ext in ('.csv', '.parquet', '.pq'):
        if ext == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        return extract_features_from_dataframe(df, window_size=10)
    else:
        data = np.load(path)
        if 'X' in data:
            return data['X']
        return data[data.files[0]]


def evaluate(features_path: str, model_dir: str):
    X = load_features(features_path)
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    scaler_path = os.path.join(model_dir, 'svm_scaler.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = SVM_SVDDModel()
    model.load(model_path, scaler_path)

    df = model.decision_function(X, scale=True)
    # convert to anomaly score where higher == more anomalous
    scores = -df
    out = os.path.join(model_dir, 'svm_scores.npz')
    np.savez(out, scores=scores)
    print(f"Saved scores to {out}")
    print(f"scores: mean={scores.mean():.6f} std={scores.std():.6f} max={scores.max():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', '-f', required=True)
    p.add_argument('--model-dir', '-m', default='svm_out')
    args = p.parse_args()
    evaluate(args.features, args.model_dir)


if __name__ == '__main__':
    main()
