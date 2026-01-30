#!/usr/bin/env python3
"""Train OneClassSVM (SVDD proxy) on features.npz and save artifacts.

Saves:
    - model: `svm_model.pkl`
    - scaler: `svm_scaler.pkl`
"""
import argparse
import os
import numpy as np
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

try:
    from svm_svdd.svm_svdd_model import SVM_SVDDModel
except Exception:
    # When running the script directly from the package folder, import the local module
    from svm_svdd_model import SVM_SVDDModel
from ros2_svdd_monitor.features import extract_features_from_dataframe


def load_features(path: str):
    # Support .npz (precomputed feature arrays) or CSV/Parquet raw logs
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        data = np.load(path)
        if 'X' in data:
            return data['X']
        return data[data.files[0]]
    elif ext in ('.csv', '.parquet', '.pq'):
        # load raw dataframe and extract sliding-window features (includes odom if present)
        if ext == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        # default window_size 10; allow caller to override via environment or later CLI
        return extract_features_from_dataframe(df, window_size=10)
    else:
        # try numpy loader as fallback
        data = np.load(path)
        if 'X' in data:
            return data['X']
        return data[data.files[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', '-f', required=True)
    p.add_argument('--out', '-o', default='svm_out')
    p.add_argument('--nu', type=float, default=0.1)
    p.add_argument('--gamma', default='scale')
    p.add_argument('--kernel', default='rbf')
    args = p.parse_args()

    X = load_features(args.features)
    print(f"Loaded features shape: {X.shape}")

    # simple split to ensure code path parity with others (not used for fitting here)
    X_train, X_val = train_test_split(X, test_size=0.05, random_state=42)

    model = SVM_SVDDModel(nu=args.nu, gamma=args.gamma, kernel=args.kernel)
    model.fit(X_train, scale=True)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, 'svm_model.pkl')
    scaler_path = os.path.join(args.out, 'svm_scaler.pkl')
    model.save(model_path, scaler_path)

    print(f"Saved model to {model_path} and scaler to {scaler_path}")


if __name__ == '__main__':
    main()
