"""SVM / SVDD wrapper (OneClassSVM) for anomaly detection.

This is adapted from the project's top-level `svdd_model.py` into
the `svm_svdd` package to keep baselines isolated and consistent
with other model folders.
"""
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class SVM_SVDDModel:
    """Wrapper for sklearn OneClassSVM acting as an SVDD baseline."""

    def __init__(self, nu: float = 0.1, gamma: str = 'scale', kernel: str = 'rbf'):
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, scale: bool = True):
        X = np.array(X)
        if scale:
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = X
        self.model.fit(Xs)
        self.is_fitted = True
        return self

    def predict(self, X, scale: bool = True):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = np.array(X)
        Xs = self.scaler.transform(X) if scale else X
        return self.model.predict(Xs)

    def decision_function(self, X, scale: bool = True):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before decision_function")
        X = np.array(X)
        Xs = self.scaler.transform(X) if scale else X
        return self.model.decision_function(Xs)

    def save(self, model_path: str, scaler_path: Optional[str] = None):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, model_path)
        if scaler_path:
            scaler_dir = os.path.dirname(scaler_path)
            if scaler_dir:
                os.makedirs(scaler_dir, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        self.is_fitted = True

    def get_params(self):
        return {'nu': self.nu, 'gamma': self.gamma, 'kernel': self.kernel, 'is_fitted': self.is_fitted}
