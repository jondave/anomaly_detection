"""
SVDD Model Wrapper using OneClassSVM

This module provides a wrapper for sklearn's OneClassSVM to use as an SVDD
(Support Vector Data Description) proxy for anomaly detection.
"""

import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class SVDDModel:
    """
    Wrapper for OneClassSVM to perform SVDD-based anomaly detection.
    """

    def __init__(self, nu=0.1, gamma='scale', kernel='rbf'):
        """
        Initialize the SVDD model.

        Args:
            nu: Upper bound on fraction of training errors and lower bound
                on fraction of support vectors (0 < nu <= 1)
            gamma: Kernel coefficient for RBF kernel
            kernel: Kernel type to be used in the algorithm
        """
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, scale=True):
        """
        Train the SVDD model on the provided data.

        Args:
            X: Training data, shape (n_samples, n_features)
            scale: Whether to apply feature scaling

        Returns:
            self
        """
        X = np.array(X)
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, X, scale=True):
        """
        Predict if samples are inliers (1) or outliers (-1).

        Args:
            X: Data to predict, shape (n_samples, n_features)
            scale: Whether to apply feature scaling

        Returns:
            Array of predictions: 1 for inliers, -1 for outliers
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        X = np.array(X)
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict(X_scaled)

    def decision_function(self, X, scale=True):
        """
        Compute the decision function (signed distance to separating hyperplane).

        Args:
            X: Data to evaluate, shape (n_samples, n_features)
            scale: Whether to apply feature scaling

        Returns:
            Array of decision function values (anomaly scores)
            Negative values indicate outliers
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        X = np.array(X)
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.decision_function(X_scaled)

    def save(self, model_path, scaler_path=None):
        """
        Save the trained model and scaler to disk.

        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler (optional)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        import os
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        joblib.dump(self.model, model_path)
        
        if scaler_path:
            scaler_dir = os.path.dirname(scaler_path)
            if scaler_dir:
                os.makedirs(scaler_dir, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        if scaler_path:
            print(f"Scaler saved to {scaler_path}")

    def load(self, model_path, scaler_path=None):
        """
        Load a trained model and scaler from disk.

        Args:
            model_path: Path to load the model from
            scaler_path: Path to load the scaler from (optional)
        """
        self.model = joblib.load(model_path)
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        self.is_fitted = True
        print(f"Model loaded from {model_path}")
        if scaler_path:
            print(f"Scaler loaded from {scaler_path}")

    def get_params(self):
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'nu': self.nu,
            'gamma': self.gamma,
            'kernel': self.kernel,
            'is_fitted': self.is_fitted
        }
