"""SVM/SVDD baseline package.

Provides a lightweight wrapper and scripts for training/evaluating
an sklearn OneClassSVM-based SVDD baseline.
"""

from .svm_svdd_model import SVM_SVDDModel

__all__ = ["SVM_SVDDModel"]
