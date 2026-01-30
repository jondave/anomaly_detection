# SVDD Anomaly Detection Methods

This repository contains Python implementations of three anomaly detection methods using SVDD (Support Vector Data Description) approaches:

1. **ae_autoencoder**: Autoencoder-based anomaly detection
2. **ms_svdd**: Multi-Scale SVDD for anomaly detection
3. **svm_svdd**: SVM-based SVDD for anomaly detection

## Repository Structure

```
.
├── ae_autoencoder/          # Autoencoder-based anomaly detection
│   ├── ae_autoencoder.py    # Autoencoder model implementation
│   ├── train_autoencoder.py # Training script
│   └── eval_autoencoder.py  # Evaluation script
│
├── ms_svdd/                 # Multi-Scale SVDD
│   ├── ms_svdd_model.py     # MS-SVDD model implementation
│   ├── train_ms_svdd.py     # Training script
│   ├── msvdd_config.yaml    # Configuration file
│   ├── msvdd_experiment.md  # Experiment documentation
│   ├── msvdd_model.pt       # Trained model file
│   └── msvdd_model_cmd_vel_imu.pt  # Alternative trained model
│
├── svm_svdd/                # SVM-based SVDD
│   ├── svm_svdd_model.py    # SVM-SVDD model implementation
│   ├── train_svm_svdd.py    # Training script
│   └── eval_svm_svdd.py     # Evaluation script
│
├── examples/                # Example scripts
│   └── generate_sample_data.py  # Generate sample data for testing
│
├── features.py              # Feature extraction utilities
├── svdd_model.py            # Base SVDD model
├── train.py                 # General training utilities
├── eval_ms_svdd.py          # MS-SVDD evaluation
├── features.npz             # Extracted features dataset
├── features_cmd_vel_imu.npz # Alternative features dataset
└── requirements.txt         # Python dependencies
```

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Methods Overview

### 1. Autoencoder (ae_autoencoder)

Autoencoder-based anomaly detection that learns to reconstruct normal patterns and flags deviations as anomalies.

- **Train**: `python ae_autoencoder/train_autoencoder.py`
- **Evaluate**: `python ae_autoencoder/eval_autoencoder.py`

### 2. Multi-Scale SVDD (ms_svdd)

Multi-scale Support Vector Data Description for anomaly detection with multiple resolution levels.

- **Train**: `python ms_svdd/train_ms_svdd.py`
- **Pre-trained models**: 
  - `msvdd_model.pt`
  - `msvdd_model_cmd_vel_imu.pt`
- **Configuration**: See `ms_svdd/msvdd_config.yaml`
- **Documentation**: See `ms_svdd/msvdd_experiment.md`

### 3. SVM SVDD (svm_svdd)

Classic SVM-based Support Vector Data Description using OneClassSVM.

- **Train**: `python svm_svdd/train_svm_svdd.py`
- **Evaluate**: `python svm_svdd/eval_svm_svdd.py`

## Data Files

- `features.npz`: Pre-extracted features for training and evaluation
- `features_cmd_vel_imu.npz`: Alternative feature dataset with command velocity and IMU data

## Feature Extraction

The `features.py` module provides utilities for extracting features from time-series data. Features include:
- Statistical measures (mean, std, min, max)
- Temporal features from sliding windows
- Cross-features capturing relationships between variables

## Examples

The `examples/` directory contains scripts for generating sample data:

```bash
python examples/generate_sample_data.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Tax, D. M., & Duin, R. P. (2004). Support vector data description. Machine learning, 54(1), 45-66.
- Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. Neural computation, 13(7), 1443-1471.
