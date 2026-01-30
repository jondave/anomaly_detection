#!/usr/bin/env python3
"""
Train SVDD Model CLI

This script trains an SVDD model (OneClassSVM) on normal operation data
exported from rosbag to CSV format.

Usage:
    ros2 run ros2_svdd_monitor train --csv <path_to_csv> --config <path_to_config>
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from ros2_svdd_monitor.svdd_model import SVDDModel
from ros2_svdd_monitor.features import extract_features_from_dataframe, get_feature_names


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(csv_path):
    """Load training data from CSV file."""
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    return df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train SVDD model for anomaly detection'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with training data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: searches for config/config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save trained model and scaler'
    )
    
    args = parser.parse_args()
    
    # Find config file
    if args.config is None:
        # Search for config in common locations
        possible_paths = [
            'config/config.yaml',
            '../config/config.yaml',
            os.path.join(os.path.dirname(__file__), '../config/config.yaml'),
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            print("Error: Could not find config.yaml. Please specify with --config")
            sys.exit(1)
    else:
        config_path = args.config
    
    print(f"Using config: {config_path}")
    config = load_config(config_path)
    
    # Load training data
    df = load_training_data(args.csv)
    
    # Extract features
    print(f"\nExtracting features with window_size={config['window_size']}...")
    try:
        features = extract_features_from_dataframe(df, window_size=config['window_size'])
    except ValueError as e:
        print(f"Error extracting features: {e}")
        print("\nExpected CSV columns:")
        print("  timestamp, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z,")
        print("  accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z")
        sys.exit(1)
    
    print(f"Extracted {features.shape[0]} feature vectors with {features.shape[1]} features each")
    
    # Print feature statistics
    feature_names = get_feature_names()
    print("\nFeature statistics:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={features[:, i].mean():.4f}, std={features[:, i].std():.4f}")
    
    # Initialize and train model
    print(f"\nTraining SVDD model (OneClassSVM)...")
    print(f"  nu={config['nu']}")
    print(f"  gamma={config['gamma']}")
    
    model = SVDDModel(
        nu=config['nu'],
        gamma=config['gamma']
    )
    
    model.fit(features, scale=config.get('feature_scaling', True))
    print("Training completed!")
    
    # Evaluate on training data
    predictions = model.predict(features, scale=config.get('feature_scaling', True))
    n_outliers = np.sum(predictions == -1)
    outlier_ratio = n_outliers / len(predictions)
    
    print(f"\nTraining set evaluation:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Outliers detected: {n_outliers} ({outlier_ratio*100:.2f}%)")
    print(f"  Inliers: {len(predictions) - n_outliers} ({(1-outlier_ratio)*100:.2f}%)")

    # Compute decision function scores on training data and suggest a threshold
    scores = model.decision_function(features, scale=config.get('feature_scaling', True))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))

    enter_pct = float(config.get('enter_threshold_percentile', 1.0))
    exit_pct = float(config.get('exit_threshold_percentile', max(enter_pct, 5.0)))
    enter_threshold = float(np.percentile(scores, enter_pct))
    exit_threshold = float(np.percentile(scores, exit_pct))
    print(f"\nDecision function stats on training set:")
    print(f"  mean={score_mean:.3f}, std={score_std:.3f}, min={score_min:.3f}, max={score_max:.3f}")
    print(f"  Recommended enter threshold @ p{enter_pct:.1f} = {enter_threshold:.3f}")
    print(f"  Recommended exit  threshold @ p{exit_pct:.1f} = {exit_threshold:.3f}")
    
    # Save model and scaler
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = os.path.join(args.output_dir, os.path.basename(os.path.expanduser(config['model_path'])))
    scaler_path = os.path.join(args.output_dir, os.path.basename(os.path.expanduser(config['scaler_path'])))
    
    print(f"\nSaving model to {model_path}...")
    model.save(model_path, scaler_path)

    # Save recommended threshold alongside the model for the monitor to consume
    try:
        import yaml
        threshold_path = os.path.join(args.output_dir, config.get('threshold_path', 'threshold.yaml'))
        threshold_payload = {
            'threshold': enter_threshold,  # backward compatibility
            'enter_threshold': enter_threshold,
            'exit_threshold': exit_threshold,
            'enter_percentile': enter_pct,
            'exit_percentile': exit_pct,
            'score_stats': {
                'mean': score_mean,
                'std': score_std,
                'min': score_min,
                'max': score_max,
            },
            'nu': config['nu'],
            'gamma': config['gamma'],
            'feature_scaling': bool(config.get('feature_scaling', True)),
            'window_size': int(config['window_size']),
        }
        with open(threshold_path, 'w') as f:
            yaml.safe_dump(threshold_payload, f)
        print(f"Saved threshold to: {threshold_path}")
    except Exception as e:
        print(f"Warning: failed to save threshold: {e}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Threshold saved to: {os.path.join(args.output_dir, config.get('threshold_path', 'threshold.yaml'))}")
    print("\nYou can now run the monitor with:")
    print(f"  ros2 run ros2_svdd_monitor monitor")


if __name__ == '__main__':
    main()
