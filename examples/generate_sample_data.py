#!/usr/bin/env python3
"""
Generate synthetic training data for testing the SVDD monitor.

This creates a CSV file with simulated robot operation data where
commanded velocities correlate with IMU measurements (normal operation).
"""

import pandas as pd
import numpy as np

def generate_sample_data(n_samples=100, output_file='example_training_data.csv'):
    """
    Generate synthetic normal operation data.
    
    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path
    """
    np.random.seed(42)
    
    # Generate commanded velocities
    linear_x = np.random.normal(1.0, 0.1, n_samples)  # Mean 1.0 m/s forward
    angular_z = np.random.normal(0.5, 0.05, n_samples)  # Mean 0.5 rad/s turning
    
    # Generate correlated IMU measurements
    # Forward acceleration should correlate with linear velocity command
    accel_x = linear_x * 2.0 + np.random.normal(0, 0.2, n_samples)
    
    # Gyro should correlate with angular velocity command
    gyro_z = angular_z * 2.0 + np.random.normal(0, 0.1, n_samples)
    
    data = {
        'timestamp': np.arange(1.0, n_samples + 1.0),
        'linear_x': linear_x,
        'linear_y': np.zeros(n_samples),
        'linear_z': np.zeros(n_samples),
        'angular_x': np.zeros(n_samples),
        'angular_y': np.zeros(n_samples),
        'angular_z': angular_z,
        'accel_x': accel_x,
        'accel_y': np.zeros(n_samples),
        'accel_z': np.zeros(n_samples),
        'gyro_x': np.zeros(n_samples),
        'gyro_y': np.zeros(n_samples),
        'gyro_z': gyro_z,
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {n_samples} samples of normal operation data")
    print(f"✓ Saved to: {output_file}")
    print(f"\nData summary:")
    print(f"  Linear velocity: mean={linear_x.mean():.2f}, std={linear_x.std():.2f}")
    print(f"  Angular velocity: mean={angular_z.mean():.2f}, std={angular_z.std():.2f}")
    print(f"  Forward accel: mean={accel_x.mean():.2f}, std={accel_x.std():.2f}")
    print(f"  Gyro Z: mean={gyro_z.mean():.2f}, std={gyro_z.std():.2f}")
    print(f"\nNext steps:")
    print(f"  1. Train the model:")
    print(f"     ros2 run ros2_svdd_monitor train --csv {output_file}")
    print(f"  2. Run the monitor:")
    print(f"     ros2 run ros2_svdd_monitor monitor")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for SVDD monitor'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='example_training_data.csv',
        help='Output CSV file path (default: example_training_data.csv)'
    )
    
    args = parser.parse_args()
    
    generate_sample_data(args.samples, args.output)
