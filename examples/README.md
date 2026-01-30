# Examples

This directory contains example usage scenarios for the ros2_svdd_monitor package.

## Generating Sample Training Data

You can use the provided script to generate synthetic training data for testing:

```python
import pandas as pd
import numpy as np

# Generate 100 samples of normal operation
np.random.seed(42)
n_samples = 100

data = {
    'timestamp': np.arange(1.0, n_samples + 1.0),
    'linear_x': np.random.normal(1.0, 0.1, n_samples),  # Mean 1.0 m/s
    'linear_y': np.zeros(n_samples),
    'linear_z': np.zeros(n_samples),
    'angular_x': np.zeros(n_samples),
    'angular_y': np.zeros(n_samples),
    'angular_z': np.random.normal(0.5, 0.05, n_samples),  # Mean 0.5 rad/s
    'accel_x': np.random.normal(2.0, 0.2, n_samples),  # Forward accel
    'accel_y': np.zeros(n_samples),
    'accel_z': np.zeros(n_samples),
    'gyro_x': np.zeros(n_samples),
    'gyro_y': np.zeros(n_samples),
    'gyro_z': np.random.normal(1.0, 0.1, n_samples),  # Yaw rate
}

df = pd.DataFrame(data)
df.to_csv('example_training_data.csv', index=False)
print("Created example_training_data.csv")
```

## Quick Test

1. Generate sample data:
   ```bash
   cd examples/
   python generate_sample_data.py
   ```

2. Train the model:
   ```bash
   ros2 run ros2_svdd_monitor train --csv examples/example_training_data.csv
   ```

3. The trained model will be saved as `svdd_model.pkl` and `scaler.pkl`

## Real Robot Usage

### Recording Data

1. Start recording with the CSV converter:
   ```bash
   ros2 run ros2_svdd_monitor rosbag_to_csv training_data.csv
   ```

2. In another terminal, operate your robot normally:
   - Drive around in normal conditions
   - Include various speeds and turning rates
   - Record for at least 5-10 minutes
   - Ensure data represents typical operation

3. Stop recording with Ctrl+C

### Training

```bash
ros2 run ros2_svdd_monitor train --csv training_data.csv --output-dir models/
```

### Monitoring

```bash
ros2 run ros2_svdd_monitor monitor
```

### Visualizing Results

In separate terminals:

```bash
# Echo anomaly detection
ros2 topic echo /svdd/anomaly

# Plot anomaly scores
rqt_plot /svdd/anomaly_score/data
```

## Typical Anomaly Scenarios

The SVDD monitor can detect:

1. **Wheel Slip**: High cmd_vel but low IMU acceleration
2. **Motor Failure**: No movement despite commands
3. **Unexpected Collision**: Sudden negative acceleration
4. **Rough Terrain**: High vibration in IMU
5. **Sensor Degradation**: Loss of correlation between command and response

## Tuning Tips

If you get too many false positives:
- Increase `nu` in config.yaml (e.g., from 0.1 to 0.15)
- Increase `anomaly_threshold` (e.g., from 0.0 to -0.1)
- Collect more diverse training data

If you miss real anomalies:
- Decrease `nu` (e.g., from 0.1 to 0.05)
- Decrease `anomaly_threshold` (e.g., from 0.0 to 0.1)
- Ensure training data only contains normal operation
