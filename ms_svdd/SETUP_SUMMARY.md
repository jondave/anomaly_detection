# Ranger Mini 2 Anomaly Detection - Feature Recording System

## Summary

You now have a complete system to record sensor data from your Ranger Mini 2 robot for anomaly detection model training. The system records **56+ features** from 4 ROS 2 topics at 10 Hz.

## Files Created

| File | Purpose |
|------|---------|
| `record_features.py` | Core recording script - subscribes to topics and logs data to CSV |
| `record_scenario.py` | Wrapper script with scenario-based workflows (normal, aggressive, fault, etc.) |
| `analyze_recorded_data.py` | Utility to analyze and visualize recorded data statistics |
| `RECORD_FEATURES_README.md` | Detailed documentation of topics and CSV format |
| `DATA_COLLECTION_GUIDE.md` | Step-by-step guide for data collection workflow |

## Data Recorded

### Topic 1: `/cmd_vel` (Commanded Velocities)
- `linear.x`, `linear.y`, `linear.z` - Desired translation speeds
- `angular.x`, `angular.y`, `angular.z` - Desired rotation speeds

### Topic 2: `/imu` (Accelerometer & Gyroscope)
- `linear_acceleration` - 3-axis acceleration measurements
- `angular_velocity` - 3-axis gyroscopic data
- `orientation` - Quaternion pose

### Topic 3: `/odom` (Odometry/Encoder Data)
- `pose` - Estimated position (x, y, z) and orientation
- `twist` - Measured linear and angular velocities from wheels

### Topic 4: `/ranger_status` (Motor & System Status)
**For each of 4 motors:**
- `rpm` - Revolutions per minute
- `current` - Amperage draw (critical for detecting jamming)
- `pulse_count` - Cumulative encoder pulses
- `driver_voltage` - Power supply to motor driver
- `driver_temperature` - Thermal status

**System-wide:**
- `battery_voltage` - Main power level
- `linear_velocity` - Actual speed achieved
- `angular_velocity` - Actual turning rate
- `steering_angle` - Steering position
- `vehicle_state` - Operating state code
- `control_mode` - Manual/autonomous mode

## Key Features for Anomaly Detection

The recorded features allow detection of:

### Motor Faults
- **Jamming** - Abnormal current spike without speed increase
- **Slipping** - Speed command vs actual RPM mismatch
- **Overheating** - Driver temperature warnings
- **Power issues** - Voltage drops under load

### Movement Anomalies
- **Unexpected acceleration** - IMU data inconsistent with commands
- **Drift** - Position estimate deviates from expected
- **Imbalance** - Asymmetric motor responses

### System Failures
- **Battery degradation** - Voltage drops unexpectedly
- **Sensor errors** - IMU/encoder reading inconsistencies

## Quick Usage

### 1. Record Normal Operation (5-10 minutes)
```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/ms_svdd
python3 record_scenario.py normal
# Drive robot normally, then Ctrl+C to stop
```

### 2. Analyze the Data
```bash
python3 analyze_recorded_data.py
# Shows motor stats, battery info, feature summary
```

### 3. Train the Model
```bash
python3 train_ms_svdd.py --data ./recorded_data/normal/*.csv \
                          --output ranger_mini_2/msvdd_model.pt
```

### 4. Run Detection
```bash
# In one terminal:
ros2 launch ranger_mini_2 ...

# In another:
python3 ms_svdd_node.py
# Publishes to /ms_svdd/anomaly and /ms_svdd/anomaly_score
```

## Recording Scenarios

Use `record_scenario.py` for guided recording:

```bash
# Normal driving patterns (smooth, controlled)
python3 record_scenario.py normal

# Aggressive driving (sharp turns, high speeds)
python3 record_scenario.py aggressive

# Over obstacles (rough terrain, bumps)
python3 record_scenario.py obstacle

# Intentional faults (for comparison/testing)
python3 record_scenario.py fault

# Custom scenario
python3 record_scenario.py custom --custom-name my_scenario
```

Each creates a subdirectory in `./recorded_data/` with scenario-specific CSVs.

## CSV Output Format

Each row contains:
```
timestamp, cmd_vel_linear_x, ..., ranger_motor0_rpm, ..., ranger_battery_voltage, ...
2026-01-30T16:42:25.123, 0.5, ..., 125, ..., 49.5, ...
```

56+ columns total, timestamped at 10 Hz (0.1 second intervals)

## Data Requirements for Training

- **Minimum**: 5-10 minutes of normal operation
- **Recommended**: 30+ minutes of varied normal driving
- **For fault detection**: Also record 10+ minutes of fault conditions

The MS-SVDD algorithm learns the "normal envelope" from the baseline data, then detects deviations.

## Troubleshooting

### No data being recorded?
Check topics are active:
```bash
ros2 topic list
ros2 topic echo /ranger_status --once
```

### Connection errors?
Ensure setup is sourced:
```bash
cd /home/agilex/agilex_ws && source install/setup.bash
```

### Missing ranger_msgs?
Rebuild:
```bash
cd /home/agilex/agilex_ws
colcon build --packages-select ranger_msgs
source install/setup.bash
```

## Next Steps

1. **Collect baseline data**: Record 30+ minutes of normal operation
2. **Analyze patterns**: Use `analyze_recorded_data.py` to understand the data
3. **Train model**: Use CSV data with `train_ms_svdd.py`
4. **Test detection**: Run `ms_svdd_node.py` while driving
5. **Record faults**: Capture fault scenarios for comparison
6. **Iterate**: Refine model with more diverse training data

## Technical Details

### Recording Process
- Subscribes to 4 ROS 2 topics simultaneously
- Waits for all 4 topics to have at least one message
- Records at 10 Hz (triggered by internal timer, not topic rate)
- Buffers 1000 rows in memory before writing to disk
- Writes CSV with all features per sample

### Features Captured
- 6 cmd_vel features (3 linear + 3 angular)
- 10 IMU features (3 accel + 3 gyro + 4 quaternion)
- 12 odom features (3 position + 4 quaternion + 3 linear vel + 2 angular vel)
- 20+ ranger_status features (motors, battery, motion info)

### Anomaly Detection
MS-SVDD (Multivariate Support Vector Data Description):
- Learns the boundary of normal operation
- Calculates distance to boundary for new samples
- Anomaly score = how far from normal boundary
- Publishes Bool (is_anomaly) and Float32 (distance_score)

## Support

For detailed documentation, see:
- [RECORD_FEATURES_README.md](RECORD_FEATURES_README.md) - Topic details
- [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) - Collection workflow
