# Quick Reference Card

## Files You Need to Know

### Core Scripts
- **record_features.py** - Raw recording (advanced)
- **record_scenario.py** - Recommended (easy to use)
- **analyze_recorded_data.py** - View statistics

### Documentation
- **SETUP_SUMMARY.md** - Start here! Overview of everything
- **RECORD_FEATURES_README.md** - Topic and CSV details
- **DATA_COLLECTION_GUIDE.md** - Complete workflow

---

## Data Flow Diagram

```
Robot Running
    ↓
ROS 2 Topics Published:
  /cmd_vel        (geometry_msgs/Twist)
  /imu            (sensor_msgs/Imu)
  /odom           (nav_msgs/Odometry)
  /ranger_status  (ranger_msgs/RangerStatus)
    ↓
record_features.py / record_scenario.py
    ↓
CSV File: recorded_data/ranger_features_*.csv
    ↓
56+ features × 10 Hz × recording_duration
    ↓
analyze_recorded_data.py (optional)
    ↓
train_ms_svdd.py
    ↓
Trained Model: ranger_mini_2/msvdd_model.pt
    ↓
ms_svdd_node.py (deployment)
    ↓
Publishes:
  /ms_svdd/anomaly       (Bool)
  /ms_svdd/anomaly_score (Float32)
```

---

## What Each Topic Provides

| Topic | Message Type | Contains | Records |
|-------|--------------|----------|---------|
| `/cmd_vel` | Twist | Commanded speeds | 6 features |
| `/imu` | Imu | Acceleration, rotation | 10 features |
| `/odom` | Odometry | Position, velocity | 12 features |
| `/ranger_status` | RangerStatus | Motors, battery, state | 20+ features |

**Total: 56+ features per sample**

---

## The 56+ Features

### Command Velocity (6)
- linear.x, linear.y, linear.z
- angular.x, angular.y, angular.z

### IMU (10)
- accel.x, accel.y, accel.z
- gyro.x, gyro.y, gyro.z
- quat.x, quat.y, quat.z, quat.w

### Odometry (12)
- pose.x, pose.y, pose.z
- quat.x, quat.y, quat.z, quat.w
- twist_linear.x, twist_linear.y, twist_linear.z
- twist_angular.x, twist_angular.y, twist_angular.z

### Motor Status (20)
Per motor (0-3):
- rpm
- current (key for detecting jamming!)
- pulse_count
- driver_voltage
- driver_temperature

### System Status (6+)
- battery_voltage
- linear_velocity
- angular_velocity
- lateral_velocity
- steering_angle
- vehicle_state
- control_mode

---

## CSV Output Example

```
timestamp,cmd_vel_linear_x,cmd_vel_linear_y,...,ranger_motor0_rpm,...
2026-01-30T16:42:25.123456,0.5,0.0,...,125,...
2026-01-30T16:42:25.223456,0.5,0.0,...,126,...
2026-01-30T16:42:25.323456,0.5,0.0,...,127,...
```

---

## Anomalies This Detects

### Motor Problems
- **Jamming**: High current without speed increase
- **Slipping**: RPM doesn't match command
- **Overheating**: Driver temperature spikes
- **Voltage drop**: Under heavy load

### Motion Issues
- **Unexpected acceleration**: IMU vs command mismatch
- **Drift**: Odometry diverges from expected
- **Imbalance**: Motors behave differently

### System Problems
- **Battery degradation**: Voltage drops during operation
- **Sensor errors**: Inconsistent readings

---

## Recording Checklist

Before recording:
- [ ] Robot is running and publishing topics
- [ ] All 4 topics available: `ros2 topic list`
- [ ] Output directory exists or will be created
- [ ] 10+ minutes of normal operation available

During recording:
- [ ] Drive robot through varied patterns
- [ ] Smooth accelerations/decelerations
- [ ] Different speeds and directions
- [ ] Let it record at least 5-10 minutes

After recording:
- [ ] Press Ctrl+C to flush data
- [ ] Check CSV file was created
- [ ] Run analyze_recorded_data.py to verify
- [ ] Proceed to model training

---

## Common Commands

```bash
# Change to script directory
cd /home/agilex/ros2_ws/src/anomaly_detection/ms_svdd

# Record normal operation (guided)
python3 record_scenario.py normal

# Record with custom output directory
python3 record_features.py /custom/path

# View all recorded data statistics
python3 analyze_recorded_data.py

# View specific CSV file stats
python3 analyze_recorded_data.py recorded_data/ranger_features_20260130_164225.csv

# Train the model
python3 train_ms_svdd.py --data ./recorded_data/normal/*.csv

# Run anomaly detection
python3 ms_svdd_node.py

# Check topics are publishing
ros2 topic list
ros2 topic echo /ranger_status --once
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: ranger_msgs | `colcon build --packages-select ranger_msgs` |
| Topics not found | Check robot is running, run `ros2 topic list` |
| No data recorded | Ensure all 4 topics are publishing before starting |
| CSV is empty | Recording requires all topics to have messages |
| File permission error | `chmod +x record_features.py` |

---

## Next Steps

1. Read **SETUP_SUMMARY.md** for full overview
2. Run `python3 record_scenario.py normal`
3. Drive robot 5-10 minutes normally
4. Run `python3 analyze_recorded_data.py`
5. Train model with `train_ms_svdd.py`
6. Deploy with `ms_svdd_node.py`

---

**Questions?** Check the documentation files in this directory.
