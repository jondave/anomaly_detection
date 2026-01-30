# Anomaly Detection Data Collection Setup

## Quick Start

You now have two scripts to record sensor data for training the MS-SVDD anomaly detection model:

### Option 1: Direct Recording
```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/ms_svdd
python3 record_features.py
```

### Option 2: Scenario-Based Recording
```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/ms_svdd
python3 record_scenario.py normal      # Record normal operation
python3 record_scenario.py aggressive  # Record aggressive driving
python3 record_scenario.py obstacle    # Record over obstacles
python3 record_scenario.py fault       # Record with faults
python3 record_scenario.py custom -n my_scenario  # Custom scenario
```

## What's Being Recorded

The scripts record **50+ features** across 4 ROS 2 topics:

### Motor Control Data (`/ranger_status`)
- **Motor voltages** - Driver voltage for each motor (0-3)
- **Motor currents** - Current draw by each motor
- **Motor RPM** - Rotation speed
- **Encoder pulses** - Cumulative pulse counts for wheel positions
- **Driver temperatures** - Thermal status of motor drivers
- **Battery voltage** - Overall system power level

### Motion Data (`/cmd_vel` + `/odom`)
- **Commanded velocities** - What was requested
- **Actual velocities** - What actually happened (from wheel encoders)
- **Linear/angular velocities** - X, Y, Z components

### Sensor Data (`/imu`)
- **Accelerations** - Linear acceleration in 3D
- **Gyroscopic data** - Angular velocity measurements
- **Orientation** - Quaternion representing robot orientation

### Position Data (`/odom`)
- **XYZ coordinates** - Robot position in space
- **Heading** - Orientation as quaternion

## Workflow for Training

### 1. Record Normal Operation (Baseline)
```bash
python3 record_scenario.py normal
# Drive the robot normally for 5-10 minutes
# Records to: ./recorded_data/normal/ranger_features_*.csv
```

### 2. (Optional) Record Fault Conditions
```bash
python3 record_scenario.py fault
# Introduce controlled faults and record
# Records to: ./recorded_data/fault/ranger_features_*.csv
```

### 3. Train the Model
```bash
python3 train_ms_svdd.py --data ./recorded_data/normal/*.csv \
                          --output ranger_mini_2/msvdd_model.pt
```

### 4. Deploy for Detection
```bash
# In one terminal, start the robot/gazebo
ros2 launch ranger_mini_2 ...

# In another terminal, run the anomaly detector
python3 ms_svdd_node.py
```

## Output Format

Each CSV file contains rows with:
- **Timestamp** - ISO format for synchronization
- **56 features** - All sensor readings at that moment

Example columns:
```
timestamp,cmd_vel_linear_x,imu_accel_x,...,ranger_motor0_rpm,ranger_motor1_current,...
2026-01-30T16:42:25.123456,0.5,0.1,...,125,2.3,...
2026-01-30T16:42:25.223456,0.5,0.09,...,126,2.2,...
...
```

## Key Features for Anomaly Detection

The model will learn to detect anomalies in:

1. **Motor irregularities**
   - Abnormal RPM patterns
   - Current spikes (motor jamming/overload)
   - Temperature warnings
   - Voltage drops

2. **Movement inconsistencies**
   - Commanded vs actual velocity mismatch (wheel slip)
   - Unexpected accelerations
   - Orientation errors

3. **Power issues**
   - Battery voltage drops
   - Uneven motor current distribution

4. **Sensor anomalies**
   - IMU reading inconsistencies
   - Encoder count anomalies

## Files Created

- **record_features.py** - Core recording script
- **record_scenario.py** - Convenience wrapper with scenarios
- **RECORD_FEATURES_README.md** - Detailed documentation
- **recorded_data/** - Directory where data is saved

## Troubleshooting

### "Failed to connect to topics"
Make sure the robot is running and publishing:
```bash
ros2 topic list
# Should show: /cmd_vel, /imu, /odom, /ranger_status
```

### "ImportError: ranger_msgs"
Rebuild the packages:
```bash
cd /home/agilex/agilex_ws
source install/setup.bash
colcon build --packages-select ranger_msgs
```

### No data recorded
- Check that all 4 topics are receiving messages
- The recorder waits for data on all topics before writing
- Use `ros2 topic echo /topic_name --once` to verify

## Next Steps

1. Record 20-30 minutes of normal operation data
2. Analyze the recorded data for patterns
3. Train the MS-SVDD model with `train_ms_svdd.py`
4. Test detection with `ms_svdd_node.py`
5. Record fault scenarios and verify detection
