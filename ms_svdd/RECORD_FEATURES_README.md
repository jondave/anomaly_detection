# Ranger Mini 2 Feature Recording for Anomaly Detection

## Overview

The `record_features.py` script records data from Ranger Mini 2 sensors to establish a baseline of normal robot operation. This recorded data is used to train the MS-SVDD (Multivariate Support Vector Data Description) model for anomaly detection.

## Topics Recorded

### 1. `/cmd_vel` (geometry_msgs/Twist)
**Command Velocity** - Desired velocities sent to the robot
- `linear.x/y/z` - Linear velocities in X, Y, Z directions (m/s)
- `angular.x/y/z` - Angular velocities around X, Y, Z axes (rad/s)

### 2. `/imu` (sensor_msgs/Imu)
**Inertial Measurement Unit** - Accelerometer and gyroscope data
- `linear_acceleration.x/y/z` - Accelerations in X, Y, Z (m/s²)
- `angular_velocity.x/y/z` - Angular velocities from gyro (rad/s)
- `orientation.x/y/z/w` - Quaternion orientation

### 3. `/odom` (nav_msgs/Odometry)
**Odometry** - Estimated position and velocity from wheel encoders
- `pose.position.x/y/z` - Estimated position (meters)
- `pose.orientation.x/y/z/w` - Estimated orientation (quaternion)
- `twist.linear.x/y/z` - Measured linear velocities (m/s)
- `twist.angular.x/y/z` - Measured angular velocities (rad/s)

### 4. `/ranger_status` (ranger_msgs/RangerStatus)
**Robot Status** - Motor control and power information
- `battery_voltage` - Battery voltage (V)
- **For each motor (0-3)**:
  - `rpm` - Motor rotations per minute
  - `current` - Current draw (A)
  - `pulse_count` - Encoder pulse count
  - `driver_voltage` - Motor driver voltage (V)
  - `driver_temperature` - Motor driver temperature (°C)
- `linear_velocity` - Actual linear velocity (m/s)
- `angular_velocity` - Actual angular velocity (rad/s)
- `lateral_velocity` - Lateral velocity (m/s)
- `steering_angle` - Steering angle (rad)
- `vehicle_state` - Vehicle state code
- `control_mode` - Control mode (0=manual, 1=auto, etc.)

## Usage

### 1. Start Recording Normal Operation

```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/ms_svdd

# Basic usage - saves to ./recorded_data/
python3 record_features.py

# Specify custom output directory
python3 record_features.py /path/to/output/directory
```

### 2. Run the Robot Through Normal Operations

While the recorder is running, drive the robot through typical use cases:
- Smooth forward/backward movement
- Turns and rotations
- Various speeds
- Different terrain if applicable
- Normal acceleration/deceleration

Record for at least 5-10 minutes of varied normal operation.

### 3. Stop Recording

Press `Ctrl+C` to stop. The script will:
- Flush all buffered data to CSV
- Create a CSV file with timestamp: `ranger_features_YYYYMMDD_HHMMSS.csv`
- Print the output file location

## Output CSV Format

The CSV file contains the following columns:

```
timestamp,
cmd_vel_linear_x, cmd_vel_linear_y, cmd_vel_linear_z,
cmd_vel_angular_x, cmd_vel_angular_y, cmd_vel_angular_z,
imu_accel_x, imu_accel_y, imu_accel_z,
imu_gyro_x, imu_gyro_y, imu_gyro_z,
imu_quat_x, imu_quat_y, imu_quat_z, imu_quat_w,
odom_pose_x, odom_pose_y, odom_pose_z,
odom_quat_x, odom_quat_y, odom_quat_z, odom_quat_w,
odom_vel_linear_x, odom_vel_linear_y, odom_vel_linear_z,
odom_vel_angular_x, odom_vel_angular_y, odom_vel_angular_z,
ranger_battery_voltage,
ranger_motor0_rpm, ranger_motor0_current, ranger_motor0_pulse,
ranger_motor0_driver_voltage, ranger_motor0_driver_temp,
ranger_motor1_rpm, ranger_motor1_current, ranger_motor1_pulse,
ranger_motor1_driver_voltage, ranger_motor1_driver_temp,
ranger_motor2_rpm, ranger_motor2_current, ranger_motor2_pulse,
ranger_motor2_driver_voltage, ranger_motor2_driver_temp,
ranger_motor3_rpm, ranger_motor3_current, ranger_motor3_pulse,
ranger_motor3_driver_voltage, ranger_motor3_driver_temp,
ranger_linear_velocity, ranger_angular_velocity, ranger_lateral_velocity,
ranger_steering_angle, ranger_vehicle_state, ranger_control_mode
```

## Training the Model

Use the recorded CSV data with `train_ms_svdd.py`:

```bash
python3 train_ms_svdd.py --data recorded_data/ranger_features_*.csv --output ranger_mini_2/msvdd_model.pt
```

## Anomaly Detection

Once trained, use `ms_svdd_node.py` to continuously monitor the robot:

```bash
ros2 run anomaly_detection ms_svdd_node
```

The node publishes:
- `/ms_svdd/anomaly` (Bool) - True if anomaly detected
- `/ms_svdd/anomaly_score` (Float32) - Distance from normal operating envelope

## Notes

- The script records at **10 Hz** (every 0.1 seconds)
- Data is buffered in memory (1000 rows) and written to disk periodically
- All timestamps are ISO format for easy parsing
- Missing motors are padded with zeros
- The recorder waits for all 4 topics to have at least one message before starting to record
