#!/usr/bin/env python3
"""
ROS2 node to record features from Ranger Mini 2 for anomaly detection training.

Records data from:
    /cmd_vel (geometry_msgs/Twist) - commanded velocities
    /imu (sensor_msgs/Imu) - inertial measurement unit data
    /odom (nav_msgs/Odometry) - odometry/position data
    /ranger_status (ranger_msgs/RangerStatus) - motor voltages, currents, RPM, encoder pulses

Data is recorded to a CSV file with timestamps and all relevant features.
"""
import os
import sys
import csv
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32

# Import ranger_msgs
from ranger_msgs.msg import RangerStatus


class FeatureRecorder(Node):
    def __init__(self, output_dir='./recorded_data', buffer_size=1000):
        super().__init__('feature_recorder')
        
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create CSV file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(self.output_dir, f'ranger_features_{timestamp}.csv')
        
        # Initialize CSV file with headers
        self.init_csv_file()
        
        # Store latest message from each topic
        self.latest_cmd_vel = None
        self.latest_imu = None
        self.latest_odom = None
        self.latest_ranger_status = None
        
        # Subscribe to topics
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.ranger_status_sub = self.create_subscription(
            RangerStatus, '/ranger_status', self.ranger_status_callback, 10)
        
        # Timer to write data periodically (10 Hz)
        self.timer = self.create_timer(0.1, self.record_data_callback)
        
        self.get_logger().info(f"Feature recorder started. Output file: {self.csv_filename}")
        self.get_logger().info("Recording from topics: /cmd_vel, /imu, /odom, /ranger_status")
        self.get_logger().info("Press Ctrl+C to stop recording...")
    
    def init_csv_file(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp',
            # Command velocity
            'cmd_vel_linear_x', 'cmd_vel_linear_y', 'cmd_vel_linear_z',
            'cmd_vel_angular_x', 'cmd_vel_angular_y', 'cmd_vel_angular_z',
            # IMU
            'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
            'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
            'imu_quat_x', 'imu_quat_y', 'imu_quat_z', 'imu_quat_w',
            # Odometry
            'odom_pose_x', 'odom_pose_y', 'odom_pose_z',
            'odom_quat_x', 'odom_quat_y', 'odom_quat_z', 'odom_quat_w',
            'odom_vel_linear_x', 'odom_vel_linear_y', 'odom_vel_linear_z',
            'odom_vel_angular_x', 'odom_vel_angular_y', 'odom_vel_angular_z',
            # Ranger Status - Battery
            'ranger_battery_voltage',
            # Ranger Status - Motor 0
            'ranger_motor0_rpm', 'ranger_motor0_current', 'ranger_motor0_pulse',
            'ranger_motor0_driver_voltage', 'ranger_motor0_driver_temp',
            # Ranger Status - Motor 1
            'ranger_motor1_rpm', 'ranger_motor1_current', 'ranger_motor1_pulse',
            'ranger_motor1_driver_voltage', 'ranger_motor1_driver_temp',
            # Ranger Status - Motor 2
            'ranger_motor2_rpm', 'ranger_motor2_current', 'ranger_motor2_pulse',
            'ranger_motor2_driver_voltage', 'ranger_motor2_driver_temp',
            # Ranger Status - Motor 3
            'ranger_motor3_rpm', 'ranger_motor3_current', 'ranger_motor3_pulse',
            'ranger_motor3_driver_voltage', 'ranger_motor3_driver_temp',
            # Ranger Status - Motion info
            'ranger_linear_velocity', 'ranger_angular_velocity', 'ranger_lateral_velocity',
            'ranger_steering_angle', 'ranger_vehicle_state', 'ranger_control_mode'
        ]
        
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def cmd_vel_callback(self, msg: Twist):
        """Store latest cmd_vel message"""
        self.latest_cmd_vel = msg
    
    def imu_callback(self, msg: Imu):
        """Store latest IMU message"""
        self.latest_imu = msg
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry message"""
        self.latest_odom = msg
    
    def ranger_status_callback(self, msg: RangerStatus):
        """Store latest ranger status message"""
        self.latest_ranger_status = msg
    
    def record_data_callback(self):
        """Periodically record data from all topics"""
        # Only record if we have at least some data
        if all([self.latest_cmd_vel, self.latest_imu, 
                self.latest_odom, self.latest_ranger_status]):
            
            row = self._extract_features()
            self.data_buffer.append(row)
            
            # Write to CSV when buffer is full
            if len(self.data_buffer) >= self.buffer_size:
                self.flush_to_csv()
    
    def _extract_features(self):
        """Extract all features from the latest messages"""
        timestamp = datetime.now().isoformat()
        
        row = [timestamp]
        
        # cmd_vel features
        cmd_vel = self.latest_cmd_vel
        row.extend([
            cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z,
            cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z
        ])
        
        # IMU features
        imu = self.latest_imu
        row.extend([
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w
        ])
        
        # Odometry features
        odom = self.latest_odom
        row.extend([
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z,
            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w,
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z,
            odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z
        ])
        
        # Ranger Status features
        ranger = self.latest_ranger_status
        row.append(ranger.battery_voltage)
        
        # Extract motor features (up to 4 motors)
        # Motors 0-3 (front-right, front-left, rear-right, rear-left)
        for i in range(4):
            if i < len(ranger.actuator_states):
                motor = ranger.actuator_states[i]
                row.extend([
                    motor.rpm, motor.current, motor.pulse_count,
                    motor.driver_voltage, motor.driver_temperature
                ])
            else:
                row.extend([0.0, 0.0, 0, 0.0, 0.0])
        
        # Additional ranger status info
        row.extend([
            ranger.linear_velocity, ranger.angular_velocity, ranger.lateral_velocity,
            ranger.steering_angle, ranger.vehicle_state, ranger.control_mode
        ])
        
        return row
    
    def flush_to_csv(self):
        """Write buffered data to CSV file"""
        if not self.data_buffer:
            return
        
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.data_buffer)
        
        num_rows = len(self.data_buffer)
        self.get_logger().info(f"Wrote {num_rows} rows to {self.csv_filename}")
        self.data_buffer.clear()
    
    def destroy_node(self):
        """Flush remaining data before shutting down"""
        self.get_logger().info("Shutting down... flushing remaining data")
        self.flush_to_csv()
        self.get_logger().info(f"Recording complete. Data saved to {self.csv_filename}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Optional: specify output directory via command line
    output_dir = './recorded_data'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    node = FeatureRecorder(output_dir=output_dir, buffer_size=1000)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
