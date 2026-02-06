#!/usr/bin/env python3
"""
Data Collection Script for GNN Training
========================================

This script collects normal robot operation data and formats it for GNN training.
It records synchronized sensor messages and creates sliding windows of raw sensor
values (not statistical features).

Output Format:
--------------
- x_train: [num_samples, num_sensors, window_size] - Input sliding windows
- y_train: [num_samples, num_sensors] - Next time step targets

Usage:
------
    # Collect 60 seconds of data at 10Hz with window size 50
    python3 collect_gnn_data.py --duration 60 --rate 10 --window_size 50
    
    # With custom output path
    python3 collect_gnn_data.py --duration 120 --output normal_robot_data.npz

Author: Research Project - Unsupervised Anomaly Detection
"""

import argparse
import numpy as np
import time
from collections import deque
from typing import List, Optional, Deque, Tuple
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import message_filters


class GNNDataCollector(Node):
    """
    ROS 2 node to collect sensor data for GNN training.
    
    Features:
    ---------
    - Synchronizes /cmd_vel, /imu/data_raw, /odom topics
    - Records raw sensor values at specified rate
    - Creates sliding windows for time series forecasting
    - Saves in .npz format compatible with train_gnn.py
    """
    
    def __init__(
        self,
        duration_sec: float,
        sample_rate_hz: float,
        window_size: int,
        output_path: str
    ):
        super().__init__('gnn_data_collector')
        
        self.duration_sec = duration_sec
        self.sample_rate_hz = sample_rate_hz
        self.window_size = window_size
        self.output_path = output_path
        
        self.sample_interval = 1.0 / sample_rate_hz
        
        # Data storage: list of feature vectors [num_sensors]
        self.sensor_data: List[np.ndarray] = []
        
        # Message counters for debugging
        self.cmd_vel_count = 0
        self.imu_count = 0
        self.odom_count = 0
        self.sync_count = 0
        
        # Sensor names (must match gnn_monitor_node.py order)
        self.sensor_names = [
            'cmd_vel.linear.x',
            'cmd_vel.angular.z',
            'imu.accel.x',
            'imu.accel.y',
            'imu.gyro.z',
            'odom.twist.linear.x',
            'odom.twist.angular.z',
            'imu.orientation.z'
        ]
        self.num_sensors = len(self.sensor_names)
        
        # Timing
        self.start_time = None
        self.last_sample_time = None
        self.samples_collected = 0
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("GNN Data Collection Node")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Duration: {duration_sec} seconds")
        self.get_logger().info(f"Sample rate: {sample_rate_hz} Hz")
        self.get_logger().info(f"Window size: {window_size}")
        self.get_logger().info(f"Number of sensors: {self.num_sensors}")
        self.get_logger().info(f"Output: {output_path}")
        self.get_logger().info("=" * 70)
        
        # Setup subscribers with message filters
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Individual callbacks to monitor topic activity
        self.cmd_vel_sub = message_filters.Subscriber(
            self,
            Twist,
            '/cmd_vel',
            qos_profile=qos_profile
        )
        self.cmd_vel_sub.registerCallback(lambda msg: self._increment_counter('cmd_vel'))
        
        self.imu_sub = message_filters.Subscriber(
            self,
            Imu,
            '/imu/data_raw',
            qos_profile=qos_profile
        )
        self.imu_sub.registerCallback(lambda msg: self._increment_counter('imu'))
        
        self.odom_sub = message_filters.Subscriber(
            self,
            Odometry,
            '/odom',
            qos_profile=qos_profile
        )
        self.odom_sub.registerCallback(lambda msg: self._increment_counter('odom'))
        
        # Time synchronizer (allow_headerless=True for cmd_vel Twist messages)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.cmd_vel_sub, self.imu_sub, self.odom_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.sensor_callback)
        
        self.get_logger().info("Waiting for sensor messages...")
        self.get_logger().info("Please ensure robot is operating normally!")
        self.get_logger().info("=" * 70)
        
        # Status timer to show topic activity
        self.status_timer = self.create_timer(3.0, self.print_status)
    
    def _increment_counter(self, topic: str):
        """Count individual topic messages for diagnostics."""
        if topic == 'cmd_vel':
            self.cmd_vel_count += 1
        elif topic == 'imu':
            self.imu_count += 1
        elif topic == 'odom':
            self.odom_count += 1
    
    def print_status(self):
        """Print periodic status update showing topic activity."""
        if self.samples_collected == 0 and self.start_time is None:
            # Still waiting for first synchronized message
            self.get_logger().info(
                f"📡 Topic Activity: /cmd_vel={self.cmd_vel_count} | "
                f"/imu/data_raw={self.imu_count} | /odom={self.odom_count} | "
                f"Synced={self.sync_count}"
            )
            
            # Provide helpful diagnostics
            if self.cmd_vel_count == 0 and self.imu_count == 0 and self.odom_count == 0:
                self.get_logger().warn("⚠ No sensor messages received yet!")
                self.get_logger().info("   Check: ros2 topic list")
                self.get_logger().info("   Check: ros2 topic hz /cmd_vel")
            elif self.sync_count == 0:
                missing = []
                if self.cmd_vel_count == 0:
                    missing.append("/cmd_vel")
                if self.imu_count == 0:
                    missing.append("/imu/data_raw")
                if self.odom_count == 0:
                    missing.append("/odom")
                
                if missing:
                    self.get_logger().warn(f"⚠ Waiting for topics: {', '.join(missing)}")
                else:
                    self.get_logger().info("✓ All topics active, waiting for synchronization...")
    
    def extract_features(
        self,
        cmd_vel: Twist,
        imu: Imu,
        odom: Odometry
    ) -> np.ndarray:
        """
        Extract raw sensor values (same as gnn_monitor_node.py).
        
        Returns:
        --------
        features : np.ndarray
            Feature vector [num_sensors]
        """
        features = np.array([
            cmd_vel.linear.x,
            cmd_vel.angular.z,
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.angular_velocity.z,
            odom.twist.twist.linear.x,
            odom.twist.twist.angular.z,
            imu.orientation.z
        ], dtype=np.float32)
        
        return features
    
    def sensor_callback(
        self,
        cmd_vel_msg: Twist,
        imu_msg: Imu,
        odom_msg: Odometry
    ):
        """
        Callback for synchronized sensor messages.
        """
        self.sync_count += 1
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Initialize start time
        if self.start_time is None:
            self.start_time = current_time
            self.last_sample_time = current_time
            self.get_logger().info("✓ Synchronized messages received! Starting collection...")
            self.get_logger().info("")
        
        # Check if duration exceeded
        elapsed = current_time - self.start_time
        if elapsed >= self.duration_sec:
            self.get_logger().info("\n" + "=" * 70)
            self.get_logger().info(f"Collection complete! Collected {self.samples_collected} samples")
            self.save_data()
            rclpy.shutdown()
            return
        
        # Sample at specified rate
        time_since_last_sample = current_time - self.last_sample_time
        if time_since_last_sample >= self.sample_interval:
            # Extract and store features
            features = self.extract_features(cmd_vel_msg, imu_msg, odom_msg)
            self.sensor_data.append(features)
            
            self.samples_collected += 1
            self.last_sample_time = current_time
            
            # Progress update
            if self.samples_collected % 50 == 0:
                progress = (elapsed / self.duration_sec) * 100
                self.get_logger().info(
                    f"Progress: {progress:.1f}% | Samples: {self.samples_collected} | "
                    f"Time: {elapsed:.1f}/{self.duration_sec:.1f}s"
                )
    
    def create_sliding_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for time series forecasting.
        
        For each position i, create:
        - Input window: data[i:i+window_size] → shape [num_sensors, window_size]
        - Target: data[i+window_size] → shape [num_sensors]
        
        Returns:
        --------
        x_train : np.ndarray
            Input windows [num_samples, num_sensors, window_size]
        y_train : np.ndarray
            Target next steps [num_samples, num_sensors]
        """
        self.get_logger().info("\nCreating sliding windows...")
        
        # Convert list to array: [num_timesteps, num_sensors]
        data_array = np.array(self.sensor_data)
        
        num_timesteps = len(data_array)
        num_windows = num_timesteps - self.window_size
        
        if num_windows <= 0:
            self.get_logger().error(
                f"Not enough data! Need at least {self.window_size + 1} samples, "
                f"but only have {num_timesteps}"
            )
            sys.exit(1)
        
        x_windows = []
        y_targets = []
        
        for i in range(num_windows):
            # Input: window from i to i+window_size
            window = data_array[i:i+self.window_size]  # [window_size, num_sensors]
            window = window.T  # [num_sensors, window_size]
            x_windows.append(window)
            
            # Target: next time step
            target = data_array[i + self.window_size]  # [num_sensors]
            y_targets.append(target)
        
        x_train = np.array(x_windows, dtype=np.float32)
        y_train = np.array(y_targets, dtype=np.float32)
        
        self.get_logger().info(f"Created {len(x_train)} sliding windows")
        self.get_logger().info(f"  - X shape: {x_train.shape}")
        self.get_logger().info(f"  - Y shape: {y_train.shape}")
        
        return x_train, y_train
    
    def save_data(self):
        """
        Save collected data to .npz file.
        """
        if len(self.sensor_data) < self.window_size + 1:
            self.get_logger().error("Insufficient data collected!")
            return
        
        self.get_logger().info("\nProcessing collected data...")
        
        # Create sliding windows
        x_train, y_train = self.create_sliding_windows()
        
        # Print statistics
        self.get_logger().info("\nData Statistics:")
        self.get_logger().info(f"  - Total timesteps collected: {len(self.sensor_data)}")
        self.get_logger().info(f"  - Training samples: {len(x_train)}")
        self.get_logger().info(f"  - Features per sample: {self.num_sensors}")
        self.get_logger().info(f"  - Window size: {self.window_size}")
        
        # Show sensor value ranges
        data_array = np.array(self.sensor_data)
        self.get_logger().info("\nSensor Value Ranges:")
        for i, name in enumerate(self.sensor_names):
            min_val = data_array[:, i].min()
            max_val = data_array[:, i].max()
            mean_val = data_array[:, i].mean()
            self.get_logger().info(
                f"  {name:25s}: [{min_val:8.4f}, {max_val:8.4f}] mean={mean_val:8.4f}"
            )
        
        # Save to file
        self.get_logger().info(f"\nSaving to: {self.output_path}")
        np.savez(
            self.output_path,
            x_train=x_train,
            y_train=y_train,
            sensor_names=self.sensor_names,
            window_size=self.window_size,
            sample_rate=self.sample_rate_hz
        )
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("✓ Data collection successful!")
        self.get_logger().info(f"✓ Saved {len(x_train)} training samples")
        self.get_logger().info(f"\nNext step: Train the GNN model")
        self.get_logger().info(f"    python3 train_gnn.py --data_path {self.output_path}")
        self.get_logger().info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Collect sensor data for GNN training"
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Data collection duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--rate',
        type=float,
        default=10.0,
        help='Sampling rate in Hz (default: 10)'
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=50,
        help='Sliding window size (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='normal_robot_data.npz',
        help='Output file path (default: normal_robot_data.npz)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    if args.rate <= 0:
        print("Error: Sample rate must be positive")
        sys.exit(1)
    
    if args.window_size <= 0:
        print("Error: Window size must be positive")
        sys.exit(1)
    
    # Estimate number of samples
    expected_samples = int(args.duration * args.rate)
    expected_windows = max(0, expected_samples - args.window_size)
    
    print("\n" + "=" * 70)
    print("GNN Data Collection Configuration")
    print("=" * 70)
    print(f"Duration:          {args.duration} seconds")
    print(f"Sample rate:       {args.rate} Hz")
    print(f"Window size:       {args.window_size}")
    print(f"Expected samples:  ~{expected_samples}")
    print(f"Expected windows:  ~{expected_windows}")
    print(f"Output file:       {args.output}")
    print("=" * 70)
    print("\nIMPORTANT: Operate the robot NORMALLY during data collection")
    print("           (drive around, typical movements, no anomalies)")
    print("=" * 70)
    
    # Initialize ROS 2
    rclpy.init()
    
    try:
        collector = GNNDataCollector(
            duration_sec=args.duration,
            sample_rate_hz=args.rate,
            window_size=args.window_size,
            output_path=args.output
        )
        
        rclpy.spin(collector)
    
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
        if collector.samples_collected > collector.window_size:
            print("Saving partial data...")
            collector.save_data()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
