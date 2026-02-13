#!/usr/bin/env python3
"""
GNN-based Real-Time Anomaly Detection Node for ROS 2
=====================================================

This ROS 2 node performs real-time anomaly detection on mobile robot sensor data
using a pre-trained Graph Neural Network. It subscribes to multiple sensor topics,
maintains a sliding window buffer, and uses the GNN to detect deviations from
normal sensor relationships.

Research Context:
-----------------
- Uses trained GNN to predict expected sensor values based on learned relationships
- Compares predictions with actual measurements to detect anomalies
- Anomalies indicate broken sensor relationships (e.g., hardware faults, attacks)

Node Information:
-----------------
- Node Name: gnn_anomaly_monitor
- Subscribed Topics:
    * /cmd_vel (geometry_msgs/Twist) - Velocity commands
    * /imu/data (sensor_msgs/Imu) - IMU measurements
    * /odom (nav_msgs/Odometry) - Odometry data
- Published Topics:
    * /anomaly_score (std_msgs/Float32) - Current anomaly score
    * /anomaly_alert (std_msgs/Bool) - Anomaly detection flag

Author: Research Project - Unsupervised Anomaly Detection in Mobile Robots
Framework: ROS 2 (Humble) + PyTorch + PyTorch Geometric
"""

import os
import sys
from pathlib import Path
from collections import deque
from typing import List, Optional, Deque, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS 2 message types
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, String

# PyTorch
import torch

# Import GNN model
from gnn_model import SensorRelationGNN


class GNNAnomalyMonitor(Node):
    """
    ROS 2 node for GNN-based real-time anomaly detection in mobile robots.
    
    Architecture:
    -------------
    1. Subscribe to sensor topics with approximate time synchronization
    2. Maintain sliding window buffer of recent sensor measurements
    3. Extract feature vector from synchronized sensor data
    4. Feed feature window to pre-trained GNN model
    5. Compare prediction with actual measurement (anomaly score)
    6. Trigger anomaly alert if score exceeds threshold
    
    Subscribed Topics:
    ------------------
    - /cmd_vel (geometry_msgs/Twist) - Velocity commands
    - /imu/data_raw (sensor_msgs/Imu) - Raw IMU measurements
    - /odom (nav_msgs/Odometry) - Odometry data
    
    The node operates in real-time and publishes both continuous anomaly scores
    and binary anomaly alerts for downstream safety systems.
    """
    
    def __init__(self):
        """
        Initialize the GNN anomaly monitor node.
        """
        super().__init__('gnn_anomaly_monitor')
        
        # ===== Node Parameters =====
        self.declare_parameter('model_path', 'gnn_checkpoint.pth')
        self.declare_parameter('window_size', 50)
        self.declare_parameter('anomaly_threshold', 0.3)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('sensor_names', [
            'cmd_vel.linear.x',
            'cmd_vel.angular.z',
            'imu.accel.x',
            'imu.accel.y',
            'imu.gyro.z',
            'odom.twist.linear.x',
            'odom.twist.angular.z',
            'imu.orientation.z'
        ])
        
        # Get parameters
        self.model_path: str = self.get_parameter('model_path').value
        self.window_size: int = self.get_parameter('window_size').value
        self.anomaly_threshold: float = self.get_parameter('anomaly_threshold').value
        self.device_name: str = self.get_parameter('device').value
        self.sensor_names: List[str] = self.get_parameter('sensor_names').value
        
        self.num_sensors = len(self.sensor_names)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("GNN Anomaly Detection Monitor")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Window size: {self.window_size}")
        self.get_logger().info(f"Anomaly threshold: {self.anomaly_threshold}")
        self.get_logger().info(f"Number of sensors: {self.num_sensors}")
        self.get_logger().info(f"Device: {self.device_name}")
        self.get_logger().info("")
        self.get_logger().info("To adjust anomaly threshold at runtime:")
        self.get_logger().info(f"  ros2 param set /gnn_anomaly_monitor anomaly_threshold 0.5")
        self.get_logger().info("=" * 70)
        
        # ===== Load Pre-trained GNN Model =====
        self.device = torch.device(self.device_name)
        self.model = self._load_model()
        
        # ===== Sliding Window Buffer =====
        # Stores recent sensor measurements: deque of shape [num_sensors]
        self.sensor_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        
        # Initialize buffer with zeros (will be filled during warmup)
        self.is_warmed_up = False
        self.warmup_counter = 0
        
        # ===== Anomaly Detection State =====
        self.anomaly_scores: List[float] = []
        self.anomaly_detected = False
        
        # ===== Manual Message Buffers for Synchronization =====
        # Store (timestamp, message) tuples
        self.cmd_vel_buffer: Deque[Tuple[float, Twist]] = deque(maxlen=100)
        self.imu_buffer: Deque[Tuple[float, Imu]] = deque(maxlen=100)
        self.odom_buffer: Deque[Tuple[float, Odometry]] = deque(maxlen=100)
        
        # Synchronization tolerance (seconds)
        self.sync_tolerance = 0.2
        
        # ===== ROS 2 Subscribers (Individual Subscriptions for Manual Sync) =====
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            qos_profile=qos_profile
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            qos_profile=qos_profile
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=qos_profile
        )
        
        # Timer to attempt synchronization
        self.sync_timer = self.create_timer(0.01, self.attempt_sync)
        
        # ===== ROS 2 Publishers =====
        self.anomaly_score_pub = self.create_publisher(
            Float32,
            '/anomaly_score',
            10
        )
        
        self.anomaly_alert_pub = self.create_publisher(
            Bool,
            '/anomaly_alert',
            10
        )
        
        self.anomaly_info_pub = self.create_publisher(
            String,
            '/anomaly_info',
            10
        )
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Node initialized successfully!")
        self.get_logger().info(f"Warming up buffer (need {self.window_size} samples)...")
        self.get_logger().info("=" * 70)
        
        # Add parameter callback for dynamic threshold adjustment
        self.add_on_set_parameters_callback(self._on_parameter_changed)
    
    def _load_model(self) -> SensorRelationGNN:
        """
        Load pre-trained GNN model from checkpoint.
        
        Returns:
        --------
        model : SensorRelationGNN
            Loaded GNN model in evaluation mode
        """
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model checkpoint not found: {self.model_path}")
            self.get_logger().error("Please train the model first using train_gnn.py")
            sys.exit(1)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            config = checkpoint['model_config']
            
            # Initialize model with saved configuration
            model = SensorRelationGNN(
                num_sensors=config['num_sensors'],
                window_size=config['window_size'],
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                top_k=config['top_k'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            self.get_logger().info("✓ Model loaded successfully!")
            self.get_logger().info(f"  - Trained epoch: {checkpoint['epoch']}")
            self.get_logger().info(f"  - Validation loss: {checkpoint['val_loss']:.6f}")
            
            return model
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            sys.exit(1)
    
    def _on_parameter_changed(self, params):
        """
        Callback for parameter changes (e.g., anomaly_threshold).
        
        Allows dynamic adjustment of threshold without restarting the node.
        """
        from rcl_interfaces.msg import SetParametersResult
        
        for param in params:
            if param.name == 'anomaly_threshold':
                old_threshold = self.anomaly_threshold
                self.anomaly_threshold = param.value
                self.get_logger().info(
                    f"📊 Anomaly threshold updated: {old_threshold:.3f} → {self.anomaly_threshold:.3f}"
                )
                return SetParametersResult(successful=True)
        
        return SetParametersResult(successful=True)
    
    def cmd_vel_callback(self, msg: Twist):
        """Callback for /cmd_vel messages."""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.cmd_vel_buffer.append((timestamp, msg))
    
    def imu_callback(self, msg: Imu):
        """Callback for /imu/data_raw messages."""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.imu_buffer.append((timestamp, msg))
    
    def odom_callback(self, msg: Odometry):
        """Callback for /odom messages."""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.odom_buffer.append((timestamp, msg))
    
    def attempt_sync(self):
        """Attempt to find synchronized messages from all three topics."""
        # Need at least one message from each topic
        if not self.cmd_vel_buffer or not self.imu_buffer or not self.odom_buffer:
            return
        
        # Use the most recent imu message as reference point
        imu_time, imu_msg = self.imu_buffer[-1]
        
        # Find cmd_vel and odom messages closest in time to imu
        cmd_vel_match = None
        odom_match = None
        
        # Find best cmd_vel match
        for cv_time, cv_msg in self.cmd_vel_buffer:
            if abs(cv_time - imu_time) <= self.sync_tolerance:
                if cmd_vel_match is None or abs(cv_time - imu_time) < abs(cmd_vel_match[0] - imu_time):
                    cmd_vel_match = (cv_time, cv_msg)
        
        # Find best odom match
        for odom_time, odom_msg in self.odom_buffer:
            if abs(odom_time - imu_time) <= self.sync_tolerance:
                if odom_match is None or abs(odom_time - imu_time) < abs(odom_match[0] - imu_time):
                    odom_match = (odom_time, odom_msg)
        
        # If we found matches for all three, process them
        if cmd_vel_match is not None and odom_match is not None:
            self.sensor_callback(cmd_vel_match[1], imu_msg, odom_match[1])
            
            # Clean up old messages to save memory
            self.cmd_vel_buffer.clear()
            self.imu_buffer.clear()
            self.odom_buffer.clear()
    
    def extract_features(
        self,
        cmd_vel: Twist,
        imu: Imu,
        odom: Odometry
    ) -> np.ndarray:
        """
        Extract feature vector from synchronized sensor messages.
        
        Feature Extraction:
        -------------------
        The feature vector combines measurements from all sensors into a
        single array. The order must match the training data configuration.
        
        Default Features (8 sensors):
        1. cmd_vel.linear.x - Forward velocity command
        2. cmd_vel.angular.z - Angular velocity command
        3. imu.accel.x - Linear acceleration X
        4. imu.accel.y - Linear acceleration Y
        5. imu.gyro.z - Angular velocity Z (gyroscope)
        6. odom.twist.linear.x - Measured forward velocity
        7. odom.twist.angular.z - Measured angular velocity
        8. imu.orientation.z - Orientation quaternion Z
        
        Parameters:
        -----------
        cmd_vel : Twist
            Velocity command message
        imu : Imu
            IMU sensor message
        odom : Odometry
            Odometry message
        
        Returns:
        --------
        features : np.ndarray
            Feature vector of shape [num_sensors]
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
        
        Pipeline:
        ---------
        1. Extract feature vector from synchronized messages
        2. Add to sliding window buffer
        3. If buffer is full (warmed up):
            a. Construct input tensor [1, num_sensors, window_size]
            b. Feed to GNN model to get prediction
            c. Calculate anomaly score (MSE between prediction and actual)
            d. Compare with threshold and publish alert if needed
        
        Parameters:
        -----------
        cmd_vel_msg : Twist
            Synchronized cmd_vel message
        imu_msg : Imu
            Synchronized IMU message
        odom_msg : Odometry
            Synchronized odometry message
        """
        # Extract features from current sensor readings
        current_features = self.extract_features(cmd_vel_msg, imu_msg, odom_msg)
        
        # Add to buffer
        self.sensor_buffer.append(current_features)
        
        # Warmup phase: collect enough samples to fill the window
        if not self.is_warmed_up:
            self.warmup_counter += 1
            
            if self.warmup_counter >= self.window_size:
                self.is_warmed_up = True
                self.get_logger().info("✓ Buffer warmed up! Starting anomaly detection...")
            else:
                if self.warmup_counter % 10 == 0:
                    self.get_logger().info(
                        f"Warmup progress: {self.warmup_counter}/{self.window_size}"
                    )
            return
        
        # ===== Anomaly Detection =====
        try:
            # Construct input tensor: [1, num_sensors, window_size]
            # Stack all samples in buffer along time dimension
            buffer_array = np.array(self.sensor_buffer)  # [window_size, num_sensors]
            buffer_array = buffer_array.T  # [num_sensors, window_size]
            buffer_array = np.expand_dims(buffer_array, axis=0)  # [1, num_sensors, window_size]
            
            # Convert to PyTorch tensor
            input_tensor = torch.FloatTensor(buffer_array).to(self.device)
            
            # Model inference (no gradient needed)
            with torch.no_grad():
                prediction, _ = self.model(input_tensor)
            
            # Get prediction for current time step
            predicted_values = prediction.cpu().numpy()[0]  # [num_sensors]
            actual_values = current_features  # [num_sensors]
            
            # Calculate anomaly score: Mean Squared Error
            anomaly_score = float(np.mean((predicted_values - actual_values) ** 2))
            self.anomaly_scores.append(anomaly_score)
            
            # Publish anomaly score
            score_msg = Float32()
            score_msg.data = anomaly_score
            self.anomaly_score_pub.publish(score_msg)
            
            # Check for anomaly
            is_anomaly = anomaly_score > self.anomaly_threshold
            
            # Publish anomaly alert
            alert_msg = Bool()
            alert_msg.data = is_anomaly
            self.anomaly_alert_pub.publish(alert_msg)
            
            # Log anomaly detection
            if is_anomaly:
                self.get_logger().warn(
                    f"⚠ ANOMALY DETECTED! Score: {anomaly_score:.6f} > "
                    f"Threshold: {self.anomaly_threshold:.6f}"
                )
                self.get_logger().warn("Sensor Relationship Broken - Possible Fault or Attack")
                
                # Find most deviant sensor
                sensor_errors = (predicted_values - actual_values) ** 2
                max_error_idx = np.argmax(sensor_errors)
                max_error_sensor = self.sensor_names[max_error_idx]
                max_error_value = sensor_errors[max_error_idx]
                
                anomaly_info_msg = String()
                anomaly_info_msg.data = (
                    f"Anomaly Score: {anomaly_score:.6f} | "
                    f"Most Deviant Sensor: {max_error_sensor} "
                    f"(Error: {max_error_value:.6f})"
                )
                self.anomaly_info_pub.publish(anomaly_info_msg)
                
                self.get_logger().warn(
                    f"  → Most deviant sensor: {max_error_sensor} "
                    f"(Error: {max_error_value:.6f})"
                )
                
                self.anomaly_detected = True
            else:
                # Normal operation
                if self.anomaly_detected:
                    self.get_logger().info("✓ System returned to normal operation")
                    self.anomaly_detected = False
                
                # Log periodically
                if len(self.anomaly_scores) % 50 == 0:
                    self.get_logger().info(
                        f"Monitoring... Score: {anomaly_score:.6f} "
                        f"(Threshold: {self.anomaly_threshold:.6f})"
                    )
        
        except Exception as e:
            self.get_logger().error(f"Error during anomaly detection: {str(e)}")
    
    def destroy_node(self):
        """
        Clean shutdown of the node.
        """
        self.get_logger().info("Shutting down GNN Anomaly Monitor...")
        self.get_logger().info(f"Total samples processed: {len(self.anomaly_scores)}")
        
        if len(self.anomaly_scores) > 0:
            avg_score = np.mean(self.anomaly_scores)
            max_score = np.max(self.anomaly_scores)
            self.get_logger().info(f"Average anomaly score: {avg_score:.6f}")
            self.get_logger().info(f"Maximum anomaly score: {max_score:.6f}")
        
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the GNN anomaly monitor node.
    """
    rclpy.init(args=args)
    
    try:
        node = GNNAnomalyMonitor()
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
