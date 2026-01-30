#!/usr/bin/env python3
"""
ROS2 node for Mahalanobis-SVDD inference using PyTorch model

Publishes:
    /ms_svdd/anomaly (std_msgs/Bool)
    /ms_svdd/anomaly_score (std_msgs/Float32)

Config (yaml or defaults): model_path, window_size, threshold
"""
import os
import sys
# Ensure package `ros2_svdd_monitor` (located under src/) is importable when
# running this file directly. Insert the parent `src` directory onto sys.path.
_here = os.path.abspath(os.path.dirname(__file__))
_pkg_parent = os.path.dirname(_here)  # .../src/ros2_svdd_monitor
_src_root = os.path.dirname(_pkg_parent)
if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
import os
import sys
import yaml
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32

try:
    from ros2_svdd_monitor.ms_svdd.ms_svdd_model import MSVDDWrapper
    from ros2_svdd_monitor.features import extract_window_features
except Exception:
    # Fallback when running the script directly from package folder
    from ms_svdd_model import MSVDDWrapper
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from features import extract_window_features


class MSVDDNode(Node):
    def __init__(self, config_path=None):
        super().__init__('ms_svdd_node')
        # simple defaults
        cfg = {
            'window_size': 10,
            'model_path': 'msvdd_model.pt',
            'threshold': None
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                usercfg = yaml.safe_load(f)
            if isinstance(usercfg, dict):
                cfg.update(usercfg)

        self.config = cfg
        self.window_size = int(self.config.get('window_size', 10))
        self.cmd_vel_window = deque(maxlen=self.window_size)
        self.imu_window = deque(maxlen=self.window_size)
        self.odom_window = deque(maxlen=self.window_size)

        model_path = os.path.expanduser(self.config.get('model_path'))
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found: {model_path}")
            raise RuntimeError('Model file missing')

        self.get_logger().info(f"Loading MSVDD model from {model_path}...")
        # we will lazy-init wrapper with input_dim inferred from scaler
        # Auto-detect compatible device
        import torch
        device = torch.device('cpu')
        if torch.cuda.is_available():
            try:
                # Check CUDA compute capability
                capability = torch.cuda.get_device_capability()
                major, minor = capability
                # PyTorch typically requires compute capability 7.0+
                if major < 7:
                    self.get_logger().warn(f'CUDA device has compute capability {major}.{minor}, but PyTorch requires 7.0+. Using CPU instead.')
                else:
                    # Test if CUDA actually works by attempting a simple operation
                    test_tensor = torch.zeros(1, device='cuda')
                    test_result = test_tensor + 1
                    _ = test_result.cpu()  # Force synchronization
                    device = torch.device('cuda')
                    self.get_logger().info('Using CUDA GPU for inference')
            except Exception as e:
                self.get_logger().warn(f'CUDA available but not compatible: {e}. Using CPU instead.')
        else:
            self.get_logger().info('CUDA not available. Using CPU for inference')
        
        self.wrapper = MSVDDWrapper(input_dim=1, device=device)
        self.wrapper.load(model_path)
        self.get_logger().info('Model loaded')

        self.threshold = self.config.get('threshold')

        self.anom_pub = self.create_publisher(Bool, '/ms_svdd/anomaly', 10)
        self.score_pub = self.create_publisher(Float32, '/ms_svdd/anomaly_score', 10)

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(Imu, '/imu/data_raw', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

    def cmd_vel_cb(self, msg: Twist):
        # pack linear.x and angular.z into 6-element vector expected by features
        v = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z]
        self.cmd_vel_window.append(v)
        self.try_analyze()

    def imu_cb(self, msg: Imu):
        imu = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
               msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.imu_window.append(imu)
        self.try_analyze()

    def odom_cb(self, msg: Odometry):
        odom = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.odom_window.append(odom)
        self.try_analyze()

    def try_analyze(self):
        # Only run when we have at least one sample in each
        if len(self.cmd_vel_window) == 0 or len(self.imu_window) == 0 or len(self.odom_window) == 0:
            return

        feat = extract_window_features(list(self.cmd_vel_window), list(self.imu_window), list(self.odom_window))
        try:
            score = float(self.wrapper.score_samples(feat.reshape(1, -1))[0])
        except Exception as e:
            # Handle common mismatch where scaler expects different feature dim
            msg = str(e)
            self.get_logger().warn(f"Initial scoring failed: {msg}")
            expected = None
            try:
                expected = len(self.wrapper.scaler.scale_)
            except Exception:
                expected = None

            if expected is not None:
                cur = feat.reshape(1, -1).shape[1]
                if cur != expected:
                    self.get_logger().warn(f"Adapting features: current={cur}, expected={expected} (trunc/pad)")
                    if cur > expected:
                        feat2 = feat[:expected]
                    else:
                        pad = np.zeros(expected - cur, dtype=feat.dtype)
                        feat2 = np.concatenate([feat, pad])
                    try:
                        score = float(self.wrapper.score_samples(feat2.reshape(1, -1))[0])
                    except Exception as e2:
                        self.get_logger().error(f"Scoring retry failed after adapt: {e2}")
                        return
                else:
                    self.get_logger().error(f"Scoring failed and feature dims match; error: {e}")
                    return
            else:
                self.get_logger().error(f"Scoring failed and expected dim unknown: {e}")
                return

        anom = False
        if self.threshold is not None:
            anom = score > float(self.threshold)

        self.anom_pub.publish(Bool(data=anom))
        self.score_pub.publish(Float32(data=float(score)))


def main(args=None):
    rclpy.init(args=args)
    try:
        cfg = None
        if args and isinstance(args, list) and len(args) > 1:
            cfg = args[1]
        node = MSVDDNode(config_path=cfg)
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
