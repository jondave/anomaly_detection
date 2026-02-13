#!/usr/bin/env python3
"""
Simple Anomaly Threshold Calibrator
====================================

This script:
1. Runs the monitor node in the background
2. Plays a rosbag of normal operation
3. Collects anomaly scores from /anomaly_score topic
4. Analyzes the distribution and recommends thresholds

Much simpler than parsing rosbag messages directly!

Usage:
------
    python3 simple_threshold_calibrator.py --rosbag_path /path/to/rosbag

Author: Research Project
"""

import argparse
import subprocess
import time
import numpy as np
import threading
from pathlib import Path
import sys

# ROS imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class AnomalyScoreCollector(Node):
    """
    Simple ROS 2 node that collects anomaly scores from the monitor node.
    """
    
    def __init__(self):
        super().__init__('threshold_calibrator')
        
        self.anomaly_scores = []
        self.start_time = time.time()
        self.start_collection = False
        
        # Subscribe to anomaly score topic
        self.sub = self.create_subscription(
            Float32,
            '/anomaly_score',
            self.anomaly_callback,
            10
        )
        
        print("✓ Collector node initialized, waiting for messages...")
    
    def anomaly_callback(self, msg: Float32):
        """Collect anomaly scores."""
        if self.start_collection:
            self.anomaly_scores.append(msg.data)
            
            # Print progress every 100 scores
            if len(self.anomaly_scores) % 100 == 0:
                elapsed = time.time() - self.start_time
                print(f"  Collected {len(self.anomaly_scores)} scores ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description='Simple threshold calibration using rosbag playback'
    )
    
    parser.add_argument(
        '--rosbag_path',
        type=str,
        required=True,
        help='Path to rosbag2 database folder'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='gnn_checkpoint.pth',
        help='Path to trained GNN checkpoint'
    )
    
    args = parser.parse_args()
    
    rosbag_path = Path(args.rosbag_path)
    if not rosbag_path.exists():
        print(f"Error: Rosbag path not found: {rosbag_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Simple Anomaly Threshold Calibrator")
    print("=" * 70)
    print(f"Rosbag: {rosbag_path}")
    print(f"Model:  {args.model_path}")
    print("=" * 70)
    print()
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create collector node
        collector = AnomalyScoreCollector()
        
        # Start ROS spinner in background thread
        def spin_ros():
            rclpy.spin(collector)
        
        spinner = threading.Thread(target=spin_ros, daemon=True)
        spinner.start()
        
        # Give node time to initialize
        time.sleep(1)
        
        # Start the monitor node
        print("Starting monitor node...")
        monitor_proc = subprocess.Popen(
            ['python3', 'gnn_monitor_node.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for monitor to warm up
        print("Waiting for monitor node to warm up (50 samples)...")
        time.sleep(5)
        
        # Start rosbag playback
        print(f"\nPlaying rosbag: {rosbag_path}")
        rosbag_proc = subprocess.Popen(
            ['ros2', 'bag', 'play', str(rosbag_path), '-r', '1.0'],  # 2x speed
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Signal to start collection (after monitor warmup)
        time.sleep(2)
        collector.start_collection = True
        print("✓ Collection started!\n")
        
        # Wait for rosbag to finish
        rosbag_proc.wait()
        print("\n✓ Rosbag playback complete")
        
        # Wait a bit more to catch remaining messages
        time.sleep(2)
        
        # Stop monitor
        monitor_proc.terminate()
        monitor_proc.wait(timeout=5)
        
        # Analyze scores
        if collector.anomaly_scores:
            analyze_scores(collector.anomaly_scores)
        else:
            print("No anomaly scores collected!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        try:
            monitor_proc.terminate()
            rosbag_proc.terminate()
        except:
            pass
    
    finally:
        rclpy.shutdown()


def analyze_scores(scores):
    """Analyze anomaly score distribution."""
    scores = np.array(scores)
    
    print("\n" + "=" * 70)
    print("ANOMALY SCORE ANALYSIS")
    print("=" * 70)
    
    # Statistics
    print(f"\nBasic Statistics ({len(scores)} samples):")
    print(f"  Mean:     {np.mean(scores):.4f}")
    print(f"  Std Dev:  {np.std(scores):.4f}")
    print(f"  Min:      {np.min(scores):.4f}")
    print(f"  Max:      {np.max(scores):.4f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(scores, p)
        print(f"  {p:2d}th:     {value:.4f}")
    
    # Recommendations
    p95 = np.percentile(scores, 95)
    p99 = np.percentile(scores, 99)
    p90 = np.percentile(scores, 90)
    
    print(f"\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 70)
    print(f"\nBased on {len(scores)} samples of normal operation:")
    print(f"  🟢 Sensitive   (90th %ile):  {p90:.4f}")
    print(f"  🟡 Moderate    (95th %ile):  {p95:.4f}  ← Recommended")
    print(f"  🔴 Conservative (99th %ile): {p99:.4f}")
    
    print(f"\nHow many samples exceeded each threshold:")
    for threshold in [p90, p95, p99]:
        exceeded = np.sum(scores > threshold)
        pct = 100.0 * exceeded / len(scores)
        print(f"  {threshold:.4f}: {exceeded:4d} samples ({pct:.1f}%)")
    
    print(f"\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Start with recommended threshold: {p95:.4f}")
    print(f"   ros2 param set /gnn_anomaly_monitor anomaly_threshold {p95:.4f}")
    print(f"\n2. Test with actual driving")
    print(f"   ros2 run anomaly_detection gnn_monitor_node.py")
    print(f"\n3. If too many false positives, increase to: {p99:.4f}")
    print(f"4. If missing real anomalies, decrease to: {p90:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
