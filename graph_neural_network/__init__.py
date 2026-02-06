"""
Graph Neural Network for Unsupervised Anomaly Detection in Mobile Robots

This package implements a Graph Neural Network (GNN) that dynamically learns
sensor relationships for anomaly detection in mobile robots using ROS 2.

Modules:
--------
- gnn_model: SensorRelationGNN model with dynamic graph learning
- train_gnn: Training script with self-supervised learning
- gnn_monitor_node: ROS 2 real-time anomaly detection node

Author: Research Project - Unsupervised Anomaly Detection
Framework: ROS 2 (Humble) + PyTorch + PyTorch Geometric
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__framework__ = "ROS 2 Humble + PyTorch Geometric"

from .gnn_model import SensorRelationGNN

__all__ = ['SensorRelationGNN']
