# Graph Neural Network for Unsupervised Anomaly Detection

This directory contains three Python scripts implementing a **Graph Neural Network (GNN)** for unsupervised anomaly detection in mobile robots using ROS 2 and PyTorch Geometric.

## 🎯 Research Overview

**Objective**: Detect anomalies in mobile robot sensor data by learning the relationships between sensors using a dynamically constructed graph structure.

**Approach**: 
- **Dynamic Graph Learning**: Unlike traditional GNNs with fixed graph topologies, this model learns sensor relationships from data using learnable node embeddings and cosine similarity
- **Self-Supervised Learning**: Uses a forecasting task (predict t+1 given t-W to t) to learn normal sensor behavior without requiring anomaly labels
- **Real-time Detection**: ROS 2 node monitors sensor streams and detects deviations from learned patterns

**Inspiration**: Graph Deviation Network (GDN) for multivariate time series anomaly detection

---

## 📁 File Structure

```
graph_neural_network/
├── gnn_model.py           # GNN model with dynamic graph learning
├── train_gnn.py           # Training script with validation
├── gnn_monitor_node.py    # ROS 2 real-time anomaly detection node
├── collect_gnn_data.py    # Data collection script (ROS 2)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🧠 Model Architecture (`gnn_model.py`)

### Key Features:
- **Dynamic Graph Construction**: Learns sensor relationships using cosine similarity between node embeddings
- **Sparse Graph**: Keeps only top-K strongest connections per node
- **GATv2Conv**: Graph attention mechanism to model sensor dependencies
- **Forecasting Head**: Predicts next time step for each sensor

### Model Components:
1. **Node Embeddings**: Each sensor has a learnable embedding vector
2. **Temporal Convolution**: 1D CNN to extract temporal patterns
3. **Graph Attention Layers**: Two-layer GATv2Conv with multi-head attention
4. **Forecasting Layer**: Fully connected layers for prediction

### Test the Model:
```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/graph_neural_network
python3 gnn_model.py
```

---

## 🏋️ Training (`train_gnn.py`)

### Data Format:
The training script expects a `.npz` file with:
- `x_train`: Shape `[num_samples, num_sensors, window_size]` - Input sliding windows
- `y_train`: Shape `[num_samples, num_sensors]` - Next time step targets

### Training Features:
- **Self-Supervised Learning**: Forecasting task (predict sensor values at t+1)
- **Validation Split**: Monitors generalization with validation set
- **Early Stopping**: Saves best model based on validation loss
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Visualization**: Automatically generates training curves and learned graph structure

### Usage:

#### Basic Training:
```bash
python3 train_gnn.py --data_path normal_robot_data.npz --epochs 100 --batch_size 32
```

#### Advanced Options:
```bash
python3 train_gnn.py \
  --data_path normal_robot_data.npz \
  --epochs 150 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --num_sensors 8 \
  --window_size 50 \
  --hidden_dim 128 \
  --top_k 3 \
  --num_heads 4 \
  --dropout 0.2 \
  --checkpoint_path gnn_checkpoint.pth \
  --save_dir training_results \
  --device cuda
```

### Output:
- **Model Checkpoint**: `gnn_checkpoint.pth` - Best model weights
- **Training History**: `training_results/training_history.json`
- **Training Curves**: `training_results/training_curves.png`
- **Learned Graph**: `training_results/learned_graph_structure.png`

---

## 🤖 ROS 2 Real-Time Monitor (`gnn_monitor_node.py`)

### Node Information:
- **Node Name**: `gnn_anomaly_monitor`
- **Framework**: ROS 2 Humble

### Subscribed Topics:
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands |
| `/imu/data_raw` | `sensor_msgs/Imu` | Raw IMU measurements |
| `/odom` | `nav_msgs/Odometry` | Odometry data |

### Published Topics:
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/anomaly_score` | `std_msgs/Float32` | Continuous anomaly score |
| `/anomaly_alert` | `std_msgs/Bool` | Binary anomaly flag |
| `/anomaly_info` | `std_msgs/String` | Anomaly details |

### Features:
- **Time Synchronization**: Uses `ApproximateTimeSynchronizer` for multi-sensor fusion
- **Sliding Window Buffer**: Maintains recent sensor history (default: 50 samples)
- **Real-time Inference**: Feeds window to GNN and compares prediction with actual
- **Anomaly Detection**: Triggers alert when prediction error exceeds threshold

### Sensor Features (Default: 8 sensors):
1. `cmd_vel.linear.x` - Forward velocity command
2. `cmd_vel.angular.z` - Angular velocity command
3. `imu.accel.x` - Linear acceleration X
4. `imu.accel.y` - Linear acceleration Y
5. `imu.gyro.z` - Angular velocity Z
6. `odom.twist.linear.x` - Measured forward velocity
7. `odom.twist.angular.z` - Measured angular velocity
8. `imu.orientation.z` - Orientation quaternion Z

### Usage:

#### Run Node:
```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/graph_neural_network
ros2 run anomaly_detection gnn_monitor_node.py
```

#### With Custom Parameters:
```bash
ros2 run anomaly_detection gnn_monitor_node.py \
  --ros-args \
  -p model_path:=gnn_checkpoint.pth \
  -p window_size:=50 \
  -p anomaly_threshold:=0.1 \
  -p device:=cpu
```

#### Monitor Anomaly Scores:
```bash
# Terminal 1: Run the node
ros2 run anomaly_detection gnn_monitor_node.py

# Terminal 2: Monitor anomaly scores
ros2 topic echo /anomaly_score

# Terminal 3: Monitor anomaly alerts
ros2 topic echo /anomaly_alert
```

---

## 🔧 Installation & Dependencies

### Required Packages:

```bash
# PyTorch (CPU version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip3 install torch-geometric torch-scatter torch-sparse torch-cluster

# ROS 2 Dependencies (should already be installed)
sudo apt install ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-nav-msgs

# Additional Python packages
pip3 install numpy matplotlib
```

### For GPU Support (CUDA):
```bash
# PyTorch (CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric (CUDA 11.8)
pip3 install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## 📊 Workflow

### Step 1: Collect Normal Data

Use the provided **`collect_gnn_data.py`** script to automatically collect and format data:

```bash
cd /home/agilex/ros2_ws/src/anomaly_detection/graph_neural_network

# Collect 60 seconds of data at 10Hz (recommended)
python3 collect_gnn_data.py --duration 60 --rate 10 --window_size 50

# For more training data, collect longer duration
python3 collect_gnn_data.py --duration 300 --rate 10 --output normal_robot_data.npz
```

**Important**: 
- Operate the robot **normally** during collection (drive around, typical movements)
- Ensure all sensors (`/cmd_vel`, `/imu/data`, `/odom`) are publishing
- Avoid anomalies or unusual behavior

The script will:
- ✅ Synchronize sensor topics automatically
- ✅ Sample at your specified rate (default: 10 Hz)
- ✅ Create sliding windows of raw sensor values
- ✅ Save in correct format for GNN training
- ✅ Display statistics and sensor value ranges

**Alternative Method** (Manual):
```bash
# Use ROS 2 bag to record
ros2 bag record /cmd_vel /imu/data /odom -o normal_operation

# Then convert using your own preprocessing script
```

### Step 2: Verify Data
```bash
# Check the generated file
python3 -c "import numpy as np; data = np.load('normal_robot_data.npz'); print('X:', data['x_train'].shape); print('Y:', data['y_train'].shape)"
```

Expected output:
```
X: (N, 8, 50)  # N training samples, 8 sensors, window size 50
Y: (N, 8)      # N target values, 8 sensors
```

### Step 3: Train GNN
```bash
python3 train_gnn.py --data_path normal_robot_data.npz --epochs 100
```

### Step 4: Real-Time Monitoring
```bash
# Source ROS 2 workspace
cd /home/agilex/ros2_ws
source install/setup.bash

# Run the monitor node
ros2 run anomaly_detection gnn_monitor_node.py
```

### Step 5: Analyze Results
- Monitor `/anomaly_score` topic for continuous assessment
- Watch `/anomaly_alert` for binary anomaly flags
- Check logs for detailed anomaly information

---

## 📈 Understanding Anomaly Scores

### Anomaly Score Calculation:
$$
\text{Anomaly Score} = \frac{1}{N} \sum_{i=1}^{N} (\text{Predicted}_i - \text{Actual}_i)^2
$$

Where $N$ is the number of sensors.

### Interpretation:
- **Low Score (< threshold)**: Sensor relationships are consistent with learned patterns → **Normal Operation**
- **High Score (> threshold)**: Sensor relationships deviate from learned patterns → **Anomaly Detected**

### Common Anomaly Causes:
1. **Hardware Faults**: Sensor malfunction or calibration drift
2. **External Disturbances**: Unexpected physical interactions
3. **Cyber Attacks**: Malicious sensor data injection
4. **Environmental Changes**: Operating outside training conditions

---

## 🔍 Troubleshooting

### Issue: Model not loading
```
Error: Model checkpoint not found
```
**Solution**: Train the model first using `train_gnn.py`

### Issue: Import error for gnn_model
```
ModuleNotFoundError: No module named 'gnn_model'
```
**Solution**: Run scripts from the `graph_neural_network` directory or add to PYTHONPATH

### Issue: Sensor synchronization timeout
```
Warning: No synchronized messages received
```
**Solution**: 
- Check if all sensor topics are publishing
- Adjust `slop` parameter in `ApproximateTimeSynchronizer` (increase tolerance)
- Verify topic names match your robot configuration

### Issue: High false positive rate
```
Too many anomalies detected
```
**Solution**:
- Increase `anomaly_threshold` parameter
- Retrain with more diverse normal data
- Check if test conditions match training conditions

---

## 📚 Research References

1. **Graph Deviation Network (GDN)**: Deng, A., & Hooi, B. (2021). "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"
2. **Graph Attention Networks**: Brody, S., Alon, U., & Yahav, E. (2021). "How Attentive are Graph Attention Networks?"
3. **ROS 2 Message Filters**: [ROS 2 Documentation](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{gnn_anomaly_detection_mobile_robots,
  title={Unsupervised Anomaly Detection in Mobile Robots using Graph Neural Networks},
  author={Your Name},
  year={2026},
  howpublished={Research Project},
  note={ROS 2 Humble + PyTorch Geometric}
}
```

---

## 🤝 Contributing

For questions, issues, or contributions, please contact the research team.

---

## 📄 License

This project is intended for research purposes. Please check with your institution regarding licensing.

---

**Happy Researching! 🚀**
