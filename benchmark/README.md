# Offline Benchmark Pipeline

This folder contains an offline rosbag benchmark that aligns the robot streams
onto one time base and produces directly comparable score files for:

- GNN
- Mahalanobis-SVDD
- Autoencoder
- OneClassSVM / SVDD

## What the script writes

For each bag:

- one CSV with shared timestamps, labels, and method scores

Across all bags:

- `combined_scores.csv`
- `benchmark_manifest.json`

## Label format

Labels are defined in YAML under `bags`.

Required fields:

- `bag_path`: path relative to `--bag-root`
- `scenario`: scenario name
- `intervals`: list of anomaly intervals

Each interval may use either:

- `start_abs` and `end_abs` in epoch seconds
- `start_rel` and `end_rel` in seconds from bag start

Optional fields:

- `event_id`
- `name`
- `notes`

Rows outside all intervals are labeled normal with `label=0`.

## Topic handling

The script prefers these topics and falls back automatically when needed:

- cmd: `/cmd_vel`, `/cmd_vel/nav`, `/cmd_vel/joy`, `/cmd_vel/collision_monitor`
- imu: `/imu/data_fused`, `/imu/data`, `/imu/data_raw`
- odom: `/odometry/local`, `/odom`, `/odometry/global`

## GNN scoring modes

`--gnn-score-mode next_step`:

- uses the intended forecasting setup from training
- window is `[t-W, ..., t-1]`
- target is `t`

`--gnn-score-mode runtime`:

- mirrors the current live node behavior
- window includes the current sample
- useful if you want offline results that match the deployed node more closely

For the paper, `next_step` is the cleaner default.

## Example command

### Gravel bags

```bash
cd /home/jcox/ros2_ws/src/anomaly_detection

python3 benchmark/offline_benchmark.py \
  --bag-root /home/jcox/ros2_ws/src/anomaly_detection/rosbags/hunter_rosbags/gravel \
  --labels /home/jcox/ros2_ws/src/anomaly_detection/benchmark/gravel_labels.yaml \
  --out-dir /home/jcox/ros2_ws/src/anomaly_detection/benchmark/results_gravel \
  --sample-rate 10 \
  --feature-window-size 10 \
  --gnn-score-mode next_step \
  --device cpu \
  --include-aligned-signals \
  --gnn-checkpoint /home/jcox/ros2_ws/src/anomaly_detection/graph_neural_network/gnn_hunter_out/gnn_checkpoint_hunter.pth \
  --msvdd-model /home/jcox/ros2_ws/src/anomaly_detection/ms_svdd/msvdd_hunter_out/msvdd_model_hunter.pt \
  --ae-model-dir /home/jcox/ros2_ws/src/anomaly_detection/ae_autoencoder/ae_hunter_out \
  --svm-model-dir /home/jcox/ros2_ws/src/anomaly_detection/svm_svdd/svm_hunter_out
```

### Tarmac bags (once recorded)

```bash
python3 benchmark/offline_benchmark.py \
  --bag-root /home/jcox/ros2_ws/src/anomaly_detection/rosbags/hunter_rosbags/tarmac \
  --labels /home/jcox/ros2_ws/src/anomaly_detection/benchmark/tarmac_labels.yaml \
  --out-dir /home/jcox/ros2_ws/src/anomaly_detection/benchmark/results_tarmac \
  --sample-rate 10 \
  --feature-window-size 10 \
  --gnn-score-mode next_step \
  --device cpu \
  --include-aligned-signals \
  --gnn-checkpoint /home/jcox/ros2_ws/src/anomaly_detection/graph_neural_network/gnn_hunter_out/gnn_checkpoint_hunter.pth \
  --msvdd-model /home/jcox/ros2_ws/src/anomaly_detection/ms_svdd/msvdd_hunter_out/msvdd_model_hunter.pt \
  --ae-model-dir /home/jcox/ros2_ws/src/anomaly_detection/ae_autoencoder/ae_hunter_out \
  --svm-model-dir /home/jcox/ros2_ws/src/anomaly_detection/svm_svdd/svm_hunter_out
```

## Output columns

Core columns:

- `bag_path`
- `timestamp_abs`
- `timestamp_rel`
- `label`
- `event_id`
- `event_name`
- `scenario`

Method columns appear when the corresponding model is supplied:

- `gnn_score`
- `msvdd_score`
- `ae_score`
- `svm_score`

Optional alarm columns appear when thresholds are supplied:

- `gnn_alarm`
- `msvdd_alarm`
- `ae_alarm`
- `svm_alarm`

Use `--include-aligned-signals` if you also want the aligned raw signals in the
CSV for plotting and error analysis.