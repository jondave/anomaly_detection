#!/usr/bin/env python3
"""Offline rosbag benchmark for anomaly-detection methods.

This script reads ROS 2 bag folders directly, aligns `/cmd_vel`, `/imu`, and
`/odometry` streams onto a shared time grid, runs all configured anomaly
detectors, applies optional interval labels, and writes directly comparable
score files.

Outputs:
- One CSV per bag with shared timestamps and score columns.
- One combined CSV across all processed bags.
- One manifest JSON with bag metadata and selected topics.

The benchmark keeps all methods on the same aligned sample timestamps. Raw score
magnitudes remain method-specific; comparability comes from matching time bases,
labels, and evaluation windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "graph_neural_network") not in sys.path:
    sys.path.insert(0, str(ROOT / "graph_neural_network"))
if str(ROOT / "ae_autoencoder") not in sys.path:
    sys.path.insert(0, str(ROOT / "ae_autoencoder"))
if str(ROOT / "ms_svdd") not in sys.path:
    sys.path.insert(0, str(ROOT / "ms_svdd"))
if str(ROOT / "svm_svdd") not in sys.path:
    sys.path.insert(0, str(ROOT / "svm_svdd"))

from features import extract_window_features
from gnn_model import SensorRelationGNN
from ae_autoencoder import build_autoencoder_from_config
from ms_svdd_model import MSVDDWrapper
from svm_svdd_model import SVM_SVDDModel

import joblib
import torch

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


CMD_TOPIC_CANDIDATES = ["/cmd_vel", "/cmd_vel/nav", "/cmd_vel/joy", "/cmd_vel/collision_monitor"]
IMU_TOPIC_CANDIDATES = ["/imu/data_fused", "/imu/data", "/imu/data_raw"]
ODOM_TOPIC_CANDIDATES = ["/odometry/local", "/odom"] # , "/odometry/global"]


@dataclass
class TopicStream:
    topic: str
    timestamps: np.ndarray
    values: np.ndarray


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of YAML file: {path}")
    return data


def choose_topic(topic_names: Sequence[str], candidates: Sequence[str], kind: str) -> str:
    for candidate in candidates:
        if candidate in topic_names:
            return candidate
    raise ValueError(f"Could not find {kind} topic. Available topics: {sorted(topic_names)}")


def load_storage_id(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"
    if metadata_path.exists():
        metadata = load_yaml(metadata_path)
        info = metadata.get("rosbag2_bagfile_information", {})
        storage_id = info.get("storage_identifier")
        if isinstance(storage_id, str) and storage_id:
            return storage_id
    return "sqlite3"


def read_bag_streams(bag_path: Path) -> Dict[str, TopicStream]:
    storage_id = load_storage_id(bag_path)

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {item.name: item.type for item in topics_and_types}
    topic_names = set(topic_type_map.keys())

    cmd_topic = choose_topic(topic_names, CMD_TOPIC_CANDIDATES, "cmd_vel")
    imu_topic = choose_topic(topic_names, IMU_TOPIC_CANDIDATES, "imu")
    odom_topic = choose_topic(topic_names, ODOM_TOPIC_CANDIDATES, "odometry")

    selected_topics = {cmd_topic, imu_topic, odom_topic}
    message_types = {topic: get_message(msg_type) for topic, msg_type in topic_type_map.items() if topic in selected_topics}

    buffers: Dict[str, List[float]] = {topic: [] for topic in selected_topics}
    values: Dict[str, List[List[float]]] = {topic: [] for topic in selected_topics}

    while reader.has_next():
        topic_name, serialized_data, timestamp_ns = reader.read_next()
        if topic_name not in selected_topics:
            continue

        message = deserialize_message(serialized_data, message_types[topic_name])
        timestamp = timestamp_ns / 1e9
        buffers[topic_name].append(timestamp)

        if topic_name == cmd_topic:
            values[topic_name].append([
                message.linear.x,
                message.linear.y,
                message.linear.z,
                message.angular.x,
                message.angular.y,
                message.angular.z,
            ])
        elif topic_name == imu_topic:
            values[topic_name].append([
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
                message.orientation.z,
            ])
        elif topic_name == odom_topic:
            values[topic_name].append([
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
                message.twist.twist.angular.x,
                message.twist.twist.angular.y,
                message.twist.twist.angular.z,
            ])

    streams = {
        "cmd": TopicStream(cmd_topic, np.asarray(buffers[cmd_topic], dtype=np.float64), np.asarray(values[cmd_topic], dtype=np.float32)),
        "imu": TopicStream(imu_topic, np.asarray(buffers[imu_topic], dtype=np.float64), np.asarray(values[imu_topic], dtype=np.float32)),
        "odom": TopicStream(odom_topic, np.asarray(buffers[odom_topic], dtype=np.float64), np.asarray(values[odom_topic], dtype=np.float32)),
    }

    for name, stream in streams.items():
        if stream.timestamps.size == 0:
            raise ValueError(f"Bag {bag_path} did not contain any messages for selected {name} topic {stream.topic}")

    return streams


def align_streams(streams: Dict[str, TopicStream], sample_rate: float) -> Dict[str, np.ndarray]:
    dt = 1.0 / sample_rate

    start_time = max(stream.timestamps[0] for stream in streams.values())
    end_time = min(stream.timestamps[-1] for stream in streams.values())
    if end_time <= start_time:
        raise ValueError("No overlapping time range across cmd, imu, and odom streams")

    timestamps = np.arange(start_time, end_time + 1e-9, dt, dtype=np.float64)
    if timestamps.size == 0:
        raise ValueError("Alignment produced zero samples")

    aligned: Dict[str, np.ndarray] = {"timestamps": timestamps}
    for key, stream in streams.items():
        indices = np.searchsorted(stream.timestamps, timestamps, side="right") - 1
        indices = np.clip(indices, 0, stream.timestamps.size - 1)
        aligned[key] = stream.values[indices]

    imu = aligned["imu"]
    sensor_matrix = np.column_stack([
        aligned["cmd"][:, 0],
        aligned["cmd"][:, 5],
        imu[:, 0],
        imu[:, 1],
        imu[:, 5],
        aligned["odom"][:, 0],
        aligned["odom"][:, 5],
        imu[:, 6],
    ]).astype(np.float32)
    aligned["sensor_matrix"] = sensor_matrix

    return aligned


def build_feature_matrix(aligned: Dict[str, np.ndarray], window_size: int) -> np.ndarray:
    cmd = aligned["cmd"]
    imu = aligned["imu"][:, :6]
    odom = aligned["odom"]
    feature_rows: List[np.ndarray] = []

    for idx in range(len(aligned["timestamps"])):
        start_idx = max(0, idx - window_size + 1)
        feature_rows.append(
            extract_window_features(
                cmd[start_idx : idx + 1],
                imu[start_idx : idx + 1],
                odom[start_idx : idx + 1],
            )
        )

    return np.asarray(feature_rows, dtype=np.float32)


class GNNScorer:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["model_config"]
        self.window_size = int(config["window_size"])
        self.device = torch.device(device)
        self.model = SensorRelationGNN(
            num_sensors=config["num_sensors"],
            window_size=config["window_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            top_k=config["top_k"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def score(self, sensor_matrix: np.ndarray, mode: str = "next_step", batch_size: int = 256) -> np.ndarray:
        num_samples = sensor_matrix.shape[0]
        scores = np.full(num_samples, np.nan, dtype=np.float32)
        if num_samples <= self.window_size:
            return scores

        windows: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        valid_indices: List[int] = []

        if mode == "runtime":
            for idx in range(self.window_size - 1, num_samples):
                window = sensor_matrix[idx - self.window_size + 1 : idx + 1].T
                windows.append(window)
                targets.append(sensor_matrix[idx])
                valid_indices.append(idx)
        elif mode == "next_step":
            for idx in range(self.window_size, num_samples):
                window = sensor_matrix[idx - self.window_size : idx].T
                windows.append(window)
                targets.append(sensor_matrix[idx])
                valid_indices.append(idx)
        else:
            raise ValueError(f"Unsupported GNN score mode: {mode}")

        if not windows:
            return scores

        windows_array = np.asarray(windows, dtype=np.float32)
        targets_array = np.asarray(targets, dtype=np.float32)

        batch_scores: List[np.ndarray] = []
        with torch.no_grad():
            for start_idx in range(0, len(windows_array), batch_size):
                batch = torch.from_numpy(windows_array[start_idx : start_idx + batch_size]).to(self.device)
                predictions, _ = self.model(batch)
                prediction_np = predictions.cpu().numpy()
                batch_target = targets_array[start_idx : start_idx + batch_size]
                batch_scores.append(np.mean((prediction_np - batch_target) ** 2, axis=1))

        flat_scores = np.concatenate(batch_scores, axis=0).astype(np.float32)
        scores[np.asarray(valid_indices, dtype=np.int64)] = flat_scores
        return scores


class AEScorer:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.device = device
        self.scaler = joblib.load(model_dir / "ae_scaler.pkl")
        state_dict = torch.load(model_dir / "ae_model.pt", map_location=device, weights_only=False)
        inferred_input_dim = int(state_dict["encoder.0.weight"].shape[1])
        self.model = build_autoencoder_from_config(inferred_input_dim)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def score(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        with torch.no_grad():
            inputs = torch.from_numpy(scaled.astype(np.float32)).to(self.device)
            recon = self.model(inputs).cpu().numpy()
        return np.mean((recon - scaled) ** 2, axis=1).astype(np.float32)


class MSVDDScorer:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.wrapper = MSVDDWrapper(input_dim=1, device=torch.device(device))
        self.wrapper.load(str(model_path))

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.wrapper.score_samples(features).astype(np.float32)


class SVMSCorer:
    def __init__(self, model_dir: Path):
        self.model = SVM_SVDDModel()
        self.model.load(str(model_dir / "svm_model.pkl"), str(model_dir / "svm_scaler.pkl"))

    def score(self, features: np.ndarray) -> np.ndarray:
        return (-self.model.decision_function(features, scale=True)).astype(np.float32)


def build_label_arrays(
    timestamps: np.ndarray,
    bag_relative_path: str,
    bag_start_time: float,
    labels_config: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp_abs": timestamps,
            "timestamp_rel": timestamps - bag_start_time,
            "label": np.zeros(len(timestamps), dtype=np.int64),
            "event_id": np.full(len(timestamps), "", dtype=object),
            "event_name": np.full(len(timestamps), "", dtype=object),
            "scenario": np.full(len(timestamps), "", dtype=object),
            "label_notes": np.full(len(timestamps), "", dtype=object),
        }
    )

    if not labels_config:
        return frame

    bag_entries = labels_config.get("bags", [])
    matched_entry: Optional[Dict[str, Any]] = None
    normalized_bag_path = bag_relative_path.replace("\\", "/")
    for entry in bag_entries:
        entry_path = str(entry.get("bag_path", "")).replace("\\", "/")
        if entry_path == normalized_bag_path:
            matched_entry = entry
            break
        if normalized_bag_path.endswith(entry_path) or entry_path.endswith(normalized_bag_path):
            matched_entry = entry

    if matched_entry is None:
        return frame

    scenario = str(matched_entry.get("scenario", ""))
    if scenario:
        frame["scenario"] = scenario

    for interval_idx, interval in enumerate(matched_entry.get("intervals", []), start=1):
        start_abs = interval.get("start_abs")
        end_abs = interval.get("end_abs")
        start_rel = interval.get("start_rel")
        end_rel = interval.get("end_rel")

        if start_abs is None and start_rel is not None:
            start_abs = bag_start_time + float(start_rel)
        if end_abs is None and end_rel is not None:
            end_abs = bag_start_time + float(end_rel)
        if start_abs is None or end_abs is None:
            continue

        mask = (frame["timestamp_abs"] >= float(start_abs)) & (frame["timestamp_abs"] <= float(end_abs))
        if not mask.any():
            continue

        event_id = str(interval.get("event_id", f"event_{interval_idx}"))
        event_name = str(interval.get("name", event_id))
        notes = str(interval.get("notes", ""))

        frame.loc[mask, "label"] = 1
        frame.loc[mask, "event_id"] = event_id
        frame.loc[mask, "event_name"] = event_name
        frame.loc[mask, "label_notes"] = notes
        if scenario:
            frame.loc[mask, "scenario"] = scenario

    return frame


def sanitize_bag_name(relative_path: str) -> str:
    return relative_path.replace("/", "__")


def discover_bags(args: argparse.Namespace, labels_config: Optional[Dict[str, Any]]) -> List[Path]:
    bag_paths: List[Path] = []

    for raw_path in args.bags:
        bag_paths.append(Path(raw_path).expanduser().resolve())

    if labels_config and args.bag_root is not None:
        bag_root = Path(args.bag_root).expanduser().resolve()
        for entry in labels_config.get("bags", []):
            relative = entry.get("bag_path")
            if not relative:
                continue
            bag_paths.append((bag_root / relative).resolve())

    unique_paths: List[Path] = []
    seen: set[Path] = set()
    for bag_path in bag_paths:
        if bag_path in seen:
            continue
        seen.add(bag_path)
        unique_paths.append(bag_path)

    if not unique_paths:
        raise ValueError("No bag paths were provided. Use --bags or --bag-root with --labels.")

    for bag_path in unique_paths:
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag path not found: {bag_path}")

    return unique_paths


def add_alarm_column(frame: pd.DataFrame, score_column: str, threshold: Optional[float]) -> None:
    if threshold is None:
        return
    alarm_column = score_column.replace("_score", "_alarm")
    frame[alarm_column] = frame[score_column].gt(float(threshold)).fillna(False).astype(np.int64)


def process_bag(
    bag_path: Path,
    bag_root: Optional[Path],
    args: argparse.Namespace,
    labels_config: Optional[Dict[str, Any]],
    scorers: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    streams = read_bag_streams(bag_path)
    aligned = align_streams(streams, args.sample_rate)
    timestamps = aligned["timestamps"]
    bag_start_time = float(timestamps[0])

    if bag_root is not None:
        relative_path = bag_path.relative_to(bag_root).as_posix()
    else:
        relative_path = bag_path.name

    result = build_label_arrays(timestamps, relative_path, bag_start_time, labels_config)
    result.insert(0, "bag_path", relative_path)

    feature_matrix = build_feature_matrix(aligned, args.feature_window_size)
    sensor_matrix = aligned["sensor_matrix"]

    if "gnn" in scorers:
        result["gnn_score"] = scorers["gnn"].score(sensor_matrix, mode=args.gnn_score_mode, batch_size=args.batch_size)
        add_alarm_column(result, "gnn_score", args.gnn_threshold)
    if "msvdd" in scorers:
        result["msvdd_score"] = scorers["msvdd"].score(feature_matrix)
        add_alarm_column(result, "msvdd_score", args.msvdd_threshold)
    if "ae" in scorers:
        result["ae_score"] = scorers["ae"].score(feature_matrix)
        add_alarm_column(result, "ae_score", args.ae_threshold)
    if "svm" in scorers:
        result["svm_score"] = scorers["svm"].score(feature_matrix)
        add_alarm_column(result, "svm_score", args.svm_threshold)

    if args.include_aligned_signals:
        result["cmd_vel_linear_x"] = aligned["cmd"][:, 0]
        result["cmd_vel_angular_z"] = aligned["cmd"][:, 5]
        result["imu_accel_x"] = aligned["imu"][:, 0]
        result["imu_accel_y"] = aligned["imu"][:, 1]
        result["imu_gyro_z"] = aligned["imu"][:, 5]
        result["odom_linear_x"] = aligned["odom"][:, 0]
        result["odom_angular_z"] = aligned["odom"][:, 5]
        result["imu_orientation_z"] = aligned["imu"][:, 6]

    metadata = {
        "bag_path": relative_path,
        "absolute_bag_path": str(bag_path),
        "selected_topics": {
            "cmd": streams["cmd"].topic,
            "imu": streams["imu"].topic,
            "odom": streams["odom"].topic,
        },
        "bag_start_time_abs": bag_start_time,
        "bag_end_time_abs": float(timestamps[-1]),
        "duration_sec": float(timestamps[-1] - timestamps[0]),
        "num_rows": int(len(result)),
        "num_labeled_rows": int(result["label"].sum()),
    }
    return result, metadata


def build_scorers(args: argparse.Namespace) -> Dict[str, Any]:
    scorers: Dict[str, Any] = {}
    if args.gnn_checkpoint:
        scorers["gnn"] = GNNScorer(Path(args.gnn_checkpoint).expanduser().resolve(), device=args.device)
    if args.msvdd_model:
        scorers["msvdd"] = MSVDDScorer(Path(args.msvdd_model).expanduser().resolve(), device=args.device)
    if args.ae_model_dir:
        scorers["ae"] = AEScorer(Path(args.ae_model_dir).expanduser().resolve(), device=args.device)
    if args.svm_model_dir:
        scorers["svm"] = SVMSCorer(Path(args.svm_model_dir).expanduser().resolve())

    if not scorers:
        raise ValueError("No model artifacts were provided. Configure at least one scorer.")
    return scorers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline rosbag benchmark for anomaly detection methods")
    parser.add_argument("--bag-root", type=str, default=None, help="Root directory for bag paths referenced by the labels YAML")
    parser.add_argument("--bags", nargs="*", default=[], help="Explicit bag folders to benchmark")
    parser.add_argument("--labels", type=str, default=None, help="YAML file with bag metadata and anomaly intervals")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for benchmark outputs")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Aligned sample rate in Hz")
    parser.add_argument("--feature-window-size", type=int, default=10, help="Sliding-window size for feature-based methods")
    parser.add_argument("--gnn-score-mode", choices=["next_step", "runtime"], default="next_step", help="GNN scoring mode: intended forecasting or current runtime behavior")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for offline GNN inference")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for GNN, AE, and MSVDD")
    parser.add_argument("--include-aligned-signals", action="store_true", help="Include aligned raw signal columns in output CSVs")

    parser.add_argument("--gnn-checkpoint", type=str, default=None, help="Path to gnn_checkpoint.pth")
    parser.add_argument("--msvdd-model", type=str, default=None, help="Path to msvdd_model.pt")
    parser.add_argument("--ae-model-dir", type=str, default=None, help="Directory containing ae_model.pt and ae_scaler.pkl")
    parser.add_argument("--svm-model-dir", type=str, default=None, help="Directory containing svm_model.pkl and svm_scaler.pkl")

    parser.add_argument("--gnn-threshold", type=float, default=None, help="Optional threshold for binary GNN alarms")
    parser.add_argument("--msvdd-threshold", type=float, default=None, help="Optional threshold for binary MSVDD alarms")
    parser.add_argument("--ae-threshold", type=float, default=None, help="Optional threshold for binary AE alarms")
    parser.add_argument("--svm-threshold", type=float, default=None, help="Optional threshold for binary SVM alarms")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_config = None
    if args.labels:
        labels_config = load_yaml(Path(args.labels).expanduser().resolve())

    bag_root = Path(args.bag_root).expanduser().resolve() if args.bag_root else None
    bag_paths = discover_bags(args, labels_config)
    scorers = build_scorers(args)

    combined_frames: List[pd.DataFrame] = []
    manifest: Dict[str, Any] = {
        "sample_rate_hz": args.sample_rate,
        "feature_window_size": args.feature_window_size,
        "gnn_score_mode": args.gnn_score_mode,
        "device": args.device,
        "bags": [],
    }

    for bag_path in bag_paths:
        try:
            frame, metadata = process_bag(bag_path, bag_root, args, labels_config, scorers)
        except Exception as exc:
            print(f"SKIPPED {bag_path.name}: {exc}")
            manifest["bags"].append({"bag_path": str(bag_path), "error": str(exc)})
            continue
        output_name = sanitize_bag_name(metadata["bag_path"]) + ".csv"
        frame.to_csv(out_dir / output_name, index=False)
        combined_frames.append(frame)
        manifest["bags"].append(metadata)
        print(f"Saved {output_name} with {len(frame)} rows")

    combined = pd.concat(combined_frames, ignore_index=True)
    combined.to_csv(out_dir / "combined_scores.csv", index=False)
    with (out_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved combined_scores.csv with {len(combined)} rows")
    print(f"Saved benchmark_manifest.json to {out_dir}")


if __name__ == "__main__":
    main()