#!/usr/bin/env python3
"""Compute benchmark metrics from combined_scores.csv.

Metrics computed per method and per scenario:
  - AUROC  (area under ROC curve)
  - AUPRC  (area under precision-recall curve)
  - Detection delay (seconds from anomaly start to first alarm, per event)
  - False-alarm rate (false positives per minute on normal-labelled segments)
    - Threshold-sweep summary (best-F1 operating point and delay tradeoff)

Threshold for alarm / detection-delay is set at the (1-FPR_target) percentile
of normal-segment scores (default FPR target = 5%).

Usage:
    python3 benchmark/compute_metrics.py \
        --results benchmark/results_hunter/combined_scores.csv \
        --out-dir  benchmark/results_hunter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


METHODS = ["gnn", "msvdd", "ae", "svm"]
SCORE_COLS = {m: f"{m}_score" for m in METHODS}


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise scenario column — fill blanks with bag_path derived name
    if "scenario" not in df.columns:
        df["scenario"] = ""
    df["scenario"] = df["scenario"].fillna("").astype(str)
    empty = df["scenario"] == ""
    if empty.any():
        df.loc[empty, "scenario"] = df.loc[empty, "bag_path"].apply(
            lambda p: str(p).split("/")[0] if "/" in str(p) else str(p)
        )
    return df


def available_methods(df: pd.DataFrame) -> List[str]:
    return [m for m in METHODS if SCORE_COLS[m] in df.columns]


def derive_thresholds(df: pd.DataFrame, methods: List[str], fpr_target: float) -> Dict[str, float]:
    """Set threshold at (100 - fpr_target*100)th percentile of normal-segment scores."""
    normal = df[df["label"] == 0]
    thresholds: Dict[str, float] = {}
    for method in methods:
        col = SCORE_COLS[method]
        valid = normal[col].dropna()
        if valid.empty:
            thresholds[method] = float("nan")
        else:
            thresholds[method] = float(np.percentile(valid, (1.0 - fpr_target) * 100.0))
    return thresholds


def auroc_auprc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (AUROC, AUPRC) ignoring NaN scores. Returns NaN if insufficient data."""
    mask = ~np.isnan(scores)
    y = labels[mask]
    s = scores[mask]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def detection_delay(
    df: pd.DataFrame,
    method: str,
    threshold: float,
) -> List[dict]:
    """Per-event detection delay in seconds. Only for events that have label==1 rows."""
    col = SCORE_COLS[method]
    if "event_id" not in df.columns:
        return []

    results = []
    for event_id, grp in df[df["label"] == 1].groupby("event_id"):
        if event_id == "":
            continue
        event_start = grp["timestamp_abs"].min()
        event_end = grp["timestamp_abs"].max()
        event_name = grp["event_name"].iloc[0] if "event_name" in grp.columns else event_id
        scenario = grp["scenario"].iloc[0] if "scenario" in grp.columns else ""

        # Look at scores from event_start onward (within the same bag)
        bag = grp["bag_path"].iloc[0]
        bag_df = df[df["bag_path"] == bag]
        window = bag_df[bag_df["timestamp_abs"] >= event_start].copy()
        window = window.sort_values("timestamp_abs")

        valid = window[col].dropna()
        above = window.loc[valid.index][window.loc[valid.index, col] >= threshold]

        if above.empty:
            delay = float("nan")
            detected = False
        else:
            first_alarm_t = above["timestamp_abs"].iloc[0]
            delay = float(first_alarm_t - event_start)
            detected = True

        results.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "scenario": scenario,
                "bag": bag,
                "event_start": event_start,
                "event_duration_s": float(event_end - event_start),
                "detected": detected,
                "delay_s": delay,
                "threshold": threshold,
            }
        )
    return results


def false_alarm_rate(df: pd.DataFrame, method: str, threshold: float) -> float:
    """False alarms per minute on normal-labelled rows."""
    col = SCORE_COLS[method]
    normal = df[df["label"] == 0].copy()
    valid = normal[col].dropna()
    if valid.empty:
        return float("nan")
    fa = (valid >= threshold).sum()
    # Estimate duration: number of samples / sample_rate — infer from timestamps
    normal_sorted = normal.sort_values("timestamp_abs")
    dt = normal_sorted["timestamp_abs"].diff().median()
    duration_min = len(valid) * dt / 60.0
    return float(fa / duration_min) if duration_min > 0 else float("nan")


def precision_recall_f1(df: pd.DataFrame, method: str, threshold: float) -> tuple[float, float, float]:
    col = SCORE_COLS[method]
    valid = df[df[col].notna()]
    if valid.empty:
        return float("nan"), float("nan"), float("nan")

    y_true = valid["label"].to_numpy().astype(np.int64)
    y_pred = (valid[col].to_numpy() >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def event_detection_summary(df: pd.DataFrame, method: str, threshold: float) -> dict:
    events = detection_delay(df, method, threshold)
    if not events:
        return {
            "event_recall": float("nan"),
            "mean_delay_s": float("nan"),
            "median_delay_s": float("nan"),
            "detected_events": 0,
            "total_events": 0,
        }

    detected_delays = [e["delay_s"] for e in events if e["detected"] and not np.isnan(e["delay_s"])]
    detected_count = len(detected_delays)
    total_events = len(events)

    return {
        "event_recall": float(detected_count / total_events) if total_events > 0 else float("nan"),
        "mean_delay_s": float(np.mean(detected_delays)) if detected_delays else float("nan"),
        "median_delay_s": float(np.median(detected_delays)) if detected_delays else float("nan"),
        "detected_events": int(detected_count),
        "total_events": int(total_events),
    }


def threshold_sweep_summary(df: pd.DataFrame, methods: List[str], n_points: int = 120) -> dict:
    summary: dict = {}
    for method in methods:
        col = SCORE_COLS[method]
        scores = df[col].dropna().to_numpy()
        if scores.size < 10:
            summary[method] = {}
            continue

        q_low, q_high = np.percentile(scores, [1, 99])
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
            summary[method] = {}
            continue

        thrs = np.linspace(q_low, q_high, n_points)
        best = None
        curve = []

        for thr in thrs:
            precision, recall, f1 = precision_recall_f1(df, method, float(thr))
            fa_per_min = false_alarm_rate(df, method, float(thr))
            event_stats = event_detection_summary(df, method, float(thr))

            row = {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_alarms_per_min": fa_per_min,
                "event_recall": event_stats["event_recall"],
                "mean_delay_s": event_stats["mean_delay_s"],
                "median_delay_s": event_stats["median_delay_s"],
            }
            curve.append(row)

            # Pick best-F1, and if tied prefer lower false alarms
            if best is None:
                best = row
            else:
                better_f1 = row["f1"] > best["f1"]
                tied_f1 = np.isclose(row["f1"], best["f1"], atol=1e-12)
                better_fa = row["false_alarms_per_min"] < best["false_alarms_per_min"]
                if better_f1 or (tied_f1 and better_fa):
                    best = row

        summary[method] = {
            "best_f1_operating_point": best,
            "curve": curve,
        }
    return summary


def compute_all_metrics(
    df: pd.DataFrame,
    methods: List[str],
    thresholds: Dict[str, float],
    fpr_target: float,
) -> dict:
    scenarios = sorted(df["scenario"].unique())
    metrics: dict = {
        "fpr_target": fpr_target,
        "thresholds": thresholds,
        "overall": {},
        "per_scenario": {},
        "detection_delays": {},
        "false_alarm_rates": {},
        "threshold_sweep": {},
    }

    for method in methods:
        col = SCORE_COLS[method]
        thr = thresholds[method]

        # Overall AUROC / AUPRC
        valid = df[col].notna()
        auroc, auprc = auroc_auprc(df.loc[valid, "label"].values, df.loc[valid, col].values)
        metrics["overall"][method] = {"auroc": auroc, "auprc": auprc}

        # Per-scenario AUROC / AUPRC
        metrics["per_scenario"][method] = {}
        for sc in scenarios:
            sub = df[df["scenario"] == sc]
            valid_sc = sub[col].notna()
            if valid_sc.sum() == 0:
                continue
            a, ap = auroc_auprc(sub.loc[valid_sc, "label"].values, sub.loc[valid_sc, col].values)
            metrics["per_scenario"][method][sc] = {"auroc": a, "auprc": ap}

        # Detection delay
        metrics["detection_delays"][method] = detection_delay(df, method, thr)

        # False alarm rate
        metrics["false_alarm_rates"][method] = false_alarm_rate(df, method, thr)

    metrics["threshold_sweep"] = threshold_sweep_summary(df, methods)

    return metrics


def format_table(metrics: dict, methods: List[str]) -> str:
    lines = []

    # ── Overall ──
    lines.append("=" * 72)
    lines.append("OVERALL  (all labeled rows)")
    lines.append("-" * 72)
    lines.append(f"{'Method':<10}  {'AUROC':>8}  {'AUPRC':>8}  {'FA/min':>8}  {'Threshold':>12}")
    lines.append("-" * 72)
    for m in methods:
        ov = metrics["overall"].get(m, {})
        thr = metrics["thresholds"].get(m, float("nan"))
        fa = metrics["false_alarm_rates"].get(m, float("nan"))
        auroc = ov.get("auroc", float("nan"))
        auprc = ov.get("auprc", float("nan"))
        lines.append(
            f"{m:<10}  {_fmt(auroc):>8}  {_fmt(auprc):>8}  {_fmt(fa):>8}  {_fmt(thr):>12}"
        )
    lines.append("")

    # ── Per scenario ──
    all_scenarios: set = set()
    for m in methods:
        all_scenarios.update(metrics["per_scenario"].get(m, {}).keys())

    for sc in sorted(all_scenarios):
        lines.append(f"SCENARIO: {sc}")
        lines.append("-" * 50)
        lines.append(f"{'Method':<10}  {'AUROC':>8}  {'AUPRC':>8}")
        lines.append("-" * 50)
        for m in methods:
            sc_data = metrics["per_scenario"].get(m, {}).get(sc, {})
            auroc = sc_data.get("auroc", float("nan"))
            auprc = sc_data.get("auprc", float("nan"))
            lines.append(f"{m:<10}  {_fmt(auroc):>8}  {_fmt(auprc):>8}")
        lines.append("")

    # ── Detection delays ──
    lines.append("=" * 72)
    lines.append("DETECTION DELAYS  (seconds from anomaly start to first alarm)")
    lines.append("-" * 72)
    lines.append(f"{'Method':<10}  {'Event':<30}  {'Detected':>8}  {'Delay (s)':>10}")
    lines.append("-" * 72)
    for m in methods:
        for ev in metrics["detection_delays"].get(m, []):
            det_str = "YES" if ev["detected"] else "NO"
            delay_str = _fmt(ev["delay_s"]) if ev["detected"] else "—"
            lines.append(
                f"{m:<10}  {ev['event_name'][:30]:<30}  {det_str:>8}  {delay_str:>10}"
            )
        lines.append("")

    # ── Best-F1 operating points ──
    lines.append("=" * 72)
    lines.append("THRESHOLD SWEEP SUMMARY  (best-F1 operating point)")
    lines.append("-" * 72)
    lines.append(
        f"{'Method':<10}  {'Thresh':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'FA/min':>8}  {'Delay(s)':>8}"
    )
    lines.append("-" * 72)
    for m in methods:
        best = metrics.get("threshold_sweep", {}).get(m, {}).get("best_f1_operating_point")
        if not best:
            lines.append(f"{m:<10}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
            continue
        lines.append(
            f"{m:<10}  {_fmt(best.get('threshold')):>8}  {_fmt(best.get('precision')):>8}"
            f"  {_fmt(best.get('recall')):>8}  {_fmt(best.get('f1')):>8}"
            f"  {_fmt(best.get('false_alarms_per_min')):>8}  {_fmt(best.get('mean_delay_s')):>8}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def save_threshold_sweep_summary_csv(metrics: dict, methods: List[str], out_path: Path) -> None:
    rows = []
    for m in methods:
        best = metrics.get("threshold_sweep", {}).get(m, {}).get("best_f1_operating_point")
        if not best:
            continue
        rows.append(
            {
                "method": m,
                "threshold": best.get("threshold"),
                "precision": best.get("precision"),
                "recall": best.get("recall"),
                "f1": best.get("f1"),
                "false_alarms_per_min": best.get("false_alarms_per_min"),
                "event_recall": best.get("event_recall"),
                "mean_delay_s": best.get("mean_delay_s"),
                "median_delay_s": best.get("median_delay_s"),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute anomaly detection benchmark metrics")
    parser.add_argument("--results", required=True, help="Path to combined_scores.csv")
    parser.add_argument("--out-dir", required=True, help="Directory to write metrics output")
    parser.add_argument("--fpr-target", type=float, default=0.05,
                        help="Target false-positive rate used to set per-method thresholds (default 0.05 = 5%%)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results).expanduser().resolve())
    methods = available_methods(df)
    if not methods:
        print("No score columns found in CSV. Ensure benchmark ran successfully.")
        return

    print(f"Loaded {len(df)} rows, methods: {methods}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")

    thresholds = derive_thresholds(df, methods, args.fpr_target)
    print(f"\nDerived thresholds (at {(1-args.fpr_target)*100:.0f}th pct of normal scores):")
    for m, t in thresholds.items():
        print(f"  {m}: {t:.6f}")

    metrics = compute_all_metrics(df, methods, thresholds, args.fpr_target)

    # Save JSON
    json_path = out_dir / "metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nSaved metrics.json to {json_path}")

    # Save and print table
    table = format_table(metrics, methods)
    table_path = out_dir / "metrics_table.txt"
    table_path.write_text(table)
    print(f"Saved metrics_table.txt to {table_path}")

    # Save experiment-friendly threshold sweep summary
    sweep_summary_csv = out_dir / "threshold_sweep_best_f1.csv"
    save_threshold_sweep_summary_csv(metrics, methods, sweep_summary_csv)
    print(f"Saved threshold_sweep_best_f1.csv to {sweep_summary_csv}")

    print("\n" + table)


if __name__ == "__main__":
    main()
