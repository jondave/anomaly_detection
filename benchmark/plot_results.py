#!/usr/bin/env python3
"""Plot benchmark results for anomaly detection methods.

Creates:
- roc_curves.png
- pr_curves.png
- score_distributions.png
- threshold_sweeps.png
- timeline_<bag>.png for each bag

Usage:
  python3 benchmark/plot_results.py \
    --results benchmark/results_hunter/combined_scores.csv \
    --out-dir benchmark/results_hunter/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


METHODS = ["gnn", "msvdd", "ae", "svm"]
COLORS = {
    "gnn": "#1f77b4",
    "msvdd": "#ff7f0e",
    "ae": "#2ca02c",
    "svm": "#d62728",
}


def available_methods(df: pd.DataFrame) -> list[str]:
    return [m for m in METHODS if f"{m}_score" in df.columns]


def clean_xy(df: pd.DataFrame, score_col: str) -> tuple[np.ndarray, np.ndarray]:
    mask = df[score_col].notna()
    return df.loc[mask, "label"].to_numpy(), df.loc[mask, score_col].to_numpy()


def plot_roc(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for m in methods:
        y, s = clean_xy(df, f"{m}_score")
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=COLORS[m], label=f"{m.upper()} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=220)
    plt.close()


def plot_pr(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for m in methods:
        y, s = clean_xy(df, f"{m}_score")
        if len(np.unique(y)) < 2:
            continue
        p, r, _ = precision_recall_curve(y, s)
        pr_auc = auc(r, p)
        plt.plot(r, p, lw=2, color=COLORS[m], label=f"{m.upper()} (AUC={pr_auc:.3f})")

    positive_rate = float(df["label"].mean()) if len(df) else 0.0
    plt.axhline(positive_rate, linestyle="--", color="k", alpha=0.5, label=f"baseline={positive_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=220)
    plt.close()


def plot_distributions(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    for i, m in enumerate(methods[:4]):
        ax = axes[i]
        col = f"{m}_score"
        normal = df.loc[df["label"] == 0, col].dropna().to_numpy()
        anom = df.loc[df["label"] == 1, col].dropna().to_numpy()

        if normal.size > 0:
            ax.hist(normal, bins=60, density=True, alpha=0.6, color="#4C78A8", label="normal")
        if anom.size > 0:
            ax.hist(anom, bins=60, density=True, alpha=0.6, color="#F58518", label="anomaly")

        ax.set_title(f"{m.upper()} score distribution")
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend()

    # Hide any unused axes if fewer than 4 methods
    for j in range(len(methods), 4):
        axes[j].axis("off")

    fig.savefig(out_dir / "score_distributions.png", dpi=220)
    plt.close(fig)


def anomaly_segments(group: pd.DataFrame) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    g = group.sort_values("timestamp_abs").copy()
    ts = g["timestamp_abs"].to_numpy()
    lb = g["label"].to_numpy()

    start = None
    for i in range(len(g)):
        if lb[i] == 1 and start is None:
            start = ts[i]
        if lb[i] == 0 and start is not None:
            segs.append((start, ts[i - 1]))
            start = None
    if start is not None:
        segs.append((start, ts[-1]))
    return segs


def plot_timelines(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    for bag, grp in df.groupby("bag_path"):
        grp = grp.sort_values("timestamp_abs")
        t0 = grp["timestamp_abs"].iloc[0]
        t = grp["timestamp_abs"].to_numpy() - t0

        fig, axes = plt.subplots(len(methods), 1, figsize=(12, 2.5 * len(methods)), sharex=True)
        if len(methods) == 1:
            axes = [axes]

        segs = anomaly_segments(grp)

        for ax, m in zip(axes, methods):
            col = f"{m}_score"
            s = grp[col].to_numpy()
            ax.plot(t, s, lw=1.2, color=COLORS[m], label=f"{m.upper()} score")
            for a, b in segs:
                ax.axvspan(a - t0, b - t0, color="#e63946", alpha=0.15)
            ax.set_ylabel(m.upper())
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

        scenario = str(grp["scenario"].iloc[0]) if "scenario" in grp.columns else ""
        fig.suptitle(f"{bag}   |   scenario={scenario}")
        axes[-1].set_xlabel("Time since bag start (s)")
        fig.tight_layout()

        safe_name = bag.replace("/", "__")
        fig.savefig(out_dir / f"timeline_{safe_name}.png", dpi=220)
        plt.close(fig)


def event_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
    if "event_id" not in df.columns:
        return []
    ev = df[(df["label"] == 1) & (df["event_id"].fillna("") != "")]
    if ev.empty:
        return []
    groups = []
    for _, g in ev.groupby(["bag_path", "event_id"], dropna=False):
        groups.append(g.sort_values("timestamp_abs"))
    return groups


def mean_detection_delay_for_threshold(df: pd.DataFrame, score_col: str, threshold: float) -> float:
    groups = event_groups(df)
    if not groups:
        return float("nan")

    delays = []
    for g in groups:
        bag = g["bag_path"].iloc[0]
        start_t = float(g["timestamp_abs"].iloc[0])
        end_t = float(g["timestamp_abs"].iloc[-1])

        bag_df = df[df["bag_path"] == bag].sort_values("timestamp_abs")
        window = bag_df[(bag_df["timestamp_abs"] >= start_t) & (bag_df["timestamp_abs"] <= end_t)]
        window = window[window[score_col].notna()]

        alarms = window[window[score_col] >= threshold]
        if alarms.empty:
            continue
        first_alarm = float(alarms["timestamp_abs"].iloc[0])
        delays.append(first_alarm - start_t)

    if not delays:
        return float("nan")
    return float(np.mean(delays))


def f1_for_threshold(df: pd.DataFrame, score_col: str, threshold: float) -> float:
    valid = df[df[score_col].notna()]
    if valid.empty:
        return float("nan")

    y_true = valid["label"].to_numpy().astype(np.int64)
    y_pred = (valid[score_col].to_numpy() >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def plot_threshold_sweeps(df: pd.DataFrame, methods: list[str], out_dir: Path, n_points: int = 80) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False, constrained_layout=True)
    ax_f1, ax_delay = axes

    for m in methods:
        col = f"{m}_score"
        scores = df[col].dropna().to_numpy()
        if scores.size < 10:
            continue

        q_low, q_high = np.percentile(scores, [1, 99])
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
            continue
        thrs = np.linspace(q_low, q_high, n_points)

        f1s = []
        delays = []
        for thr in thrs:
            f1s.append(f1_for_threshold(df, col, float(thr)))
            delays.append(mean_detection_delay_for_threshold(df, col, float(thr)))

        ax_f1.plot(thrs, f1s, color=COLORS[m], lw=2, label=m.upper())
        ax_delay.plot(thrs, delays, color=COLORS[m], lw=2, label=m.upper())

    ax_f1.set_title("Threshold Sweep: F1 vs Threshold")
    ax_f1.set_ylabel("F1")
    ax_f1.grid(alpha=0.25)
    ax_f1.legend()

    ax_delay.set_title("Threshold Sweep: Mean Detection Delay vs Threshold")
    ax_delay.set_xlabel("Threshold")
    ax_delay.set_ylabel("Delay (s)")
    ax_delay.grid(alpha=0.25)
    ax_delay.legend()

    fig.savefig(out_dir / "threshold_sweeps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results", required=True, help="Path to combined_scores.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.results).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = available_methods(df)
    if not methods:
        raise RuntimeError("No method score columns found in results CSV")

    plot_roc(df, methods, out_dir)
    plot_pr(df, methods, out_dir)
    plot_distributions(df, methods, out_dir)
    plot_threshold_sweeps(df, methods, out_dir)
    plot_timelines(df, methods, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
