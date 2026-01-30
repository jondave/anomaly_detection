#!/usr/bin/env python3
"""Evaluate a trained Mahalanobis-SVDD (MSVDD) model and save anomaly scores.

Outputs: `<model_dir>/msvdd_scores.npz` containing `scores` (per-sample distances).
"""
import argparse
import os
import numpy as np

import torch

from ms_svdd_model import MSVDDWrapper


def load_features(path: str) -> np.ndarray:
    data = np.load(path)
    if 'X' in data:
        return data['X']
    return data[data.files[0]]


def evaluate(features_path: str, model_dir: str, model_name: str = 'msvdd_model.pt', device: str = 'cpu'):
    X = load_features(features_path)
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize wrapper with dummy input_dim; `load` will re-create model to match checkpoint when possible
    wrapper = MSVDDWrapper(input_dim=1, device=torch.device(device))
    wrapper.load(model_path)

    # compute scores (Mahalanobis distances); higher == more anomalous
    scores = wrapper.score_samples(X)

    out = os.path.join(model_dir, 'msvdd_scores.npz')
    np.savez(out, scores=scores)
    print(f"Saved scores to {out}")
    print(f"scores: mean={scores.mean():.6f} std={scores.std():.6f} max={scores.max():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', '-f', required=True)
    p.add_argument('--model-dir', '-m', default='msvdd_out')
    p.add_argument('--model-name', default='msvdd_model.pt')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    evaluate(args.features, args.model_dir, args.model_name, args.device)


if __name__ == '__main__':
    main()
