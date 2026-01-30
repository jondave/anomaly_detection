#!/usr/bin/env python3
"""Evaluate trained autoencoder on features and save reconstruction scores.

Produces `ae_scores.npz` containing `scores` (reconstruction MSE per sample) and optionally prints basic stats.
"""
import argparse
import os

import joblib
import numpy as np
import torch

from ae_autoencoder import build_autoencoder_from_config


def load_features(path: str) -> np.ndarray:
    data = np.load(path)
    if 'X' in data:
        return data['X']
    return data[data.files[0]]


def evaluate(features_path: str, model_dir: str, device='cpu'):
    X = load_features(features_path)
    scaler_path = os.path.join(model_dir, 'ae_scaler.pkl')
    model_path = os.path.join(model_dir, 'ae_model.pt')
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError('Model or scaler not found in ' + model_dir)

    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)

    input_dim = X.shape[1]
    model = build_autoencoder_from_config(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(Xs.astype('float32')).to(device)
        recon = model(xb).cpu().numpy()

    # per-sample MSE
    mse = ((recon - Xs) ** 2).mean(axis=1)
    out_path = os.path.join(model_dir, 'ae_scores.npz')
    np.savez(out_path, scores=mse)
    print(f"Saved scores to {out_path}")
    print(f"scores: mean={mse.mean():.6f} std={mse.std():.6f} max={mse.max():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', '-f', required=True)
    p.add_argument('--model-dir', '-m', default='ae_out')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    evaluate(args.features, args.model_dir, device=args.device)


if __name__ == '__main__':
    main()
