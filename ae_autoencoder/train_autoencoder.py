#!/usr/bin/env python3
"""Train a feed-forward autoencoder on features.npz

Saves:
    - model: `ae_model.pt`
    - scaler: `ae_scaler.pkl`
    - training stats: `ae_train_stats.npz`
"""
import argparse
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ae_autoencoder import build_autoencoder_from_config


def load_features(path: str) -> np.ndarray:
    data = np.load(path)
    if 'X' in data:
        return data['X']
    # assume CSV-like saved as array
    return data[data.files[0]]


def train(X: np.ndarray, out_dir: str, epochs=100, batch_size=256, lr=1e-3, device='cpu') -> None:
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # train/val split
    X_train, X_val = train_test_split(Xs, test_size=0.1, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train.astype('float32')))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype('float32')))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    model = build_autoencoder_from_config(input_dim)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float('inf')
    best_epoch = 0
    stats = {'train_loss': [], 'val_loss': []}

    patience = 10

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss /= max(1, n)

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                loss = loss_fn(recon, xb)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        val_loss /= max(1, n)

        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)

        print(f"Epoch {ep:03d}: train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            # save model + scaler
            torch.save(model.state_dict(), os.path.join(out_dir, 'ae_model.pt'))
            joblib.dump(scaler, os.path.join(out_dir, 'ae_scaler.pkl'))
        elif ep - best_epoch >= patience:
            print(f"Early stopping at epoch {ep} (best {best_epoch}, val {best_val:.6f})")
            break

    np.savez(os.path.join(out_dir, 'ae_train_stats.npz'), **stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', '-f', required=True, help='Path to features.npz')
    p.add_argument('--out', '-o', default='ae_out', help='Output directory')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    X = load_features(args.features)
    print(f"Loaded features shape: {X.shape}")
    train(X, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)


if __name__ == '__main__':
    main()
