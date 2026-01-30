"""Training utility for Mahalanobis-SVDD using extracted features.

Usage:
  python train_ms_svdd.py --features path/to/features.npz --out model.pt

The features .npz should contain an array named 'X' with shape (N, D).
If given a CSV, the script will attempt to read it with pandas and use
all numeric columns as features.
"""
import argparse
import numpy as np
import os
import sys

try:
    import torch
except Exception:
    print("PyTorch is required. Install with pip install torch")
    raise

from ms_svdd_model import MSVDDWrapper


def load_features(path):
    path = os.path.expanduser(path)
    if path.endswith('.npz') or path.endswith('.npz'):
        data = np.load(path)
        if 'X' in data:
            return data['X']
        else:
            # try first array
            arrs = [data[k] for k in data]
            return arrs[0]
    else:
        # try CSV via numpy loadtxt (simple fallback)
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.select_dtypes(include=[np.number]).values
        except Exception as e:
            print(f"Failed to load features from {path}: {e}")
            sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='Path to .npz or .csv with features')
    p.add_argument('--out', required=True, help='Output model path (.pt)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default=None, help="Device to use: 'cpu' or 'cuda' (default: auto)")
    args = p.parse_args()

    X = load_features(args.features)
    print(f'Loaded features shape: {X.shape}')

    # Respect requested device if provided
    device = None
    if args.device is not None:
        try:
            device = torch.device(args.device)
        except Exception:
            device = None

    wrapper = MSVDDWrapper(input_dim=X.shape[1], embed_dim=32, device=device)
    wrapper.train(X,
              epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              variance_reg=1e-3,
              patience=5)
    wrapper.save(args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()
