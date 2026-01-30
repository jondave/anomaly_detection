"""
Mahalanobis-SVDD PyTorch implementation (minimal)

Provides a small MLP backbone to produce embeddings, a learnable
precision matrix parameterization for Mahalanobis distance, and a
wrapper with training utilities suitable for offline training and
ROS2 inference.
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler


class MLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=32):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MahalanobisSVDD(nn.Module):
    """Model that maps inputs to embeddings and computes Mahalanobis distance

    Precision matrix P is parameterized as A^T A + eps*I (guaranteed PD).
    The center `c` is computed as the mean of embeddings of the training set
    (or updated during training).
    """
    def __init__(self, input_dim, embed_dim=32, hidden_dims=(128,64), eps=1e-6):
        super().__init__()
        self.backbone = MLPBackbone(input_dim, hidden_dims, embed_dim)
        # A parameter to build precision matrix: shape (embed_dim, embed_dim)
        # initialize A with a slightly larger scale so precision isn't tiny
        self.A = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)
        self.eps = eps
        # center c is not registered as param (computed externally or set as buffer)
        self.register_buffer('center', torch.zeros(embed_dim))

    def forward(self, x):
        z = self.backbone(x)
        return z

    def precision_matrix(self):
        # P = A^T A + eps * I
        A = self.A
        P = torch.matmul(A.t(), A)
        diag_eps = self.eps * torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        return P + diag_eps

    def mahalanobis_distance(self, z):
        # z: (batch, dim) or (dim,)
        c = self.center
        delta = z - c.view(1, -1)
        P = self.precision_matrix()
        # compute quadratic form for each sample
        dist = torch.sum((delta @ P) * delta, dim=-1)
        return dist


class MSVDDWrapper:
    def __init__(self, input_dim, embed_dim=32, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = MahalanobisSVDD(input_dim, embed_dim).to(self.device)
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit_scaler(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        Xs = self.scaler.transform(X)
        return torch.tensor(Xs, dtype=torch.float32, device=self.device)

    def train(self, X, epochs=50, batch_size=64, lr=1e-3, radius_init=1.0,
              variance_reg=1e-3, patience=None, tol=1e-8, verbose=True):
        """Train the backbone and precision to minimize Mahalanobis-SVDD objective.

        Args:
            X: numpy array (n_samples, n_features)
        """
        X = np.array(X)
        self.fit_scaler(X)
        Xtorch = self.transform(X)

        opt = optim.Adam(self.model.parameters(), lr=lr)
        # compute center c as mean of embeddings (warmup)
        with torch.no_grad():
            Z = self.model(Xtorch)
            c = torch.mean(Z, dim=0)
            self.model.center.copy_(c)

        # initialize radius R from distribution of initial distances (90th percentile)
        with torch.no_grad():
            Z_full = self.model(Xtorch)
            d0 = self.model.mahalanobis_distance(Z_full).cpu().numpy()
        try:
            r_val = float(np.sqrt(max(1e-12, np.percentile(d0, 90))))
        except Exception:
            r_val = float(radius_init)
        R = torch.tensor(float(r_val), device=self.device)
        if verbose:
            print(f"Initial radius R set to {R.item():.6g} (from 90th percentile)")

        dataset = torch.utils.data.TensorDataset(Xtorch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                z = self.model(batch)
                # compute Mahalanobis distances
                dist = self.model.mahalanobis_distance(z)
                # soft SVDD objective: minimize mean( max(0, dist - R^2) )
                hinge = torch.mean(torch.relu(dist - R**2))

                # variance regularizer to prevent collapse: encourage non-zero variance
                var = torch.var(z, dim=0).mean()
                reg = variance_reg / (var + 1e-9)

                loss = hinge + reg

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= len(X)

            # early stopping based on training loss if requested
            if epoch_loss + tol < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % max(1, epochs//10) == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} (hinge~{hinge.item():.6g} reg~{reg.item():.6g})")

            if patience is not None and epochs_no_improve >= int(patience):
                if verbose:
                    print(f"Early stopping after {epoch+1} epochs (no improvement {epochs_no_improve} >= patience)")
                break

        # recompute center after training
        with torch.no_grad():
            Z = self.model(self.transform(X))
            c = torch.mean(Z, dim=0)
            self.model.center.copy_(c)

        self.is_trained = True

    def score_samples(self, X):
        """Return Mahalanobis distances (higher => more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError('Model not trained or loaded')
        Xt = self.transform(X)
        with torch.no_grad():
            z = self.model(Xt)
            d = self.model.mahalanobis_distance(z)
        return d.cpu().numpy()

    def save(self, model_path):
        model_path = os.path.expanduser(model_path)
        dirname = os.path.dirname(model_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        state = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler
        }
        torch.save(state, model_path)

    def load(self, model_path):
        # Allow loading of non-tensor objects (scaler was saved as sklearn object).
        # `weights_only=False` permits unpickling the full checkpoint (trusted local file).
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch may not support weights_only kwarg; fall back to legacy call.
            state = torch.load(model_path, map_location=self.device)
        model_state = state.get('model_state', state)

        # Try to infer input_dim and embed_dim from saved state dict
        input_dim = None
        embed_dim = None

        # Common key for first linear layer weight in backbone: 'backbone.net.0.weight'
        if 'backbone.net.0.weight' in model_state:
            w0 = model_state['backbone.net.0.weight']
            try:
                input_dim = int(w0.shape[1])
            except Exception:
                input_dim = None

        # Try to infer embed_dim from parameter 'A' if present
        if 'A' in model_state:
            A = model_state['A']
            try:
                embed_dim = int(A.shape[0])
            except Exception:
                embed_dim = None

        # Fallback: infer embed_dim from last linear in backbone
        if embed_dim is None:
            # find backbone weights and take last linear out features
            keys = [k for k in model_state.keys() if k.startswith('backbone.net') and k.endswith('.weight')]
            if keys:
                keys_sorted = sorted(keys, key=lambda x: int(x.split('.')[2]))
                last_w = model_state[keys_sorted[-1]]
                try:
                    embed_dim = int(last_w.shape[0])
                except Exception:
                    embed_dim = None

        # If we successfully inferred input_dim/embed_dim, re-create the model to match
        if input_dim is not None and embed_dim is not None:
            self.model = MahalanobisSVDD(input_dim, embed_dim).to(self.device)

        # Finally load the state dict into model
        self.model.load_state_dict(model_state)
        # load scaler if present
        self.scaler = state.get('scaler', self.scaler)
        self.is_trained = True
