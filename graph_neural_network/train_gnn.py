"""
Training Script for Graph Neural Network Sensor Relationship Learning
======================================================================

This script implements self-supervised training for the SensorRelationGNN model
using a forecasting task. The model learns to predict the next time step based
on a sliding window of sensor measurements, which allows it to capture normal
sensor behavior patterns.

Research Approach:
------------------
- Task: Time series forecasting (predict t+1 given t-W to t)
- Learning Paradigm: Self-supervised (no anomaly labels needed during training)
- Training Data: Only normal robot operation data
- Loss Function: Mean Squared Error (MSE) between predicted and actual values

Usage:
------
    python train_gnn.py --data_path normal_data.npz --epochs 100 --batch_size 32

Author: Research Project - Unsupervised Anomaly Detection in Mobile Robots
Framework: PyTorch + PyTorch Geometric
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from gnn_model import SensorRelationGNN


class SensorTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sensor time series with sliding windows.
    
    Data Format:
    ------------
    - Input (X): [num_samples, num_sensors, window_size] - Historical sensor data
    - Target (Y): [num_samples, num_sensors] - Next time step sensor values
    
    The dataset implements a forecasting task where the model learns to predict
    the next time step given a window of historical measurements.
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Parameters:
        -----------
        x_data : np.ndarray
            Input windows of shape [num_samples, num_sensors, window_size]
        y_data : np.ndarray
            Target next step of shape [num_samples, num_sensors]
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        
        assert len(self.x_data) == len(self.y_data), "X and Y must have same number of samples"
        
        print(f"Dataset initialized: {len(self)} samples")
        print(f"  - Input shape: {self.x_data.shape}")
        print(f"  - Target shape: {self.y_data.shape}")
    
    def __len__(self) -> int:
        return len(self.x_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


class GNNTrainer:
    """
    Trainer class for SensorRelationGNN with validation and checkpointing.
    
    Features:
    ---------
    - Training with MSE loss
    - Validation monitoring with early stopping
    - Best model checkpointing
    - Learning rate scheduling
    - Training history logging and visualization
    """
    
    def __init__(
        self,
        model: SensorRelationGNN,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Parameters:
        -----------
        model : SensorRelationGNN
            The GNN model to train
        device : torch.device
            Device to train on (cuda or cpu)
        learning_rate : float
            Initial learning rate
        weight_decay : float
            L2 regularization coefficient
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function: MSE for forecasting task
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler: Reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader for training data
        
        Returns:
        --------
        avg_loss : float
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions, _ = self.model(batch_x)
            
            # Compute loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Parameters:
        -----------
        val_loader : DataLoader
            DataLoader for validation data
        
        Returns:
        --------
        avg_loss : float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions, _ = self.model(batch_x)
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_path: str = "gnn_checkpoint.pth",
        save_dir: str = "training_results"
    ) -> Dict:
        """
        Full training loop with validation and checkpointing.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Number of epochs to train
        checkpoint_path : str
            Path to save the best model checkpoint
        save_dir : str
            Directory to save training results
        
        Returns:
        --------
        history : Dict
            Training history dictionary
        """
        print("=" * 70)
        print("Starting GNN Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("=" * 70)
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_config': {
                        'num_sensors': self.model.num_sensors,
                        'window_size': self.model.window_size,
                        'embedding_dim': self.model.embedding_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'top_k': self.model.top_k,
                        'num_heads': self.model.num_heads,
                        'dropout': self.model.dropout
                    }
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  → Best model saved! (Val Loss: {val_loss:.6f})")
        
        print("=" * 70)
        print(f"Training Completed!")
        print(f"Best Validation Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
        print(f"Model saved to: {checkpoint_path}")
        print("=" * 70)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        # Visualize learned graph structure
        self.visualize_learned_graph(save_dir)
        
        return history
    
    def plot_training_curves(self, save_dir: str):
        """
        Plot training and validation loss curves.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.axvline(x=self.best_epoch - 1, color='r', linestyle='--', 
                    label=f'Best Epoch ({self.best_epoch})')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('GNN Training Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Training curves saved to: {plot_path}")
    
    def visualize_learned_graph(self, save_dir: str):
        """
        Visualize the learned sensor relationship graph (adjacency matrix).
        
        Parameters:
        -----------
        save_dir : str
            Directory to save the visualization
        """
        adj_matrix = self.model.get_learned_graph().cpu().numpy()
        
        plt.figure(figsize=(8, 7))
        im = plt.imshow(adj_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Connection Strength')
        plt.title('Learned Sensor Relationship Graph', fontsize=14, fontweight='bold')
        plt.xlabel('Sensor ID', fontsize=12)
        plt.ylabel('Sensor ID', fontsize=12)
        
        # Add grid
        plt.xticks(range(self.model.num_sensors))
        plt.yticks(range(self.model.num_sensors))
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        graph_path = os.path.join(save_dir, "learned_graph_structure.png")
        plt.savefig(graph_path, dpi=300)
        plt.close()
        print(f"Learned graph structure saved to: {graph_path}")


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from .npz file.
    
    Expected Format:
    ----------------
    - x_train: [num_samples, num_sensors, window_size]
    - y_train: [num_samples, num_sensors]
    
    Parameters:
    -----------
    data_path : str
        Path to .npz file containing training data
    
    Returns:
    --------
    x_train : np.ndarray
        Input windows
    y_train : np.ndarray
        Target next step values
    """
    print(f"Loading data from: {data_path}")
    
    data = np.load(data_path)
    x_train = data['x_train']
    y_train = data['y_train']
    
    print(f"Data loaded successfully:")
    print(f"  - X shape: {x_train.shape}")
    print(f"  - Y shape: {y_train.shape}")
    
    return x_train, y_train


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(
        description="Train GNN for Sensor Relationship Learning"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='normal_robot_data.npz',
        help='Path to training data (.npz file)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    # Model arguments
    parser.add_argument(
        '--num_sensors',
        type=int,
        default=8,
        help='Number of sensor nodes (default: 8)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=50,
        help='Sliding window size (default: 50)'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Node embedding dimension (default: 64)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden layer dimension (default: 128)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Top-K connections per node (default: 3)'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4,
        help='Number of attention heads (default: 4)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (default: 0.2)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay (default: 1e-5)'
    )
    
    # Output arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='gnn_checkpoint.pth',
        help='Path to save model checkpoint (default: gnn_checkpoint.pth)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='training_results',
        help='Directory to save training results (default: training_results)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate CUDA availability and compatibility
    if args.device == 'cuda':
        try:
            if not torch.cuda.is_available():
                print("⚠ CUDA is not available. Falling back to CPU.")
                args.device = 'cpu'
            else:
                # Test if CUDA actually works with a simple operation
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
        except (RuntimeError, torch.cuda.device.RuntimeError) as e:
            print(f"⚠ CUDA is not compatible with your GPU: {e}")
            print("Falling back to CPU for training...")
            args.device = 'cpu'
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    x_train, y_train = load_data(args.data_path)
    
    # Infer num_sensors and window_size from data if not provided
    if args.num_sensors == 8:  # Default value
        args.num_sensors = x_train.shape[1]
    if args.window_size == 50:  # Default value
        args.window_size = x_train.shape[2]
    
    print(f"\nAutodetected parameters from data:")
    print(f"  - Number of sensors: {args.num_sensors}")
    print(f"  - Window size: {args.window_size}")
    
    # Create dataset
    full_dataset = SensorTimeSeriesDataset(x_train, y_train)
    
    # Train-validation split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize model
    model = SensorRelationGNN(
        num_sensors=args.num_sensors,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    print(f"\nModel initialized:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    device = torch.device(args.device)
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir
    )
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
