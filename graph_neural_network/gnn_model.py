"""
Graph Neural Network Model for Sensor Relationship Learning
============================================================

This module implements a dynamic graph learning approach for mobile robot
sensor anomaly detection. Unlike traditional GNNs with fixed graph structures,
this model learns the sensor relationships (graph topology) dynamically using
node embeddings and cosine similarity.

Research Context:
-----------------
- Inspired by Graph Deviation Networks (GDN) for multivariate time series anomaly detection
- Learns temporal and spatial (graph) relationships between sensors
- Self-supervised learning via forecasting the next time step

Author: Research Project - Unsupervised Anomaly Detection in Mobile Robots
Framework: PyTorch + PyTorch Geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from typing import Tuple, Optional


class SensorRelationGNN(nn.Module):
    """
    A Graph Neural Network that dynamically learns sensor relationships.
    
    Architecture:
    -------------
    1. Node Embedding Layer: Each sensor has a learnable embedding vector
    2. Dynamic Graph Construction: Cosine similarity between embeddings → adjacency matrix
    3. Graph Attention Layer (GATv2Conv): Learn sensor relationships with attention
    4. Temporal Convolution: Process time series information
    5. Forecasting Head: Predict next time step for each sensor
    
    Parameters:
    -----------
    num_sensors : int
        Number of sensor nodes (e.g., 6-8 for cmd_vel, imu, odom)
    window_size : int
        Length of the temporal sliding window
    embedding_dim : int
        Dimension of node embeddings for graph structure learning
    hidden_dim : int
        Hidden dimension for temporal processing
    top_k : int
        Number of strongest connections to keep per node (sparse graph)
    num_heads : int
        Number of attention heads in GATv2Conv
    dropout : float
        Dropout rate for regularization
    """
    
    def __init__(
        self,
        num_sensors: int = 8,
        window_size: int = 50,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        top_k: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super(SensorRelationGNN, self).__init__()
        
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_heads = num_heads
        self.dropout = dropout
        
        # ===== 1. Node Embedding Layer =====
        # Each sensor gets a learnable embedding to represent its "identity"
        # These embeddings are used to compute the graph structure dynamically
        self.node_embeddings = nn.Parameter(
            torch.randn(num_sensors, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        
        # ===== 2. Temporal Feature Extraction =====
        # For each sensor, extract features from its time series window
        # Approach: Global average pooling + FC layer to get hidden_dim features per sensor
        # Input: [batch_size, num_sensors, window_size]
        # Mean pooling across time: [batch_size, num_sensors]
        # FC expansion: [batch_size, num_sensors, hidden_dim]
        self.temporal_fc = nn.Linear(self.window_size, self.hidden_dim)
        
        # ===== 3. Graph Attention Network (GATv2Conv) =====
        # Learn relationships between sensors with attention mechanism
        # Input: hidden_dim features per node
        # Output: hidden_dim features per node (aggregated via attention)
        self.gat_conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.gat_conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # ===== 4. Forecasting Head =====
        # Map from hidden representation to next time step prediction
        self.fc1 = nn.Linear(hidden_dim + window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Predict scalar value for next time step
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def compute_dynamic_graph(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically compute the graph structure using node embeddings.
        
        Strategy:
        ---------
        1. Compute cosine similarity between all pairs of sensor embeddings
        2. For each sensor, keep only top-k strongest connections (sparse graph)
        3. Convert dense adjacency matrix to edge_index format for PyG
        
        Parameters:
        -----------
        batch_size : int
            Batch size for expanding the graph to each sample
        
        Returns:
        --------
        edge_index : torch.Tensor
            Edge list in COO format [2, num_edges]
        edge_weight : torch.Tensor
            Edge weights (similarity scores) [num_edges]
        """
        # Normalize embeddings for cosine similarity
        node_emb_norm = F.normalize(self.node_embeddings, p=2, dim=-1)  # [num_sensors, embedding_dim]
        
        # Compute cosine similarity: A_ij = cos(emb_i, emb_j)
        adjacency_matrix = torch.mm(node_emb_norm, node_emb_norm.t())  # [num_sensors, num_sensors]
        
        # Remove self-loops by masking diagonal
        mask = torch.eye(self.num_sensors, device=adjacency_matrix.device).bool()
        adjacency_matrix = adjacency_matrix.masked_fill(mask, -1e9)
        
        # Keep only top-k strongest connections per node (sparse graph)
        # This reduces noise and focuses on the most important sensor relationships
        top_k_values, top_k_indices = torch.topk(adjacency_matrix, self.top_k, dim=-1)
        
        # Create sparse adjacency matrix
        sparse_adj = torch.zeros_like(adjacency_matrix)
        row_indices = torch.arange(self.num_sensors).view(-1, 1).expand(-1, self.top_k)
        sparse_adj[row_indices, top_k_indices] = top_k_values
        
        # Make graph undirected (symmetrize)
        sparse_adj = (sparse_adj + sparse_adj.t()) / 2.0
        
        # Convert to PyTorch Geometric edge format
        edge_index, edge_weight = dense_to_sparse(sparse_adj)
        
        return edge_index, edge_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Learn sensor relationships and forecast next time step.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_sensors, window_size]
        
        Returns:
        --------
        predictions : torch.Tensor
            Forecasted next time step for each sensor [batch_size, num_sensors]
        edge_index : torch.Tensor
            Learned graph structure (for visualization/interpretation)
        """
        batch_size = x.size(0)
        
        # ===== Step 1: Extract Temporal Features =====
        # For each sensor, apply FC layer to its window to get hidden_dim features
        # Input: [batch_size, num_sensors, window_size]
        # Output: [batch_size, num_sensors, hidden_dim]
        graph_node_features = self.temporal_fc(x)  # [batch_size, num_sensors, hidden_dim]
        graph_node_features = F.relu(graph_node_features)
        
        # ===== Step 2: Dynamically Construct Graph Structure =====
        edge_index, edge_weight = self.compute_dynamic_graph(batch_size)
        
        # ===== Step 3: Graph Neural Network Processing =====
        # Reshape for PyTorch Geometric: [batch_size * num_sensors, hidden_dim]
        node_features = graph_node_features.view(batch_size * self.num_sensors, self.hidden_dim)
        
        # Expand edge_index for batched processing
        edge_index_batch = self._expand_edge_index_for_batch(edge_index, batch_size)
        
        # First GAT layer with multi-head attention
        node_features = self.gat_conv1(node_features, edge_index_batch)
        node_features = F.elu(node_features)
        node_features = self.dropout_layer(node_features)
        
        # Second GAT layer
        node_features = self.gat_conv2(node_features, edge_index_batch)
        node_features = self.layer_norm(node_features)
        
        # Reshape back: [batch_size, num_sensors, hidden_dim]
        node_features = node_features.view(batch_size, self.num_sensors, self.hidden_dim)
        
        # ===== Step 4: Forecasting Head =====
        # Concatenate graph features with original temporal information
        x_flat = x.view(batch_size, self.num_sensors, -1)  # [batch_size, num_sensors, window_size]
        combined_features = torch.cat([node_features, x_flat], dim=-1)
        
        # Predict next time step
        hidden = F.relu(self.fc1(combined_features))
        hidden = self.dropout_layer(hidden)
        predictions = self.fc2(hidden).squeeze(-1)  # [batch_size, num_sensors]
        
        return predictions, edge_index
    
    def _expand_edge_index_for_batch(self, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand edge_index to handle batched graphs.
        
        For batched processing in PyG, we need to offset node indices for each
        sample in the batch.
        
        Parameters:
        -----------
        edge_index : torch.Tensor
            Original edge index [2, num_edges]
        batch_size : int
            Number of graphs in batch
        
        Returns:
        --------
        edge_index_batch : torch.Tensor
            Expanded edge index [2, num_edges * batch_size]
        """
        edge_index_list = []
        for i in range(batch_size):
            offset = i * self.num_sensors
            edge_index_list.append(edge_index + offset)
        
        edge_index_batch = torch.cat(edge_index_list, dim=-1)
        return edge_index_batch
    
    def get_learned_graph(self) -> torch.Tensor:
        """
        Retrieve the learned graph structure (adjacency matrix).
        
        Useful for visualization and interpretation of sensor relationships.
        
        Returns:
        --------
        adjacency_matrix : torch.Tensor
            Learned adjacency matrix [num_sensors, num_sensors]
        """
        with torch.no_grad():
            node_emb_norm = F.normalize(self.node_embeddings, p=2, dim=-1)
            adjacency_matrix = torch.mm(node_emb_norm, node_emb_norm.t())
            
            # Apply top-k sparsification
            mask = torch.eye(self.num_sensors, device=adjacency_matrix.device).bool()
            adjacency_matrix = adjacency_matrix.masked_fill(mask, 0.0)
            
            top_k_values, top_k_indices = torch.topk(adjacency_matrix, self.top_k, dim=-1)
            sparse_adj = torch.zeros_like(adjacency_matrix)
            row_indices = torch.arange(self.num_sensors).view(-1, 1).expand(-1, self.top_k)
            sparse_adj[row_indices, top_k_indices] = top_k_values
            
            # Symmetrize
            sparse_adj = (sparse_adj + sparse_adj.t()) / 2.0
        
        return sparse_adj


if __name__ == "__main__":
    """
    Simple test to verify model construction and forward pass.
    """
    print("=" * 60)
    print("Testing SensorRelationGNN Model")
    print("=" * 60)
    
    # Model parameters
    num_sensors = 8
    window_size = 50
    batch_size = 16
    
    # Initialize model
    model = SensorRelationGNN(
        num_sensors=num_sensors,
        window_size=window_size,
        embedding_dim=64,
        hidden_dim=128,
        top_k=3,
        num_heads=4,
        dropout=0.2
    )
    
    print(f"\nModel Architecture:")
    print(f"  - Number of sensors: {num_sensors}")
    print(f"  - Window size: {window_size}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    x = torch.randn(batch_size, num_sensors, window_size)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    predictions, edge_index = model(x)
    print(f"Output shape: {predictions.shape}")
    print(f"Learned graph edges: {edge_index.shape[1]} edges")
    
    # Visualize learned adjacency matrix
    adj_matrix = model.get_learned_graph()
    print(f"\nLearned Adjacency Matrix:")
    print(adj_matrix.cpu().numpy())
    
    print("\n✓ Model test passed successfully!")
