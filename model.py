"""GNN model for DDI prediction using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SAGEConv(nn.Module):
    """GraphSAGE convolutional layer with optional self-loops."""

    def __init__(
        self,
        embedding_dim: int,
        with_self: bool = True,
        degree_norm: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_self = with_self
        self.degree_norm = degree_norm

        # Linear layer to transform concatenated embeddings
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Aggregates neighborhood information and updates node embeddings.

        Args:
            x: Node embeddings of shape [n_nodes, embedding_dim]
            senders: Source node indices
            receivers: Target node indices
            n_nodes: Total number of nodes

        Returns:
            Updated node embeddings
        """
        device = x.device

        # Add self-loops if enabled
        if self.with_self:
            self_loop_indices = torch.arange(n_nodes, device=device)
            senders = torch.cat([senders, self_loop_indices])
            receivers = torch.cat([receivers, self_loop_indices])

        # Get sender and receiver embeddings
        sender_embeddings = x[senders]  # [num_edges, embedding_dim]

        # Apply degree normalization if enabled
        if self.degree_norm:
            # Compute degrees
            out_degree = torch.zeros(n_nodes, device=device)
            out_degree.scatter_add_(0, receivers, torch.ones_like(receivers, dtype=x.dtype))
            in_degree = torch.zeros(n_nodes, device=device)
            in_degree.scatter_add_(0, senders, torch.ones_like(senders, dtype=x.dtype))

            # Normalize by degree
            out_degree_norm = torch.rsqrt(torch.clamp(out_degree, min=1.0))
            in_degree_norm = torch.rsqrt(torch.clamp(in_degree, min=1.0))

            sender_embeddings = sender_embeddings * out_degree_norm[senders].unsqueeze(1)

        # Aggregate: mean of sender embeddings for each receiver
        aggregated = torch.zeros(n_nodes, self.embedding_dim, device=device)
        aggregated.index_add_(0, receivers, sender_embeddings)

        # Count edges per receiver
        edge_count = torch.zeros(n_nodes, device=device)
        edge_count.scatter_add_(0, receivers, torch.ones_like(receivers, dtype=x.dtype))
        edge_count = torch.clamp(edge_count, min=1.0)

        # Compute mean
        aggregated = aggregated / edge_count.unsqueeze(1)

        # Apply degree normalization to aggregated if enabled
        if self.degree_norm:
            aggregated = aggregated * in_degree_norm.unsqueeze(1)

        # Concatenate original and aggregated embeddings
        combined = torch.cat([x, aggregated], dim=1)

        # Project back to embedding dimension
        output = self.linear(combined)

        return output


class NodeEncoder(nn.Module):
    """Encodes nodes into embeddings using a two-layer GraphSAGE model."""

    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        last_layer_self: bool,
        degree_norm: bool,
        dropout_rate: float,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.last_layer_self = last_layer_self
        self.degree_norm = degree_norm
        self.dropout_rate = dropout_rate

        # Node embeddings for all nodes
        self.node_embeddings = nn.Embedding(n_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        # Graph convolution layers
        self.conv1 = SAGEConv(
            embedding_dim,
            with_self=True,
            degree_norm=degree_norm,
        )
        self.conv2 = SAGEConv(
            embedding_dim,
            with_self=last_layer_self,
            degree_norm=degree_norm,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(
        self,
        node_ids: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Encodes the nodes of a graph into embeddings.

        Args:
            node_ids: Global node IDs
            senders: Source node indices
            receivers: Target node indices
            training: Whether in training mode

        Returns:
            Node embeddings
        """
        # Get embeddings for all nodes in graph
        x = self.node_embeddings(node_ids)

        # First convolution layer
        x = self.conv1(x, senders, receivers, self.n_nodes)
        x = self.relu(x)
        if training:
            x = self.dropout(x)

        # Second convolution layer
        x = self.conv2(x, senders, receivers, self.n_nodes)

        return x


class LinkPredictor(nn.Module):
    """Predicts interaction scores for pairs of node embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        n_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        # Build MLP layers
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Final output layer
        layers.append(nn.Linear(embedding_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        sender_embeddings: torch.Tensor,
        receiver_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes scores for node pairs.

        Args:
            sender_embeddings: Embeddings of sender nodes
            receiver_embeddings: Embeddings of receiver nodes

        Returns:
            Scores for each pair
        """
        # Element-wise multiplication
        x = sender_embeddings * receiver_embeddings

        # MLP transformation
        x = self.mlp(x)

        # Squeeze to 1D
        return x.squeeze(-1)


class DdiModel(nn.Module):
    """Graph-based model for predicting drug-drug interactions (DDIs)."""

    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        dropout_rate: float,
        last_layer_self: bool,
        degree_norm: bool,
        n_mlp_layers: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.last_layer_self = last_layer_self
        self.degree_norm = degree_norm
        self.n_mlp_layers = n_mlp_layers

        # Node encoder
        self.node_encoder = NodeEncoder(
            n_nodes=n_nodes,
            embedding_dim=embedding_dim,
            last_layer_self=last_layer_self,
            degree_norm=degree_norm,
            dropout_rate=dropout_rate,
        )

        # Link predictor
        self.link_predictor = LinkPredictor(
            embedding_dim=embedding_dim,
            n_layers=n_mlp_layers,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        node_ids: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        pairs: Dict[str, torch.Tensor],
        training: bool = False,
        is_pred: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Generates interaction scores for node pairs.

        Args:
            node_ids: Global node IDs
            senders: Source node indices for edges
            receivers: Target node indices for edges
            pairs: Dictionary with 'pos' and 'neg' keys containing node pair indices
            training: Whether in training mode
            is_pred: Whether in prediction mode (for inference)

        Returns:
            Scores for node pairs
        """
        # Compute node embeddings
        h = self.node_encoder(node_ids, senders, receivers, training=training)

        if is_pred:
            # Inference mode: score arbitrary node pairs
            scores = self.link_predictor(
                h[pairs[:, 0]], h[pairs[:, 1]]
            )
        else:
            # Training mode: score positive and negative pairs
            pos_pairs = pairs["pos"]
            neg_pairs = pairs["neg"]

            scores = {
                "pos": self.link_predictor(
                    h[pos_pairs[:, 0]], h[pos_pairs[:, 1]]
                ),
                "neg": self.link_predictor(
                    h[neg_pairs[:, 0]], h[neg_pairs[:, 1]]
                ),
            }

        return scores

    @staticmethod
    def add_mean_embedding(embeddings: torch.Tensor) -> torch.Tensor:
        """Concatenates a mean embedding to the existing embeddings."""
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        return torch.cat([embeddings, mean_embedding], dim=0)
