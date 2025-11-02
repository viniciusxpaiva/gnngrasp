import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


class IntermediateSubgraphClassifier(nn.Module):
    """
    Graph-level classifier for intermediate subgraphs (i.e., sub-subgraphs),
    typically extracted from larger subgraphs that are known to contain binding sites.

    This model predicts whether a given sub-subgraph contains at least one binding site residue,
    using GATv2 layers for attention-based message passing and global pooling for graph representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
    ):
        """
        Initialize the IntermediateSubgraphClassifier.

        Args:
            input_dim (int): Dimensionality of node input features (e.g., 1280 for ESM embeddings).
            hidden_dim (int): Hidden dimensionality used throughout the network.
            output_dim (int): Output dimensionality (typically 1 for binary classification).
            num_heads (int): Number of attention heads in the first GATv2 layer.
            dropout (float): Dropout probability for regularization.
        """
        super(IntermediateSubgraphClassifier, self).__init__()

        # First GATv2Conv layer: multi-head attention
        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,  # Output shape: [hidden_dim * num_heads]
        )

        # Second GATv2Conv layer: single-head attention
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=True,
        )

        # Global pooling to convert node embeddings to graph-level embedding
        self.pool = global_add_pool  # Sum pooling across nodes per graph

        # Fully connected MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),  # Binary output (e.g., logits)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the intermediate subgraph classifier.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch vector assigning each node to a graph [num_nodes].

        Returns:
            Tensor: Output logits per graph [num_graphs, output_dim].
        """
        x = self.gat1(x, edge_index)
        x = F.relu(x)

        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # Pool node features into a graph-level vector
        x = self.pool(x, batch)  # [num_graphs, hidden_dim]

        # Final binary classification
        out = self.mlp(x)  # [num_graphs, output_dim]
        return out
