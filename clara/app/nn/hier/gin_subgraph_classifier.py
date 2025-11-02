import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    BatchNorm,
    LayerNorm,
)


class GINSubgraphClassifier(nn.Module):
    """
    Graph-level classifier for protein subgraphs using GINConv layers.

    The model predicts whether a subgraph contains at least one binding site,
    supporting configurable depth and normalization strategy.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
        num_layers,
        norm_type,
        pool_type,
    ):
        """
        Initialize the GIN-based Subgraph Classifier.

        Args:
            input_dim (int): Node feature dimension (e.g., 1280 for ESM embeddings).
            hidden_dim (int): Hidden dimension for GIN layers and MLP.
            output_dim (int): Output size (1 for binary classification).
            dropout (float): Dropout rate for regularization.
            num_layers (int): Number of GIN layers.
            norm_type (str): Type of normalization ('batch' or 'layer').
            pool_type (str): Type of pooling function ('add', 'mean' or 'max').
        """
        super(GINSubgraphClassifier, self).__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ["batch", "layer"], "norm_type must be 'batch' or 'layer'"

        self.dropout = dropout
        self.num_layers = num_layers

        # === Build GIN layers dynamically ===
        self.gin_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            gin_layer = GINConv(mlp)
            self.gin_layers.append(gin_layer)

            norm_layer = (
                BatchNorm(hidden_dim) if norm_type == "batch" else LayerNorm(hidden_dim)
            )
            self.norm_layers.append(norm_layer)

        # === Pooling function ===
        if pool_type == "add":
            self.pool = global_add_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(
                f"[ERROR] Unsupported pool type '{pool_type}'. Use 'add', 'mean', or 'max'."
            )

        # === MLP Classifier ===
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),  # Output logits
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (Tensor): Edge connectivity [2, num_edges].
            batch (Tensor): Batch vector mapping each node to its graph [num_nodes].

        Returns:
            Tensor: Graph-level logits [num_graphs, output_dim].
        """
        for gin, norm in zip(self.gin_layers, self.norm_layers):
            x = gin(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Global pooling to obtain graph-level embedding
        x = self.pool(x, batch)  # [num_graphs, hidden_dim]

        # Final MLP classifier
        out = self.mlp(x)  # [num_graphs, output_dim]
        return out
