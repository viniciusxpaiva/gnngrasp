import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, LayerNorm


class GINNodeClassifier(nn.Module):
    """
    Node-level classifier that predicts binding site residues within subgraphs,
    using GINConv layers and an MLP.

    Supports dynamic number of layers and choice of normalization (BatchNorm or LayerNorm).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        num_layers,
        norm_type,
    ):
        """
        Initialize the NodeClassifier.

        Args:
            input_dim (int): Dimension of input node features (e.g., 1280 for ESM embeddings).
            hidden_dim (int): Hidden dimension used in GIN layers and MLP.
            dropout (float): Dropout rate for regularization.
            num_layers (int): Number of GIN layers (minimum 1).
            norm_type (str): Normalization type ('batch' or 'layer').
        """
        super(GINNodeClassifier, self).__init__()

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

        # === Final MLP classifier for per-node prediction ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary output (logits)
        )

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (Tensor): Edge list [2, num_edges].

        Returns:
            Tensor: Logits for each node [num_nodes, 1].
        """
        for gin, norm in zip(self.gin_layers, self.norm_layers):
            x = gin(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        out = self.classifier(x)  # No sigmoid; handled by BCEWithLogitsLoss
        return out
