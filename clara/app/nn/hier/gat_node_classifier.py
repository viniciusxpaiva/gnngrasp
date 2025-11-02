import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, LayerNorm


class GATNodeClassifier(nn.Module):
    """
    Node-level classifier to predict binding site residues within subgraphs.

    The model supports multiple GATv2Conv layers combined with either BatchNorm or LayerNorm,
    followed by an MLP for per-node binary classification.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_layers,
        norm_type,
    ):
        """
        Initialize the GAT-based Node Classifier.

        Args:
            input_dim (int): Dimension of node input features (e.g., ESM embeddings).
            hidden_dim (int): Hidden dimension used in GAT layers and MLP.
            num_heads (int): Number of attention heads in the first GAT layer.
            dropout (float): Dropout rate for regularization.
            num_layers (int): Number of GAT layers.
            norm_type (str): Type of normalization ('batch' or 'layer').
        """
        super(GATNodeClassifier, self).__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ["batch", "layer"], "norm_type must be 'batch' or 'layer'"

        self.dropout = dropout
        self.num_layers = num_layers

        # === Build GAT layers ===
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim = input_dim
                out_dim = hidden_dim
                heads = num_heads
            else:
                in_dim = hidden_dim * num_heads if layer_idx == 1 else hidden_dim
                out_dim = hidden_dim
                heads = 1  # Single head after first layer

            gat_layer = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                dropout=dropout,
                concat=True,
            )
            self.gat_layers.append(gat_layer)

            norm_dim = out_dim * heads if layer_idx == 0 else out_dim
            norm_layer = (
                BatchNorm(norm_dim) if norm_type == "batch" else LayerNorm(norm_dim)
            )
            self.norm_layers.append(norm_layer)

        # === Determine final embedding dimension ===
        if num_layers == 1:
            mlp_input_dim = hidden_dim * num_heads
        else:
            mlp_input_dim = hidden_dim

        # === MLP classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Logits for BCEWithLogitsLoss
        )

    def forward(self, x, edge_index):
        """
        Forward pass for node classification.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (Tensor): Edge list [2, num_edges].

        Returns:
            Tensor: Logits for each node [num_nodes, 1].
        """
        for gat, norm in zip(self.gat_layers, self.norm_layers):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        out = self.classifier(x)  # [num_nodes, 1]
        return out
