import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm


class GCNNodeClassifier(nn.Module):
    """
    Node-level classifier to predict binding-site residues within subgraphs using GCN layers.

    Architecture:
      [GCN -> Norm -> ReLU] x L  →  MLP(Linear→ReLU→Dropout→Linear→1)
    Produces per-node logits suitable for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
    ):
        """
        Args:
            input_dim  : Dimension of node input features.
            hidden_dim : Hidden dimension for GCN layers and MLP.
            dropout    : Dropout probability used in MLP (and optionally between layers).
            num_layers : Number of GCN layers (>=1).
            norm_type  : 'batch' or 'layer' normalization after each GCN layer.
        """
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"

        self.dropout = dropout
        self.num_layers = num_layers

        # --- Build GCN stack ---
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            self.gcn_layers.append(GCNConv(in_channels=in_dim, out_channels=out_dim))
            norm = BatchNorm(out_dim) if norm_type == "batch" else LayerNorm(out_dim)
            self.norm_layers.append(norm)
            in_dim = out_dim  # next layer input

        # Final embedding dimension after the last GCN block:
        mlp_input_dim = hidden_dim

        # --- Per-node MLP classifier to produce a single logit per node ---
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [num_nodes, input_dim] node features.
            edge_index : [2, num_edges] graph connectivity (COO format).

        Returns:
            logits : [num_nodes, 1] per-node logits (NO sigmoid).
        """
        for gcn, norm in zip(self.gcn_layers, self.norm_layers):
            x = gcn(x, edge_index)  # message passing
            x = norm(x)  # feature-wise normalization
            x = F.relu(x)  # non-linearity

        logits = self.classifier(x)  # [N, 1]
        return logits
