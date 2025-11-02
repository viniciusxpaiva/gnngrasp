import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, LayerNorm


class SAGENodeClassifier(nn.Module):
    """
    Node-level classifier based on GraphSAGE.

    Stack of SAGEConv → Norm → ReLU (with optional dropout), followed by
    a small MLP head that produces one logit per node for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        norm_type: str = "batch",  # 'batch' | 'layer'
        aggr: str = "mean",  # 'mean' | 'max' | 'add' | 'sum' | 'lstm' (PyG supports multiple)
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"

        self.num_layers = num_layers
        self.dropout = dropout

        # === Build SAGE layers ===
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            SAGEConv(in_channels=input_dim, out_channels=hidden_dim, aggr=aggr)
        )
        self.norms.append(
            BatchNorm(hidden_dim) if norm_type == "batch" else LayerNorm(hidden_dim)
        )

        # Remaining layers (hidden_dim → hidden_dim)
        for _ in range(1, num_layers):
            self.convs.append(
                SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr=aggr)
            )
            self.norms.append(
                BatchNorm(hidden_dim) if norm_type == "batch" else LayerNorm(hidden_dim)
            )

        # Final embedding size is hidden_dim for any num_layers >= 1
        mlp_in = hidden_dim

        # === MLP classifier (per-node) ===
        self.classifier = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 1 logit per node → use with BCEWithLogitsLoss
        )

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): [num_nodes, input_dim]
            edge_index (LongTensor): [2, num_edges]
        Returns:
            Tensor: [num_nodes, 1] node logits
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)  # message passing with chosen aggregator
            x = norm(x)  # BN/LN over node features
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)  # [N, 1]
