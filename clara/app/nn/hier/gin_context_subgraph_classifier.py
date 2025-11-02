import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    BatchNorm,
    LayerNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class GINContextSubgraphClassifier(nn.Module):
    """
    Graph-level classifier for subgraphs using multi-layer GINConv.

    Roles:
      (1) Standalone subgraph classifier (via `forward`).
      (2) Context provider for node-level models — exposes subgraph/node embeddings
          via `get_subgraph_embeddings`, `get_node_embeddings`, or `forward_with_embeddings`.

    Design notes:
      - GIN is strong for multiset discrimination (sum aggregator + MLP).
      - Simple and fast; often an excellent cost/benefit as context encoder.
      - Lightweight residuals when widths match; optional dropout between blocks.
      - Global pooling selectable: add / mean / max.

    `graph_emb_dim`:
      - Width of the subgraph embedding from this model (set to `hidden_dim`).
      - Use it as the `context_dim` for your node-level context model.
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
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"

        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self._use_layer_dropout = True  # consistent with other context encoders

        # ---- Build GIN layers + norms ----
        self.gin_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(mlp, train_eps=True)
            self.gin_layers.append(conv)

            norm = (
                BatchNorm(hidden_dim) if norm_type == "batch" else LayerNorm(hidden_dim)
            )
            self.norm_layers.append(norm)

            in_dim = hidden_dim

        # ---- Pool selection ----
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

        # ---- Final embedding width ----
        self.graph_emb_dim = hidden_dim

        # ---- Graph-level MLP head (logits) ----
        self.graph_head = nn.Sequential(
            nn.Linear(self.graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim),  # logits
        )

    # ---------------- Public API ----------------

    def forward(self, x, edge_index, batch):
        """Graph-level classification: returns logits [num_subgraphs, output_dim]."""
        Z, _ = self._encode_subgraphs(x, edge_index, batch)
        return self.graph_head(Z)

    def forward_with_embeddings(self, x, edge_index, batch):
        """
        Single pass that returns:
          - graph-level logits
          - subgraph embeddings (Z)
          - node embeddings (H)
        """
        Z, H = self._encode_subgraphs(x, edge_index, batch)
        logits = self.graph_head(Z)
        return logits, Z, H

    def get_subgraph_embeddings(self, x, edge_index, batch):
        """Return subgraph embeddings [num_subgraphs, graph_emb_dim]."""
        Z, _ = self._encode_subgraphs(x, edge_index, batch)
        return Z

    def get_node_embeddings(self, x, edge_index):
        """Return node embeddings [num_nodes, hidden_dim] after the GIN stack."""
        return self._encode_nodes(x, edge_index)

    # ---------------- Core blocks ----------------

    def _encode_nodes(self, x, edge_index):
        """
        GIN stack with optional light residuals and dropout.
        Residuals are applied only when input/output widths match.
        """
        h = x
        for conv, norm in zip(self.gin_layers, self.norm_layers):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

            if h_in.shape[1] == h.shape[1]:
                h = h + h_in

            if self._use_layer_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _encode_subgraphs(self, x, edge_index, batch):
        """Node embeddings → global pooling → subgraph embeddings."""
        H = self._encode_nodes(x, edge_index)  # [N, hidden_dim]
        Z = self.pool(H, batch)  # [B, hidden_dim]
        return Z, H
