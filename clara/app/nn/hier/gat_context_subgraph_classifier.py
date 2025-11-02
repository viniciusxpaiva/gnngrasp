import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    BatchNorm,
    LayerNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class GATContextSubgraphClassifier(nn.Module):
    """
    Graph-level classifier for subgraphs using multi-layer GATv2Conv.

    Roles:
      (1) Standalone subgraph classifier (forward -> logits)
      (2) Context provider for node-level models (get node/subgraph embeddings)

    Features:
      - Multi-head attention ONLY on the first layer (concat=True).
      - Configurable normalization ('batch' | 'layer') per block.
      - Optional lightweight residual when channel sizes match.
      - Dropout between blocks and inside the MLP head.
      - Global pooling: add | mean | max.
      - Convenience methods to work directly with PyG Data/Batch.

    Notes:
      - `graph_emb_dim` is the subgraph embedding width produced by the encoder.
      - If you use a node-level context model that expects `context_dim`,
        set `context_dim == graph_emb_dim`.
      - Supports feature selection via `self.x_key` (defaults to "x").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
        pool_type: str,
    ):
        """
        Args:
            input_dim  : Node feature dimension.
            hidden_dim : Hidden width for blocks and head.
            output_dim : Output dim (1 for binary classification).
            num_heads  : #heads in the FIRST GAT layer (concat=True).
            dropout    : Dropout prob (blocks + head).
            num_layers : #GAT layers (>=1).
            norm_type  : 'batch' or 'layer'.
            pool_type  : 'add' | 'mean' | 'max'.
        """
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        assert norm_type in ("batch", "layer"), "norm_type must be 'batch' or 'layer'"

        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self._use_layer_dropout = True

        # Optional external feature key (can be set after init)
        # e.g., model.x_key = "x_view_struct"
        self.x_key = "x"

        # --- GAT blocks ---
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_dim = input_dim
                out_dim = hidden_dim
                heads = num_heads
            else:
                # After first layer we collapse to single-head width `hidden_dim`
                # First block output width is hidden_dim * num_heads (concat=True)
                in_dim = hidden_dim * num_heads if layer_idx == 1 else hidden_dim
                out_dim = hidden_dim
                heads = 1

            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                dropout=self.dropout,
                concat=True,  # keep channels widened in the 1st layer
            )
            self.gat_layers.append(conv)

            norm_dim = out_dim * heads if layer_idx == 0 else out_dim
            norm = BatchNorm(norm_dim) if norm_type == "batch" else LayerNorm(norm_dim)
            self.norm_layers.append(norm)

        # --- Global pooling ---
        if pool_type == "add":
            self.pool = global_add_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"[ERROR] Unsupported pool type '{pool_type}'.")

        # --- Final embedding width ---
        # If only 1 GAT layer (multi-head), width is hidden_dim * heads; otherwise hidden_dim.
        self.graph_emb_dim = hidden_dim * num_heads if num_layers == 1 else hidden_dim

        # --- Graph-level head ---
        self.graph_head = nn.Sequential(
            nn.Linear(self.graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim),  # logits (no activation)
        )

    # ---------------------------------------------------------
    # Core encoder blocks
    # ---------------------------------------------------------

    def encode_nodes(self, x, edge_index, edge_attr=None):
        """
        Encode node embeddings through the GAT stack.

        Residuals:
          - Add h := h + h_in only when channel sizes match.

        Dropout:
          - If `_use_layer_dropout` is True, apply dropout after activations.
        """
        h = x
        for conv, norm in zip(self.gat_layers, self.norm_layers):
            h_in = h
            # GATv2 supports optional edge_attr; pass-through if provided
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.relu(h)

            if h_in.shape[1] == h.shape[1]:
                h = h + h_in  # lightweight residual

            if self._use_layer_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def encode_subgraphs(self, x, edge_index, batch, edge_attr=None):
        """
        Node encoder + global pooling -> subgraph embeddings.
        """
        H = self.encode_nodes(x, edge_index, edge_attr=edge_attr)  # [N, d_node]
        Z = self.pool(H, batch)  # [B, d_graph]
        return Z, H

    # ---------------------------------------------------------
    # Public forward methods (tensor API)
    # ---------------------------------------------------------

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Graph-level logits for a (batched) set of subgraphs.
        """
        Z, _ = self.encode_subgraphs(x, edge_index, batch, edge_attr=edge_attr)
        logits = self.graph_head(Z)
        return logits

    def get_subgraph_embeddings(self, x, edge_index, batch, edge_attr=None):
        """
        Subgraph embeddings before the classification head.
        """
        Z, _ = self.encode_subgraphs(x, edge_index, batch, edge_attr=edge_attr)
        return Z

    def get_node_embeddings(self, x, edge_index, edge_attr=None):
        """
        Node embeddings after the GAT stack.
        """
        return self.encode_nodes(x, edge_index, edge_attr=edge_attr)

    # ---------------------------------------------------------
    # Feature selection helpers (Data/Batch API)
    # ---------------------------------------------------------

    def _select_x_from_data(self, data: Data) -> torch.Tensor:
        """
        Choose node features from a Data/Batch using `self.x_key` (defaults to '.x').
        Falls back to '.x' if the key is missing.
        """
        key = getattr(self, "x_key", "x")
        x = getattr(data, key, None)
        if x is None:
            # fallback to .x if custom key missing
            x = getattr(data, "x", None)
        if x is None:
            raise AttributeError(
                f"No node feature found. Tried '{key}' and 'x' in Data/Batch."
            )
        return x

    # ---------------------------------------------------------
    # Convenience methods (Data/Batch API)
    # ---------------------------------------------------------

    def forward_from_data(self, data: Data) -> torch.Tensor:
        """
        Forward pass using a PyG Data/Batch object.

        Expects:
          - data.edge_index [2, E]
          - data.batch [N]
          - optional: data.edge_attr [E, *]
          - features in data.<x_key> or data.x
        """
        x = self._select_x_from_data(data)
        Z, _ = self.encode_subgraphs(
            x=x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_attr=getattr(data, "edge_attr", None),
        )
        return self.graph_head(Z)

    def get_subgraph_embeddings_from_data(self, data: Data) -> torch.Tensor:
        """
        Convenience: subgraph embeddings directly from Data/Batch.
        """
        x = self._select_x_from_data(data)
        Z, _ = self.encode_subgraphs(
            x=x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_attr=getattr(data, "edge_attr", None),
        )
        return Z

    def get_node_embeddings_from_data(self, data: Data) -> torch.Tensor:
        """
        Convenience: node embeddings directly from Data/Batch.
        """
        x = self._select_x_from_data(data)
        return self.encode_nodes(
            x=x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
        )
