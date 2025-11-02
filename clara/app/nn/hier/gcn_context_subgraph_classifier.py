import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    BatchNorm,
    LayerNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class GCNContextSubgraphClassifier(nn.Module):
    """
    Graph-level classifier for subgraphs using a multi-layer GCN (GCNConv).

    It can be used either as:
        (1) A standalone subgraph classifier (forward -> logits), or
        (2) A context provider for node-level models (get node/subgraph embeddings).

    Key features:
        - Flexible number of GCN layers.
        - Configurable normalization (BatchNorm or LayerNorm).
        - Configurable pooling (add / mean / max).
        - MLP head for graph-level classification.

    Note:
        `num_heads` is kept in the constructor to preserve a drop-in compatible API
        with the GAT version, but is not used by GCNConv.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        norm_type: str,
        pool_type: str,
    ):
        """
        Initialize the GCN-based Subgraph Classifier.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Hidden dimension for intermediate layers.
            output_dim (int): Output dimension (1 for binary classification).
            num_heads (int): Unused; present for API compatibility with GAT version.
            dropout (float): Dropout probability for regularization.
            num_layers (int): Number of GCN layers (>= 1).
            norm_type (str): Type of normalization ('batch' or 'layer').
            pool_type (str): Type of global pooling ('add', 'mean', or 'max').
        """
        super().__init__()

        assert num_layers >= 1, "Number of layers must be >= 1"
        assert norm_type in [
            "batch",
            "layer",
        ], "Normalization must be 'batch' or 'layer'"

        self.dropout = dropout
        self.num_layers = num_layers

        # === Build GCN layers and normalization layers ===
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim

            self.gcn_layers.append(
                GCNConv(in_channels=in_dim, out_channels=out_dim, normalize=True)
            )

            norm_layer = (
                BatchNorm(out_dim) if norm_type == "batch" else LayerNorm(out_dim)
            )
            self.norm_layers.append(norm_layer)

        # === Pooling function for graph-level embedding ===
        if pool_type == "add":
            self.pool = global_add_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pool type '{pool_type}'")

        # === Final embedding dimension after the stack ===
        self.graph_emb_dim = hidden_dim

        # === Graph-level MLP classifier ===
        self.graph_head = nn.Sequential(
            nn.Linear(self.graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),  # logits (e.g., for BCEWithLogitsLoss)
        )

    # ---------------------------------------------------------
    # Core building blocks
    # ---------------------------------------------------------

    def encode_nodes(self, x, edge_index, edge_weight=None):
        """
        Compute node embeddings after the GCN stack.

        Args:
            x (Tensor): Input node features [num_nodes, input_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            edge_weight (Tensor, optional): Edge weights [num_edges]. Default: None.

        Returns:
            Tensor: Node embeddings [num_nodes, hidden_dim].
        """
        h = x
        for conv, norm in zip(self.gcn_layers, self.norm_layers):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def encode_subgraphs(self, x, edge_index, batch, edge_weight=None):
        """
        Compute subgraph-level embeddings via node embeddings + global pooling.

        Args:
            x (Tensor): Input node features [num_nodes, input_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch assignment of nodes to subgraphs [num_nodes].
            edge_weight (Tensor, optional): Edge weights [num_edges]. Default: None.

        Returns:
            Z (Tensor): Subgraph embeddings [num_subgraphs, hidden_dim].
            H (Tensor): Node embeddings [num_nodes, hidden_dim].
        """
        H = self.encode_nodes(x, edge_index, edge_weight=edge_weight)  # [N, hidden_dim]
        Z = self.pool(H, batch)  # [B, hidden_dim]
        return Z, H

    # ---------------------------------------------------------
    # Public forward methods
    # ---------------------------------------------------------

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Standard forward for graph-level classification.

        Args:
            x (Tensor): Node features [num_nodes, input_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch assignment of nodes to subgraphs [num_nodes].
            edge_weight (Tensor, optional): Edge weights [num_edges]. Default: None.

        Returns:
            Tensor: Graph-level logits [num_subgraphs, output_dim].
        """
        Z, _ = self.encode_subgraphs(x, edge_index, batch, edge_weight=edge_weight)
        logits = self.graph_head(Z)  # [num_subgraphs, output_dim]
        return logits

    def get_subgraph_embeddings(self, x, edge_index, batch, edge_weight=None):
        """
        Return subgraph-level embeddings (before the classification head).
        Useful for bias fusion with node-level GNNs.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch (Tensor): Batch assignment of nodes to subgraphs.
            edge_weight (Tensor, optional): Edge weights.

        Returns:
            Tensor: Subgraph embeddings [num_subgraphs, hidden_dim].
        """
        Z, _ = self.encode_subgraphs(x, edge_index, batch, edge_weight=edge_weight)
        return Z

    def get_node_embeddings(self, x, edge_index, edge_weight=None):
        """
        Return node-level embeddings (after the GCN stack).
        Useful for node-level tasks or visualization.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_weight (Tensor, optional): Edge weights.

        Returns:
            Tensor: Node embeddings [num_nodes, hidden_dim].
        """
        return self.encode_nodes(x, edge_index, edge_weight=edge_weight)

    # ---------------------------------------------------------
    # Feature selection helpers
    # ---------------------------------------------------------

    def _select_x_from_data(self, data: Data) -> torch.Tensor:
        """
        Choose node features from a Data/Batch object using the configured `x_key`,
        falling back to `.x` if the key is missing.

        Returns:
            Tensor: Node feature matrix [num_nodes, input_dim_like].
        """
        x = getattr(data, "x", None)
        if x is None:
            raise AttributeError(f"'x' found in the provided Data/Batch.")
        return x

    # ---------------------------------------------------------
    # Convenience methods (NEW): work directly with Data/Batch
    # ---------------------------------------------------------

    def forward_from_data(self, data: Data) -> torch.Tensor:
        """
        Forward pass for graph-level classification using a PyG Data/Batch.
        It selects features via `x_key` (falls back to `.x`), then runs the model.

        Expected attributes on `data`:
            - edge_index [2, E]
            - batch [N]  (when working with batched subgraphs)
            - optional: edge_weight [E]
            - one of: data.<x_key> or data.x  with shape [N, Din]

        Returns:
            Tensor: Graph-level logits [num_subgraphs, output_dim].
        """
        x = self._select_x_from_data(data)
        Z, _ = self.encode_subgraphs(
            x=x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_weight=getattr(data, "edge_weight", None),
        )
        return self.graph_head(Z)

    def get_subgraph_embeddings_from_data(self, data: Data) -> torch.Tensor:
        """
        Convenience method returning subgraph embeddings from a Data/Batch.

        Returns:
            Tensor: Subgraph embeddings [num_subgraphs, hidden_dim].
        """
        x = self._select_x_from_data(data)
        Z, _ = self.encode_subgraphs(
            x=x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_weight=getattr(data, "edge_weight", None),
        )
        return Z

    def get_node_embeddings_from_data(self, data: Data) -> torch.Tensor:
        """
        Convenience method returning node embeddings from a Data/Batch.

        Returns:
            Tensor: Node embeddings [num_nodes, hidden_dim].
        """
        x = self._select_x_from_data(data)
        return self.encode_nodes(
            x=x,
            edge_index=data.edge_index,
            edge_weight=getattr(data, "edge_weight", None),
        )
