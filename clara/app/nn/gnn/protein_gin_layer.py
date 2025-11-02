import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class ProteinGINLayer(MessagePassing):
    """
    A GIN-style GNN layer for protein graphs.

    This layer performs:
    - Summation of messages from neighboring nodes (optionally includes edge features).
    - Transformation of aggregated message + self feature via an MLP.
    - Suitable for node-level classification where structural differences are important.
    """

    def __init__(
        self, node_in_channels, edge_in_channels, hidden_channels, dropout_prob
    ):
        """
        Initialize the GIN layer.

        Args:
            node_in_channels (int): Number of input node features.
            edge_in_channels (int): Number of input edge features.
            hidden_channels (int): Number of hidden/output units.
            dropout_prob (float): Dropout probability for MLPs.
        """
        super().__init__(aggr="add")  # GIN uses summation aggregation

        self.edge_in_channels = edge_in_channels

        # Learnable epsilon (initialized as 0.0)
        self.eps = nn.Parameter(torch.zeros(1))

        # MLP for transforming aggregated neighborhood features
        self.mlp = nn.Sequential(
            nn.Linear(node_in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x_node_features, edge_index, edge_features):
        """
        Forward pass of the layer.

        Args:
            x_node_features (Tensor): Node features [num_nodes, node_in_channels].
            edge_index (Tensor): Edge indices [2, num_edges].
            edge_features (Tensor): Edge features (ignored in GIN).

        Returns:
            Tensor: Updated node features [num_nodes, hidden_channels].
        """
        return self.propagate(edge_index=edge_index, x=x_node_features)

    def message(self, x_j):
        """
        GIN simply passes neighbor features as message.

        Args:
            x_j (Tensor): Source node features [num_edges, node_in_channels].

        Returns:
            Tensor: Messages passed to each target node.
        """
        return x_j

    def update(self, aggr_out, x):
        """
        Combine self-feature and aggregated messages (summation), then apply MLP.

        Args:
            aggr_out (Tensor): Aggregated messages [num_nodes, node_in_channels].
            x (Tensor): Original node features [num_nodes, node_in_channels].

        Returns:
            Tensor: Updated node features [num_nodes, hidden_channels].
        """
        out = (1 + self.eps) * x + aggr_out  # GIN: add self-loop explicitly
        return self.mlp(out)
