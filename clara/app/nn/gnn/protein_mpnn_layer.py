import torch
from torch_geometric.nn import MessagePassing


class ProteinMPNNLayer(MessagePassing):
    """
    A single MPNN layer for protein graphs.

    This layer performs:
    - Message passing between nodes, combining sender node features and edge features.
    - Aggregation of received messages.
    - Update of node features based on original features and aggregated messages.
    """

    def __init__(
        self, node_in_channels, edge_in_channels, hidden_channels, dropout_prob
    ):
        """
        Initialize the GNN layer.

        Args:
            node_in_channels (int): Number of input node features.
            edge_in_channels (int): Number of input edge features.
            hidden_channels (int): Number of output hidden units.
        """
        super().__init__(aggr="add")  # Mean aggregation of neighbor messages

        self.edge_in_channels = edge_in_channels

        # MLP to compute messages from neighbors (source node + edge features)
        self.mlp_message = torch.nn.Sequential(
            torch.nn.Linear(node_in_channels + edge_in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )

        # MLP to update node features (original node + aggregated messages)
        self.mlp_update = torch.nn.Sequential(
            torch.nn.Linear(node_in_channels + hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x_node_features, edge_index, edge_features):
        """
        Forward pass through the GNN layer.

        Args:
            x_node_features (Tensor): Node feature matrix [num_nodes, node_in_channels].
            edge_index (Tensor): Graph connectivity matrix [2, num_edges].
            edge_features (Tensor): Edge feature matrix [num_edges, edge_in_channels].

        Returns:
            Tensor: Updated node features [num_nodes, hidden_channels].
        """
        return self.propagate(
            edge_index=edge_index, x=x_node_features, edge_attr=edge_features
        )

    def message(self, x_j, edge_attr):
        """
        Construct messages from source nodes to target nodes.

        Args:
            x_j (Tensor): Source node features for each edge.
            edge_attr (Tensor): Edge features.

        Returns:
            Tensor: Messages for each edge.
        """
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros(
                (x_j.size(0), self.edge_in_channels), device=x_j.device
            )

        combined_features = torch.cat([x_j, edge_attr], dim=-1)
        message = self.mlp_message(combined_features)
        return message

    def update(self, aggr_out, x):
        """
        Update node features after aggregation.

        Args:
            aggr_out (Tensor): Aggregated messages for each node.
            x (Tensor): Original node features.

        Returns:
            Tensor: Updated node features.
        """
        combined_features = torch.cat([x, aggr_out], dim=-1)
        updated_node = self.mlp_update(combined_features)
        return updated_node
