import torch
from app.nn.gnn.protein_mpnn_layer import ProteinMPNNLayer
from app.nn.gnn.protein_gat_layer import ProteinGATLayer
from app.nn.gnn.protein_gin_layer import ProteinGINLayer
from torch_geometric.nn import global_mean_pool


class EgoProteinGNN(torch.nn.Module):
    """
    A multi-layer GNN model for protein graphs that supports both standard GNN
    and GAT-style attention layers. It combines node embeddings and optionally
    handcrafted features, followed by stacked message passing layers.
    """

    def __init__(
        self,
        node_emb_in_channels,
        edge_in_channels,
        hidden_channels,
        num_gnn_layers,
        layer_type,
        dropout,
    ):
        """
        Initialize the GNN model.

        Args:
            node_emb_in_channels (int): Input dimension of node embeddings.
            edge_in_channels (int): Input dimension of edge features.
            hidden_channels (int): Hidden layer dimension.
            num_gnn_layers (int): Number of stacked GNN layers.
            layer_type (str): Type of GNN layer to use ("GNN" or "GAT").
        """
        super().__init__()

        self.edge_in_channels = edge_in_channels

        # Select GNN layer type
        if layer_type.upper() == "GAT":
            LayerClass = ProteinGATLayer
        elif layer_type.upper() == "MPNN":
            LayerClass = ProteinMPNNLayer
        elif layer_type.upper() == "GIN":
            LayerClass = ProteinGINLayer
        else:
            raise ValueError(
                f"[ERROR] Unknown layer type: {layer_type}. Use 'MPNN', 'GIN' or 'GAT'."
            )

        # Initialize GNN layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            LayerClass(node_emb_in_channels, edge_in_channels, hidden_channels, dropout)
        )

        for _ in range(num_gnn_layers - 1):
            self.layers.append(
                LayerClass(hidden_channels, edge_in_channels, hidden_channels, dropout)
            )

        # Final classification layer (logits per node)
        out_channels = 2
        self.output_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_embeddings, edge_index, edge_features, batch):
        """
        Forward pass for graph-level classification (one prediction per ego-graph).

        Args:
            x_embeddings (Tensor): Node embeddings [num_nodes, node_feat_dim]
            edge_index (Tensor): Edge indices [2, num_edges]
            edge_features (Tensor or None): Edge features [num_edges, edge_feat_dim]
            batch (Tensor): Batch vector mapping each node to its ego-graph [num_nodes]

        Returns:
            Tensor: Output logits per graph [num_graphs, num_classes]
        """
        h = x_embeddings
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)

        # Pool node embeddings to get graph-level representation
        graph_embedding = global_mean_pool(h, batch)

        # Classification
        logits = self.output_layer(graph_embedding)
        return logits
