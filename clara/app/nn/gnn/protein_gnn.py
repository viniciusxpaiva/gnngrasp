import torch
from app.nn.gnn.protein_mpnn_layer import ProteinMPNNLayer
from app.nn.gnn.protein_gat_layer import ProteinGATLayer
from app.nn.gnn.protein_gin_layer import ProteinGINLayer


class ProteinGNN(torch.nn.Module):
    """
    A configurable multi-layer Graph Neural Network (GNN) for protein graphs.

    This model supports different message-passing architectures (MPNN, GAT, GIN),
    allows the optional use of edge features, and supports multi-head attention
    in GAT-style layers. The model is suitable for node-level classification tasks,
    such as identifying binding sites in protein structures.
    """

    def __init__(
        self,
        embeddings_dim: int,
        property_dim: int,
        projected_embd_dim: int,
        hidden_channels: int,
        num_gnn_layers: int,
        layer_type: str,
        dropout: float,
        use_edge_attr: bool,
        num_heads: int,
        use_embd_projection: bool,
    ):
        """
        Initializes the ProteinGNN model with configurable GNN architecture.

        Args:
            embeddings_dim (int): Dimensionality of raw input embeddings (e.g., from ProstT5).
            projected_embd_dim (int): Output dimension of the embedding projector.
            edge_dim (int): Dimensionality of input edge features.
            hidden_channels (int): Hidden dimension for GNN layers.
            num_gnn_layers (int): Number of stacked GNN layers.
            layer_type (str): Type of GNN layer ("MPNN", "GAT", or "GIN").
            dropout (float): Dropout probability used in each GNN layer.
            use_edge_attr (bool): Whether to use edge features in message passing.
            num_heads (int): Number of attention heads (for GAT only).
            use_embd_projection (bool): If True, apply projection to node embeddings.
        """

        super().__init__()

        self.use_edge_attr = use_edge_attr
        self.use_embd_projection = use_embd_projection
        self.num_heads = num_heads
        self.layer_type = layer_type.upper()

        if use_embd_projection:
            # Project high-dimensional embeddings to a lower-dimensional space
            self.embedding_projector = torch.nn.Linear(
                embeddings_dim, projected_embd_dim
            )
            gnn_input_dim = projected_embd_dim + property_dim
        else:
            gnn_input_dim = embeddings_dim

        """

        if use_embd_projection:
            # Project high-dimensional embeddings to a lower-dimensional space
            self.embedding_projector = torch.nn.Linear(property_dim, projected_embd_dim)
            gnn_input_dim = projected_embd_dim + embeddings_dim
        else:
            gnn_input_dim = embeddings_dim

        """

        # Select GNN layer class and calculate output dimension
        if self.layer_type == "GAT":
            LayerClass = ProteinGATLayer
            out_dim = num_heads * hidden_channels
        elif self.layer_type == "MPNN":
            LayerClass = ProteinMPNNLayer
            out_dim = hidden_channels
        elif self.layer_type == "GIN":
            LayerClass = ProteinGINLayer
            out_dim = hidden_channels
        else:
            raise ValueError(f"[ERROR] Unknown layer type: {layer_type}")

        self.layers = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = gnn_input_dim if i == 0 else out_dim

            if self.layer_type == "GAT":
                self.layers.append(
                    LayerClass(
                        node_in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        dropout_prob=dropout,
                        num_heads=num_heads,
                        use_edge_attr=use_edge_attr,
                    )
                )
            else:
                self.layers.append(
                    LayerClass(
                        node_in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        dropout_prob=dropout,
                    )
                )

        self.output_layer = torch.nn.Linear(out_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        # x_properties: torch.Tensor,
        edge_index: torch.Tensor,
        # edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:

        if self.use_embd_projection:
            x_embd_proj = self.embedding_projector(x)
            # h = torch.cat(
            #    [x_embd_proj, x_properties], dim=1
            # )  # Concatenate with physicochemical properties
            print(h)
        else:
            h = x

        for layer in self.layers:
            h = layer(h, edge_index)

        return self.output_layer(h)
