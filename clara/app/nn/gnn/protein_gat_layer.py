import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class ProteinGATLayer(MessagePassing):
    """
    A GAT-style layer with multi-head attention and residual connections,
    dynamically supporting edge features in the attention mechanism if provided.
    If edge features are not passed, it defaults to standard GAT behavior.
    """

    def __init__(
        self,
        node_in_channels: int,
        hidden_channels: int,
        dropout_prob: float,
        num_heads: int,
        use_edge_attr: bool,
    ):
        super().__init__(aggr=None)  # Manual aggregation across heads

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.output_dim = num_heads * hidden_channels
        self.input_dim = node_in_channels
        self.use_edge_attr = use_edge_attr

        # Head-specific linear projections for node features
        self.linear_msg = nn.ModuleList(
            [nn.Linear(node_in_channels, hidden_channels) for _ in range(num_heads)]
        )

        # Head-specific attention MLPs
        self.attn_mlp = nn.ModuleList()
        for _ in range(num_heads):
            input_size = 2 * hidden_channels
            self.attn_mlp.append(
                nn.Sequential(
                    nn.Linear(input_size, 1),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )

        self.attn_dropout = nn.Dropout(dropout_prob)

        # Residual projection if dimensions mismatch
        if self.output_dim != node_in_channels:
            self.residual_proj = nn.Linear(node_in_channels, self.output_dim)
        else:
            self.residual_proj = nn.Identity()

        # Final node feature update
        self.mlp_update = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(
        self,
        x_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GAT layer.

        Args:
            x_node_features (Tensor): Node features [num_nodes, node_in_channels].
            edge_index (Tensor): Edge index tensor [2, num_edges].
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim].

        Returns:
            Tensor: Updated node features [num_nodes, output_dim].
        """
        return self.propagate(
            edge_index=edge_index, x=x_node_features, edge_attr=edge_attr
        )

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute attention-weighted messages per head, optionally using edge features.

        Args:
            x_i (Tensor): Target node features [num_edges, node_in_channels].
            x_j (Tensor): Source node features [num_edges, node_in_channels].
            index (Tensor): Indices of target nodes [num_edges].
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim].

        Returns:
            Tensor: Messages stacked by head [num_heads, num_edges, hidden_channels].
        """
        head_messages = []

        for head_idx in range(self.num_heads):
            proj_i = self.linear_msg[head_idx](x_i)
            proj_j = self.linear_msg[head_idx](x_j)

            # Optionally concatenate edge features into attention input
            if self.use_edge_attr and edge_attr is not None:
                concat_input = torch.cat([proj_i, proj_j, edge_attr], dim=-1)
            else:
                concat_input = torch.cat([proj_i, proj_j], dim=-1)

            attn_scores = self.attn_mlp[head_idx](concat_input)
            attn_scores = softmax(attn_scores, index=index)
            attn_scores = self.attn_dropout(attn_scores)

            weighted_msg = proj_j * attn_scores
            head_messages.append(weighted_msg)

        return torch.stack(head_messages, dim=0)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr=None,
        dim_size=None,
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors per attention head.

        Args:
            inputs (Tensor): Messages [num_heads, num_edges, hidden_channels].
            index (Tensor): Target node indices [num_edges].
            dim_size (int): Total number of nodes.

        Returns:
            Tensor: Aggregated node features [num_nodes, output_dim].
        """
        num_nodes = dim_size if dim_size is not None else index.max().item() + 1

        aggregated_per_head = [
            torch.zeros(
                num_nodes, self.hidden_channels, device=inputs.device
            ).index_add_(0, index, inputs[head])
            for head in range(self.num_heads)
        ]

        return torch.cat(aggregated_per_head, dim=-1)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Final node update with residual connection and MLP.

        Args:
            aggr_out (Tensor): Aggregated features [num_nodes, output_dim].
            x (Tensor): Original node features [num_nodes, node_in_channels].

        Returns:
            Tensor: Updated node features [num_nodes, output_dim].
        """
        x_res = self.residual_proj(x)
        update_input = torch.cat([x_res, aggr_out], dim=-1)
        updated = self.mlp_update(update_input)
        return updated + x_res
