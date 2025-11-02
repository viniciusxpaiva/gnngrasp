import torch
import torch.nn as nn
from app.nn.cnn.protein_cnn import ProteinCNN
from app.nn.mlp.protein_mlp import ProteinMLP


class DeepGraspModel(nn.Module):
    """
    End-to-end architecture combining GNN, CNN, and MLP modules for residue-level binding site prediction.

    - The GNN module captures 3D spatial relationships between residues.
    - The CNN models sequential dependencies in the residue order.
    - The MLP performs final classification into binary labels (binding/non-binding).
    """

    def __init__(
        self,
        gnn_module: nn.Module,
        cnn_channels: int,
        cnn_kernel_size: int,
        cnn_dropout: float,
        mlp_hidden: int,
        mlp_dropout: float,
    ):
        """
        Initialize the DeepGraspModel with a preconfigured GNN and modular CNN+MLP.

        Args:
            gnn_module (nn.Module): A ProteinGNN instance.
            cnn_channels (int): Number of channels in CNN output.
            cnn_kernel_size (int): Kernel size for the CNN convolution.
            mlp_hidden (int): Hidden size in the MLP classifier.
            mlp_dropout (float): Dropout rate.
        """
        super().__init__()
        self.gnn = gnn_module

        # Disable the GNN's final classification layer (if it exists)
        gnn_output_dim = self.gnn.output_layer.in_features
        self.gnn.output_layer = nn.Identity()

        # Modular CNN and MLP components
        self.cnn = ProteinCNN(
            in_channels=gnn_output_dim,
            cnn_channels=cnn_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_dropout=cnn_dropout,
        )
        self.mlp = ProteinMLP(
            in_dim=cnn_channels,
            mlp_hidden_dim=mlp_hidden,
            mlp_dropout=mlp_dropout,
        )

    def forward(
        self,
        x_embeddings: torch.Tensor,
        x_properties: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: runs GNN → CNN → MLP.

        Args:
            x_embeddings (Tensor): Input embeddings [num_nodes, embedding_dim].
            x_properties (Tensor): Additional node properties [num_nodes, property_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            edge_features (Tensor, optional): Edge attributes [num_edges, edge_dim].

        Returns:
            Tensor: Node-wise logits [num_nodes, 2]
        """
        # Step 1: GNN → Node embeddings [N, D]
        h = self.gnn.extract_embeddings(
            x_embeddings, x_properties, edge_index, edge_features
        )

        # Step 2: CNN expects [B, D, N], apply 1D conv
        h = h.T.unsqueeze(0)  # [1, D, N]
        h = self.cnn(h)  # [1, C, N]

        # Step 3: MLP classification per residue
        h = h.squeeze(0).T  # [N, C]
        return self.mlp(h)  # [N, 2] logits
