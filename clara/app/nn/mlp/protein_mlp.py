import torch
import torch.nn as nn


class ProteinMLP(nn.Module):
    """
    A simple MLP head for residue-wise binary classification.
    Takes feature vectors (e.g., from CNN) and outputs class logits.
    """

    def __init__(self, in_dim: int, mlp_hidden_dim: int, mlp_dropout: float):
        """
        Initialize the MLP classifier.

        Args:
            in_dim (int): Input feature dimension (e.g., CNN output channels).
            mlp_hidden_dim (int): Size of hidden layer.
            mlp_dropout (float): Dropout rate.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(
                mlp_hidden_dim, 2
            ),  # Binary classification (binding / non-binding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape [seq_len, in_dim].

        Returns:
            Tensor: Output logits [seq_len, 2].
        """
        return self.mlp(x)
